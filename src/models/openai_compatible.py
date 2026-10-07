"""OpenAI SDK 相容 API 模型介面模組。

支援以 OpenAI 相容 Chat Completions 端點進行推論（Groq、Ollama（本地）、
OpenRouter 等）。安全性與健壯性規則：

- **API Key 安全性**：金鑰一律以 ``os.getenv(...)`` 自環境變數讀取，禁止
  寫死。本地端點（如 Ollama）於環境變數缺失時允許佔位字串（預設
  ``"ollama"``）；雲端 provider 於金鑰缺失時 fail-fast（``ValueError``）。
- **第一層例外防護**：對可重試錯誤（429 限流／5xx 伺服器錯誤／逾時／連線
  異常）以 ``tenacity`` 指數退避重試（最多 ``max_retries`` 次，第 n 次
  重試等待 ``backoff_factor ** (n - 1)`` 秒，對齊
  ``configs/eval_config.yaml`` 之 ``evaluation.retry``）；重試耗盡或遇
  不可重試錯誤（401／400／404 等）時，內部 catch 例外並回傳
  ``"ERROR: <ExceptionType>: <message>"`` 格式字串，確保單次推論不拋出
  未捕獲例外。
- SDK 層重試關閉（``max_retries=0``），確保 tenacity 為唯一重試層，避免
  隱藏性重試倍乘。
"""

from __future__ import annotations

import logging
import math
import os
from typing import Mapping, Optional

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    Retrying,
    RetryCallState,
    RetryError,
    before_log,
    retry_if_exception_type,
    stop_after_attempt,
)

from src.models.base import ERROR_PREFIX, BaseModelInterface

LOGGER = logging.getLogger(__name__)

#: Provider 預設表：驅動類型 → 預設端點／API key 環境變數／金鑰策略。
#: ``key_required`` 為 ``"true"`` 者必須由環境提供真實金鑰（缺失時於
#: 建構期拋出 ``ValueError``）；其餘（本地端點）於金鑰環境變數缺失時
#: 使用 ``key_placeholder`` 佔位字串。
PROVIDER_PRESETS: Mapping[str, Mapping[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env_var": "GROQ_API_KEY",
        "key_required": "true",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env_var": "OLLAMA_API_KEY",
        "key_placeholder": "ollama",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env_var": "OPENROUTER_API_KEY",
        "key_required": "true",
    },
    "openai_compatible": {
        "base_url": "",
        "api_key_env_var": "OPENAI_API_KEY",
        "key_placeholder": "openai-compatible",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env_var": "GEMINI_API_KEY",
    },
}

#: 可重試例外型別：429 限流、5xx 伺服器錯誤、連線／逾時異常
#: （``APITimeoutError`` 為 ``APIConnectionError`` 子類）。
_RETRYABLE_EXCEPTIONS: tuple[type[openai.OpenAIError], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

#: 內建預設值（對齊 ``configs/eval_config.yaml`` 之
#: ``evaluation.retry`` 與 generation 區塊）。
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 2.0
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 128
DEFAULT_TOP_P: float = 1.0


class OpenAICompatibleInterface(BaseModelInterface):
    """OpenAI SDK 相容 API 推論介面（Groq／Ollama／OpenRouter 等）。

    Attributes:
        model_id: API 端模型識別碼（如 ``llama-3.1-8b-instant``）。
        provider: 解析後的驅動類型（``groq``／``ollama``／``openrouter``／
            ``openai_compatible``）。
        base_url: 解析後的 API 端點。

    Example:
        >>> interface = OpenAICompatibleInterface(  # doctest: +SKIP
        ...     model_name="groq-llama3.1", model_id="llama-3.1-8b-instant"
        ... )
        >>> interface.predict("...prompt...")  # doctest: +SKIP
        'The correct answer is (A)'
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        provider: str = "groq",
        base_url: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
        client: Optional[OpenAI] = None,
    ) -> None:
        """建立 OpenAI 相容客戶端與 tenacity 重試策略。

        Args:
            model_name: 模型名稱（供報告顯示）。
            model_id: API 端模型識別碼（必填、非空）。
            provider: 驅動類型；必須為 ``PROVIDER_PRESETS`` 的鍵。
            base_url: API 端點；省略時使用 provider 預設值。
            api_key_env_var: API key 環境變數名稱；省略時使用 provider
                預設值。
            max_retries: 單次推論的最大重試次數（不含首次呼叫）；
                負值收攏為 0。
            backoff_factor: 指數退避係數；第 n 次重試等待
                ``backoff_factor ** (n - 1)`` 秒；不得為負。
            timeout_seconds: 單次請求逾時（秒）；必須為正數。
            temperature: 取樣溫度（0.0 為貪婪解碼、可重現）；須為有限數值。
            max_tokens: 單次生成之最大新 tokens 數；須為正整數（>= 1）。
            top_p: 核取樣機率門檻；須為有限數值。
            client: 自訂 ``OpenAI`` 客戶端（供單元測試注入，取代預設
                建構；提供時跳過端點／金鑰解析邏輯）。

        Raises:
            ValueError: 當 model_id 為空、provider 未知、base_url 缺失、
                雲端 provider 的 API key 環境變數缺失、backoff_factor
                為負、timeout_seconds 非正數、max_tokens 非正整數、
                或 temperature／top_p 非有限數值時。
        """
        super().__init__(model_name)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}.")
        self._model_id: str = model_id.strip()

        provider_key: str = str(provider).strip().lower()
        preset: Mapping[str, str] = PROVIDER_PRESETS.get(provider_key, {})
        if not preset:
            raise ValueError(
                f"Unknown provider {provider!r}; expected one of "
                f"{sorted(PROVIDER_PRESETS)}."
            )
        self._provider: str = provider_key

        resolved_base_url: str = (base_url or preset["base_url"]).strip()
        if not resolved_base_url:
            raise ValueError(
                f"Provider {provider_key!r} has no default endpoint; "
                "an explicit base_url must be provided."
            )
        self._base_url: str = resolved_base_url
        self._max_retries: int = max(0, int(max_retries))
        if backoff_factor < 0:
            raise ValueError(f"backoff_factor must be >= 0, got {backoff_factor!r}.")
        self._backoff_factor: float = float(backoff_factor)
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}.")
        self._timeout_seconds: float = float(timeout_seconds)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError(
                f"max_tokens must be an integer, got {type(max_tokens).__name__}."
            )
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}.")
        self._max_tokens: int = max_tokens
        for field, value in (("temperature", temperature), ("top_p", top_p)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{field} must be a finite number, got {type(value).__name__}."
                )
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite, got {value!r}.")
        self._temperature: float = float(temperature)
        self._top_p: float = float(top_p)

        if client is not None:
            self._client: OpenAI = client
            LOGGER.info(
                "Model %s uses an injected OpenAI client (%s).",
                self.model_name,
                self._model_id,
            )
        else:
            env_var: str = (api_key_env_var or preset["api_key_env_var"]).strip()
            key_from_env: str = os.getenv(env_var, "").strip()
            if key_from_env:
                api_key: str = key_from_env
            elif preset.get("key_required") == "true":
                raise ValueError(
                    f"API key environment variable {env_var!r} is not set. "
                    "Export a real key before running cloud inference; "
                    "keys must never be hardcoded in code or config files."
                )
            else:
                api_key = preset.get("key_placeholder", "no-key-required")
                LOGGER.info(
                    "Environment variable %s is not set; using placeholder "
                    "key %r for local endpoint %s.",
                    env_var,
                    api_key,
                    self._base_url,
                )
            self._client = OpenAI(
                base_url=resolved_base_url,
                api_key=api_key,
                timeout=self._timeout_seconds,
                max_retries=0,  # SDK 層重試關閉；tenacity 為唯一重試層
            )

        self._retrying: Retrying = Retrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=self._backoff_wait,
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            before_sleep=before_log(LOGGER, logging.WARNING),
            reraise=False,
        )
        LOGGER.debug(
            "OpenAICompatibleInterface ready: model=%s provider=%s "
            "base_url=%s max_retries=%d backoff_factor=%s timeout=%ss",
            self._model_id,
            self._provider,
            self._base_url,
            self._max_retries,
            self._backoff_factor,
            self._timeout_seconds,
        )

    @property
    def model_id(self) -> str:
        """API 端模型識別碼。"""
        return self._model_id

    @property
    def provider(self) -> str:
        """解析後的驅動類型。"""
        return self._provider

    @property
    def base_url(self) -> str:
        """解析後的 API 端點。"""
        return self._base_url

    def predict(self, prompt: str) -> str:
        """呼叫 OpenAI 相容 Chat Completions API 並回傳原始輸出。

        第一層防護流程：可重試例外（429／5xx／逾時／連線異常）以
        指數退避重試至多 ``max_retries`` 次；重試耗盡或遇不可重試
        例外時，內部 catch 並回傳 ``ERROR: <ExceptionType>:
        <message>`` 哨兵字串。

        Args:
            prompt: 完整組裝後的提示文字。

        Returns:
            模型原始輸出文字；回應無 choices 或內容為空時回傳空
            字串（由評測層判定 INVALID）。

        Raises:
            TypeError: 當 ``prompt`` 非字串時。
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}.")
        try:
            raw_output: str = self._retrying(self._invoke_chat, prompt)
        except RetryError as exc:
            return self._build_error_output(exc, exhausted=True)
        except Exception as exc:
            return self._build_error_output(exc, exhausted=False)
        return raw_output

    def _backoff_wait(self, retry_state: RetryCallState) -> float:
        """指數退避等待：第 n 次重試等待 ``backoff_factor ** (n - 1)`` 秒。

        Args:
            retry_state: tenacity 重試狀態；``attempt_number`` 為已失敗
                的嘗試次數（1 起算）。

        Returns:
            等待秒數；對齊 ``evaluation.retry.backoff_factor`` 語意
            （預設 2.0 → 1s／2s／4s）。
        """
        attempt: int = max(1, int(retry_state.attempt_number))
        return float(self._backoff_factor ** (attempt - 1))

    def _invoke_chat(self, prompt: str) -> str:
        """送出單一 Chat Completions 請求。

        Args:
            prompt: 提示文字（以單則 user 訊息送出）。

        Returns:
            第一則 choice 的內容文字（空／None 時回傳空字串）。

        Raises:
            openai.OpenAIError: API／傳輸層例外；可重試型別由
                tenacity 捕捉重試，不可重試型別由 ``predict`` 末梢
                防禦兜底。
        """
        completion: ChatCompletion = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
            timeout=self._timeout_seconds,
        )
        if not completion.choices:
            LOGGER.warning(
                "Model %s returned no choices; treating output as empty.",
                self._model_id,
            )
            return ""
        content: Optional[str] = completion.choices[0].message.content
        if content is None:
            LOGGER.warning(
                "Model %s returned empty content; treating output as empty.",
                self._model_id,
            )
            return ""
        return content

    def _build_error_output(self, exc: BaseException, exhausted: bool) -> str:
        """組裝 ``ERROR:`` 哨兵字串並記錄失敗原因。

        Args:
            exc: 最終例外；``RetryError`` 會展開為底層原因以呈現更
                精確的錯誤訊息。
            exhausted: 是否為重試耗盡後之失敗（相對不可重試直接
                失敗）。

        Returns:
            ``ERROR: <ExceptionType>: <message>`` 格式字串。
        """
        cause: BaseException = exc
        if isinstance(exc, RetryError) and exc.last_attempt is not None:
            underlying: Optional[BaseException] = exc.last_attempt.exception()
            if underlying is not None:
                cause = underlying
        if exhausted:
            LOGGER.error(
                "Model %s (%s) still failed after %d retries: %s",
                self.model_name,
                self._model_id,
                self._max_retries,
                cause,
            )
        else:
            LOGGER.error(
                "Model %s (%s) inference failed (non-retryable): %s",
                self.model_name,
                self._model_id,
                cause,
            )
        return f"{ERROR_PREFIX}{type(cause).__name__}: {cause}"
