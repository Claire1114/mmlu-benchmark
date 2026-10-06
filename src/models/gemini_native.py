from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models.base import ERROR_PREFIX, BaseModelInterface

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 2.0


class GeminiNativeInterface(BaseModelInterface):
    """Google Gemini 原生 REST API 驅動介面。"""

    def __init__(
        self,
        model_name: str,
        model_id: str,
        api_key_env_var: Optional[str] = "GEMINI_API_KEY",
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = 0.0,
        max_tokens: int = 128,
        top_p: float = 1.0,
    ) -> None:
        super().__init__(model_name)
        self.model_id = model_id.strip()
        env_var = api_key_env_var or "GEMINI_API_KEY"
        self.api_key = os.environ.get(env_var, "").strip()
        if not self.api_key:
            raise ValueError(f"Environment variable {env_var!r} is missing or empty.")

        self.timeout = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max(0, max_retries)
        self.backoff_factor = backoff_factor

        # 原生 generateContent 端點
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent?key={self.api_key}"
        )

    def _call_api(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": self.top_p,
            },
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.endpoint, headers=headers, json=payload)

        # 429 或 5xx 丟出例外以觸發 tenacity 重試
        if resp.status_code in (429, 500, 502, 503, 504):
            resp.raise_for_status()

        if resp.status_code != 200:
            return f"{ERROR_PREFIX}HTTP {resp.status_code} - {resp.text}"

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            LOGGER.warning("Gemini returned empty or unexpected structure: %s", data)
            return ""

    def predict(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}.")

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=self.backoff_factor, min=1),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        )
        def _execute() -> str:
            return self._call_api(prompt)

        try:
            return _execute()
        except Exception as exc:
            LOGGER.error("Gemini inference failed after retries: %s", exc)
            return f"{ERROR_PREFIX}{type(exc).__name__} - {exc}"
