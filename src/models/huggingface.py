"""Hugging Face 本地模型推論介面模組。

以 ``transformers.pipeline`` 建立本地 text-generation 驅動
（Qwen、Llama 等本地模型推論）：

- **Token 安全性**：環境存在 ``HF_TOKEN`` 時以 ``os.getenv("HF_TOKEN", None)``
  讀取並傳入 ``transformers.pipeline``，禁止寫死。
- **Chat Template 動態套用**：pipeline 之 tokenizer 提供
  ``chat_template`` 時，自動以 ``apply_chat_template`` 將 prompt 轉為模型
  對話模板格式（Instruct 模型必備）；套用失敗或回傳非字串／空白結果時
  回退原 prompt。
- **Prompt 裁剪**：text-generation pipeline 以 ``return_full_text=False``
  請求，預期僅回傳新生成文字；若輸出意外仍含輸入 prompt 前綴，則自動
  進行防禦性裁剪，僅回傳新生成文字。
- **第一層例外防禦**：``predict()`` 內部以 try...except 包覆全部推論
  路徑（權重載入失敗、OOM、輸出結構異常等），一律回傳
  ``"ERROR: HuggingFace Inference Failed - <error_message>"`` 哨兵字串，
  確保單次推論不拋出未捕獲例外。

本模組僅負責推論輸入／輸出，不執行答案解析、正則過濾或指標計算
（職責屬 ``src/evaluator.py``）。
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping, Sequence
from typing import Dict, Optional, Union

from transformers import pipeline
from transformers.pipelines import Pipeline  # type: ignore[attr-defined]

from src.models.base import BaseModelInterface

LOGGER = logging.getLogger(__name__)

#: 預設最大新生成 tokens 數（對齊 ``configs/eval_config.yaml``
#: generation.max_tokens）。
DEFAULT_MAX_NEW_TOKENS: int = 128

#: Hugging Face Hub token 之環境變數名稱。
HF_TOKEN_ENV_VAR: str = "HF_TOKEN"


class HuggingFacePipelineInterface(BaseModelInterface):
    """Hugging Face 本地模型推論介面（``transformers.pipeline``）。

    Example:
        >>> interface = HuggingFacePipelineInterface(  # doctest: +SKIP
        ...     model_name="qwen2.5-0.5b", model_id="Qwen/Qwen2.5-0.5B"
        ... )
        >>> interface.predict("...prompt...")  # doctest: +SKIP
        'The correct answer is (A)'
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        device: Optional[Union[int, str]] = None,
        torch_dtype: Optional[object] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        """初始化並建立 text-generation pipeline。

        Args:
            model_name: 模型名稱（供報告顯示）。
            model_id: HF Hub 模型識別碼（必填、非空）。
            device: 推論裝置；整數 GPU 編號或 ``"cuda"``／``"cpu"`` 等
                字串；``None`` 為自動選擇。
            torch_dtype: torch 資料型別（或其字串識別碼）；``None`` 為
                模型預設。以 ``object`` 註記因 torch dtype 非 Python
                內建型別。
            max_new_tokens: 單次生成之最大新 tokens 數；須為正整數。
            temperature: 取樣溫度；0.0 為貪婪解碼；須為有限非負數值。

        Raises:
            ValueError: 當 ``model_id`` 為空、``device`` 型別無效、
                ``max_new_tokens`` 非正整數或 ``temperature`` 非有限
                非負數值時。
        """
        super().__init__(model_name)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}.")
        self._model_id: str = model_id.strip()
        if device is not None and (
            isinstance(device, bool) or not isinstance(device, (int, str))
        ):
            raise ValueError(f"device must be an int, str, or None, got {device!r}.")
        self._device: Optional[Union[int, str]] = device
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError(
                "max_new_tokens must be an integer, "
                f"got {type(max_new_tokens).__name__}."
            )
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens!r}.")
        self._max_new_tokens: int = max_new_tokens
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0.0
        ):
            raise ValueError(
                f"temperature must be a finite number >= 0, got {temperature!r}."
            )
        self._temperature: float = float(temperature)

        model_kwargs: Optional[Dict[str, object]] = None
        if torch_dtype is not None:
            model_kwargs = {"torch_dtype": torch_dtype}
        hf_token: Optional[str] = os.getenv(HF_TOKEN_ENV_VAR) or None
        LOGGER.info(
            "Loading Hugging Face pipeline for model %s (device=%s, hf_token=%s).",
            model_id,
            device,
            "set" if hf_token else "not set",
        )
        self._pipeline: Pipeline = pipeline(
            "text-generation",
            model=model_id,
            device=device,
            model_kwargs=model_kwargs,
            token=hf_token,
        )

    def predict(self, prompt: str) -> str:
        """執行推論並回傳已裁剪 prompt 前綴之生成文字。

        Args:
            prompt: 完整組裝後的提示文字。

        Returns:
            模型新生成文字（已裁剪輸入 prompt 前綴）；pipeline 輸出為空
            時回傳空字串（由評測層判定 INVALID）；推論失敗時回傳
            ``"ERROR: HuggingFace Inference Failed - <error_message>"``
            哨兵字串。

        Raises:
            TypeError: 當 ``prompt`` 非字串時（程式錯誤，fail-fast）。
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}.")
        try:
            return self._generate(prompt)
        except Exception as exc:
            LOGGER.exception(
                "Hugging Face inference failed for model %s (%s).",
                self.model_name,
                self._model_id,
            )
            return f"ERROR: HuggingFace Inference Failed - {exc}"

    def _format_prompt(self, prompt: str) -> str:
        """若 tokenizer 具備 chat_template 則動態套用對話模板，否則維持原樣。

        Args:
            prompt: 提示文字。

        Returns:
            套用 chat template 後的 prompt；當 tokenizer 無
            ``chat_template``、``apply_chat_template`` 拋出例外、或結果
            非有效非空字串時，回傳原始 prompt（並記錄警告）。
        """
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted: object = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                LOGGER.warning(
                    "Chat template application failed for model %s; "
                    "falling back to the raw prompt.",
                    self._model_id,
                )
                return prompt
            if not isinstance(formatted, str) or not formatted.strip():
                LOGGER.warning(
                    "Chat template produced an unusable result for model %s; "
                    "falling back to the raw prompt.",
                    self._model_id,
                )
                return prompt
            return formatted
        return prompt

    def _generate(self, prompt: str) -> str:
        # 1. 轉為對應模型的 Chat Template（Instruct 模型必備）
        formatted_prompt = self._format_prompt(prompt)

        generation_kwargs: Dict[str, object] = {
            "max_new_tokens": self._max_new_tokens,
            "return_full_text": False,
            "clean_up_tokenization_spaces": False,
        }
        if self._temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self._temperature
        else:
            generation_kwargs["do_sample"] = False

        # 2. 傳入 formatted_prompt 推論
        result: object = self._pipeline(formatted_prompt, **generation_kwargs)

        if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
            LOGGER.warning(
                "Model %s returned an unexpected result container %s.",
                self._model_id,
                type(result).__name__,
            )
            return ""
        if not result:
            LOGGER.warning("Model %s returned no generation records.", self._model_id)
            return ""
        first: object = result[0]
        if not isinstance(first, Mapping):
            raise ValueError(
                f"Unexpected pipeline output record type: {type(first).__name__}."
            )
        text: object = first.get("generated_text", first.get("output_text"))
        if not isinstance(text, str):
            raise ValueError(
                "Pipeline output record has no string 'generated_text' "
                f"(got {type(text).__name__})."
            )

        # 3. 若有模型未遵循 return_full_text=False，以 formatted_prompt 進行裁切
        if text.startswith(formatted_prompt):
            return text[len(formatted_prompt) :].strip()

        return text.strip()
