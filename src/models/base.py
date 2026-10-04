"""模型推論介面基礎模組。

定義所有模型驅動（mock／OpenAI 相容 API 等）必須實作的統一契約：

- ``predict(prompt) -> str``：單題推論，回傳模型原始輸出文字；API／傳輸層
  失敗由驅動內建第一層防護（重試 ＋ ``ERROR: <error>`` 哨兵字串）處理，
  單次推論不拋出未捕獲例外。
- ``predict_batch(samples)``：循序批次推論，組裝 Runner 預測紀錄
  （``question_id`` / ``raw_output`` / ``latency``）。

本模組僅負責推論輸入／輸出，不執行答案解析、正則過濾或指標計算
（職責屬 ``src/evaluator.py``）。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

#: 錯誤輸出哨兵前綴：單次推論防禦耗盡後一律回傳 ``ERROR: <error_message>``。
#: 評測層將此輸出視為 INVALID（判定答錯），確保管線不中斷。
ERROR_PREFIX: str = "ERROR: "


def _coerce_question_id(value: object) -> str:
    """安全轉換 question_id 為字串；``None`` 映射為空字串。"""
    if value is None:
        return ""
    return str(value)


def _coerce_prompt(value: object) -> Optional[str]:
    """安全轉換 ``formatted_prompt`` 為可用提示字串。

    ``None`` 或純空白字串映射為 ``None``（呼叫端將產出錯誤紀錄並跳過模型
    呼叫）；非字串型別以 ``str()`` 強制轉換並記錄警告。
    """
    if value is None:
        return None
    if isinstance(value, str):
        prompt: str = value
    else:
        LOGGER.warning(
            "Non-string formatted_prompt of type %s; coercing to str.",
            type(value).__name__,
        )
        prompt = str(value)
    if not prompt.strip():
        return None
    return prompt


class BaseModelInterface(ABC):
    """模型推論介面抽象基底。

    子類須實作 :meth:`predict`；:meth:`predict_batch` 提供預設循序批次
    實作（不使用執行緒池，防止 VRAM OOM 或雲端限流放大）。

    Attributes:
        model_name: 模型名稱（來自設定檔 ``models[].name``，供報告顯示）。
    """

    def __init__(self, model_name: str) -> None:
        """驗證並儲存模型名稱。

        Args:
            model_name: 非空模型名稱字串。

        Raises:
            ValueError: 當 model_name 非字串或為空／純空白時。
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"model_name must be a non-empty string, got {model_name!r}."
            )
        self.model_name: str = model_name

    @abstractmethod
    def predict(self, prompt: str) -> str:
        """執行單題推論。

        Args:
            prompt: 完整組裝後的提示文字。

        Returns:
            模型原始輸出文字。API／傳輸層失敗時由第一層防護回傳
            ``ERROR: <error>`` 哨兵字串（子類必須保證外部呼叫不拋出
            未捕獲例外）。

        Raises:
            TypeError: 當 ``prompt`` 非字串時（程式錯誤，fail-fast）。
        """

    def predict_batch(
        self,
        samples: Sequence[Mapping[str, object]],
    ) -> List[Dict[str, object]]:
        """循序批次推論並組裝 Runner 預測紀錄。

        每筆樣本僅讀取 ``question_id`` 與 ``formatted_prompt``；
        ``target_letter`` 等其餘欄位不讀取、不出現在輸出。缺失有效
        提示的紀錄不呼叫模型，直接產出錯誤紀錄；``predict`` 拋出的
        意外例外由末梢防禦捕捉並轉為 ``ERROR:`` 哨兵字串。

        Args:
            samples: ``MMLUDatasetLoader.get_samples()`` 輸出（mapping
                清單）；模型層依契約保證不讀取目標欄位。

        Returns:
            預測紀錄清單（與 ``samples`` 等長且同序），每筆固定含三鍵：
            ``question_id``（str）、``raw_output``（str）、
            ``latency``（float 秒，含重試耗時；錯誤紀錄為 ``0.0``）。
        """
        predictions: List[Dict[str, object]] = []
        for index, record in enumerate(samples):
            if not isinstance(record, Mapping):
                LOGGER.warning(
                    "Sample #%d is not a mapping (got %s); emitting an error record.",
                    index,
                    type(record).__name__,
                )
                predictions.append(
                    self._error_record(
                        "", f"malformed sample record: {type(record).__name__}"
                    )
                )
                continue

            question_id: str = _coerce_question_id(record.get("question_id"))
            prompt: Optional[str] = _coerce_prompt(record.get("formatted_prompt"))
            if prompt is None:
                if record.get("formatted_prompt") is None:
                    reason: str = "missing 'formatted_prompt' in sample"
                else:
                    reason = "empty 'formatted_prompt' in sample"
                LOGGER.warning(
                    "Sample %r has no usable prompt (%s); skipping the model call.",
                    question_id or f"index {index}",
                    reason,
                )
                predictions.append(self._error_record(question_id, reason))
                continue

            start: float = time.perf_counter()
            try:
                raw_output: str = self.predict(prompt)
            except Exception as exc:
                LOGGER.exception(
                    "Model %s predict() unexpectedly failed for sample %r "
                    "(last line of defense).",
                    self.model_name,
                    question_id,
                )
                raw_output = f"{ERROR_PREFIX}{type(exc).__name__}: {exc}"
            latency: float = time.perf_counter() - start
            predictions.append(
                {
                    "question_id": question_id,
                    "raw_output": raw_output,
                    "latency": latency,
                }
            )
        return predictions

    @staticmethod
    def _error_record(question_id: str, reason: str) -> Dict[str, object]:
        """組裝延遲為 0.0 的錯誤預測紀錄。"""
        return {
            "question_id": question_id,
            "raw_output": f"{ERROR_PREFIX}{reason}",
            "latency": 0.0,
        }
