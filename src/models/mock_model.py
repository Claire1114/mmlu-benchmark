"""Mock 模型介面模組。

供單元測試與離線驗證使用：零網路呼叫、零模型權重載入。支援兩種輸出
模式：

- ``fixed``：回傳固定字串（預設 ``"The correct answer is (A)"``）。
- ``random``：隨機擇取 A–D 之一並回傳 ``"The correct answer is (X)"``；
  可設定 ``seed`` 取得可重現的回歸測試輸出序列。
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from src.models.base import BaseModelInterface

#: 四個有效選項字母。
_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")

#: 預設固定答案（與 Prompt 模板標準格式對齊）。
DEFAULT_FIXED_ANSWER: str = "The correct answer is (A)"

#: 有效輸出模式。
_MODE_FIXED: str = "fixed"
_MODE_RANDOM: str = "random"


class MockModelInterface(BaseModelInterface):
    """Mock 模型介面：回傳固定字串或隨機選項格式文字。

    Attributes:
        mode: 輸出模式（``"fixed"`` 或 ``"random"``，正規化為小寫）。

    Example:
        >>> MockModelInterface().predict("any prompt")
        'The correct answer is (A)'
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        mode: str = _MODE_FIXED,
        fixed_answer: str = DEFAULT_FIXED_ANSWER,
        seed: Optional[int] = None,
    ) -> None:
        """初始化 Mock 模型。

        Args:
            model_name: 模型名稱（供報告顯示）。
            mode: 輸出模式，``"fixed"`` 或 ``"random"``（大小寫不敏感）。
            fixed_answer: fixed 模式回傳的字串。
            seed: random 模式隨機種子；相同種子產生相同輸出序列。

        Raises:
            ValueError: 當 mode 非有效值時。
        """
        normalized_mode: str = str(mode).strip().lower()
        if normalized_mode not in (_MODE_FIXED, _MODE_RANDOM):
            raise ValueError(
                f"mode must be one of {_MODE_FIXED!r} or {_MODE_RANDOM!r}, "
                f"got {mode!r}."
            )
        super().__init__(model_name)
        self._mode: str = normalized_mode
        self._fixed_answer: str = fixed_answer
        self._rng: random.Random = random.Random(seed)

    @property
    def mode(self) -> str:
        """目前輸出模式（已正規化為小寫）。"""
        return self._mode

    def predict(self, prompt: str) -> str:
        """回傳 Mock 輸出。

        Args:
            prompt: 提示文字（Mock 模型不解析內容，僅驗證型別）。

        Returns:
            fixed 模式回傳 ``fixed_answer``；random 模式回傳
            ``"The correct answer is (X)"``（X 為 A–D 隨機擇取）。

        Raises:
            TypeError: 當 ``prompt`` 非字串時。
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}.")
        if self._mode == _MODE_FIXED:
            return self._fixed_answer
        letter: str = self._rng.choice(_LETTERS)
        return f"The correct answer is ({letter})"
