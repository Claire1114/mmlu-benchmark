"""模型推論介面模組（Step 4 第一階段）。

統一匯出模型驅動介面與工廠：

- ``BaseModelInterface``：抽象基底（``predict`` 單題推論 ＋
  ``predict_batch`` 循序批次與 Runner 預測紀錄組裝）。
- ``MockModelInterface``：離線 Mock 模型（固定／隨機選項格式）。
- ``OpenAICompatibleInterface``：OpenAI SDK 相容 API（Groq／Ollama／
  OpenRouter 等），內建指數退避重試與 ``ERROR:`` 哨兵兜底（第一層
  例外防護）。
- ``ModelProtocol``：結構化協議；``build_model_interface``：設定工廠；
  ``SUPPORTED_TYPES``／``PROVIDER_PRESETS``：註冊表常數。
"""

from src.models.base import ERROR_PREFIX, BaseModelInterface
from src.models.interfaces import (
    ModelProtocol,
    SUPPORTED_TYPES,
    build_model_interface,
)
from src.models.mock_model import MockModelInterface
from src.models.openai_compatible import (
    PROVIDER_PRESETS,
    OpenAICompatibleInterface,
)

__all__ = [
    "ERROR_PREFIX",
    "BaseModelInterface",
    "MockModelInterface",
    "OpenAICompatibleInterface",
    "ModelProtocol",
    "PROVIDER_PRESETS",
    "SUPPORTED_TYPES",
    "build_model_interface",
]
