"""模型介面註冊與設定工廠模組。

- ``ModelProtocol``：結構化協議（``typing.Protocol``）；暴露 ``model_name``
  屬性與 ``predict(prompt: str) -> str`` 方法者，即結構同構於模型推論介面。
- ``build_model_interface``：由單一 ``models[].`` 設定區塊與全域
  ``evaluation.`` 設定區塊建構具體模型介面（執行參數單一來源為
  ``configs/eval_config.yaml``）。

Step 4 支援類型：``mock``、``openai_compatible``、``groq``、``ollama``、
``openrouter``、``huggingface``／``hf_pipeline``（HF 本地 pipeline 驅動）；
``gemini`` 排程於 Step 4 後續階段，現行拋出 ``NotImplementedError``。
"""

from __future__ import annotations

import logging
import math
from typing import Mapping, Optional, Protocol, Union, runtime_checkable

from src.models.base import BaseModelInterface
from src.models.huggingface import (
    DEFAULT_MAX_NEW_TOKENS,
    HuggingFacePipelineInterface,
)
from src.models.mock_model import MockModelInterface
from src.models.openai_compatible import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOP_P,
    OpenAICompatibleInterface,
)

LOGGER = logging.getLogger(__name__)

#: Step 4 支援的模型類型。
SUPPORTED_TYPES: frozenset[str] = frozenset(
    {
        "mock",
        "openai_compatible",
        "groq",
        "ollama",
        "openrouter",
        "huggingface",
        "hf_pipeline",
    }
)

#: 排程於 Step 4 後續階段的模型類型（Gemini 雲端驅動）。
_LATER_PHASE_TYPES: frozenset[str] = frozenset({"gemini"})


@runtime_checkable
class ModelProtocol(Protocol):
    """模型推論介面結構化協議。

    類別無需顯式繼承本協議；只要暴露 ``model_name`` 屬性與
    ``predict(prompt: str) -> str`` 方法，``isinstance`` 檢查
    （runtime_checkable）即可通過。

    Note:
        runtime_checkable 之 ``isinstance`` 僅驗證方法／屬性存在，
        不驗證簽名型別（Python 語言限制）。
    """

    model_name: str

    def predict(self, prompt: str) -> str:
        """執行單題推論，回傳模型原始輸出文字。"""
        ...


def _require_str(config: Mapping[str, object], field: str) -> str:
    """自設定區塊讀取必要非空字串。

    Args:
        config: 設定區塊。
        field: 欄位名稱（用於錯誤訊息）。

    Returns:
        去空白後的字串值。

    Raises:
        ValueError: 當欄位缺失、非字串或為空／純空白時。
    """
    value: object = config.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"models[].{field} is required and must be a non-empty string, "
            f"got {value!r}."
        )
    return value.strip()


def _optional_str(config: Mapping[str, object], field: str) -> Optional[str]:
    """讀取選填字串；缺失、空白或非字串時回傳 ``None``。

    Args:
        config: 設定區塊。
        field: 欄位名稱。

    Returns:
        去空白後的字串；缺失／空白時 ``None``；非字串時記錄警告並
        回傳 ``None``。
    """
    value: object = config.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        LOGGER.warning(
            "models[].%s should be a string (got %s); ignoring it.",
            field,
            type(value).__name__,
        )
        return None
    text: str = value.strip()
    return text or None


def _as_mapping_block(value: object, field: str) -> Mapping[str, object]:
    """讀取選填 mapping 區塊；缺失或非法時回傳空 mapping。

    Args:
        value: 設定區塊值。
        field: 區塊名稱（用於日誌訊息）。

    Returns:
        原 mapping；``None`` 或非 mapping 型別時回傳空 mapping
        （後者同時記錄警告）。
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    LOGGER.warning(
        "Config block %r should be a mapping (got %s); ignoring it.",
        field,
        type(value).__name__,
    )
    return {}


def _coerce_int(value: object, default: int, field: str) -> int:
    """將設定值轉換為整數。

    缺失（``None``）回傳預設值；bool、無法解析字串與其餘非數值型別
    記錄警告並回傳預設值；負值記錄警告並回傳預設值。

    Args:
        value: 原始設定值。
        default: 缺失／非法時的預設值。
        field: 欄位名稱（用於日誌訊息）。

    Returns:
        非負整數。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        LOGGER.warning(
            "Config %s value %r is not a number; using default %d.",
            field,
            value,
            default,
        )
        return default
    if isinstance(value, int):
        number: int = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value.strip())
        except ValueError:
            LOGGER.warning(
                "Config %s value %r is not an integer; using default %d.",
                field,
                value,
                default,
            )
            return default
    else:
        LOGGER.warning(
            "Config %s has unexpected type %s; using default %d.",
            field,
            type(value).__name__,
            default,
        )
        return default
    if number < 0:
        LOGGER.warning(
            "Config %s value %d is negative; using default %d.",
            field,
            number,
            default,
        )
        return default
    return number


def _coerce_float(value: object, default: float, field: str) -> float:
    """將設定值轉換為有限浮點數。

    缺失（``None``）回傳預設值；bool、無法解析字串、非數值型別與
    非有限值（NaN／Inf）記錄警告並回傳預設值。

    Args:
        value: 原始設定值。
        default: 缺失／非法時的預設值。
        field: 欄位名稱（用於日誌訊息）。

    Returns:
        有限浮點數。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        LOGGER.warning(
            "Config %s value %r is not a number; using default %s.",
            field,
            value,
            default,
        )
        return default
    if isinstance(value, (int, float)):
        number: float = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            LOGGER.warning(
                "Config %s value %r is not a number; using default %s.",
                field,
                value,
                default,
            )
            return default
    else:
        LOGGER.warning(
            "Config %s has unexpected type %s; using default %s.",
            field,
            type(value).__name__,
            default,
        )
        return default
    if not math.isfinite(number):
        LOGGER.warning(
            "Config %s value %s is not finite; using default %s.",
            field,
            number,
            default,
        )
        return default
    return number


def build_model_interface(
    model_cfg: Mapping[str, object],
    evaluation_cfg: Optional[Mapping[str, object]] = None,
) -> BaseModelInterface:
    """由設定建構模型介面。

    Args:
        model_cfg: 單一模型設定區塊（``models[]`` 元素），必含
            ``type``／``name``；OpenAI 相容類型另需 ``model_id``。
        evaluation_cfg: 全域 ``evaluation.`` 設定區塊；提供
            ``retry``（max_retries／backoff_factor／timeout_seconds）；
            省略時使用內建預設值。

    Returns:
        具體模型介面實體（``BaseModelInterface`` 子類）。

    Raises:
        ValueError: 當 ``type``／``name``／``model_id`` 缺失或非法、
            或雲端 provider 之 API key 環境變數缺失時。
        NotImplementedError: 當 type 為 Step 4 後續階段排程驅動
            （``gemini``）時。
    """
    if not isinstance(model_cfg, Mapping):
        raise ValueError(
            f"model_cfg must be a mapping, got {type(model_cfg).__name__}."
        )
    if evaluation_cfg is None:
        evaluation_cfg = {}

    raw_type: object = model_cfg.get("type")
    model_type: str = str(raw_type).strip().lower() if raw_type is not None else ""
    if not model_type:
        raise ValueError("models[].type is required.")
    if model_type in _LATER_PHASE_TYPES:
        raise NotImplementedError(
            f"Model type {model_type!r} is not implemented in Step 4 "
            "Phase 1; planned for a later phase."
        )
    if model_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported model type {model_type!r}; supported: "
            f"{sorted(SUPPORTED_TYPES)}."
        )

    name: str = _require_str(model_cfg, "name")

    if model_type == "mock":
        raw_mode: object = model_cfg.get("mock_mode")
        mode: str = str(raw_mode).strip().lower() if raw_mode is not None else "fixed"
        raw_seed: object = model_cfg.get("seed")
        if raw_seed is None:
            seed: Optional[int] = None
        elif isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            seed = raw_seed
        else:
            LOGGER.warning(
                "models[].seed value %r is invalid; using unseeded randomness.",
                raw_seed,
            )
            seed = None
        return MockModelInterface(model_name=name, mode=mode, seed=seed)

    model_id: str = _require_str(model_cfg, "model_id")
    generation: Mapping[str, object] = _as_mapping_block(
        model_cfg.get("generation"), "models[].generation"
    )

    if model_type in ("huggingface", "hf_pipeline"):
        raw_device: object = model_cfg.get("device")
        if isinstance(raw_device, int) and not isinstance(raw_device, bool):
            device: Optional[Union[int, str]] = raw_device
        elif isinstance(raw_device, str):
            device = raw_device
        elif raw_device is None:
            device = None
        else:
            LOGGER.warning(
                "models[].device value %r is invalid; using the default device.",
                raw_device,
            )
            device = None
        return HuggingFacePipelineInterface(
            model_name=name,
            model_id=model_id,
            device=device,
            torch_dtype=model_cfg.get("torch_dtype"),
            max_new_tokens=_coerce_int(
                model_cfg.get("max_new_tokens"),
                DEFAULT_MAX_NEW_TOKENS,
                "models[].max_new_tokens",
            ),
            temperature=_coerce_float(
                generation.get("temperature"),
                0.0,
                "models[].generation.temperature",
            ),
        )

    retry: Mapping[str, object] = _as_mapping_block(
        evaluation_cfg.get("retry"), "evaluation.retry"
    )
    return OpenAICompatibleInterface(
        model_name=name,
        model_id=model_id,
        provider=model_type,
        base_url=_optional_str(model_cfg, "base_url"),
        api_key_env_var=_optional_str(model_cfg, "api_key_env_var"),
        max_retries=_coerce_int(
            retry.get("max_retries"),
            DEFAULT_MAX_RETRIES,
            "evaluation.retry.max_retries",
        ),
        backoff_factor=_coerce_float(
            retry.get("backoff_factor"),
            DEFAULT_BACKOFF_FACTOR,
            "evaluation.retry.backoff_factor",
        ),
        timeout_seconds=_coerce_float(
            retry.get("timeout_seconds"),
            DEFAULT_TIMEOUT_SECONDS,
            "evaluation.retry.timeout_seconds",
        ),
        temperature=_coerce_float(
            generation.get("temperature"),
            DEFAULT_TEMPERATURE,
            "models[].generation.temperature",
        ),
        max_tokens=_coerce_int(
            generation.get("max_tokens"),
            DEFAULT_MAX_TOKENS,
            "models[].generation.max_tokens",
        ),
        top_p=_coerce_float(
            generation.get("top_p"),
            DEFAULT_TOP_P,
            "models[].generation.top_p",
        ),
    )
