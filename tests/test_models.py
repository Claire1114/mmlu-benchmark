"""模型介面（``src/models/``）離線單元測試。

所有案例零網路呼叫：OpenAI 相容介面以注入 mock client
（``MagicMock`` ＋ ``SimpleNamespace`` 回應）驗證；tenacity sleep
以 ``MagicMock`` 取代以消除真實退避延遲；不觸發模型權重下載。
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from src.evaluator import Evaluator
from src.models import (
    ERROR_PREFIX,
    BaseModelInterface,
    ModelProtocol,
    MockModelInterface,
    OpenAICompatibleInterface,
    SUPPORTED_TYPES,
    build_model_interface,
)

_CHAT_URL: str = "http://api.test.invalid/v1/chat/completions"


def _http_request() -> httpx.Request:
    """建立 mock 用 HTTP 請求物件。"""
    return httpx.Request("POST", _CHAT_URL)


def _http_response(status_code: int) -> httpx.Response:
    """建立 mock 用 HTTP 回應物件。"""
    return httpx.Response(status_code, request=_http_request())


def _rate_limit_error() -> openai.RateLimitError:
    """建立 429 RateLimitError 實體（純記憶體，零網路）。"""
    return openai.RateLimitError(
        message="429 rate limit",
        response=_http_response(429),
        body=None,
    )


def _server_error() -> openai.InternalServerError:
    """建立 500 InternalServerError 實體。"""
    return openai.InternalServerError(
        message="500 internal error",
        response=_http_response(500),
        body=None,
    )


def _server_error_5xx(status_code: int) -> openai.InternalServerError:
    """建立指定 5xx 狀態碼之 InternalServerError 實體（純記憶體，零網路）。

    OpenAI SDK（1.x）將所有 >= 500 狀態碼統一映射為
    ``InternalServerError``；此 helper 模擬 502/503/504 經 SDK 映射後
    之例外型別，用於鎖定重試分類行為。
    """
    return openai.InternalServerError(
        message=f"{status_code} server/gateway error",
        response=httpx.Response(status_code, request=_http_request()),
        body=None,
    )


def _auth_error() -> openai.AuthenticationError:
    """建立 401 AuthenticationError 實體（不可重試）。"""
    return openai.AuthenticationError(
        message="401 unauthorized",
        response=_http_response(401),
        body=None,
    )


def _bad_request_error() -> openai.BadRequestError:
    """建立 400 BadRequestError 實體（不可重試）。"""
    return openai.BadRequestError(
        message="400 bad request",
        response=_http_response(400),
        body=None,
    )


def _timeout_error() -> openai.APITimeoutError:
    """建立 APITimeoutError 實體（可重試；APIConnectionError 子類）。"""
    return openai.APITimeoutError(request=_http_request())


def _completion(content: Any = "The correct answer is (A)") -> SimpleNamespace:
    """建立 Chat Completions 回應物件（僅屬性存取，零網路）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _empty_choices() -> SimpleNamespace:
    """建立無 choices 的回應物件。"""
    return SimpleNamespace(choices=[])


def _is_exception(value: Any) -> bool:
    """判斷物件是否為例外實體（``BaseException`` 實例）。"""
    return isinstance(value, BaseException)


def _mock_client(*side_effects: Any) -> MagicMock:
    """建立 mock OpenAI client。

    單一非例外項目：以 ``return_value`` 固定回傳該回應；
    單一例外項目：每次呼叫皆拋出；多項目清單：依序拋出／回傳。
    """
    client: MagicMock = MagicMock()
    create = client.chat.completions.create
    if len(side_effects) == 1 and not _is_exception(side_effects[0]):
        create.return_value = side_effects[0]
    elif len(side_effects) == 1:
        create.side_effect = side_effects[0]
    else:
        create.side_effect = list(side_effects)
    return client


def _zero_sleep(iface: OpenAICompatibleInterface) -> MagicMock:
    """以 ``MagicMock`` 取代 tenacity sleep（零真實延遲）並回傳。"""
    sleep_mock: MagicMock = MagicMock()
    iface._retrying.sleep = sleep_mock
    return sleep_mock


def _samples(n: int = 3) -> List[Dict[str, object]]:
    """建立 n 筆對齊 ``get_samples()`` 結構的評測樣本。"""
    return [
        {
            "question_id": f"q{i:03d}",
            "formatted_prompt": f"prompt-{i}",
            "target_letter": "A",
            "subject": "philosophy",
            "category": "Humanities",
        }
        for i in range(n)
    ]


class _StubModel(BaseModelInterface):
    """記錄提示呼叫並回傳固定文字（測試基底類行為用）。"""

    def __init__(self, model_name: str = "stub-model") -> None:
        super().__init__(model_name)
        self.prompts: List[str] = []

    def predict(self, prompt: str) -> str:
        """回傳固定文字並記錄收到的提示。"""
        self.prompts.append(prompt)
        return "stub-output"


class _ExplodingModel(BaseModelInterface):
    """always 拋出意外例外（測試末梢防禦用）。"""

    def __init__(self) -> None:
        super().__init__("exploding-model")

    def predict(self, prompt: str) -> str:
        """拋出 RuntimeError。"""
        raise RuntimeError("boom")


class TestBaseInterface:
    """BaseModelInterface：抽象性與 predict_batch 行為。"""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseModelInterface("name")

    def test_invalid_model_name_raises(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            _StubModel(model_name="   ")

    def test_predict_batch_preserves_order_and_keys(self) -> None:
        model = _StubModel()
        records = model.predict_batch(_samples(3))
        assert [r["question_id"] for r in records] == ["q000", "q001", "q002"]
        for record in records:
            assert set(record) == {"question_id", "raw_output", "latency"}
            assert record["raw_output"] == "stub-output"
            assert isinstance(record["latency"], float)
            assert record["latency"] >= 0.0

    def test_predict_receives_formatted_prompt_only(self) -> None:
        model = _StubModel()
        model.predict_batch(_samples(2))
        assert model.prompts == ["prompt-0", "prompt-1"]

    def test_target_letter_never_leaks_into_output(self) -> None:
        model = _StubModel()
        records = model.predict_batch(_samples(1))
        assert "target_letter" not in records[0]

    def test_non_string_question_id_coerced_to_str(self) -> None:
        model = _StubModel()
        records = model.predict_batch([{"question_id": 42, "formatted_prompt": "p"}])
        assert records[0]["question_id"] == "42"

    def test_missing_prompt_yields_error_record_without_call(self) -> None:
        model = _StubModel()
        records = model.predict_batch([{"question_id": "q1"}])
        assert records[0]["raw_output"].startswith(ERROR_PREFIX)
        assert "formatted_prompt" in records[0]["raw_output"]
        assert records[0]["latency"] == 0.0
        assert model.prompts == []

    def test_empty_prompt_yields_error_record(self) -> None:
        model = _StubModel()
        records = model.predict_batch(
            [{"question_id": "q1", "formatted_prompt": "   "}]
        )
        assert records[0]["raw_output"].startswith(ERROR_PREFIX)
        assert model.prompts == []

    def test_non_mapping_sample_yields_error_record(self) -> None:
        model = _StubModel()
        records = model.predict_batch([42])  # type: ignore[arg-type]
        assert records[0]["question_id"] == ""
        assert records[0]["raw_output"].startswith(ERROR_PREFIX)
        assert records[0]["latency"] == 0.0

    def test_unexpected_exception_converted_to_error_sentinel(self) -> None:
        model = _ExplodingModel()
        records = model.predict_batch(_samples(1))
        raw = records[0]["raw_output"]
        assert raw.startswith(f"{ERROR_PREFIX}RuntimeError")
        assert "boom" in raw


class TestMockModel:
    """MockModelInterface：固定／隨機模式與可重現性。"""

    def test_fixed_mode_default_answer(self) -> None:
        model = MockModelInterface()
        assert model.predict("any prompt") == "The correct answer is (A)"

    def test_fixed_mode_custom_answer(self) -> None:
        model = MockModelInterface(fixed_answer="The correct answer is (C)")
        assert model.predict("p") == "The correct answer is (C)"

    def test_random_mode_always_valid_format(self) -> None:
        model = MockModelInterface(mode="random", seed=123)
        for _ in range(50):
            assert re.fullmatch(r"The correct answer is \([A-D]\)", model.predict("p"))

    def test_random_mode_covers_multiple_letters(self) -> None:
        model = MockModelInterface(mode="random", seed=99)
        seen = {model.predict("p") for _ in range(200)}
        # 200 次抽樣僅出同一選項之機率 ≈ 4 * 0.25^200 ≈ 0
        assert len(seen) >= 2

    def test_seeded_random_is_reproducible(self) -> None:
        m1 = MockModelInterface(mode="random", seed=42)
        first = [m1.predict("p") for _ in range(20)]
        m2 = MockModelInterface(mode="random", seed=42)
        second = [m2.predict("p") for _ in range(20)]
        assert first == second
        assert len(set(first)) >= 2

    def test_mode_is_case_insensitive(self) -> None:
        model = MockModelInterface(mode="RANDOM", seed=1)
        assert model.mode == "random"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            MockModelInterface(mode="chaos")

    def test_non_str_prompt_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            MockModelInterface().predict(123)  # type: ignore[arg-type]

    def test_model_name_default_and_custom(self) -> None:
        assert MockModelInterface().model_name == "mock-model"
        assert MockModelInterface("x").model_name == "x"

    def test_implements_model_protocol(self) -> None:
        assert isinstance(MockModelInterface(), ModelProtocol)


class TestOpenAICompatibleConstruction:
    """API Key 安全性、端點解析與參數校驗。"""

    def test_groq_preset_reads_key_from_env(self) -> None:
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test-key"}),
            patch("src.models.openai_compatible.OpenAI") as openai_cls,
        ):
            iface = OpenAICompatibleInterface(
                model_name="groq", model_id="llama-3.1-8b-instant"
            )
        kwargs = openai_cls.call_args.kwargs
        assert kwargs["api_key"] == "gsk-test-key"
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert kwargs["max_retries"] == 0  # SDK 層重試關閉
        assert kwargs["timeout"] == 30.0
        assert iface.provider == "groq"
        assert iface.model_id == "llama-3.1-8b-instant"
        assert iface.base_url == "https://api.groq.com/openai/v1"

    def test_missing_cloud_key_fails_fast(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                OpenAICompatibleInterface(model_name="groq", model_id="m")

    def test_ollama_local_uses_placeholder_key(self) -> None:
        with (
            patch.dict(os.environ, {"OLLAMA_API_KEY": ""}),
            patch("src.models.openai_compatible.OpenAI") as openai_cls,
        ):
            OpenAICompatibleInterface(
                model_name="ollama-local", model_id="llama3.2", provider="ollama"
            )
        kwargs = openai_cls.call_args.kwargs
        assert kwargs["api_key"] == "ollama"
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    def test_explicit_base_url_and_env_var_override(self) -> None:
        with (
            patch.dict(os.environ, {"MY_CUSTOM_KEY": "ck-1"}),
            patch("src.models.openai_compatible.OpenAI") as openai_cls,
        ):
            OpenAICompatibleInterface(
                model_name="custom",
                model_id="m",
                provider="openai_compatible",
                base_url="http://10.0.0.5:8080/v1",
                api_key_env_var="MY_CUSTOM_KEY",
            )
        kwargs = openai_cls.call_args.kwargs
        assert kwargs["api_key"] == "ck-1"
        assert kwargs["base_url"] == "http://10.0.0.5:8080/v1"

    def test_generic_provider_requires_explicit_base_url(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with pytest.raises(ValueError, match="base_url"):
                OpenAICompatibleInterface(
                    model_name="x",
                    model_id="m",
                    provider="openai_compatible",
                )

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            OpenAICompatibleInterface(model_name="x", model_id="m", provider="azure")

    def test_empty_model_id_raises(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with pytest.raises(ValueError, match="model_id"):
                OpenAICompatibleInterface(
                    model_name="x", model_id="  ", provider="groq"
                )

    def test_negative_backoff_factor_raises(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with pytest.raises(ValueError, match="backoff_factor"):
                OpenAICompatibleInterface(
                    model_name="x", model_id="m", backoff_factor=-1
                )

    def test_non_positive_timeout_raises(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with pytest.raises(ValueError, match="timeout_seconds"):
                OpenAICompatibleInterface(
                    model_name="x", model_id="m", timeout_seconds=0
                )

    def test_injected_client_skips_env_lookup(self) -> None:
        client = _mock_client(_completion("ok"))
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            iface = OpenAICompatibleInterface(
                model_name="x", model_id="m", client=client
            )
        assert iface.predict("p") == "ok"

    def test_max_retries_negative_clamped_to_zero(self) -> None:
        client = _mock_client(_rate_limit_error())
        iface = OpenAICompatibleInterface(
            model_name="x", model_id="m", max_retries=-3, client=client
        )
        _zero_sleep(iface)
        result = iface.predict("p")
        assert result.startswith(ERROR_PREFIX)
        assert client.chat.completions.create.call_count == 1

    def test_non_positive_max_tokens_raises(self) -> None:
        for bad_tokens in (0, -5):
            with pytest.raises(ValueError, match="max_tokens"):
                OpenAICompatibleInterface(
                    model_name="x",
                    model_id="m",
                    max_tokens=bad_tokens,
                    client=_mock_client(_completion()),
                )

    def test_non_integer_max_tokens_raises(self) -> None:
        for bad_tokens in (128.0, "128", True):
            with pytest.raises(ValueError, match="max_tokens"):
                OpenAICompatibleInterface(
                    model_name="x",
                    model_id="m",
                    max_tokens=bad_tokens,  # type: ignore[arg-type]
                    client=_mock_client(_completion()),
                )

    def test_non_finite_generation_params_raise(self) -> None:
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError, match="temperature"):
                OpenAICompatibleInterface(
                    model_name="x",
                    model_id="m",
                    temperature=bad_value,
                    client=_mock_client(_completion()),
                )
            with pytest.raises(ValueError, match="top_p"):
                OpenAICompatibleInterface(
                    model_name="x",
                    model_id="m",
                    top_p=bad_value,
                    client=_mock_client(_completion()),
                )


class TestOpenAICompatiblePredict:
    """第一層防禦：成功路徑、重試行為與 ERROR 哨兵。"""

    def test_success_returns_raw_content(self) -> None:
        client = _mock_client(_completion("The correct answer is (B)"))
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        assert iface.predict("p") == "The correct answer is (B)"

    def test_request_passes_generation_params(self) -> None:
        client = _mock_client(_completion())
        iface = OpenAICompatibleInterface(
            model_name="g",
            model_id="model-x",
            temperature=0.5,
            max_tokens=256,
            top_p=0.9,
            timeout_seconds=45.0,
            client=client,
        )
        iface.predict("hello")
        client.chat.completions.create.assert_called_once_with(
            model="model-x",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.5,
            max_tokens=256,
            top_p=0.9,
            timeout=45.0,
        )

    def test_empty_choices_returns_empty_string(self) -> None:
        client = _mock_client(_empty_choices())
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        assert iface.predict("p") == ""

    def test_none_content_returns_empty_string(self) -> None:
        client = _mock_client(_completion(content=None))
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        assert iface.predict("p") == ""

    def test_non_str_prompt_raises_type_error(self) -> None:
        client = _mock_client(_completion())
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        with pytest.raises(TypeError):
            iface.predict(123)  # type: ignore[arg-type]
        assert client.chat.completions.create.call_count == 0

    def test_rate_limit_retried_then_success(self) -> None:
        client = _mock_client(_rate_limit_error(), _completion("ok"))
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        sleep = _zero_sleep(iface)
        assert iface.predict("p") == "ok"
        assert client.chat.completions.create.call_count == 2
        sleep.assert_called_once_with(1.0)

    def test_timeout_retried_then_success(self) -> None:
        client = _mock_client(_timeout_error(), _completion("ok"))
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        _zero_sleep(iface)
        assert iface.predict("p") == "ok"
        assert client.chat.completions.create.call_count == 2

    def test_server_error_retried_then_success(self) -> None:
        client = _mock_client(_server_error(), _completion("ok"))
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        _zero_sleep(iface)
        assert iface.predict("p") == "ok"
        assert client.chat.completions.create.call_count == 2

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    def test_gateway_5xx_status_codes_are_retried(self, status_code: int) -> None:
        """5xx 重試分類回歸鎖定：502/503/504 經 SDK 映射為 InternalServerError。"""
        client = _mock_client(_server_error_5xx(status_code))
        iface = OpenAICompatibleInterface(
            model_name="g", model_id="m", max_retries=2, client=client
        )
        _zero_sleep(iface)
        result = iface.predict("p")
        assert result.startswith(f"{ERROR_PREFIX}InternalServerError")
        # 首次呼叫 + 2 次重試
        assert client.chat.completions.create.call_count == 3

    def test_exhausted_retries_returns_error_sentinel(self) -> None:
        client = _mock_client(_rate_limit_error())
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        sleep = _zero_sleep(iface)
        result = iface.predict("p")
        assert result.startswith(f"{ERROR_PREFIX}RateLimitError")
        assert "429 rate limit" in result
        # 首次呼叫 + 3 次重試
        assert client.chat.completions.create.call_count == 4
        assert [c.args[0] for c in sleep.call_args_list] == [1.0, 2.0, 4.0]

    def test_non_retryable_auth_error_returns_immediately(self) -> None:
        client = _mock_client(_auth_error())
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        sleep = _zero_sleep(iface)
        result = iface.predict("p")
        assert result.startswith(f"{ERROR_PREFIX}AuthenticationError")
        assert client.chat.completions.create.call_count == 1
        sleep.assert_not_called()

    def test_bad_request_not_retried(self) -> None:
        client = _mock_client(_bad_request_error())
        iface = OpenAICompatibleInterface(model_name="g", model_id="m", client=client)
        _zero_sleep(iface)
        result = iface.predict("p")
        assert result.startswith(f"{ERROR_PREFIX}BadRequestError")
        assert client.chat.completions.create.call_count == 1

    def test_zero_max_retries_single_attempt(self) -> None:
        client = _mock_client(_rate_limit_error())
        iface = OpenAICompatibleInterface(
            model_name="g", model_id="m", max_retries=0, client=client
        )
        result = iface.predict("p")
        assert result.startswith(ERROR_PREFIX)
        assert client.chat.completions.create.call_count == 1

    def test_custom_backoff_factor_schedule(self) -> None:
        client = _mock_client(_rate_limit_error())
        iface = OpenAICompatibleInterface(
            model_name="g",
            model_id="m",
            max_retries=2,
            backoff_factor=1.5,
            client=client,
        )
        sleep = _zero_sleep(iface)
        iface.predict("p")
        assert [c.args[0] for c in sleep.call_args_list] == [1.0, 1.5]

    def test_error_sentinel_flows_to_evaluator_as_invalid(self) -> None:
        """雙層防護整合：第一層 ERROR 哨兵 → 第二層判定 INVALID。"""
        client = _mock_client(_rate_limit_error())
        iface = OpenAICompatibleInterface(
            model_name="g", model_id="m", max_retries=0, client=client
        )
        raw = iface.predict("The correct answer is (A)")
        assert raw.startswith(ERROR_PREFIX)
        assert Evaluator().extract_answer(raw) is None


class TestBackoffWait:
    """指數退避等待排程（white-box：直接呼叫 wait 函式）。"""

    def test_exponential_schedule_matches_backoff_factor(self) -> None:
        client = _mock_client(_completion())
        iface = OpenAICompatibleInterface(
            model_name="g", model_id="m", backoff_factor=2.0, client=client
        )
        wait = iface._retrying.wait
        waits = [wait(SimpleNamespace(attempt_number=n)) for n in (1, 2, 3, 4)]
        assert waits == [1.0, 2.0, 4.0, 8.0]


class TestBuildModelInterface:
    """build_model_interface 工廠：型別分派與設定 coercing。"""

    def test_mock_type(self) -> None:
        iface = build_model_interface({"type": "mock", "name": "m1"})
        assert isinstance(iface, MockModelInterface)
        assert iface.model_name == "m1"

    def test_mock_type_with_random_mode_and_seed(self) -> None:
        iface = build_model_interface(
            {"type": "mock", "name": "m1", "mock_mode": "random", "seed": 7}
        )
        assert isinstance(iface, MockModelInterface)
        assert iface.mode == "random"

    def test_groq_type_with_retry_and_generation_config(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            iface = build_model_interface(
                {
                    "type": "groq",
                    "name": "g1",
                    "model_id": "llama-3.1-8b-instant",
                    "generation": {
                        "temperature": 0.5,
                        "max_tokens": 256,
                        "top_p": 0.9,
                    },
                },
                {
                    "retry": {
                        "max_retries": 2,
                        "backoff_factor": 3.0,
                        "timeout_seconds": 60.0,
                    }
                },
            )
        assert isinstance(iface, OpenAICompatibleInterface)
        assert iface.model_id == "llama-3.1-8b-instant"
        # 行為驗證：retry/generation 參數真實生效（注入 mock client）
        mock_client = _mock_client(_completion("x"))
        iface._client = mock_client
        iface._retrying.sleep = MagicMock()
        assert iface.predict("p") == "x"
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "p"}],
            temperature=0.5,
            max_tokens=256,
            top_p=0.9,
            timeout=60.0,
        )
        # white-box：backoff 排程 3.0^(n-1) 與停止條件（2 次重試）
        assert iface._retrying.wait(SimpleNamespace(attempt_number=2)) == 3.0
        stop = iface._retrying.stop
        assert stop(SimpleNamespace(attempt_number=2)) is False
        assert stop(SimpleNamespace(attempt_number=3)) is True

    def test_ollama_type_without_key(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_API_KEY": ""}):
            iface = build_model_interface(
                {"type": "ollama", "name": "o1", "model_id": "llama3.2"}
            )
        assert isinstance(iface, OpenAICompatibleInterface)
        assert iface.base_url == "http://localhost:11434/v1"

    def test_openrouter_type(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-1"}):
            iface = build_model_interface(
                {
                    "type": "openrouter",
                    "name": "or1",
                    "model_id": "anthropic/claude-3",
                }
            )
        assert isinstance(iface, OpenAICompatibleInterface)
        assert iface.provider == "openrouter"

    def test_openai_compatible_type_requires_base_url(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with pytest.raises(ValueError, match="base_url"):
                build_model_interface(
                    {
                        "type": "openai_compatible",
                        "name": "x",
                        "model_id": "m",
                    }
                )

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValueError, match="type"):
            build_model_interface({"name": "x"})

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model type"):
            build_model_interface({"type": "vertex", "name": "x", "model_id": "m"})

    def test_missing_model_id_raises(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with pytest.raises(ValueError, match="model_id"):
                build_model_interface({"type": "groq", "name": "x"})

    def test_missing_name_raises(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with pytest.raises(ValueError, match="name"):
                build_model_interface({"type": "groq", "model_id": "m"})

    def test_later_phase_types_raise_not_implemented(self) -> None:
        for later_type in ("huggingface", "gemini"):
            with pytest.raises(NotImplementedError):
                build_model_interface(
                    {"type": later_type, "name": "x", "model_id": "m"}
                )

    def test_invalid_retry_values_fall_back_to_defaults(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            iface = build_model_interface(
                {"type": "groq", "name": "g", "model_id": "m"},
                {
                    "retry": {
                        "max_retries": "abc",
                        "backoff_factor": "xyz",
                        "timeout_seconds": None,
                    }
                },
            )
        # white-box：非法 retry 值回退預設（backoff 2.0、3 次重試）
        assert iface._retrying.wait(SimpleNamespace(attempt_number=1)) == 1.0
        stop = iface._retrying.stop
        assert stop(SimpleNamespace(attempt_number=3)) is False
        assert stop(SimpleNamespace(attempt_number=4)) is True

    def test_none_evaluation_config_uses_defaults(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            iface = build_model_interface(
                {"type": "groq", "name": "g", "model_id": "m"}
            )
        assert iface._retrying.wait(SimpleNamespace(attempt_number=2)) == 2.0

    def test_supported_types_constant(self) -> None:
        assert SUPPORTED_TYPES == {
            "mock",
            "openai_compatible",
            "groq",
            "ollama",
            "openrouter",
        }

    def test_generated_interfaces_satisfy_protocol(self) -> None:
        iface = build_model_interface({"type": "mock", "name": "m"})
        assert isinstance(iface, ModelProtocol)
