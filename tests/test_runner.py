"""循序推論執行模組（``src/runner.py``）全離線單元測試。

所有案例零網路、零權重下載：
- 模型以 ``StubModel``（``BaseModelInterface`` 子類，模擬
  ``predict_batch`` 契約並記錄呼叫順序）取代；
- ``Evaluator`` 使用真實實體（純計算、無副作用，非外部端點）；
- ``time.sleep`` 以 ``MagicMock`` 取代以消除真實延遲；
- tqdm 以 ``TQDM_DISABLE`` 環境變數停用，鎖定非 TTY 環境行為；
- VRAM 釋放行為以 patch ``torch.cuda`` 驗證（無真實 GPU）。
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Mapping, Optional, Sequence
from unittest.mock import MagicMock, call, patch

import pytest

import src.runner
from src.evaluator import Evaluator, EvaluationResult
from src.models import BaseModelInterface
from src.runner import BenchmarkRunner, ModelRunOutput, release_vram

_TARGETS: List[str] = ["A", "B", "C", "D"]


def _make_samples(count: int = 3) -> List[Dict[str, object]]:
    """建立標準樣本紀錄（對齊 ``MMLUDatasetLoader.get_samples`` 輸出）。"""
    samples: List[Dict[str, object]] = []
    for index in range(count):
        samples.append(
            {
                "question_id": f"q-{index}",
                "formatted_prompt": f"prompt-{index}",
                "target_letter": _TARGETS[index % 4],
                "subject": "high_school_mathematics"
                if index % 2 == 0
                else "philosophy",
                "category": "STEM" if index % 2 == 0 else "Humanities",
            }
        )
    return samples


class StubModel(BaseModelInterface):
    """確定性模型替身：模擬模型層 predict_batch 契約並記錄呼叫。

    Attributes:
        predict_calls: 經單題 predict 之 prompts（Runner 契約應恒為
            空；用於斷言一律走 predict_batch）。
        batches: 每次 predict_batch 呼叫之 question_id 清單
            （用於斷言呼叫順序與批次大小）。
    """

    def __init__(
        self,
        model_name: str = "stub-model",
        raw_output: str = "The correct answer is (A)",
        latency: float = 0.5,
        latency_sequence: Optional[Sequence[float]] = None,
        records_per_call: int = 1,
        drop_keys: Sequence[str] = (),
        bad_latency: Optional[object] = None,
        explode: Optional[Exception] = None,
    ) -> None:
        """初始化替身並記錄契約違規模擬參數。

        Args:
            model_name: 模型名稱。
            raw_output: 每題回傳之固定 raw_output。
            latency: 每題注入之固定 latency。
            latency_sequence: 逐題 latency 序列（優先使用；用盡後回退
                ``latency``）。
            records_per_call: 每筆輸入紀錄回傳之紀錄數（模擬契約違規）。
            drop_keys: 需移除之紀錄鍵（模擬契約違規）。
            bad_latency: 非 None 時覆寫 latency（模擬契約違規）。
            explode: 非 None 時 predict_batch 拋出此例外。
        """
        super().__init__(model_name)
        self.predict_calls: List[str] = []
        self.batches: List[List[object]] = []
        self._raw_output = raw_output
        self._latency = latency
        self._latency_sequence: List[float] = (
            list(latency_sequence) if latency_sequence else []
        )
        self._latency_index = 0
        self._records_per_call = records_per_call
        self._drop_keys: set[str] = set(drop_keys)
        self._bad_latency = bad_latency
        self._explode = explode

    def predict(self, prompt: str) -> str:
        """單題推論（Runner 不直接呼叫；用於斷言呼叫路徑）。"""
        self.predict_calls.append(prompt)
        return self._raw_output

    def predict_batch(
        self, samples: Sequence[Mapping[str, object]]
    ) -> List[Dict[str, object]]:
        """模擬模型層契約：逐題回傳三鍵紀錄（或模擬違規）。"""
        if self._explode is not None:
            raise self._explode
        self.batches.append([record.get("question_id") for record in samples])
        records: List[Dict[str, object]] = []
        for record in samples:
            prompt: object = record.get("formatted_prompt")
            usable_prompt: bool = isinstance(prompt, str) and bool(prompt.strip())
            for _ in range(self._records_per_call):
                if usable_prompt:
                    if self._latency_index < len(self._latency_sequence):
                        latency_value: object = self._latency_sequence[
                            self._latency_index
                        ]
                        self._latency_index += 1
                    else:
                        latency_value = self._latency
                    if self._bad_latency is not None:
                        latency_value = self._bad_latency
                    record_out: Dict[str, object] = {
                        "question_id": record.get("question_id", ""),
                        "raw_output": self._raw_output,
                        "latency": latency_value,
                    }
                else:
                    # 模擬模型層髒資料行為：不呼叫模型、直接產出
                    # latency 為 0.0 之 ERROR 紀錄。
                    record_out = {
                        "question_id": record.get("question_id", ""),
                        "raw_output": "ERROR: missing 'formatted_prompt' in sample",
                        "latency": 0.0,
                    }
                for key in self._drop_keys:
                    record_out.pop(key, None)
                records.append(record_out)
        return records


class TestRunnerConstruction:
    """BenchmarkRunner 建構期 fail-fast。"""

    def test_init_defaults(self) -> None:
        runner = BenchmarkRunner(Evaluator())
        assert runner.request_delay == 0.0
        assert isinstance(runner.evaluator, Evaluator)

    @pytest.mark.parametrize(
        "bad_delay",
        [-0.1, float("nan"), float("inf"), float("-inf"), True, "0.5", None],
    )
    def test_init_rejects_invalid_request_delay(self, bad_delay: object) -> None:
        with pytest.raises(ValueError, match="request_delay"):
            BenchmarkRunner(
                Evaluator(),
                request_delay=bad_delay,  # type: ignore[arg-type]
            )

    def test_init_accepts_int_zero(self) -> None:
        runner = BenchmarkRunner(Evaluator(), request_delay=0)  # type: ignore[arg-type]
        assert runner.request_delay == 0.0

    def test_init_rejects_non_evaluator(self) -> None:
        with pytest.raises(TypeError, match="evaluator"):
            BenchmarkRunner(MagicMock())  # type: ignore[arg-type]


class TestRunModelSequentialFlow:
    """逐題循序流轉與契約。"""

    def test_calls_predict_batch_once_per_sample_in_order(self) -> None:
        model = StubModel()
        runner = BenchmarkRunner(Evaluator())
        output = runner.run_model(model, _make_samples(5))
        assert model.batches == [[f"q-{i}"] for i in range(5)]
        assert [record["question_id"] for record in output.predictions] == [
            f"q-{i}" for i in range(5)
        ]

    def test_uses_batch_contract_never_single_predict(self) -> None:
        model = StubModel()
        BenchmarkRunner(Evaluator()).run_model(model, _make_samples(2))
        assert model.predict_calls == []

    def test_returns_model_run_output(self) -> None:
        model = StubModel(model_name="stub-x")
        output = BenchmarkRunner(Evaluator()).run_model(model, _make_samples(3))
        assert isinstance(output, ModelRunOutput)
        assert output.model_name == "stub-x"
        assert isinstance(output.evaluation, EvaluationResult)
        assert output.total_seconds >= 0.0
        assert len(output.predictions) == 3

    def test_does_not_mutate_input_samples(self) -> None:
        samples = _make_samples(3)
        before: List[Dict[str, object]] = [dict(sample) for sample in samples]
        BenchmarkRunner(Evaluator()).run_model(StubModel(), samples)
        assert samples == before

    def test_source_has_no_concurrency_primitives(self) -> None:
        source: str = inspect.getsource(src.runner)
        for forbidden in (
            "asyncio",
            "threading",
            "multiprocessing",
            "concurrent.futures",
            "ThreadPoolExecutor",
        ):
            assert forbidden not in source

    def test_accepts_tuple_samples(self) -> None:
        model = StubModel()
        output = BenchmarkRunner(Evaluator()).run_model(model, tuple(_make_samples(2)))
        assert [record["question_id"] for record in output.predictions] == [
            "q-0",
            "q-1",
        ]


class TestRequestDelay:
    """request_delay 休眠控制。"""

    def test_sleeps_between_questions_but_not_after_last(self) -> None:
        runner = BenchmarkRunner(Evaluator(), request_delay=0.7)
        with patch("src.runner.time.sleep") as sleep_mock:
            runner.run_model(StubModel(), _make_samples(4))
        assert sleep_mock.call_count == 3
        for invocation in sleep_mock.call_args_list:
            assert invocation.args == (0.7,)

    def test_zero_delay_never_sleeps(self) -> None:
        with patch("src.runner.time.sleep") as sleep_mock:
            BenchmarkRunner(Evaluator()).run_model(StubModel(), _make_samples(3))
        sleep_mock.assert_not_called()

    def test_single_sample_never_sleeps(self) -> None:
        runner = BenchmarkRunner(Evaluator(), request_delay=0.7)
        with patch("src.runner.time.sleep") as sleep_mock:
            runner.run_model(StubModel(), _make_samples(1))
        sleep_mock.assert_not_called()

    def test_sleep_ordered_after_each_inference_except_last(self) -> None:
        model = StubModel()
        events: List[str] = []
        original = model.predict_batch

        def recording(
            samples: Sequence[Mapping[str, object]],
        ) -> List[Dict[str, object]]:
            records = original(samples)
            events.extend(str(item) for item in model.batches[-1])
            return records

        model.predict_batch = recording  # type: ignore[method-assign]
        runner = BenchmarkRunner(Evaluator(), request_delay=0.01)
        with patch(
            "src.runner.time.sleep",
            side_effect=lambda _seconds: events.append("sleep"),
        ):
            runner.run_model(model, _make_samples(3))
        assert events == ["q-0", "sleep", "q-1", "sleep", "q-2"]


class TestLatency:
    """單題 latency 傳遞與聚合。"""

    def test_preserves_per_question_latency(self) -> None:
        model = StubModel(latency_sequence=[0.25, 0.5, 0.75])
        output = BenchmarkRunner(Evaluator()).run_model(model, _make_samples(3))
        assert [record["latency"] for record in output.predictions] == [
            0.25,
            0.5,
            0.75,
        ]

    def test_evaluation_average_latency_matches_injected_values(self) -> None:
        model = StubModel(latency_sequence=[0.25, 0.5, 0.75])
        output = BenchmarkRunner(Evaluator()).run_model(model, _make_samples(3))
        assert output.evaluation.average_latency == 0.5

    def test_default_latency_applied_to_all_records(self) -> None:
        model = StubModel(latency=1.25)
        output = BenchmarkRunner(Evaluator()).run_model(model, _make_samples(2))
        assert all(record["latency"] == 1.25 for record in output.predictions)


class TestFieldAlignment:
    """介面欄位對齊（Runner -> Evaluator）。"""

    def test_prediction_records_have_exact_three_keys(self) -> None:
        model = StubModel()
        output = BenchmarkRunner(Evaluator()).run_model(model, _make_samples(3))
        for record in output.predictions:
            assert set(record) == {"question_id", "raw_output", "latency"}
            assert isinstance(record["question_id"], str)
            assert isinstance(record["raw_output"], str)
            assert isinstance(record["latency"], float)

    def test_question_id_preserved_in_order(self) -> None:
        samples = _make_samples(4)
        output = BenchmarkRunner(Evaluator()).run_model(StubModel(), samples)
        assert [record["question_id"] for record in output.predictions] == [
            sample["question_id"] for sample in samples
        ]

    def test_end_to_end_accuracy_with_real_evaluator(self) -> None:
        samples = _make_samples(3)  # 目標字母：A, B, C
        model = StubModel(raw_output="The correct answer is (A)")
        output = BenchmarkRunner(Evaluator()).run_model(model, samples)
        evaluation = output.evaluation
        assert evaluation.total == 3
        assert evaluation.valid == 3
        assert evaluation.correct == 1
        assert evaluation.invalid == 0
        assert evaluation.overall_accuracy == pytest.approx(1 / 3)

    def test_non_str_question_id_still_aligns(self) -> None:
        samples = _make_samples(2)
        samples[0]["question_id"] = 7
        samples[1]["question_id"] = 8
        output = BenchmarkRunner(Evaluator()).run_model(StubModel(), samples)
        assert [record["question_id"] for record in output.predictions] == [7, 8]
        # Evaluator 以 str 對齊雙邊；total 不受型別影響。
        assert output.evaluation.total == 2


class TestEdgeCases:
    """異常／空值處理。"""

    def test_empty_samples(self) -> None:
        model = StubModel()
        runner = BenchmarkRunner(Evaluator(), request_delay=0.5)
        with patch("src.runner.time.sleep") as sleep_mock:
            output = runner.run_model(model, [])
        assert output.predictions == []
        assert output.evaluation.total == 0
        assert output.evaluation.overall_accuracy == 0.0
        assert model.batches == []
        sleep_mock.assert_not_called()

    def test_missing_formatted_prompt_yields_error_record(self) -> None:
        samples = _make_samples(2)
        del samples[0]["formatted_prompt"]
        output = BenchmarkRunner(Evaluator()).run_model(StubModel(), samples)
        first = output.predictions[0]
        assert str(first["raw_output"]).startswith("ERROR:")
        assert first["latency"] == 0.0
        assert output.evaluation.invalid == 1
        assert output.evaluation.correct == 0
        assert output.evaluation.total == 2

    def test_model_predict_batch_explode_propagates_but_releases_vram(self) -> None:
        model = StubModel(explode=RuntimeError("boom"))
        runner = BenchmarkRunner(Evaluator())
        with patch("src.runner.release_vram") as release_mock:
            with pytest.raises(RuntimeError, match="boom"):
                runner.run_model(model, _make_samples(3))
        release_mock.assert_called_once()

    @pytest.mark.parametrize("records_per_call", [0, 2])
    def test_contract_violation_record_count(self, records_per_call: int) -> None:
        model = StubModel(records_per_call=records_per_call)
        with pytest.raises(RuntimeError, match="exactly 1"):
            BenchmarkRunner(Evaluator()).run_model(model, _make_samples(1))

    @pytest.mark.parametrize("drop_key", ["question_id", "raw_output", "latency"])
    def test_contract_violation_missing_key(self, drop_key: str) -> None:
        model = StubModel(drop_keys=(drop_key,))
        with pytest.raises(RuntimeError, match="missing key"):
            BenchmarkRunner(Evaluator()).run_model(model, _make_samples(1))

    @pytest.mark.parametrize("bad_latency", ["fast", float("nan"), True, 1.5e400])
    def test_contract_violation_bad_latency(self, bad_latency: object) -> None:
        model = StubModel(bad_latency=bad_latency)
        with pytest.raises(RuntimeError, match="latency"):
            BenchmarkRunner(Evaluator()).run_model(model, _make_samples(1))


class TestVRAMRelease:
    """VRAM 釋放行為。"""

    def test_run_model_releases_vram_once(self) -> None:
        with patch("src.runner.release_vram") as release_mock:
            BenchmarkRunner(Evaluator()).run_model(StubModel(), _make_samples(2))
        release_mock.assert_called_once()

    def test_release_vram_empty_cache_when_cuda_available(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as empty_mock,
        ):
            release_vram()
        empty_mock.assert_called_once()

    def test_release_vram_noop_when_cuda_unavailable(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.empty_cache") as empty_mock,
        ):
            release_vram()
        empty_mock.assert_not_called()

    def test_release_vram_swallows_empty_cache_failure(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.empty_cache",
                side_effect=RuntimeError("driver glitch"),
            ),
        ):
            release_vram()  # 不得拋出例外。


class TestRunBenchmark:
    """循序模型迴圈編排。"""

    def test_builds_and_runs_models_sequentially(self) -> None:
        model_configs = [
            {"type": "mock", "name": "m-1"},
            {"type": "mock", "name": "m-2"},
        ]
        evaluation_config = {"retry": {"max_retries": 3}}
        first = StubModel(model_name="m-1")
        second = StubModel(model_name="m-2")
        with patch(
            "src.runner.build_model_interface", side_effect=[first, second]
        ) as factory:
            results = BenchmarkRunner(Evaluator()).run_benchmark(
                _make_samples(2), model_configs, evaluation_config
            )
        assert [result.model_name for result in results] == ["m-1", "m-2"]
        assert all(result.evaluation.total == 2 for result in results)
        assert factory.call_args_list == [
            call(model_configs[0], evaluation_config),
            call(model_configs[1], evaluation_config),
        ]

    def test_empty_model_configs_returns_empty(self) -> None:
        with patch("src.runner.build_model_interface") as factory:
            results = BenchmarkRunner(Evaluator()).run_benchmark(
                _make_samples(2), [], {}
            )
        assert results == []
        factory.assert_not_called()

    def test_build_failure_fails_fast_and_stops(self) -> None:
        model_configs = [
            {"type": "mock", "name": "m-1"},
            {"type": "mock", "name": "m-2"},
            {"type": "mock", "name": "m-3"},
        ]
        with patch(
            "src.runner.build_model_interface",
            side_effect=[StubModel(model_name="m-1"), ValueError("bad config")],
        ) as factory:
            with pytest.raises(ValueError, match="bad config"):
                BenchmarkRunner(Evaluator()).run_benchmark(
                    _make_samples(2), model_configs, {}
                )
        # m-2 建構失敗即中止，m-3 不得被嘗試。
        assert factory.call_count == 2

    def test_end_to_end_with_real_mock_drivers_offline(self) -> None:
        model_configs = [
            {"type": "mock", "name": "mock-fixed"},
            {"type": "mock", "name": "mock-random", "mock_mode": "random", "seed": 7},
        ]
        results = BenchmarkRunner(Evaluator()).run_benchmark(
            _make_samples(4), model_configs, {"retry": {}}
        )
        assert [result.model_name for result in results] == [
            "mock-fixed",
            "mock-random",
        ]
        assert all(result.evaluation.total == 4 for result in results)


class TestProgressAndLogging:
    """progress log 與非 TTY 相容性。"""

    def test_run_model_safe_when_tqdm_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TQDM_DISABLE", "1")
        output = BenchmarkRunner(Evaluator()).run_model(StubModel(), _make_samples(3))
        assert len(output.predictions) == 3

    def test_run_model_logs_completion_summary(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="src.runner"):
            BenchmarkRunner(Evaluator()).run_model(StubModel(), _make_samples(2))
        joined = "\n".join(caplog.messages)
        assert "stub-model" in joined
        assert "inference complete" in joined
