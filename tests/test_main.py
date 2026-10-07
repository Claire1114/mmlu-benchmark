"""根目錄 CLI 進入點（``main.py``）全離線單元測試（Step 6）。

所有案例零網路、零權重下載：
- DataLoader 以 ``StubLoader``（記憶體偽樣本）取代，不觸碰 Hugging Face；
- 模型工廠以 ``patch.object(main, "build_model_interface")`` 回傳
  ``StubModel`` 或拋出模擬 API 錯誤（401/網路超時等）；
- ``main.sleep`` 以 MagicMock 取代，斷言條件節流行為；
- ``main.release_vram``／``main.open_journal`` 以 patch 斷言 VRAM 釋放
  與即時 append＋flush 落地行為；
- 設定檔與全部產物皆建立於 ``tmp_path``，不污染工作區。

覆蓋重點（對齊任務驗收標準）：CLI 參數解析、配置檔不存在 fail-fast、
時間戳目錄自動建立、單模拋錯容錯隔離跳過（建構／評估階段）、單題層
ERROR 哨兵隔離（契約違規／單題例外／非有限 latency）、即時 append＋flush
寫入行為（執行中模型崩潰不遺失紀錄）、條件節流（本機 0／API 節流／末題
不睡）、CSV/JSON/JSONL 輸出結構完整性（含中文非轉義落盤、latest 別名追蹤）、
I/O 容錯（日誌檔建立失敗降級終端機-only／journal・指標寫入失敗 → exit 1）
與無效 log 層級回落。
"""

from __future__ import annotations

import csv
import json
import logging
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

import main
from main import parse_args
from src.evaluator import Evaluator
from src.models import BaseModelInterface

TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{6})")


# ---------------------------------------------------------------------------
# 離線測試替身
# ---------------------------------------------------------------------------
class StubLoader:
    """``MMLUDatasetLoader`` 之記憶體替身（零網路、零磁碟）。

    固定 2 科目（STEM/subj_alpha、Humanities/subj_beta）× ``sample_size``
    題；答案循環 A/B/C/D，使「全答 A」的 fixed mock 可產出確定性指標。
    """

    def __init__(
        self,
        config_path: str = "configs/eval_config.yaml",
        num_shots: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.config_path = config_path
        self._num_shots: int = 0 if num_shots is None else int(num_shots)
        self.skipped_rows = 0
        self._subjects: List[Dict[str, str]] = [
            {"category": "STEM", "subject": "subj_alpha", "focus": "focus-a"},
            {"category": "Humanities", "subject": "subj_beta", "focus": "focus-b"},
        ]

    def iter_subjects(self) -> List[Dict[str, str]]:
        """回傳偽科目清單（與 ``MMLUDatasetLoader.iter_subjects`` 同構）。"""
        return [dict(entry) for entry in self._subjects]

    def load_data(
        self,
        subject: Optional[str] = None,
        split: Optional[str] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """回傳 ``sample_size`` 筆偽正規化列（缺失時預設 2 筆）。"""
        count: int = 2 if sample_size is None else int(sample_size)
        rows: List[Dict[str, Any]] = []
        for index in range(count):
            rows.append(
                {
                    "question_id": f"{subject}__{index:06d}__stubid",
                    "question": f"question-{index}",
                    "choices": ["opt-a", "opt-b", "opt-c", "opt-d"],
                    "answer_letter": "ABCD"[index % 4],
                    "subject": subject,
                }
            )
        return rows

    def format_prompt(self, item: Mapping[str, Any]) -> str:
        """組裝可辨識前綴之偽 Prompt。"""
        return f"PROMPT[{item.get('subject')}/{item.get('question_id')}]"


class StubModel(BaseModelInterface):
    """確定性模型替身（對齊模型層 ``predict_batch`` 三鍵契約）。

    Attributes:
        records_per_call: 每題回傳紀錄數（>1 用於模擬契約違規）。
        explode: 非 None 時 ``predict_batch`` 拋出此例外（模擬模型層崩潰）。
    """

    def __init__(
        self,
        model_name: str = "stub-model",
        raw_output: str = "The correct answer is (A)",
        latency: float = 0.01,
        records_per_call: int = 1,
        explode: Optional[Exception] = None,
    ) -> None:
        super().__init__(model_name)
        self._raw_output = raw_output
        self._latency = latency
        self._records_per_call = records_per_call
        self._explode = explode

    def predict(self, prompt: str) -> str:
        """單題推論（main.py 不直接呼叫；保留以滿足抽象契約）。"""
        return self._raw_output

    def predict_batch(
        self, samples: Sequence[Mapping[str, object]]
    ) -> List[Dict[str, object]]:
        """每題回傳 ``records_per_call`` 筆三鍵紀錄；可模擬例外拋出。"""
        if self._explode is not None:
            raise self._explode
        records: List[Dict[str, object]] = []
        for sample in samples:
            record: Dict[str, object] = {
                "question_id": str(sample.get("question_id")),
                "raw_output": self._raw_output,
                "latency": self._latency,
            }
            records.extend([record] * self._records_per_call)
        return records


class FlakyStubModel(BaseModelInterface):
    """確定性單題失敗替身（模擬瞬時 API 500 之單題層例外）。

    僅對 ``question_id`` 等於 ``crash_on`` 之題目拋出 ``RuntimeError``；
    其餘題目回傳正常三鍵紀錄。
    """

    def __init__(
        self,
        model_name: str = "flaky-model",
        crash_on: Optional[str] = None,
        raw_output: str = "The correct answer is (A)",
        latency: float = 0.01,
    ) -> None:
        super().__init__(model_name)
        self._crash_on = crash_on
        self._raw_output = raw_output
        self._latency = latency

    def predict(self, prompt: str) -> str:
        """單題推論（main.py 不直接呼叫；保留以滿足抽象契約）。"""
        return self._raw_output

    def predict_batch(
        self, samples: Sequence[Mapping[str, object]]
    ) -> List[Dict[str, object]]:
        """``crash_on`` 匹配之題目拋出 ``RuntimeError``，其餘回傳三鍵紀錄。"""
        records: List[Dict[str, object]] = []
        for sample in samples:
            question_id: str = str(sample.get("question_id"))
            if question_id == self._crash_on:
                raise RuntimeError("simulated transient API 500")
            records.append(
                {
                    "question_id": question_id,
                    "raw_output": self._raw_output,
                    "latency": self._latency,
                }
            )
        return records


class CjkStubLoader(StubLoader):
    """含中文字樣本之替身延伸（驗證 JSONL ``ensure_ascii=False`` 落盤）。"""

    def load_data(
        self,
        subject: Optional[str] = None,
        split: Optional[str] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """為父類偽列之題幹加上中文前綴。"""
        rows: List[Dict[str, Any]] = super().load_data(
            subject=subject, split=split, sample_size=sample_size, seed=seed
        )
        for row in rows:
            row["question"] = f"問題：{row.get('question')}"
        return rows

    def format_prompt(self, item: Mapping[str, Any]) -> str:
        """中文前綴＋父類偽 Prompt。"""
        return f"問題 {super().format_prompt(item)}"


class TrackingFile:
    """包裹真實檔案物件，計數 write/flush 呼叫次數（即時落地驗證）。

    Args:
        handle: 被包裹之檔案物件。
        fail_on_write: 大於 0 時於第 N 次 ``write`` 拋出 ``OSError``
            （模擬磁碟 I/O 故障）；0 表示正常。
    """

    def __init__(self, handle: Any, fail_on_write: int = 0) -> None:
        self._handle = handle
        self.writes = 0
        self.flushes = 0
        self.closed = False
        self.fail_on_write = fail_on_write

    def write(self, text: str) -> int:
        self.writes += 1
        if self.fail_on_write and self.writes == self.fail_on_write:
            raise OSError("simulated disk I/O failure")
        return self._handle.write(text)

    def flush(self) -> None:
        self.flushes += 1
        self._handle.flush()

    def __enter__(self) -> "TrackingFile":
        return self

    def __exit__(self, *exc: object) -> bool:
        self._handle.close()
        self.closed = True
        return False


# ---------------------------------------------------------------------------
# 設定檔與執行輔助
# ---------------------------------------------------------------------------
def make_config(
    tmp_path: Path,
    filename: str = "eval_config.yaml",
    *,
    models: Optional[List[Dict[str, Any]]] = None,
    request_delay: float = 0.0,
    smoke_size: int = 2,
    demo_size: int = 3,
    log_level: str = "INFO",
    modes: Optional[Dict[str, Any]] = None,
) -> Path:
    """寫入一份最小有效評測設定 YAML 至 ``tmp_path``，回傳其路徑。"""
    if models is None:
        models = [
            {"name": "mock-a", "type": "mock"},
            {"name": "mock-b", "type": "mock", "mock_mode": "random", "seed": 7},
        ]
    if modes is None:
        modes = {
            "smoke_test": {"sample_size_per_subject": smoke_size, "purpose": "smoke"},
            "demo": {"sample_size_per_subject": demo_size, "purpose": "demo"},
        }
    config: Dict[str, Any] = {
        "project": {"name": "test-project", "version": "0.1.0", "seed": 42},
        "dataset": {
            "name": "cais/mmlu",
            "split": "test",
            "prompt_template": (
                "QA about {subject}. Question: {question} "
                "A. {choice_A} B. {choice_B} C. {choice_C} D. {choice_D}"
            ),
            "active_mode": "smoke_test",
            "modes": modes,
        },
        "models": models,
        "evaluation": {
            "batch_size": 1,
            "request_delay": request_delay,
            "answer_regex": r"(?i)\bthe correct answer is\s*\(?\b([A-D])\b\)?",
        },
        "logging": {"level": log_level},
    }
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def run_main(
    base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: Sequence[str],
    *,
    factory: Optional[Any] = None,
    loader_cls: Optional[Any] = None,
) -> int:
    """於隔離環境執行 ``main.main``，回傳 exit code。

    Args:
        base_dir: 執行當前目錄（``logs``／``results`` 等皆落此處）。
        monkeypatch: pytest monkeypatch。
        argv: CLI 參數清單。
        factory: 取代 ``main.build_model_interface`` 的 MagicMock
            （None 表示不 patch 工廠；預設 mock 型別可直接由真工廠建构）。
        loader_cls: 取代 ``main.MMLUDatasetLoader`` 之類別
            （預設為本檔記憶體替身 ``StubLoader``，零網路）。
    """
    monkeypatch.chdir(base_dir)
    monkeypatch.setenv("TQDM_DISABLE", "1")
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                main,
                "MMLUDatasetLoader",
                loader_cls if loader_cls is not None else StubLoader,
            )
        )
        if factory is not None:
            stack.enter_context(patch.object(main, "build_model_interface", factory))
        return main.main(list(argv))


def find_artifacts(directory: Path, prefix: str, suffix: str) -> List[Path]:
    """列出目錄下符合 ``{prefix}...{suffix}`` 之檔案（排序）。"""
    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and path.name.endswith(suffix)
    ]
    return sorted(files)


def find_timestamped(directory: Path, prefix: str, suffix: str) -> List[Path]:
    """列出嚴格符合 ``{prefix}YYYYMMDD_HHMMSS{suffix}`` 之檔案（排序）。

    與 :func:`find_artifacts` 不同：排除 ``metrics_summary_latest.*`` 等非
    時間戳記別名檔案，避免列舉指標報告時混入 latest 別名。
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\d{{8}}_\d{{6}}{re.escape(suffix)}$")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and pattern.match(path.name)
    ]
    return sorted(files)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """讀取 JSONL 檔案為字典清單（忽略空行）。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_timestamp(filename: str) -> str:
    """自產出檔名抽出 ``YYYYMMDD_HHMMSS`` 時間戳記。"""
    match = TIMESTAMP_PATTERN.search(filename)
    assert match is not None, f"no timestamp in {filename!r}"
    return match.group(1)


@pytest.fixture(autouse=True)
def _isolate_root_logging() -> Any:
    """隔離 root logger：測試前後存還 handlers 與層級，防止 dual logging 交叉污染。"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# CLI 參數解析
# ---------------------------------------------------------------------------
class TestParseArgs:
    """``parse_args`` 預設值與邊界驗證。"""

    def test_defaults(self) -> None:
        options = parse_args(["--mode", "smoke_test"])
        assert options.config_path == "configs/eval_config.yaml"
        assert options.mode == "smoke_test"
        assert options.model_names is None
        assert options.delay is None
        assert options.limit is None
        assert options.output_dir == "results"

    def test_all_flags(self) -> None:
        options = parse_args(
            [
                "--mode",
                "demo",
                "--config",
                "alt.yaml",
                "--models",
                "a,b",
                "c",
                "--delay",
                "0.5",
                "--limit",
                "7",
                "--output-dir",
                "custom_out",
            ]
        )
        assert options.mode == "demo"
        assert options.config_path == "alt.yaml"
        assert options.model_names == ("a", "b", "c")
        assert options.delay == 0.5
        assert options.limit == 7
        assert options.output_dir == "custom_out"

    def test_models_comma_only_and_dedup(self) -> None:
        options = parse_args(["--mode", "smoke_test", "--models", "a,a,b"])
        assert options.model_names == ("a", "b")

    def test_delay_negative_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--mode", "smoke_test", "--delay", "-0.1"])
        assert exc_info.value.code == 2

    def test_delay_nan_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--mode", "smoke_test", "--delay", "nan"])
        assert exc_info.value.code == 2

    def test_delay_inf_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--mode", "smoke_test", "--delay", "inf"])
        assert exc_info.value.code == 2

    def test_limit_non_positive_rejected(self) -> None:
        for bad in ("0", "-3"):
            with pytest.raises(SystemExit) as exc_info:
                parse_args(["--mode", "smoke_test", "--limit", bad])
            assert exc_info.value.code == 2

    def test_mode_required(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])

    def test_unknown_flag_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--mode", "smoke_test", "--bogus"])


# ---------------------------------------------------------------------------
# 配置 fail-fast（exit code 2，且無目錄副作用）
# ---------------------------------------------------------------------------
class TestConfigFailFast:
    """配置檔缺失／結構錯誤時 fail-fast，不建立任何 logs/results 目錄。"""

    def test_missing_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code = run_main(
            tmp_path, monkeypatch, ["--mode", "smoke_test", "--config", "missing.yaml"]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "Configuration error" in captured.err
        assert "not found" in captured.err
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "results").exists()

    def test_config_not_mapping(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("42\n", encoding="utf-8")
        code = run_main(
            tmp_path, monkeypatch, ["--mode", "smoke_test", "--config", str(bad)]
        )
        assert code == 2
        assert "must be a mapping" in capsys.readouterr().err

    def test_mode_not_defined_in_modes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = make_config(
            tmp_path,
            modes={"smoke_test": {"sample_size_per_subject": 2, "purpose": "s"}},
        )
        code = run_main(
            tmp_path, monkeypatch, ["--mode", "demo", "--config", str(path)]
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "not defined in dataset.modes" in err
        assert "smoke_test" in err

    def test_unknown_model_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(path), "--models", "ghost"],
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "ghost" in err
        assert "mock-a" in err

    def test_sample_size_not_positive(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = make_config(
            tmp_path,
            modes={"smoke_test": {"sample_size_per_subject": 0, "purpose": "s"}},
        )
        code = run_main(
            tmp_path, monkeypatch, ["--mode", "smoke_test", "--config", str(path)]
        )
        assert code == 2
        assert "sample_size_per_subject" in capsys.readouterr().err

    def test_empty_models(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = make_config(tmp_path, models=[])
        code = run_main(
            tmp_path, monkeypatch, ["--mode", "smoke_test", "--config", str(path)]
        )
        assert code == 2
        assert "models" in capsys.readouterr().err

    def test_config_request_delay_negative(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = make_config(tmp_path, request_delay=-1.0)
        code = run_main(
            tmp_path, monkeypatch, ["--mode", "smoke_test", "--config", str(path)]
        )
        assert code == 2
        assert "request_delay" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 時間戳目錄分流與產物生成
# ---------------------------------------------------------------------------
class TestArtifactGeneration:
    """logs/{mode}/ 與 results/{mode}/ 自動建立＋五類產出＋latest 別名。"""

    def test_happy_path_creates_all_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
        )
        assert code == 0
        log_dir = tmp_path / "logs" / "smoke_test"
        results_dir = tmp_path / "results" / "smoke_test"
        assert log_dir.is_dir()
        assert results_dir.is_dir()
        logs = find_artifacts(log_dir, "pipeline_", ".log")
        preds = find_artifacts(results_dir, "predictions_", ".jsonl")
        # 嚴格匹配時間戳記檔（排除 metrics_summary_latest.* 別名）
        metrics_json = find_timestamped(results_dir, "metrics_summary_", ".json")
        metrics_csv = find_timestamped(results_dir, "metrics_summary_", ".csv")
        assert len(logs) == 1
        assert len(preds) == 1
        assert len(metrics_json) == 1
        assert len(metrics_csv) == 1
        assert (results_dir / "metrics_summary_latest.json").is_file()
        assert (results_dir / "metrics_summary_latest.csv").is_file()
        timestamp = extract_timestamp(preds[0].name)
        assert logs[0].name == f"pipeline_{timestamp}.log"
        assert preds[0].name == f"predictions_{timestamp}.jsonl"
        assert metrics_json[0].name == f"metrics_summary_{timestamp}.json"
        assert metrics_csv[0].name == f"metrics_summary_{timestamp}.csv"

    def test_demo_mode_drives_sampling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = make_config(tmp_path, smoke_size=2, demo_size=3)
        code = run_main(
            tmp_path, monkeypatch, ["--mode", "demo", "--config", str(config_path)]
        )
        assert code == 0
        results_dir = tmp_path / "results" / "demo"
        preds = find_artifacts(results_dir, "predictions_", ".jsonl")
        # 2 模型 × 6 題（demo 抽樣筆數 3 題/科目 × 2 科目）
        assert len(read_jsonl(preds[0])) == 12
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        assert summary["metadata"]["mode"] == "demo"
        assert summary["metadata"]["sample_size_per_subject"] == 3
        assert summary["metadata"]["total_samples"] == 6

    def test_latest_alias_matches_timestamped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        json_ts = find_timestamped(results_dir, "metrics_summary_", ".json")[0]
        csv_ts = find_timestamped(results_dir, "metrics_summary_", ".csv")[0]
        assert json.loads(json_ts.read_text(encoding="utf-8")) == json.loads(
            (results_dir / "metrics_summary_latest.json").read_text(encoding="utf-8")
        )
        assert csv_ts.read_text(encoding="utf-8") == (
            results_dir / "metrics_summary_latest.csv"
        ).read_text(encoding="utf-8")

    def test_dual_logging_terminal_and_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
        )
        assert code == 0
        log_file = find_artifacts(
            tmp_path / "logs" / "smoke_test", "pipeline_", ".log"
        )[0]
        content = log_file.read_text(encoding="utf-8")
        assert "MMLU benchmark pipeline starting" in content
        assert "mock-a" in content
        assert "Pipeline finished" in content
        # 終端機側同步輸出（StreamHandler → stdout）
        terminal_out = capsys.readouterr().out
        assert "MMLU benchmark pipeline starting" in terminal_out

    def test_custom_output_dir_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            [
                "--mode",
                "smoke_test",
                "--config",
                str(config_path),
                "--output-dir",
                "custom_out",
            ],
        )
        assert code == 0
        custom_dir = tmp_path / "custom_out" / "smoke_test"
        assert custom_dir.is_dir()
        assert find_artifacts(custom_dir, "predictions_", ".jsonl")
        # 預設 results/ 未被使用
        assert not (tmp_path / "results").exists()
        # 日誌固定於根目錄 logs/{mode}/（不隨 output-dir 移動）
        assert (tmp_path / "logs" / "smoke_test").is_dir()

    def test_log_level_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = make_config(tmp_path, log_level="DEBUG")
        captured: Dict[str, int] = {}

        def factory(
            model_cfg: Mapping[str, Any], evaluation_cfg: Mapping[str, Any]
        ) -> Any:
            captured["level"] = logging.getLogger().level
            return StubModel(model_name=str(model_cfg.get("name")))

        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=factory,
        )
        assert code == 0
        assert captured["level"] == logging.DEBUG


def _stub_factory() -> MagicMock:
    """回傳可偵測呼叫之 mock 工廠（依名稱建立 StubModel）。"""
    return MagicMock(
        side_effect=lambda cfg, ev_cfg: StubModel(model_name=str(cfg.get("name")))
    )


# ---------------------------------------------------------------------------
# 條件節流與 VRAM 釋放
# ---------------------------------------------------------------------------
class TestThrottling:
    """本機模型零延遲；API 模型題間節流、末題不睡；release_vram 每模必調。"""

    def test_local_models_never_sleep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = [
            {"name": "hf-1", "type": "huggingface", "model_id": "some/model"},
            {"name": "mock-1", "type": "mock"},
        ]
        config_path = make_config(tmp_path, models=models)
        with patch.object(main, "sleep") as sleep_mock:
            code = run_main(
                tmp_path,
                monkeypatch,
                [
                    "--mode",
                    "smoke_test",
                    "--config",
                    str(config_path),
                    "--delay",
                    "0.5",
                ],
                factory=_stub_factory(),
            )
        assert code == 0
        sleep_mock.assert_not_called()

    def test_api_model_sleeps_between_requests_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = [{"name": "groq-1", "type": "groq", "model_id": "some/model"}]
        config_path = make_config(tmp_path, models=models, smoke_size=3)
        with patch.object(main, "sleep") as sleep_mock:
            code = run_main(
                tmp_path,
                monkeypatch,
                [
                    "--mode",
                    "smoke_test",
                    "--config",
                    str(config_path),
                    "--delay",
                    "0.25",
                ],
                factory=_stub_factory(),
            )
        assert code == 0
        # 6 題（3 題/科目 × 2 科目）→ 僅題間 5 次休眠（末題不睡）
        assert sleep_mock.call_count == 5
        assert sleep_mock.call_args_list == [call(0.25)] * 5

    def test_cli_delay_overrides_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = [{"name": "groq-1", "type": "groq", "model_id": "some/model"}]
        config_path = make_config(tmp_path, models=models, request_delay=5.0)
        with patch.object(main, "sleep") as sleep_mock:
            code = run_main(
                tmp_path,
                monkeypatch,
                [
                    "--mode",
                    "smoke_test",
                    "--config",
                    str(config_path),
                    "--delay",
                    "0.1",
                ],
                factory=_stub_factory(),
            )
        assert code == 0
        assert sleep_mock.call_args_list
        assert all(c.args[0] == 0.1 for c in sleep_mock.call_args_list)

    def test_config_delay_used_when_cli_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        models = [{"name": "groq-1", "type": "groq", "model_id": "some/model"}]
        config_path = make_config(tmp_path, models=models, request_delay=0.3)
        with patch.object(main, "sleep") as sleep_mock:
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=_stub_factory(),
            )
        assert code == 0
        assert sleep_mock.call_args_list
        assert all(c.args[0] == 0.3 for c in sleep_mock.call_args_list)

    def test_release_vram_once_per_model_including_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = make_config(tmp_path)  # mock-a、mock-b 兩模型

        def factory(
            model_cfg: Mapping[str, Any], evaluation_cfg: Mapping[str, Any]
        ) -> Any:
            if str(model_cfg.get("name")) == "mock-a":
                raise RuntimeError("401 Unauthorized")
            return StubModel(model_name="mock-b")

        with patch.object(main, "release_vram") as release_mock:
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=factory,
            )
        assert code == 0
        assert release_mock.call_count == 2


# ---------------------------------------------------------------------------
# 單模容錯隔離
# ---------------------------------------------------------------------------
class TestFaultIsolation:
    """單模例外（建構／評估／單題契約）隔離跳過，管線持續結算其餘模型。"""

    def test_model_build_failure_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """模型 1 建構拋 401 → 標記 FAILED、跳過；模型 2 正常結算、exit 0。"""
        config_path = make_config(tmp_path)

        def factory(
            model_cfg: Mapping[str, Any], evaluation_cfg: Mapping[str, Any]
        ) -> Any:
            if str(model_cfg.get("name")) == "mock-a":
                raise RuntimeError("401 Unauthorized: invalid API key")
            return StubModel(model_name="mock-b")

        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=factory,
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        # 僅模型 2 之 4 題落盤（失敗模型無推論紀錄）
        assert len(jsonl) == 4
        assert all(entry["model"] == "mock-b" for entry in jsonl)
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        entries = summary["models"]
        assert len(entries) == 2
        assert entries[0]["status"] == "failed"
        assert "401 Unauthorized" in entries[0]["error"]
        assert entries[0]["metrics"] is None
        assert entries[1]["status"] == "completed"
        # CSV：失敗列指標留空、成功列正常
        csv_text = find_timestamped(results_dir, "metrics_summary_", ".csv")[
            0
        ].read_text(encoding="utf-8")
        rows = list(csv.DictReader(csv_text.splitlines()))
        assert rows[0]["status"] == "failed"
        assert rows[0]["overall_accuracy"] == ""
        assert rows[1]["status"] == "completed"
        assert float(rows[1]["overall_accuracy"]) == pytest.approx(0.5)

    def test_all_models_failed_exit_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = make_config(tmp_path)

        def factory(
            model_cfg: Mapping[str, Any], evaluation_cfg: Mapping[str, Any]
        ) -> Any:
            raise OSError("network unreachable")

        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=factory,
        )
        assert code == 1
        results_dir = tmp_path / "results" / "smoke_test"
        # 指標報告仍產出（兩列皆 failed），jsonl 存在但無紀錄
        assert (
            read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0]) == []
        )
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        assert all(entry["status"] == "failed" for entry in summary["models"])
        csv_text = find_timestamped(results_dir, "metrics_summary_", ".csv")[
            0
        ].read_text(encoding="utf-8")
        rows = list(csv.DictReader(csv_text.splitlines()))
        assert len(rows) == 2
        assert all(
            row["status"] == "failed" and "network unreachable" in row["error"]
            for row in rows
        )

    def test_per_question_contract_violation_sentinel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``predict_batch`` 回傳 2 筆 → 單題 ``ERROR:`` 哨兵；模型仍 completed、全題 INVALID。"""
        config_path = make_config(tmp_path)
        factory = MagicMock(
            side_effect=lambda cfg, ev_cfg: StubModel(
                model_name=str(cfg.get("name")), records_per_call=2
            )
        )
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=factory,
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        assert len(jsonl) == 8
        assert all(
            entry["raw_output"].startswith("ERROR: ")
            and "returned 2 record(s)" in entry["raw_output"]
            and entry["latency"] == 0.0
            for entry in jsonl
        )
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        assert all(entry["status"] == "completed" for entry in summary["models"])
        for entry in summary["models"]:
            metrics: Dict[str, Any] = entry["metrics"]
            assert metrics["total_samples"] == 4
            assert metrics["valid_samples"] == 0
            assert metrics["correct_samples"] == 0
            assert metrics["invalid_samples"] == 4
            assert metrics["overall_accuracy"] == 0.0
            assert metrics["valid_accuracy"] == 0.0
            assert metrics["invalid_parsing_rate"] == 1.0
            assert metrics["domain_accuracy"] == {"STEM": 0.0, "Humanities": 0.0}
            assert metrics["recall_std"] == 0.0
            assert metrics["average_latency_seconds"] == 0.0

    def test_per_question_exception_isolated_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """單題推論拋錯 → ``ERROR:`` 哨兵；模型持續執行、該題於評估層判 INVALID。"""
        config_path = make_config(tmp_path)

        def factory(
            model_cfg: Mapping[str, Any], evaluation_cfg: Mapping[str, Any]
        ) -> Any:
            if str(model_cfg.get("name")) == "mock-a":
                return FlakyStubModel(
                    model_name="mock-a", crash_on="subj_alpha__000001__stubid"
                )
            return StubModel(model_name=str(model_cfg.get("name")))

        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=factory,
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        assert len(jsonl) == 8
        # mock-a 4 題：q1 正常 → q2 哨兵 → q3/q4 正常（模型持續執行）
        assert jsonl[0]["raw_output"] == "The correct answer is (A)"
        assert jsonl[1]["raw_output"] == (
            "ERROR: RuntimeError: simulated transient API 500"
        )
        assert jsonl[1]["latency"] == 0.0
        assert jsonl[2]["raw_output"] == "The correct answer is (A)"
        assert jsonl[3]["raw_output"] == "The correct answer is (A)"
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        first = summary["models"][0]
        assert first["status"] == "completed"
        metrics: Dict[str, Any] = first["metrics"]
        assert metrics["total_samples"] == 4
        assert metrics["valid_samples"] == 3
        assert metrics["correct_samples"] == 2
        assert metrics["invalid_samples"] == 1
        assert metrics["overall_accuracy"] == pytest.approx(0.5)
        assert metrics["valid_accuracy"] == pytest.approx(2 / 3)
        assert metrics["invalid_parsing_rate"] == pytest.approx(0.25)

    def test_non_finite_latency_becomes_sentinel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``latency`` 為 NaN → 單題 ``ERROR:`` 哨兵（NaN 不落盤、評估以 0.0 計）。"""
        config_path = make_config(tmp_path)
        factory = MagicMock(
            side_effect=lambda cfg, ev_cfg: StubModel(
                model_name=str(cfg.get("name")), latency=float("nan")
            )
        )
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=factory,
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        assert all(
            entry["raw_output"].startswith("ERROR: ")
            and "failed the finite-number check" in entry["raw_output"]
            and entry["latency"] == 0.0
            for entry in jsonl
        )
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        assert all(entry["status"] == "completed" for entry in summary["models"])
        for entry in summary["models"]:
            metrics: Dict[str, Any] = entry["metrics"]
            assert metrics["invalid_samples"] == 4
            assert metrics["invalid_parsing_rate"] == 1.0
            assert metrics["overall_accuracy"] == 0.0
            assert metrics["average_latency_seconds"] == 0.0

    def test_evaluator_failure_fails_model_but_keeps_journal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """模型 1 評估階段例外 → 隔離 FAILED；其 4 題紀錄已落盤不遺失、模型 2 正常。"""
        config_path = make_config(tmp_path)
        config_text = config_path.read_text(encoding="utf-8")
        config_dict: Dict[str, Any] = yaml.safe_load(config_text)
        real_evaluator = Evaluator.from_config(config_dict)
        state: Dict[str, int] = {"calls": 0}

        def fake_evaluate(
            samples: Sequence[Mapping[str, object]],
            predictions: Sequence[Mapping[str, object]],
            model_name: str = "",
        ) -> Any:
            state["calls"] += 1
            if state["calls"] == 1:
                raise ValueError("simulated evaluator crash")
            return real_evaluator.evaluate(samples, predictions, model_name=model_name)

        fake_evaluator = MagicMock()
        fake_evaluator.evaluate.side_effect = fake_evaluate
        with patch.object(main, "Evaluator") as evaluator_cls:
            evaluator_cls.from_config.return_value = fake_evaluator
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=_stub_factory(),
            )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        # 模型 1 之 4 題於 evaluate 拋錯前已即時寫入 → 不遺失
        assert [entry["model"] for entry in jsonl] == [
            "mock-a",
            "mock-a",
            "mock-a",
            "mock-a",
            "mock-b",
            "mock-b",
            "mock-b",
            "mock-b",
        ]
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        failed, completed = summary["models"]
        assert failed["status"] == "failed"
        assert "ValueError: simulated evaluator crash" in failed["error"]
        assert failed["metrics"] is None
        assert completed["status"] == "completed"
        assert completed["metrics"]["total_samples"] == 4
        csv_text = find_timestamped(results_dir, "metrics_summary_", ".csv")[
            0
        ].read_text(encoding="utf-8")
        rows = list(csv.DictReader(csv_text.splitlines()))
        assert rows[0]["status"] == "failed"
        assert rows[0]["overall_accuracy"] == ""
        assert rows[1]["status"] == "completed"


# ---------------------------------------------------------------------------
# 即時落盤（append + flush）
# ---------------------------------------------------------------------------
class TestImmediatePersistence:
    """逐題 ``write()``＋``flush()``；執行中落盤崩潰時已寫入紀錄不遺失。"""

    def test_every_question_appends_and_flushes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """2 模型 × 4 題 → 8 次寫入與 8 次 flush；journal 於管線結束時關閉。"""
        config_path = make_config(tmp_path)
        opened: Dict[str, TrackingFile] = {}

        def fake_open_journal(path: Path) -> TrackingFile:
            tracker = TrackingFile(path.open("a", encoding="utf-8"))
            opened["tracker"] = tracker
            return tracker

        with patch.object(main, "open_journal", side_effect=fake_open_journal):
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=_stub_factory(),
            )
        assert code == 0
        tracker = opened["tracker"]
        assert tracker.writes == 8
        assert tracker.flushes == 8
        assert tracker.closed
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        assert [entry["model"] for entry in jsonl] == [
            "mock-a",
            "mock-a",
            "mock-a",
            "mock-a",
            "mock-b",
            "mock-b",
            "mock-b",
            "mock-b",
        ]

    def test_partial_records_survive_mid_run_io_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """模型 1 第 2 次落盤失敗（模擬磁碟 I/O 崩潰）→ 隔離 FAILED；
        第 1 題紀錄已 flush 落盤不遺失；模型 2 正常結算、exit 0。

        註：``predict_batch`` 之例外屬單題層（一律轉 ERROR 哨兵、模型不敗），
        推論迴圈中唯一可達之模型層崩潰路徑為落盤（journal write）例外。
        """
        config_path = make_config(tmp_path)
        opened: Dict[str, TrackingFile] = {}

        def fake_open_journal(path: Path) -> TrackingFile:
            tracker = TrackingFile(path.open("a", encoding="utf-8"), fail_on_write=2)
            opened["tracker"] = tracker
            return tracker

        with patch.object(main, "open_journal", side_effect=fake_open_journal):
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=_stub_factory(),
            )
        assert code == 0
        tracker = opened["tracker"]
        # 模型 1：write #1 成功＋flush #1；write #2 拋錯（無 flush）
        # 模型 2：write #3–#6 成功＋4 次 flush
        assert tracker.writes == 6
        assert tracker.flushes == 5
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        assert len(jsonl) == 5
        first = jsonl[0]
        assert first["model"] == "mock-a"
        assert first["question_id"] == "subj_alpha__000000__stubid"
        assert first["raw_output"] == "The correct answer is (A)"
        assert first["latency"] == pytest.approx(0.01)
        assert all(entry["model"] == "mock-b" for entry in jsonl[1:])
        summary = json.loads(
            find_timestamped(results_dir, "metrics_summary_", ".json")[0].read_text(
                encoding="utf-8"
            )
        )
        assert summary["models"][0]["status"] == "failed"
        assert "OSError: simulated disk I/O failure" in summary["models"][0]["error"]
        assert summary["models"][0]["metrics"] is None
        assert summary["models"][1]["status"] == "completed"


# ---------------------------------------------------------------------------
# 輸出格式驗證（JSONL / JSON / CSV 結構）
# ---------------------------------------------------------------------------
class TestOutputFormats:
    """三類產出物（逐題 JSONL、匯總 JSON、匯總 CSV）之結構與數值驗證。"""

    def test_json_summary_metadata_and_model_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """匯總 JSON：``metadata`` 欄位完整＋``models`` 條目結構與確定性指標數值。"""
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=_stub_factory(),
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        json_ts = find_timestamped(results_dir, "metrics_summary_", ".json")[0]
        summary = json.loads(json_ts.read_text(encoding="utf-8"))
        metadata: Dict[str, Any] = summary["metadata"]
        assert metadata["mode"] == "smoke_test"
        assert metadata["mode_purpose"] == "smoke"
        assert metadata["project"] == "test-project"
        assert metadata["config_version"] == "0.1.0"
        assert metadata["sample_size_per_subject"] == 2
        assert metadata["total_samples"] == 4
        assert metadata["subjects"] == [
            {"category": "STEM", "subject": "subj_alpha"},
            {"category": "Humanities", "subject": "subj_beta"},
        ]
        assert metadata["request_delay"] == 0.0
        assert metadata["delay_source"] == "config"
        assert metadata["output_dir"] == "results"
        assert TIMESTAMP_PATTERN.search(str(metadata["timestamp"])) is not None
        assert str(metadata["log_file"]).endswith(
            f"pipeline_{metadata['timestamp']}.log"
        )
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", str(metadata["generated_at"])
        )
        models: List[Dict[str, Any]] = summary["models"]
        assert [entry["model_name"] for entry in models] == ["mock-a", "mock-b"]
        for entry in models:
            assert set(entry) == {
                "model_name",
                "status",
                "error",
                "total_seconds",
                "metrics",
            }
            assert entry["status"] == "completed"
            assert entry["error"] is None
            assert isinstance(entry["total_seconds"], float)
            assert entry["total_seconds"] >= 0.0
            metrics: Dict[str, Any] = entry["metrics"]
            assert metrics["model"] == entry["model_name"]
            assert metrics["total_samples"] == 4
            assert metrics["valid_samples"] == 4
            assert metrics["correct_samples"] == 2
            assert metrics["invalid_samples"] == 0
            assert metrics["overall_accuracy"] == pytest.approx(0.5)
            assert metrics["valid_accuracy"] == pytest.approx(0.5)
            assert metrics["invalid_parsing_rate"] == 0.0
            assert metrics["domain_accuracy"] == {"STEM": 0.5, "Humanities": 0.5}
            assert metrics["domain_accuracy_mean"] == pytest.approx(0.5)
            assert metrics["subject_accuracy"] == {"subj_alpha": 0.5, "subj_beta": 0.5}
            assert metrics["option_recall"] == {"A": 1.0, "B": 0.0}
            assert metrics["recall_std"] == pytest.approx(0.5)
            assert metrics["average_latency_seconds"] == pytest.approx(0.01)
            assert set(metrics["category_scores"]) == {"STEM", "Humanities"}
            assert metrics["category_scores"]["STEM"]["accuracy"] == pytest.approx(0.5)
            assert set(metrics["subject_scores"]) == {"subj_alpha", "subj_beta"}
            assert metrics["subject_scores"]["subj_alpha"]["accuracy"] == pytest.approx(
                0.5
            )

    def test_csv_schema_and_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """匯總 CSV：表頭精確順序（核心欄＋領域/科目動態欄）與每模型一列數值。"""
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=_stub_factory(),
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        csv_ts = find_timestamped(results_dir, "metrics_summary_", ".csv")[0]
        reader = csv.DictReader(csv_ts.read_text(encoding="utf-8").splitlines())
        assert list(reader.fieldnames) == [
            "model",
            "status",
            "error",
            "total_samples",
            "valid_samples",
            "correct_samples",
            "invalid_samples",
            "missing_predictions",
            "overall_accuracy",
            "valid_accuracy",
            "invalid_parsing_rate",
            "domain_accuracy_mean",
            "average_latency_seconds",
            "recall_std",
            "total_seconds",
            "domain_accuracy_STEM",
            "domain_accuracy_Humanities",
            "subject_accuracy_subj_alpha",
            "subject_accuracy_subj_beta",
        ]
        rows = list(reader)
        assert [row["model"] for row in rows] == ["mock-a", "mock-b"]
        row = rows[0]
        assert row["status"] == "completed"
        assert row["error"] == ""
        assert row["total_samples"] == "4"
        assert row["valid_samples"] == "4"
        assert row["correct_samples"] == "2"
        assert row["invalid_samples"] == "0"
        assert float(row["overall_accuracy"]) == pytest.approx(0.5)
        assert float(row["valid_accuracy"]) == pytest.approx(0.5)
        assert float(row["invalid_parsing_rate"]) == 0.0
        assert float(row["domain_accuracy_mean"]) == pytest.approx(0.5)
        assert float(row["average_latency_seconds"]) == pytest.approx(0.01)
        assert float(row["recall_std"]) == pytest.approx(0.5)
        assert float(row["total_seconds"]) >= 0.0
        assert float(row["domain_accuracy_STEM"]) == pytest.approx(0.5)
        assert float(row["domain_accuracy_Humanities"]) == pytest.approx(0.5)
        assert float(row["subject_accuracy_subj_alpha"]) == pytest.approx(0.5)
        assert float(row["subject_accuracy_subj_beta"]) == pytest.approx(0.5)

    def test_jsonl_record_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """逐題 JSONL：固定 8 欄＋順序與樣本（科目/領域/Prompt/目標答案）對齊。"""
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=_stub_factory(),
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        entries = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        assert len(entries) == 8
        expected_keys = {
            "model",
            "question_id",
            "subject",
            "category",
            "prompt",
            "target_letter",
            "raw_output",
            "latency",
        }
        assert all(set(entry) == expected_keys for entry in entries)
        for index in range(4):
            entry = entries[index]
            assert entry["model"] == "mock-a"
            subject, category = (
                ("subj_alpha", "STEM") if index < 2 else ("subj_beta", "Humanities")
            )
            local = index % 2
            assert entry["question_id"] == f"{subject}__{local:06d}__stubid"
            assert entry["subject"] == subject
            assert entry["category"] == category
            assert entry["prompt"] == f"PROMPT[{subject}/{entry['question_id']}]"
            assert entry["target_letter"] == "AB"[local]
            assert entry["raw_output"] == "The correct answer is (A)"
            assert entry["latency"] == pytest.approx(0.01)
        assert [entry["model"] for entry in entries[4:]] == ["mock-b"] * 4

    def test_journal_preserves_cjk_without_unicode_escaping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSONL 以 ``ensure_ascii=False``：中文字樣本／Prompt 以原始字元落盤。"""
        config_path = make_config(tmp_path)
        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=_stub_factory(),
            loader_cls=CjkStubLoader,
        )
        assert code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        path = find_artifacts(results_dir, "predictions_", ".jsonl")[0]
        raw = path.read_text(encoding="utf-8")
        assert "問題" in raw
        # ensure_ascii=True 會轉為 \u554f（問）／\u984c（題）
        assert "\\u554f" not in raw
        assert "\\u984c" not in raw
        entries = read_jsonl(path)
        assert (
            entries[0]["prompt"] == "問題 PROMPT[subj_alpha/subj_alpha__000000__stubid]"
        )

    def test_latest_alias_tracks_most_recent_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``metrics_summary_latest.*`` 每次執行覆寫：兩連跑後匹配最新時間戳記報告。"""
        config_path = make_config(tmp_path)
        argv = ["--mode", "smoke_test", "--config", str(config_path)]
        first_code = run_main(tmp_path, monkeypatch, argv, factory=_stub_factory())
        second_code = run_main(tmp_path, monkeypatch, argv, factory=_stub_factory())
        assert first_code == 0
        assert second_code == 0
        results_dir = tmp_path / "results" / "smoke_test"
        # 同秒兩次執行會覆寫同一時間戳記檔，故允許 1 或 2 個
        json_ts_files = find_timestamped(results_dir, "metrics_summary_", ".json")
        assert 1 <= len(json_ts_files) <= 2
        newest_json = max(json_ts_files, key=lambda p: extract_timestamp(p.name))
        latest_json = results_dir / "metrics_summary_latest.json"
        assert latest_json.read_text(encoding="utf-8") == newest_json.read_text(
            encoding="utf-8"
        )
        csv_ts_files = find_timestamped(results_dir, "metrics_summary_", ".csv")
        newest_csv = max(csv_ts_files, key=lambda p: extract_timestamp(p.name))
        assert (results_dir / "metrics_summary_latest.csv").read_text(
            encoding="utf-8"
        ) == newest_csv.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# I/O 容錯與日誌降級（Phase B 敵對審計）
# ---------------------------------------------------------------------------
class TestIOFaultHandling:
    """日誌檔建立失敗降級終端機-only；journal／指標 I/O 失敗乾淨 exit 1。"""

    def test_log_file_failure_degrades_to_terminal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """模擬日誌檔權限失敗 → 管線正常完成、終端機-only 日誌（降級不中斷）。"""
        config_path = make_config(tmp_path)
        with patch.object(
            main.logging, "FileHandler", side_effect=OSError("permission denied")
        ):
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=_stub_factory(),
            )
        assert code == 0
        # 日誌目錄已建立但無日誌檔；root 僅剩一個終端機 handler
        assert not list((tmp_path / "logs" / "smoke_test").glob("*.log"))
        assert len(logging.getLogger().handlers) == 1
        terminal_out = capsys.readouterr().out
        assert "MMLU benchmark pipeline starting" in terminal_out
        assert "degrading to terminal-only logging" in terminal_out
        assert "Pipeline finished" in terminal_out

    def test_journal_open_failure_exit_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """journal 開檔失敗 → 乾淨錯誤 exit 1；未執行任何模型推論。"""
        config_path = make_config(tmp_path)
        calls: List[int] = []

        def factory(
            model_cfg: Mapping[str, Any], evaluation_cfg: Mapping[str, Any]
        ) -> Any:
            calls.append(1)
            return StubModel(model_name=str(model_cfg.get("name")))

        with patch.object(
            main, "open_journal", side_effect=OSError("read-only file system")
        ):
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=factory,
            )
        assert code == 1
        assert calls == []
        results_dir = tmp_path / "results" / "smoke_test"
        assert find_artifacts(results_dir, "predictions_", ".jsonl") == []
        assert find_timestamped(results_dir, "metrics_summary_", ".json") == []
        log_file = find_artifacts(
            tmp_path / "logs" / "smoke_test", "pipeline_", ".log"
        )[0]
        log_text = log_file.read_text(encoding="utf-8")
        assert "Failed to open predictions journal" in log_text
        assert "read-only file system" in log_text

    def test_metrics_write_failure_keeps_predictions_exit_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """指標寫入失敗 → exit 1；推論 journal 紀錄已全數落盤不遺失。"""
        config_path = make_config(tmp_path)
        with patch.object(
            main, "write_metrics_files", side_effect=OSError("simulated disk full")
        ):
            code = run_main(
                tmp_path,
                monkeypatch,
                ["--mode", "smoke_test", "--config", str(config_path)],
                factory=_stub_factory(),
            )
        assert code == 1
        results_dir = tmp_path / "results" / "smoke_test"
        jsonl = read_jsonl(find_artifacts(results_dir, "predictions_", ".jsonl")[0])
        assert len(jsonl) == 8
        assert find_timestamped(results_dir, "metrics_summary_", ".json") == []
        assert find_timestamped(results_dir, "metrics_summary_", ".csv") == []
        assert not (results_dir / "metrics_summary_latest.json").exists()
        log_file = find_artifacts(
            tmp_path / "logs" / "smoke_test", "pipeline_", ".log"
        )[0]
        log_text = log_file.read_text(encoding="utf-8")
        assert "Failed to write metrics summary artifacts" in log_text

    def test_invalid_log_level_falls_back_to_info(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """設定檔 ``logging.level`` 無效值 → root 回落 INFO、管線不阻斷。"""
        config_path = make_config(tmp_path, log_level="BOGUS")
        captured: Dict[str, int] = {}

        def factory(
            model_cfg: Mapping[str, Any], evaluation_cfg: Mapping[str, Any]
        ) -> Any:
            captured["level"] = logging.getLogger().level
            return StubModel(model_name=str(model_cfg.get("name")))

        code = run_main(
            tmp_path,
            monkeypatch,
            ["--mode", "smoke_test", "--config", str(config_path)],
            factory=factory,
        )
        assert code == 0
        assert captured["level"] == logging.INFO
