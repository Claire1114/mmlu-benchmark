"""MMLU Benchmark Pipeline CLI 進入點（Step 6）。

串接已驗收之核心模組（``src/dataset_loader.py``、``src/models/``、
``src/evaluator.py``、``src/runner.py``）為完整端到端評測管線，並嚴格
遵循單一方向管線生命週期（.clinerules §2）：

1. One-time Loading：以 DataLoader 公開 API（``iter_subjects()`` ＋
   ``load_data()`` ＋ ``format_prompt()``）組合一次性載入，抽樣筆數由
   ``--mode`` 驅動（對齊 ``dataset.modes[mode].sample_size_per_subject``
   語意）；樣本跨模型全程重用、不重讀資料。
2. Sequential Model Loop：依設定逐模型迭代（``for model in models``）；
   單一模型例外（API 401/429、網路超時、設定錯誤等）記錄日誌並跳過，
   絕不讓程式崩潰，保護其餘模型正常結算。
3. Step per Model：``build_model_interface()`` → 逐題循序推論（即時
   append ＋ ``flush()``）→ ``finally`` 區塊明確呼叫 ``release_vram()`` →
   ``Evaluator.evaluate()`` 指標交接。
4. Aggregation：持久化時間戳記 JSONL／JSON／CSV 產出，並維護
   ``metrics_summary_latest`` 別名供後續視覺化存取。

關鍵設計：
- 單模隔離雙層：單題層將三鍵契約違規或意外例外轉為 ``ERROR:`` 哨兵
  紀錄（模型持續執行、該題於評估層計 INVALID）；模型層將建構／評估
  例外標記該模型 FAILED 並續跑下一模型。
- 即時落地：推論結果逐題以 ``write()`` ＋ ``flush()`` 寫入
  ``predictions_{timestamp}.jsonl``，中斷或模型崩潰皆不遺失已寫入資料。
- 條件節流：本機模型（``huggingface``／``hf_pipeline``／``mock``）強制
  延遲 0.0；僅 API 類模型套用 ``--delay``（或 ``evaluation.request_delay``）；
  末題之後不休眠。
- Dual Logging：root logger 同時掛載終端機（stdout）與
  ``logs/{mode}/pipeline_{timestamp}.log`` 檔案雙 handler；日誌檔建立
  失敗時優雅降級為終端機-only 並記錄警告，不中斷管線。
- I/O 容錯：journal 開檔或指標寫入失敗 → 記錄乾淨錯誤並 exit 1
  （已寫入之推論資料不遺失）。

Exit codes:
- 0: 管線完成且至少一個模型成功（單模失敗被隔離並於匯總標記 FAILED）。
- 1: 致命錯誤（樣本載入失敗／全數模型失敗／I/O 失敗／未預料例外）。
- 2: CLI 或設定誤用（配置檔不存在／mode 未定義／模型名未知／delay、
  limit、request_delay 非法）。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, IO, List, Mapping, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm  # type: ignore[import-untyped]

from src.dataset_loader import MMLUDatasetLoader
from src.evaluator import Evaluator, EvaluationResult
from src.models import ERROR_PREFIX, BaseModelInterface, build_model_interface
from src.runner import PREDICTION_RECORD_KEYS, release_vram

LOGGER = logging.getLogger("main")

DEFAULT_CONFIG_PATH: str = "configs/eval_config.yaml"
DEFAULT_OUTPUT_DIR: str = "results"
LOGS_ROOT_NAME: str = "logs"
TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
DEFAULT_LOG_LEVEL: str = "INFO"

#: 本地／離線模型類型：請求節流強制為零（無 API 限流問題）。
LOCAL_MODEL_TYPES: frozenset[str] = frozenset({"huggingface", "hf_pipeline", "mock"})

#: CSV 核心欄位（領域／科目準確率欄位由 :func:`build_csv_rows` 動態追加）。
CSV_CORE_FIELDS: Tuple[str, ...] = (
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
)

_VALID_LEVELS: frozenset[int] = frozenset(
    {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
)


class UsageError(Exception):
    """CLI／設定驗證錯誤（對應 exit code 2）。"""


@dataclass(frozen=True)
class CliOptions:
    """經驗證的 CLI 執行參數。

    Attributes:
        config_path: 設定檔路徑。
        mode: 評測模式（須定義於設定檔 ``dataset.modes``）。
        model_names: 指定要評測的模型名稱；``None`` 表示設定檔全數模型。
        delay: 請求節流秒數（覆寫 ``evaluation.request_delay``）；
            ``None`` 表示沿用設定檔。
        limit: 全域樣本上限（正整數）；``None`` 表示不限制。
        output_dir: 結果產出根目錄。
    """

    config_path: str
    mode: str
    model_names: Optional[Tuple[str, ...]]
    delay: Optional[float] = None
    shots: Optional[int] = None
    limit: Optional[int] = None
    output_dir: str = DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class ExecutionPlan:
    """由設定檔＋CLI 選項導出之驗證後執行計劃。

    Attributes:
        sample_size: 每科目抽樣筆數（來自 ``dataset.modes[mode]``）。
        mode_purpose: 該模式之目的說明（供匯總報告引用）。
        model_configs: 經篩選後之模型設定區塊（保持設定檔宣告順序）。
        evaluation_cfg_raw: 全域 ``evaluation:`` 設定區塊（透傳給模型工廠）。
        base_delay: 基礎請求節流秒數（CLI 覆寫或設定檔值）。
    """

    sample_size: int
    mode_purpose: str
    model_configs: Tuple[Mapping[str, object], ...]
    evaluation_cfg_raw: Mapping[str, object]
    base_delay: float


def parse_args(argv: Optional[Sequence[str]] = None) -> CliOptions:
    """解析並驗證 CLI 參數。

    Args:
        argv: 參數清單；``None`` 時使用 ``sys.argv[1:]``。

    Returns:
        型別化之 ``CliOptions``。

    Raises:
        SystemExit: 參數缺失、``--delay`` 非有限非負數、``--limit`` 非正整數、
            或未知旗標時（argparse 慣例 exit code 2）。
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MMLU Benchmark Pipeline CLI 進入點。",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"設定檔路徑（預設 {DEFAULT_CONFIG_PATH}）。",
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="評測模式；須定義於設定檔 dataset.modes（標準值 smoke_test / demo）。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help="指定要評測的模型名稱（空白或逗號分隔）；預設為設定檔全數模型。",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="覆寫請求節流秒數（evaluation.request_delay）；本機模型恒為 0。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="全域樣本上限（正整數；依科目順序截取）。",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="每科目 Few-Shot 範例數量。0 為 Zero-Shot，>0 為 Few-Shot。預設從 config 讀取 num_shots。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="結果產出根目錄（預設 results；實際落點為 <output-dir>/{mode}/）。",
    )
    args = parser.parse_args(argv)

    if args.delay is not None and (
        math.isnan(args.delay) or math.isinf(args.delay) or args.delay < 0.0
    ):
        parser.error(
            f"--delay must be a finite non-negative number, got {args.delay!r}."
        )
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be a positive integer, got {args.limit!r}.")

    model_names: Optional[Tuple[str, ...]] = None
    if args.models:
        names: List[str] = []
        for token in args.models:
            for piece in str(token).split(","):
                piece = piece.strip()
                if piece and piece not in names:
                    names.append(piece)
        if not names:
            parser.error("--models received no valid model name(s).")
        model_names = tuple(names)

    return CliOptions(
        config_path=str(args.config),
        mode=str(args.mode),
        model_names=model_names,
        delay=args.delay,
        shots=args.shots,
        limit=args.limit,
        output_dir=str(args.output_dir),
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """讀取 YAML 設定檔並做最小結構驗證（fail-fast）。

    Args:
        config_path: 設定檔路徑。

    Returns:
        解析後的設定字典。

    Raises:
        FileNotFoundError: 當檔案不存在時。
        ValueError: 當 YAML 內容不是對應表時。
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded: Any = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Evaluation config must be a mapping.")
    return loaded


def _block_name(block: Mapping[str, object]) -> str:
    """讀取模型區塊之 ``name``（缺失／非字串時回傳空字串）。"""
    raw: object = block.get("name")
    return raw.strip() if isinstance(raw, str) else ""


def resolve_execution_plan(
    config: Mapping[str, object], options: CliOptions
) -> ExecutionPlan:
    """驗證設定檔與 CLI 選項之交集，產出執行計劃（fail-fast）。

    驗證項目：``dataset.modes`` 非空且包含 ``--mode``、該模式之
    ``sample_size_per_subject`` 為正整數、``models[]`` 為非空清單且各區塊
    具非空 ``name``、``--models`` 指定之名皆存在、``evaluation.request_delay``
    為有限非負數（未指定 ``--delay`` 時）。

    Args:
        config: 已載入之設定字典。
        options: CLI 選項。

    Returns:
        驗證後之 ``ExecutionPlan``。

    Raises:
        UsageError: 任一驗證項目不符時（訊息含可用值清單）。
    """
    dataset_raw: object = config.get("dataset")
    if not isinstance(dataset_raw, Mapping):
        raise UsageError("'dataset' must be a mapping.")
    modes_raw: object = dataset_raw.get("modes")
    if not isinstance(modes_raw, Mapping) or not modes_raw:
        raise UsageError("'dataset.modes' must be a non-empty mapping.")
    if options.mode not in modes_raw:
        raise UsageError(
            f"mode {options.mode!r} is not defined in dataset.modes; "
            f"available: {sorted(str(key) for key in modes_raw.keys())}."
        )
    mode_block: object = modes_raw.get(options.mode)
    if not isinstance(mode_block, Mapping):
        raise UsageError(f"dataset.modes[{options.mode!r}] must be a mapping.")
    raw_size: object = mode_block.get("sample_size_per_subject")
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size <= 0:
        raise UsageError(
            f"dataset.modes[{options.mode!r}].sample_size_per_subject must be a "
            f"positive integer, got {raw_size!r}."
        )

    models_raw: object = config.get("models")
    if not isinstance(models_raw, (list, tuple)) or not models_raw:
        raise UsageError("'models' must be a non-empty list of model blocks.")
    blocks: List[Mapping[str, object]] = []
    known_names: List[str] = []
    for block in models_raw:
        if not isinstance(block, Mapping):
            raise UsageError("every 'models[]' entry must be a mapping.")
        name: str = _block_name(block)
        if not name:
            raise UsageError("every 'models[]' entry requires a non-empty 'name'.")
        blocks.append(block)
        known_names.append(name)
    if options.model_names is not None:
        unknown = [name for name in options.model_names if name not in known_names]
        if unknown:
            raise UsageError(
                f"unknown model name(s) {unknown}; available: {known_names}."
            )
        wanted = set(options.model_names)
        selected: List[Mapping[str, object]] = [
            block for block in blocks if _block_name(block) in wanted
        ]
    else:
        selected = blocks

    evaluation_raw: object = config.get("evaluation")
    evaluation_cfg: Mapping[str, object] = (
        evaluation_raw if isinstance(evaluation_raw, Mapping) else {}
    )
    if options.delay is not None:
        base_delay: float = options.delay
    else:
        raw_delay: object = evaluation_cfg.get("request_delay", 0.0)
        if isinstance(raw_delay, bool) or not isinstance(raw_delay, (int, float)):
            raise UsageError(
                f"evaluation.request_delay must be a non-negative number, "
                f"got {raw_delay!r}."
            )
        as_float: float = float(raw_delay)
        if math.isnan(as_float) or math.isinf(as_float) or as_float < 0.0:
            raise UsageError(
                f"evaluation.request_delay must be a finite non-negative number, "
                f"got {raw_delay!r}."
            )
        base_delay = as_float

    return ExecutionPlan(
        sample_size=int(raw_size),
        mode_purpose=str(mode_block.get("purpose") or ""),
        model_configs=tuple(selected),
        evaluation_cfg_raw=evaluation_cfg,
        base_delay=base_delay,
    )


def resolve_logging_level(config: Mapping[str, object]) -> str:
    """讀取設定檔 ``logging.level`` 層級名稱（缺失時以 INFO 為預設）。

    Args:
        config: 已載入之設定字典。

    Returns:
        層級名稱字串（未正規化；由 ``setup_dual_logging`` 解析）。
    """
    logging_cfg: object = config.get("logging")
    raw: object = logging_cfg.get("level") if isinstance(logging_cfg, Mapping) else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return DEFAULT_LOG_LEVEL


def setup_dual_logging(log_path: Path, level_name: str = DEFAULT_LOG_LEVEL) -> None:
    """於 root logger 掛載終端機（stdout）＋日誌檔案之雙 handler。

    冪等：掛載前先移除既有的 root handlers（含先前執行的檔案 handler），
    避免同一行程重複設定造成重複輸出。

    日誌檔建立失敗（權限／路徑錯誤）時優雅降級為終端機-only 並記錄
    警告，不中斷管線。

    Args:
        log_path: 日誌檔案路徑（父目錄須已存在）。
        level_name: 層級名稱（DEBUG/INFO/WARNING/ERROR/CRITICAL）；
            無法辨識時回落 INFO。
    """
    normalized: str = level_name.strip().upper()
    level: int = getattr(logging, normalized, -1)
    if level not in _VALID_LEVELS:
        LOGGER.warning("Unknown logging level %r; falling back to INFO.", level_name)
        level = logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    file_handler_available: bool = True
    try:
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    except OSError:
        file_handler_available = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)
    if not file_handler_available:
        LOGGER.warning(
            "Log file %s is unavailable; degrading to terminal-only logging.",
            log_path,
        )


def resolve_request_delay(model_cfg: Mapping[str, object], base_delay: float) -> float:
    """計算單一模型之有效請求延遲（條件節流）。

    本機／離線模型（``huggingface``／``hf_pipeline``／``mock``）強制 0.0；
    僅 API 類模型套用 ``base_delay``。

    Args:
        model_cfg: 單一 ``models[]`` 設定區塊。
        base_delay: 基礎節流秒數（CLI ``--delay`` 或設定檔值）。

    Returns:
        該模型實際套用的延遲秒數。
    """
    model_type: object = model_cfg.get("type")
    normalized: str = str(model_type).strip().lower() if model_type is not None else ""
    if normalized in LOCAL_MODEL_TYPES:
        return 0.0
    return base_delay


def _safe_str(value: object) -> str:
    """安全轉換樣本欄位為字串（``None`` 映射為空字串）。"""
    if value is None:
        return ""
    return str(value)


def open_journal(path: Path) -> IO[str]:
    """以 append 模式開啟 JSONL 落盤檔案（即時落地入口）。

    Args:
        path: ``predictions_{timestamp}.jsonl`` 目標檔案路徑。

    Returns:
        文字模式檔案物件（append 模式；呼叫端負責關閉）。
    """
    return path.open("a", encoding="utf-8")


def _append_journal(journal: IO[str], entry: Mapping[str, object]) -> None:
    """寫入一列 JSONL 並立即 flush（防中斷資料遺失）。

    Args:
        journal: 已開啟之 JSONL 檔案物件。
        entry: 單一問題之落盤紀錄。
    """
    journal.write(json.dumps(dict(entry), ensure_ascii=False) + "\n")
    journal.flush()


def _build_samples(
    loader: MMLUDatasetLoader,
    sample_size: int,
    subject_entries: Sequence[Mapping[str, str]],
    limit: Optional[int],
) -> List[Dict[str, object]]:
    """以公開 API 組合一次性載入評測樣本。

    語意對齊 ``MMLUDatasetLoader.get_samples()``：依科目宣告順序載入、
    跳過缺 ``question_id``／``answer_letter`` 之列、以 ``format_prompt`` 組裝
    Prompt；唯一差異是抽樣筆數由 CLI ``--mode`` 驅動（而非 config
    ``active_mode``），以達成不修改已驗收 src 之約束。

    Args:
        loader: DataLoader 實體。
        sample_size: 每科目抽樣筆數（由 ``--mode`` 導出）。
        subject_entries: ``iter_subjects()`` 輸出。
        limit: 全域樣本上限；``None`` 表示不限制。

    Returns:
        評測樣本清單（每筆含 ``question_id``／``formatted_prompt``／
        ``target_letter``／``subject``／``category``）。
    """
    samples: List[Dict[str, object]] = []
    skipped_total = 0
    for entry in subject_entries:
        subject: str = _safe_str(entry.get("subject")).strip()
        if not subject:
            continue
        rows = loader.load_data(subject=subject, sample_size=sample_size)
        skipped_total += int(loader.skipped_rows)
        for row in rows:
            question_id: object = row.get("question_id")
            answer_letter: object = row.get("answer_letter")
            if question_id is None or answer_letter is None:
                continue
            samples.append(
                {
                    "question_id": str(question_id),
                    "formatted_prompt": loader.format_prompt(row),
                    "target_letter": str(answer_letter),
                    "subject": str(row.get("subject") or subject),
                    "category": _safe_str(entry.get("category")),
                }
            )
    if limit is not None:
        samples = samples[:limit]
    if skipped_total:
        LOGGER.info("Skipped %d malformed row(s) during normalization.", skipped_total)
    return samples


def _predict_one_record(
    model: BaseModelInterface, sample: Mapping[str, object]
) -> Dict[str, object]:
    """執行單題推論並驗證模型層三鍵契約（單題層容錯）。

    契約違規或意外例外一律轉為 ``ERROR:`` 哨兵紀錄（``latency`` 為 0.0），
    模型持續執行；該題於評估層的 INVALID 判定由 ``Evaluator`` 負責。

    Args:
        model: 模型推論介面。
        sample: 單一評測樣本。

    Returns:
        三鍵預測紀錄（``question_id``／``raw_output``／``latency``）；
        絕不拋出例外。
    """
    question_id: str = _safe_str(sample.get("question_id"))

    def _error_record(reason: str) -> Dict[str, object]:
        return {
            "question_id": question_id,
            "raw_output": f"{ERROR_PREFIX}{reason}",
            "latency": 0.0,
        }

    try:
        records: List[Dict[str, object]] = model.predict_batch([sample])
        if len(records) != 1:
            return _error_record(
                f"predict_batch returned {len(records)} record(s); expected exactly 1."
            )
        record: Mapping[str, object] = records[0]
        missing: frozenset[str] = PREDICTION_RECORD_KEYS - record.keys()
        if missing:
            return _error_record(f"prediction record missing key(s) {sorted(missing)}.")
        latency: object = record.get("latency")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
        ):
            # 措辭刻意避免獨立 A–D 字母：評估層 Tier 2 fallback（``\b([A-D])\b``
            # 大小寫不敏感）會把「... not a finite ...」的「a」誤判為選項答案，
            # 使哨兵紀錄被計為有效「A」。
            return _error_record(
                f"prediction latency {latency!r} failed the finite-number check."
            )
        raw_output: object = record.get("raw_output")
        if not isinstance(raw_output, str):
            LOGGER.warning(
                "Model %s returned non-string raw_output (%s); coercing to string.",
                model.model_name,
                type(raw_output).__name__,
            )
            raw_output = "" if raw_output is None else str(raw_output)
        return {
            "question_id": question_id,
            "raw_output": raw_output,
            "latency": float(latency),
        }
    except Exception as exc:
        LOGGER.exception(
            "Model %s single-sample inference failed for question %r; "
            "recording an ERROR sentinel and continuing.",
            model.model_name,
            question_id,
        )
        return _error_record(f"{type(exc).__name__}: {exc}")


def _journal_entry(
    model_name: str,
    sample: Mapping[str, object],
    record: Mapping[str, object],
) -> Dict[str, object]:
    """組裝單一問題的 JSONL 落盤紀錄（逐題明細＋原始字串）。

    Args:
        model_name: 模型顯示名稱。
        sample: 評測樣本（提供科目／領域／Prompt／目標答案）。
        record: 三鍵預測紀錄。

    Returns:
        可 JSON 序列化之落盤字典。
    """
    return {
        "model": model_name,
        "question_id": record.get("question_id", ""),
        "subject": _safe_str(sample.get("subject")),
        "category": _safe_str(sample.get("category")),
        "prompt": _safe_str(sample.get("formatted_prompt")),
        "target_letter": _safe_str(sample.get("target_letter")),
        "raw_output": record.get("raw_output", ""),
        "latency": record.get("latency", 0.0),
    }


@dataclass
class ModelResult:
    """單一模型之執行結果（完成／失敗＋錯誤資訊＋耗時＋指標）。

    Attributes:
        model_name: 模型顯示名稱（``models[].name``）。
        status: ``"completed"`` 或 ``"failed"``。
        error: 失敗原因（``"completed"`` 時為 ``None``）。
        total_seconds: 該模型區塊之牆鐘耗時（含節流休眠）。
        metrics: 評估指標字典（``EvaluationResult.to_dict()``）；失敗時
            為 ``None``。
    """

    model_name: str
    status: str
    error: Optional[str]
    total_seconds: float
    metrics: Optional[Dict[str, object]]

    def to_entry(self) -> Dict[str, object]:
        """轉為 JSON 可序列化之匯總條目。

        Returns:
            含模型名、狀態、錯誤、耗時與指標的字典。
        """
        return {
            "model_name": self.model_name,
            "status": self.status,
            "error": self.error,
            "total_seconds": self.total_seconds,
            "metrics": self.metrics,
        }


def execute_model(
    model_cfg: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    evaluation_cfg: Mapping[str, object],
    base_delay: float,
    evaluator: Evaluator,
    journal: IO[str],
) -> ModelResult:
    """執行單一模型之完整推論＋即時落盤＋評估週期（單模隔離）。

    執行流程（Step per Model）：
        1. ``build_model_interface`` 建構模型（失敗 → FAILED、跳過）。
        2. 逐題循序推論：``predict_batch([sample])`` → 三鍵契約檢查 →
           即時 append＋flush → 條件節流休眠（本機模型 0.0、末題不睡）。
        3. ``finally`` 區塊明確呼叫 ``release_vram()``（含例外中断路徑）。
        4. ``Evaluator.evaluate()`` 指標交接（失敗 → FAILED、跳過）。

    Args:
        model_cfg: 單一 ``models[]`` 設定區塊。
        samples: 預載評測樣本。
        evaluation_cfg: 全域 ``evaluation:`` 設定區塊（透傳給模型工廠）。
        base_delay: 基礎請求節流秒數（CLI 覆寫或設定檔值）。
        evaluator: 評估器實體。
        journal: 已開啟之 JSONL 落盤檔案物件。

    Returns:
        ``ModelResult``（絕不將例外拋回呼叫端）。
    """
    model_name: str = _safe_str(model_cfg.get("name")) or "unnamed-model"
    delay: float = resolve_request_delay(model_cfg, base_delay)
    start_wall: float = perf_counter()
    predictions: List[Dict[str, object]] = []
    metrics: Optional[Dict[str, object]] = None
    status: str = "completed"
    error: Optional[str] = None
    try:
        model: BaseModelInterface = build_model_interface(model_cfg, evaluation_cfg)
        total: int = len(samples)
        progress = tqdm(
            samples,
            total=total,
            desc=f"model:{model.model_name}",
            unit="q",
            mininterval=0.5,
        )
        for index, sample in enumerate(progress):
            record: Dict[str, object] = _predict_one_record(model, sample)
            predictions.append(record)
            _append_journal(journal, _journal_entry(model.model_name, sample, record))
            if index < total - 1 and delay > 0.0:
                sleep(delay)
        evaluation: EvaluationResult = evaluator.evaluate(
            samples, predictions, model_name=model.model_name
        )
        metrics = evaluation.to_dict()
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception(
            "Model %s failed and is isolated/skipped: %s", model_name, error
        )
    finally:
        release_vram()
    total_seconds: float = perf_counter() - start_wall
    return ModelResult(
        model_name=model_name,
        status=status,
        error=error,
        total_seconds=total_seconds,
        metrics=metrics,
    )


def _as_mapping(value: object) -> Dict[str, object]:
    """安全轉為字串鍵對應表；非 mapping 時回落空字典。"""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _metric_number(
    metrics: Mapping[str, object], key: str, default: float = 0.0
) -> float:
    """安全讀取指標數值（非數值／非有限數回落 ``default``）。

    Args:
        metrics: 指標字典（``EvaluationResult.to_dict()``）。
        key: 指標欄位名稱。
        default: 缺失或非法時的回落值。

    Returns:
        有限浮點數值。
    """
    value: object = metrics.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(default)
    number: float = float(value)
    if not math.isfinite(number):
        return float(default)
    return number


def build_csv_rows(
    model_results: Sequence[ModelResult],
    categories: Sequence[str],
    subjects: Sequence[str],
) -> Tuple[List[str], List[Dict[str, object]]]:
    """組裝 CSV 表頭與每模型一列之資料列。

    失敗模型列之指標欄位一律留空（保留 ``status``／``error``）。

    Args:
        model_results: 全模型結果（執行順序）。
        categories: 領域名稱（首次出現順序）。
        subjects: 科目名稱（首次出現順序）。

    Returns:
        ``(fields, rows)``：CSV 欄位順序與資料列清單。
    """
    fields: List[str] = list(CSV_CORE_FIELDS)
    fields.extend(f"domain_accuracy_{category}" for category in categories)
    fields.extend(f"subject_accuracy_{subject}" for subject in subjects)
    rows: List[Dict[str, object]] = []
    for result in model_results:
        if result.status == "completed" and result.metrics is not None:
            domain_accuracy = _as_mapping(result.metrics.get("domain_accuracy"))
            subject_accuracy = _as_mapping(result.metrics.get("subject_accuracy"))
            row: Dict[str, object] = {
                "model": result.model_name,
                "status": "completed",
                "error": "",
                "total_samples": result.metrics.get("total_samples", ""),
                "valid_samples": result.metrics.get("valid_samples", ""),
                "correct_samples": result.metrics.get("correct_samples", ""),
                "invalid_samples": result.metrics.get("invalid_samples", ""),
                "missing_predictions": result.metrics.get("missing_predictions", 0), 
                "overall_accuracy": result.metrics.get("overall_accuracy", ""),
                "valid_accuracy": result.metrics.get("valid_accuracy", ""),
                "invalid_parsing_rate": result.metrics.get("invalid_parsing_rate", ""),
                "domain_accuracy_mean": result.metrics.get("domain_accuracy_mean", ""),
                "average_latency_seconds": result.metrics.get(
                    "average_latency_seconds", ""
                ),
                "recall_std": result.metrics.get("recall_std", ""),
                "total_seconds": result.total_seconds,
            }
            for category in categories:
                row[f"domain_accuracy_{category}"] = domain_accuracy.get(category, "")
            for subject in subjects:
                row[f"subject_accuracy_{subject}"] = subject_accuracy.get(subject, "")
        else:
            row = {
                "model": result.model_name,
                "status": "failed",
                "error": result.error or "",
            }
            for field in fields:
                row.setdefault(field, "")
        rows.append(row)
    return fields, rows


def _render_csv(fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    """以 ``csv`` 模組渲染 CSV 字串（含表頭列）。

    Args:
        fields: 欄位順序。
        rows: 資料列清單。

    Returns:
        完整 CSV 內容字串（Unix 換行）。
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fields))
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue()


def write_metrics_files(
    summary: Mapping[str, object],
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    results_dir: Path,
    timestamp: str,
) -> Tuple[Path, Path, Path, Path]:
    """寫入時間戳記＋latest 別名之 JSON／CSV 匯總報告。

    ``metrics_summary_latest.*`` 每次執行覆寫，作為後續視覺化之固定存取點。

    Args:
        summary: ``metrics_summary`` JSON 載荷（``metadata`` ＋ ``models``）。
        fields: CSV 表頭欄位順序。
        rows: CSV 資料列（每模型一列）。
        results_dir: ``results/{mode}/`` 目錄（須已存在）。
        timestamp: 本次執行之 ``YYYYMMDD_HHMMSS`` 時間戳記。

    Returns:
        四個產出路徑 ``(json_ts, csv_ts, json_latest, csv_latest)``。
    """
    json_ts = results_dir / f"metrics_summary_{timestamp}.json"
    csv_ts = results_dir / f"metrics_summary_{timestamp}.csv"
    json_latest = results_dir / "metrics_summary_latest.json"
    csv_latest = results_dir / "metrics_summary_latest.csv"
    payload: str = json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n"
    json_ts.write_text(payload, encoding="utf-8")
    json_latest.write_text(payload, encoding="utf-8")
    csv_payload: str = _render_csv(fields, rows)
    csv_ts.write_text(csv_payload, encoding="utf-8")
    csv_latest.write_text(csv_payload, encoding="utf-8")
    return json_ts, csv_ts, json_latest, csv_latest


def _domain_subject_axes(
    subject_entries: Sequence[Mapping[str, str]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """自科目條目依首次出現順序抽出（領域軸、科目軸）。

    Args:
        subject_entries: ``iter_subjects()`` 輸出。

    Returns:
        ``(categories, subjects)`` 名稱元組。
    """
    categories: List[str] = []
    subjects: List[str] = []
    for entry in subject_entries:
        category: str = _safe_str(entry.get("category")).strip()
        subject: str = _safe_str(entry.get("subject")).strip()
        if category and category not in categories:
            categories.append(category)
        if subject and subject not in subjects:
            subjects.append(subject)
    return tuple(categories), tuple(subjects)


def run_pipeline(options: CliOptions) -> int:
    """執行完整評測管線，回傳行程 exit code。

    對齊 .clinerules §2 單一方向管線生命週期：
        設定驗證（fail-fast，exit 2）→ 時間戳記目錄分流＋Dual Logging →
        One-time Loading（失敗 exit 1）→ Sequential Model Loop
        （即時落盤＋單模隔離＋finally 釋放 VRAM）→ Aggregation 持久化；
        I/O 失敗（journal 開檔／指標寫入）→ exit 1，日誌檔建立失敗降級
        為終端機-only。

    Args:
        options: 經 ``parse_args`` 驗證之 CLI 選項。

    Returns:
        Exit code：0（≥1 模型成功）／1（致命錯誤）／2（CLI 或設定誤用）。
    """
    try:
        config: Dict[str, Any] = load_config(options.config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        plan: ExecutionPlan = resolve_execution_plan(config, options)
    except UsageError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    log_dir = Path(LOGS_ROOT_NAME) / options.mode
    results_dir = Path(options.output_dir) / options.mode
    timestamp: str = datetime.now().strftime(TIMESTAMP_FORMAT)
    log_path = log_dir / f"pipeline_{timestamp}.log"
    predictions_path = results_dir / f"predictions_{timestamp}.jsonl"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Failed to create output directories: {exc}", file=sys.stderr)
        return 1
    setup_dual_logging(log_path, resolve_logging_level(config))
    LOGGER.info(
        "MMLU benchmark pipeline starting: mode=%s config=%s output_dir=%s "
        "models=%s base_delay=%ss limit=%s",
        options.mode,
        options.config_path,
        options.output_dir,
        [_block_name(block) for block in plan.model_configs],
        plan.base_delay,
        options.limit,
    )

    try:
        loader = MMLUDatasetLoader(options.config_path, num_shots=options.shots)
        subject_entries = loader.iter_subjects()
        samples = _build_samples(
            loader, plan.sample_size, subject_entries, options.limit
        )
    except Exception as exc:
        LOGGER.exception("Failed to load MMLU samples: %s", exc)
        return 1
    if not samples:
        LOGGER.error(
            "No samples available after loading/sampling; aborting without model runs."
        )
        return 1
    LOGGER.info(
        "One-time loading complete: %d sample(s) across %d subject(s) "
        "(sample_size_per_subject=%d, limit=%s).",
        len(samples),
        len(subject_entries),
        plan.sample_size,
        options.limit,
    )

    evaluator = Evaluator.from_config(config)
    results: List[ModelResult] = []
    try:
        journal_file = open_journal(predictions_path)
    except OSError as exc:
        LOGGER.error("Failed to open predictions journal %s: %s", predictions_path, exc)
        return 1
    with journal_file:
        for index, model_cfg in enumerate(plan.model_configs):
            LOGGER.info(
                "Model %d/%d: %s (type=%s, effective_delay=%.3fs)",
                index + 1,
                len(plan.model_configs),
                _block_name(model_cfg) or f"model-{index + 1}",
                str(model_cfg.get("type")),
                resolve_request_delay(model_cfg, plan.base_delay),
            )
            results.append(
                execute_model(
                    model_cfg=model_cfg,
                    samples=samples,
                    evaluation_cfg=plan.evaluation_cfg_raw,
                    base_delay=plan.base_delay,
                    evaluator=evaluator,
                    journal=journal_file,
                )
            )

    categories, subjects = _domain_subject_axes(subject_entries)
    metadata: Dict[str, object] = {
        "project": _safe_str(config.get("project", {}).get("name")),
        "config_version": _safe_str(config.get("project", {}).get("version")),
        "config_path": options.config_path,
        "mode": options.mode,
        "mode_purpose": plan.mode_purpose,
        "sample_size_per_subject": plan.sample_size,
        "num_shots": getattr(loader, "_num_shots", 0),
        "shots_source": "cli" if options.shots is not None else "config",
        "total_samples": len(samples),
        "subjects": [
            {
                "category": _safe_str(entry.get("category")),
                "subject": _safe_str(entry.get("subject")),
            }
            for entry in subject_entries
        ],
        "request_delay": plan.base_delay,
        "delay_source": "cli" if options.delay is not None else "config",
        "output_dir": options.output_dir,
        "log_file": str(log_path),
        "timestamp": timestamp,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary: Dict[str, object] = {
        "metadata": metadata,
        "models": [result.to_entry() for result in results],
    }
    fields, rows = build_csv_rows(results, categories, subjects)
    try:
        write_metrics_files(summary, fields, rows, results_dir, timestamp)
    except OSError as exc:
        LOGGER.error(
            "Failed to write metrics summary artifacts to %s: %s", results_dir, exc
        )
        return 1

    completed = [result for result in results if result.status == "completed"]
    for result in results:
        if result.status == "completed" and result.metrics is not None:
            LOGGER.info(
                "Model summary: %s accuracy=%.4f total=%d avg_latency=%.4fs "
                "status=completed",
                result.model_name,
                _metric_number(result.metrics, "overall_accuracy"),
                int(_metric_number(result.metrics, "total_samples")),
                _metric_number(result.metrics, "average_latency_seconds"),
            )
        else:
            LOGGER.error(
                "Model summary: %s status=failed error=%s",
                result.model_name,
                result.error,
            )
    if completed:
        LOGGER.info(
            "Pipeline finished: %d/%d model(s) completed. Artifacts: %s, %s",
            len(completed),
            len(results),
            log_path,
            results_dir,
        )
        return 0
    LOGGER.error(
        "Pipeline finished with failures: 0/%d model(s) completed.", len(results)
    )
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 進入點。

    Args:
        argv: 參數清單；``None`` 時使用 ``sys.argv[1:]``。

    Returns:
        行程 exit code（語意見 :func:`run_pipeline`）。
    """
    options: CliOptions = parse_args(argv)
    return run_pipeline(options)


if __name__ == "__main__":
    sys.exit(main())
