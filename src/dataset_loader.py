"""MMLU 資料集載入、標籤映射與 Prompt 組裝模組。"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

try:
    # 選用依賴：datasets 未安裝或缺型別 stub
    from datasets import load_dataset  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    load_dataset = None

LOGGER = logging.getLogger(__name__)

LABEL_TO_LETTER: Mapping[int, str] = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_SET = frozenset(LABEL_TO_LETTER.values())
EXPECTED_CHOICE_COUNT = 4

DEFAULT_PROMPT_TEMPLATE: str = (
    "The following are multiple choice questions (with answers) about {subject}.\n"
    "\n"
    "Question: {question}\n"
    "A. {choice_A}\n"
    "B. {choice_B}\n"
    "C. {choice_C}\n"
    "D. {choice_D}\n"
    "\n"
    "Format your output strictly as: 'The correct answer is (X)' where X is A, B, C, or D.\n"
    "Answer:"
)

# 設定檔 dataset.prompt_template 缺失時的內建 fallback 範本；哨兵值用於佔位符預驗證。
PROMPT_PLACEHOLDERS: Mapping[str, str] = {
    "subject": "subject-sentinel",
    "question": "question-sentinel",
    "choice_A": "A-sentinel",
    "choice_B": "B-sentinel",
    "choice_C": "C-sentinel",
    "choice_D": "D-sentinel",
}

# project.seed 缺失時的預設全域種子。
DEFAULT_SEED: int = 42


class MMLUDatasetLoader:
    """負責載入 MMLU、正規化欄位並產出結構化評測樣本。

    依 ``dataset.categories`` 解析多科目領域結構，以
    ``dataset.active_mode`` 決定每科目抽樣筆數，並以 ``project.seed``
    建立可重現的抽樣隨機流；Prompt 由 ``dataset.prompt_template`` 組裝。

    Attributes:
        config_path: YAML 設定檔路徑。
        config: 解析後的設定內容。
        skipped_rows: 最近一次 ``load_data`` 因驗證失敗而略過的列數。
        _subjects: 由 ``dataset.categories`` 解析出的科目清單（category/subject/focus）。
        _seed: 全域抽樣種子（``project.seed``）。
        _prompt_template: 已驗證的 Prompt 範本字串。
    """

    def __init__(
        self, config_path: str = "configs/eval_config.yaml", num_shots: Optional[int] = None
    ) -> None:
        """載入 YAML 設定檔並初始化內部快取。

        同時解析並驗證：``dataset.categories`` 多子集結構、``dataset.active_mode``
        抽樣模式、``project.seed`` 全域種子、``dataset.prompt_template``
        範本佔位符，以及 ``dataset.cache_dir`` 快取目錄。

        Args:
            config_path: 評測設定檔路徑，預設為 ``configs/eval_config.yaml``。

        Raises:
            FileNotFoundError: 當設定檔不存在時。
            ValueError: 當 YAML 內容不是對應表，或必要結構
                （categories / active_mode / seed / prompt 佔位符）缺失或非法時。
        """
        self.config_path: str = config_path
        self.config: Dict[str, Any] = self._load_config(config_path)
        self._records: List[Dict[str, Any]] = []
        self.skipped_rows: int = 0
        dataset_cfg: Dict[str, Any] = self._dataset_config()
        self._dataset_name: str = str(dataset_cfg.get("name") or "cais/mmlu")
        self._split: str = str(dataset_cfg.get("split") or "test").strip() or "test"
        self._cache_dir: Optional[str] = self._resolve_cache_dir(dataset_cfg)
        self._seed: int = self._resolve_seed()
        self._prompt_template: str = self._resolve_prompt_template(dataset_cfg)
        self._subjects: List[Dict[str, str]] = self._parse_subjects(dataset_cfg)
        # ★ num_shots: priority CLI > YAML config > default 0
        self._num_shots: int = (
            num_shots
            if num_shots is not None
            else self.config.get("dataset", {}).get("num_shots", 0)
        )
        self._active_mode: str = self._validate_active_mode()
        LOGGER.info(
            "MMLUDatasetLoader ready: dataset=%s split=%s active_mode=%s subjects=%s seed=%s num_shots=%s",
            self._dataset_name,
            self._split,
            self._active_mode,
            [entry["subject"] for entry in self._subjects],
            self._seed,
            self._num_shots,
        )

    def load_data(
        self,
        subject: Optional[str] = None,
        split: Optional[str] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """自 Hugging Face Hub 載入 MMLU 子集並正規化資料列。

        使用 ``datasets.load_dataset`` 讀取資料，將數字標籤映射為 A-D。
        無法安全正規化的列會被略過（嚴格拒絕髒資料），並累計至
        ``skipped_rows``，同時輸出警告日誌。

        當有效列數超過 ``sample_size`` 時，以由 ``(seed, subject)`` 派生的
        隨機流抽取確定性子集（同輸入跨執行結果一致）；抽樣列保留原始
        相對順序，使日誌與報表呈現穩定。

        Args:
            subject: MMLU 科目名稱；必填（新版設定檔不再含單一 ``dataset.subject`` 鍵）。
            split: 資料切分名稱；預設採用設定檔 ``dataset.split``。
            sample_size: 最多保留的筆數；``None`` 表示改用目前 ``active_mode``
                抽樣模式的 ``sample_size_per_subject``；非正數回傳空清單。
            seed: 抽樣種子；``None`` 表示使用設定檔 ``project.seed``。

        Returns:
            正規化後的資料列清單。空資料集會回傳空清單。

        Raises:
            ImportError: 當 ``datasets`` 套件不可用時。
            ValueError: 當資料集載入失敗、subject 缺失、split 不存在、
                抽樣模式無效，或 ``sample_size``／``seed`` 非整數時。
        """
        resolved_subject: str = str(subject or "").strip()
        if not resolved_subject:
            raise ValueError("MMLU subject is missing from arguments.")
        resolved_split: str = str(split or self._split).strip() or self._split

        if sample_size is None:
            resolved_size: int = self.resolve_sample_size()
        else:
            if isinstance(sample_size, bool) or not isinstance(sample_size, int):
                raise ValueError(
                    f"sample_size must be an integer, got {sample_size!r}."
                )
            resolved_size = sample_size

        if load_dataset is None:
            raise ImportError("The 'datasets' package is required to load MMLU.")

        try:
            raw_dataset: Any = load_dataset(
                self._dataset_name,
                resolved_subject,
                **self._load_kwargs(resolved_split),
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to load dataset '{self._dataset_name}' for subject '{resolved_subject}'."
            ) from exc

        records: List[Dict[str, Any]] = self._coerce_records(
            raw_dataset, split=resolved_split
        )
        normalized: List[Dict[str, Any]] = []
        skipped = 0
        for index, row in enumerate(records):
            item: Optional[Dict[str, Any]] = self._normalize_row(
                row=row,
                fallback_subject=resolved_subject,
                index=index,
            )
            if item is None:
                skipped += 1
                continue
            normalized.append(item)

        self.skipped_rows = skipped
        if skipped:
            LOGGER.warning(
                "Skipped %s/%s MMLU rows during normalization for subject=%r.",
                skipped,
                len(records),
                resolved_subject,
            )

        if resolved_size <= 0:
            normalized = []
        elif len(normalized) > resolved_size:
            # seed=None 為文件化預設值（使用設定檔 project.seed），須先
            # resolve 再驗證；``self._seed`` 已於 __init__ 驗證為整數。
            resolved_seed: int = self._seed if seed is None else seed
            if isinstance(resolved_seed, bool) or not isinstance(resolved_seed, int):
                raise ValueError(f"seed must be an integer, got {seed!r}.")
            normalized = self._seeded_sample(
                normalized, resolved_size, resolved_seed, resolved_subject
            )
        elif normalized and len(normalized) < resolved_size:
            LOGGER.warning(
                "Subject %r has only %s valid rows; keeping all instead of sampling %s.",
                resolved_subject,
                len(normalized),
                resolved_size,
            )

        self._records = normalized
        return list(self._records)

    def format_prompt(self, item: Mapping[str, Any]) -> str:
        """將單一題目組裝為結構化多選一 Prompt。

        支援 Zero-Shot (num_shots=0) 與 Few-Shot (num_shots>0) 兩種模式：
        - Zero-Shot：直接使用設定檔之 prompt template，不附加範例。
        - Few-Shot：在目標測試題目前，插入 ``num_shots`` 筆來自 ``dev`` split 的範例。
          每個範例僅包含問題、選項與標準答案；目標題目使用完整範本（包含輸出格式指令與 Answer: 提示）。

        Args:
            item: 單筆題目資料，包含 ``question``、``choices``、``subject`` 等欄位。

        Returns:
            符合評測規範之 Prompt 字串。
        """
        subject: str = str(item.get("subject") or "unknown")
        question: str = str(item.get("question") or "")
        choice_a, choice_b, choice_c, choice_d = self._extract_choices(item)

        # Zero-Shot 模式：直接套用標準範本
        if self._num_shots == 0:
            return self._prompt_template.format(
                subject=subject,
                question=question,
                choice_A=choice_a,
                choice_B=choice_b,
                choice_C=choice_c,
                choice_D=choice_d,
            )

        # Few-Shot 模式：自 dev split 載入範例
        exemplars: List[Dict[str, Any]] = self._load_dev_exemplars(subject, self._num_shots)
        parts: List[str] = []

        # 1. 組裝 Few-Shot 範例：乾淨的「問題 + 選項 + 標準答案」，不套用含格式指令的 template
        for ex in exemplars:
            ex_q: str = str(ex.get("question") or "")
            ex_a, ex_b, ex_c, ex_d = self._extract_choices(ex)
            ex_ans: str = str(ex.get("answer_letter") or "").strip()
            
            parts.append(
                f"Question: {ex_q}\n"
                f"A. {ex_a}\n"
                f"B. {ex_b}\n"
                f"C. {ex_c}\n"
                f"D. {ex_d}\n"
                f"Answer: The correct answer is ({ex_ans})"
            )

        # 2. 組裝目標題目：套用 self._prompt_template（內部已自帶開頭引言與結尾格式指令）
        target_part = self._prompt_template.format(
            subject=subject,
            question=question,
            choice_A=choice_a,
            choice_B=choice_b,
            choice_C=choice_c,
            choice_D=choice_d,
        )
        parts.append(target_part)

        # 3. 雙換行合併所有段落
        return "\n\n".join(parts)


    def _load_dev_exemplars(self, subject: str, k: int) -> List[Dict[str, Any]]:
        """從 ``dev`` split 載入 ``k`` 筆 Few-Shot 範例。

        關鍵設計：
        - 來源必須是 ``dev`` split，而非 ``test`` split，以防止資料外洩。
        - 使用 SHA-512 seed（``project.seed :: subject``）確定性抽取前 ``k`` 題。
        - 範例包含完整欄位（question, choices, answer_letter），含標準答案。
        - 若 dev 題目數量 < ``k``，則回傳全部可用題目而不報錯。

        Args:
            subject: MMLU 科目名稱。
            k: 要抽取的範例筆數。

        Returns:
            長度為 ``min(k, available_dev_count)`` 的範例清單。
        """
        from src.dataset_loader import load_dataset  # local import to avoid circular

        # 決定使用的 split 為 dev
        dev_split: str = "dev"

        try:
            raw_dataset = load_dataset(
                self._dataset_name,
                subject,
                **self._load_kwargs(dev_split),
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed to load dev dataset for subject=%s: %s", subject, exc
            )
            return []

        # 正規化資料列
        records: List[Dict[str, Any]] = self._coerce_records(raw_dataset, split=dev_split)
        normalized: List[Dict[str, Any]] = []
        for row in records:
            item: Optional[Dict[str, Any]] = self._normalize_row(
                row=row, fallback_subject=subject, index=0
            )
            if item is not None:
                normalized.append(item)

        if not normalized:
            LOGGER.warning("No dev records found for subject=%s", subject)
            return []

        # 使用由 (seed, subject) 派生的種子確定性抽取前 k 題
        rng = random.Random(f"{self._seed}::{subject}")
        # 取得所有題目的索引，並隨機抽取 k 筆（或全部如果不足）
        all_indices: List[int] = list(range(len(normalized)))
        chosen_count: int = min(k, len(all_indices))
        chosen_indices: List[int] = sorted(rng.sample(all_indices, chosen_count))
        exemplars: List[Dict[str, Any]] = [normalized[i] for i in chosen_indices]

        LOGGER.debug(
            "Loaded %d Few-Shot exemplars from dev split for subject=%s (out of %d available)",
            len(exemplars),
            subject,
            len(normalized),
        )
        return exemplars

    def get_samples(self) -> List[Dict[str, Any]]:
        """回傳全部設定科目的可供評測樣本清單。

        依 ``dataset.categories`` 的宣告順序逐科載入，使用目前
        ``dataset.active_mode`` 抽樣模式的抽樣筆數與 ``project.seed``，
        並以 ``dataset.prompt_template`` 範本組裝 Prompt。

        Returns:
            每筆包含 ``question_id``、``formatted_prompt``、``target_letter``、
            ``subject`` 與 ``category`` 的字典清單。
        """
        sample_size: int = self.resolve_sample_size()
        samples: List[Dict[str, Any]] = []
        for entry in self._subjects:
            records: List[Dict[str, Any]] = self.load_data(
                subject=entry["subject"],
                split=self._split,
                sample_size=sample_size,
                seed=self._seed,
            )
            for item in records:
                question_id = item.get("question_id")
                answer_letter = item.get("answer_letter")
                if question_id is None or answer_letter is None:
                    continue
                samples.append(
                    {
                        "question_id": str(question_id),
                        "formatted_prompt": self.format_prompt(item),
                        "target_letter": str(answer_letter),
                        "subject": str(item.get("subject") or entry["subject"]),
                        "category": entry["category"],
                    }
                )
        return samples

    def iter_subjects(self) -> List[Dict[str, str]]:
        """回傳 ``dataset.categories`` 設定的評測科目清單（YAML 宣告順序）。

        Returns:
            每筆含 ``category``（領域名）、``subject``（MMLU 子集名）與
            ``focus``（考察重點說明，可能為空字串）的字典清單。

        Note:
            結構驗證於 ``__init__`` 執行；設定無效時實例無法建立，
            故此處不再重複驗證。
        """
        return list(self._subjects)

    def resolve_sample_size(self) -> int:
        """回傳目前 ``dataset.active_mode`` 抽樣模式的每科目抽樣筆數。

        Example:
            ``active_mode: "smoke_test"`` 時回傳
            ``dataset.modes.smoke_test.sample_size_per_subject``。

        Raises:
            ValueError: 當 ``dataset.modes`` 缺失或為空、``active_mode``
                缺失或未定義於 ``modes``、或對應的
                ``sample_size_per_subject`` 非正整數時。
        """
        dataset_cfg: Dict[str, Any] = self._dataset_config()
        active_mode: str = self._validate_active_mode()
        mode_cfg: Dict[str, Any] = dataset_cfg["modes"][active_mode]
        raw_size: Any = mode_cfg.get("sample_size_per_subject")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise ValueError(
                f"dataset.modes[{active_mode!r}].sample_size_per_subject must be an integer, "
                f"got {raw_size!r}."
            )
        if raw_size <= 0:
            raise ValueError(
                f"dataset.modes[{active_mode!r}].sample_size_per_subject must be positive, "
                f"got {raw_size}."
            )
        return int(raw_size)

    @staticmethod
    def map_numeric_to_letter(label: Any) -> str:
        """將答案標籤精確映射為 A-D。

        接受 ``0..3`` 整數、可解析為整數的字串（如 ``" 2 "``）、以及字母
        ``A``-``D``（大小寫不敏感）。明確拒絕 ``bool``、含小數的 ``float``、
        以及無法安全轉換的值。

        Args:
            label: 原始答案標籤。

        Returns:
            對應的選項字母 ``A``、``B``、``C`` 或 ``D``。

        Raises:
            ValueError: 當標籤無法映射為 A-D 時。
        """
        if isinstance(label, bool):
            raise ValueError(f"Boolean answer label is not allowed: {label!r}.")

        if isinstance(label, str):
            text = label.strip()
            if text.upper() in LETTER_SET:
                return text.upper()
            if text.isdigit() or (text.startswith(("+", "-")) and text[1:].isdigit()):
                numeric_label = int(text)
            else:
                raise ValueError(f"Answer label cannot be mapped: {label!r}.")
        elif isinstance(label, int):
            numeric_label = label
        elif isinstance(label, float):
            if not math.isfinite(label) or not label.is_integer():
                raise ValueError(f"Non-integer float answer label: {label!r}.")
            numeric_label = int(label)
        else:
            # 相容 numpy 整數等；仍拒絕無法轉成「整數語義」者。
            try:
                as_float = float(label)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Answer label cannot be mapped: {label!r}.") from exc
            if not math.isfinite(as_float) or not float(as_float).is_integer():
                raise ValueError(f"Answer label cannot be mapped: {label!r}.")
            numeric_label = int(as_float)

        letter: Optional[str] = LABEL_TO_LETTER.get(numeric_label)
        if letter is None:
            raise ValueError(f"Answer label is out of range: {label!r}.")
        return letter

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """讀取並驗證 YAML 設定檔。

        Args:
            config_path: 設定檔路徑。

        Returns:
            解析後的設定字典。

        Raises:
            FileNotFoundError: 當檔案不存在時。
            ValueError: 當內容不是字典時。
        """
        path: Path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded: Any = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Evaluation config must be a mapping.")
        return loaded

    def _coerce_records(self, raw_dataset: Any, split: str) -> List[Dict[str, Any]]:
        """將 Hugging Face Dataset 或序列轉為字典清單。

        Args:
            raw_dataset: ``load_dataset`` 回傳物件，或可迭代的列資料。
            split: 期望的資料切分名稱；DatasetDict 缺少該鍵時應失敗。

        Returns:
            以字典表示的資料列。無法轉譯時回傳空清單。

        Raises:
            ValueError: 當輸入為 DatasetDict 且缺少指定 split 時。
        """
        if raw_dataset is None:
            return []
        if isinstance(raw_dataset, dict):
            if split not in raw_dataset:
                raise ValueError(
                    f"Dataset mapping missing required split {split!r}; "
                    f"available={list(raw_dataset.keys())!r}."
                )
            return self._coerce_records(raw_dataset[split], split=split)
        try:
            rows: List[Any] = list(raw_dataset)
        except TypeError:
            return []
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def _normalize_row(
        self,
        row: Mapping[str, Any],
        fallback_subject: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """正規化單列資料；驗證失敗時回傳 ``None``。

        Args:
            row: 原始資料列。
            fallback_subject: 當列中缺少 ``subject`` 時使用的科目名稱。
            index: 列索引，用於產生 ``question_id``。

        Returns:
            正規化後的資料字典；若答案或選項無法驗證則回傳 ``None``。
        """
        subject: str = (
            str(row.get("subject") or fallback_subject).strip() or fallback_subject
        )
        question: str = str(row.get("question") or "").strip()
        try:
            choices: List[str] = list(self._extract_choices(row))
            answer_letter: str = self.map_numeric_to_letter(row.get("answer"))
        except ValueError as exc:
            LOGGER.debug("Skip row index=%s reason=%s", index, exc)
            return None

        question_id = self._build_question_id(subject, index, question, choices)
        return {
            "question_id": question_id,
            "question": question,
            "choices": choices,
            "answer": row.get("answer"),
            "answer_letter": answer_letter,
            "subject": subject,
        }

    @staticmethod
    def _build_question_id(
        subject: str,
        index: int,
        question: str,
        choices: Sequence[str],
    ) -> str:
        """建立低碰撞風險的題目識別碼。

        Args:
            subject: 科目名稱。
            index: 列索引。
            question: 題幹文字。
            choices: 四個選項字串。

        Returns:
            格式為 ``{subject}__{index:06d}__{sha1前8碼}`` 的識別碼。
        """
        digest_src = "|".join([question, *choices])
        digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:8]
        return f"{subject}__{index:06d}__{digest}"

    def _extract_choices(self, item: Mapping[str, Any]) -> Tuple[str, str, str, str]:
        """擷取恰好 4 個非 None 選項字串；絕不使用空字串填補。

        Args:
            item: 單筆題目資料。

        Returns:
            長度固定為 4 的選項 tuple。

        Raises:
            ValueError: 當選項數量不等於 4、含 ``None``、或結構無法解析時。
        """
        raw_choices: Any = item.get("choices")
        values: List[Any]

        if isinstance(raw_choices, Mapping):
            values = [
                raw_choices["A"] if "A" in raw_choices else raw_choices.get(0),
                raw_choices["B"] if "B" in raw_choices else raw_choices.get(1),
                raw_choices["C"] if "C" in raw_choices else raw_choices.get(2),
                raw_choices["D"] if "D" in raw_choices else raw_choices.get(3),
            ]
        elif isinstance(raw_choices, (list, tuple)):
            values = list(raw_choices)
        else:
            # 無合法 choices 結構時，視為髒資料（不填補空字串）。
            raise ValueError(
                f"Choices must be a sequence or mapping, got {type(raw_choices)!r}."
            )

        if len(values) != EXPECTED_CHOICE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_CHOICE_COUNT} choices, got {len(values)}."
            )
        if any(value is None for value in values):
            raise ValueError(f"Choice list contains None: {values!r}.")

        return str(values[0]), str(values[1]), str(values[2]), str(values[3])

    @staticmethod
    def _seeded_sample(
        rows: Sequence[Dict[str, Any]],
        k: int,
        seed: int,
        subject: str,
    ) -> List[Dict[str, Any]]:
        """以由 ``(seed, subject)`` 派生的種子從列池中確定性抽取 ``k`` 列。

        使用字串化 ``random.Random`` 作為種子（SHA-512 派生、不受程序層
        hash 隨機化影響），確保同輸入跨執行可重現同一抽樣結果。
        抽樣列保留原始相對順序（依索引升序），使日誌與報表呈現穩定。

        Args:
            rows: 完整列池（呼叫端保證 ``0 < k < len(rows)``）。
            k: 抽樣筆數。
            seed: 全域種子（通常來自 ``project.seed``）。
            subject: 科目名稱，使不同科目的隨機流彼此解耦。

        Returns:
            抽樣後的列清單，長度為 ``k``。
        """
        rng = random.Random(f"{seed}::{subject}")
        chosen_indices: List[int] = sorted(rng.sample(range(len(rows)), k))
        return [rows[index] for index in chosen_indices]

    def _resolve_seed(self) -> int:
        """讀取 ``project.seed`` 全域種子。

        Returns:
            整數種子；缺失時回退 ``DEFAULT_SEED``（42）並輸出警告日誌。

        Raises:
            ValueError: 當 seed 存在但非整數時。
        """
        project_cfg: Any = self.config.get("project")
        raw: Any = project_cfg.get("seed") if isinstance(project_cfg, dict) else None
        if raw is None:
            LOGGER.warning(
                "project.seed is missing; falling back to default seed %s.",
                DEFAULT_SEED,
            )
            return DEFAULT_SEED
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"project.seed must be an integer, got {raw!r}.")
        return int(raw)

    def _resolve_prompt_template(self, dataset_cfg: Mapping[str, Any]) -> str:
        """讀取並驗證 ``dataset.prompt_template``。

        缺失或空值時輸出警告並回退內建預設範本；再以哨兵值呼叫一次
        ``str.format`` 預驗證佔位符，使未知佔位符（如 ``{choice_E}``）
        或格式錯誤於初始化時即失敗（fail-fast）。

        Args:
            dataset_cfg: ``dataset`` 區塊內容。

        Returns:
            已驗證的範本字串。

        Raises:
            ValueError: 當範本非字串、或佔位符無效時。
        """
        raw: Any = dataset_cfg.get("prompt_template")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            LOGGER.warning(
                "dataset.prompt_template is missing; using built-in default template."
            )
            raw = DEFAULT_PROMPT_TEMPLATE
        if not isinstance(raw, str):
            raise ValueError(
                f"dataset.prompt_template must be a string, got {type(raw).__name__}."
            )
        try:
            raw.format(**PROMPT_PLACEHOLDERS)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"dataset.prompt_template contains invalid placeholders: {exc}. "
                f"Expected fields: {sorted(PROMPT_PLACEHOLDERS)}."
            ) from exc
        return raw

    @staticmethod
    def _resolve_cache_dir(dataset_cfg: Mapping[str, Any]) -> Optional[str]:
        """讀取 ``dataset.cache_dir``；缺失或空值時回傳 ``None``。"""
        raw: Any = dataset_cfg.get("cache_dir")
        if raw is None:
            return None
        text: str = str(raw).strip()
        return text or None

    def _load_kwargs(self, split: str) -> Dict[str, Any]:
        """建立 ``load_dataset`` 關鍵字參數；設定 ``cache_dir`` 時注入快取目錄。

        優先使用安裝之 datasets 版本支援的 ``cache_dir`` 參數，
        否則退而使用 ``download_cache``；並預設設定
        ``HF_DATASETS_CACHE`` 環境變數（已設定時不覆寫），避免重複下載。

        Args:
            split: 資料切分名稱。

        Returns:
            供 ``load_dataset`` 使用的關鍵字參數字典（必含 ``split``）。
        """
        kwargs: Dict[str, Any] = {"split": split}
        if not self._cache_dir or load_dataset is None:
            return kwargs
        try:
            supported: set[str] = set(inspect.signature(load_dataset).parameters)
        except Exception:  # pragma: no cover - 簽名內省異常時退回環境變數
            supported = set()
        if "cache_dir" in supported:
            kwargs["cache_dir"] = self._cache_dir
        elif "download_cache" in supported:
            kwargs["download_cache"] = self._cache_dir
        os.environ.setdefault("HF_DATASETS_CACHE", self._cache_dir)
        return kwargs

    def _dataset_config(self) -> Dict[str, Any]:
        """回傳 ``dataset`` 區塊並驗證其為對應表。

        Raises:
            ValueError: 當區塊缺失或不是對應表時。
        """
        dataset_cfg: Any = self.config.get("dataset")
        if not isinstance(dataset_cfg, dict):
            raise ValueError("Configuration must contain a 'dataset' mapping.")
        return dataset_cfg

    def _validate_active_mode(self) -> str:
        """驗證 ``dataset.active_mode`` 抽樣模式，回傳其名稱。

        於 ``__init__`` 與 ``resolve_sample_size`` 共用，使無效模式
        （缺失、未定義於 ``modes``、或 ``modes`` 結構異常）皆於
        載入器初始化時即失敗（fail-fast）。

        Returns:
            通過驗證的 ``active_mode`` 字串。

        Raises:
            ValueError: 當 ``dataset.modes`` 缺失或為空、``active_mode``
                缺失或未定義於 ``modes`` 時。
        """
        dataset_cfg: Dict[str, Any] = self._dataset_config()
        active_mode: Any = dataset_cfg.get("active_mode")
        modes: Any = dataset_cfg.get("modes")
        if not isinstance(modes, dict) or not modes:
            raise ValueError(
                "dataset.modes must be a non-empty mapping of mode -> parameters."
            )
        if not isinstance(active_mode, str) or not active_mode.strip():
            raise ValueError(
                f"dataset.active_mode is missing; available modes: {sorted(modes)}."
            )
        mode_cfg: Any = modes.get(active_mode)
        if not isinstance(mode_cfg, dict):
            raise ValueError(
                f"dataset.active_mode {active_mode!r} is not defined in dataset.modes; "
                f"available modes: {sorted(modes)}."
            )
        return active_mode

    def _parse_subjects(self, dataset_cfg: Mapping[str, Any]) -> List[Dict[str, str]]:
        """解析 ``dataset.categories.<領域>.subjects`` 為有序（領域、科目、focus）清單。

        保留 YAML 宣告順序；對結構異常（缺失或空 ``name``、
        subjects 非清單、跨領域重複科目名）一律拒絕。

        Args:
            dataset_cfg: ``dataset`` 區塊內容。

        Returns:
            有序的科目清單（``category`` / ``subject`` / ``focus``）。

        Raises:
            ValueError: 當 categories 結構缺失或異常時。
        """
        categories: Any = dataset_cfg.get("categories")
        if not isinstance(categories, dict) or not categories:
            raise ValueError("dataset.categories must be a non-empty mapping.")
        subjects: List[Dict[str, str]] = []
        seen: Dict[str, str] = {}
        for category_name, block in categories.items():
            if not isinstance(block, dict):
                raise ValueError(
                    f"dataset.categories[{category_name!r}] must be a mapping."
                )
            raw_subjects: Any = block.get("subjects")
            if not isinstance(raw_subjects, list) or not raw_subjects:
                raise ValueError(
                    f"dataset.categories[{category_name!r}].subjects must be a non-empty list."
                )
            for entry in raw_subjects:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Subject entry under category {category_name!r} must be a mapping."
                    )
                name: str = str(entry.get("name") or "").strip()
                if not name:
                    raise ValueError(
                        f"Subject entry under category {category_name!r} has a missing or empty 'name'."
                    )
                if name in seen:
                    raise ValueError(
                        f"Duplicate MMLU subject {name!r} found in categories "
                        f"{seen[name]!r} and {category_name!r}."
                    )
                seen[name] = str(category_name)
                subjects.append(
                    {
                        "category": str(category_name),
                        "subject": name,
                        "focus": str(entry.get("focus") or "").strip(),
                    }
                )
        return subjects
