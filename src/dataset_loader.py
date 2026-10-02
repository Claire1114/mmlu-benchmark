"""MMLU 資料集載入、標籤映射與 Prompt 組裝模組。"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

LABEL_TO_LETTER: Mapping[int, str] = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_SET = frozenset(LABEL_TO_LETTER.values())
EXPECTED_CHOICE_COUNT = 4

PROMPT_TEMPLATE: str = (
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


class MMLUDatasetLoader:
    """負責載入 MMLU、正規化欄位並產出結構化評測樣本。

    Attributes:
        config_path: YAML 設定檔路徑。
        config: 解析後的設定內容。
        skipped_rows: 最近一次 ``load_data`` 因驗證失敗而略過的列數。
    """

    def __init__(self, config_path: str = "configs/eval_config.yaml") -> None:
        """載入 YAML 設定檔並初始化內部快取。

        Args:
            config_path: 評測設定檔路徑，預設為 ``configs/eval_config.yaml``。

        Raises:
            FileNotFoundError: 當設定檔不存在時。
            ValueError: 當 YAML 內容不是對應表時。
        """
        self.config_path: str = config_path
        self.config: Dict[str, Any] = self._load_config(config_path)
        self._records: List[Dict[str, Any]] = []
        self.skipped_rows: int = 0

    def load_data(
        self,
        subject: Optional[str] = None,
        split: str = "test",
        sample_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """自 Hugging Face Hub 載入 MMLU 子集並正規化資料列。

        使用 ``datasets.load_dataset`` 讀取資料，將數字標籤映射為 A-D。
        無法安全正規化的列會被略過（嚴格拒絕髒資料），並累計至
        ``skipped_rows``，同時輸出警告日誌。

        Args:
            subject: MMLU 科目名稱。若為 ``None``，改用設定檔中的科目。
            split: 資料切分名稱，預設為 ``test``。
            sample_size: 最多保留的筆數。``None`` 表示不額外截斷。

        Returns:
            正規化後的資料列清單。空資料集會回傳空清單。

        Raises:
            ImportError: 當 ``datasets`` 套件不可用時。
            ValueError: 當資料集載入失敗、必要設定缺失、或 split 不存在時。
        """
        dataset_cfg: Dict[str, Any] = dict(self.config.get("dataset") or {})
        dataset_name: str = str(dataset_cfg.get("name") or "cais/mmlu")
        resolved_subject: str = str(subject or dataset_cfg.get("subject") or "").strip()
        if not resolved_subject:
            raise ValueError("MMLU subject is missing from arguments and config.")

        if load_dataset is None:
            raise ImportError("The 'datasets' package is required to load MMLU.")

        try:
            raw_dataset: Any = load_dataset(dataset_name, resolved_subject, split=split)
        except Exception as exc:
            raise ValueError(
                f"Failed to load dataset '{dataset_name}' for subject '{resolved_subject}'."
            ) from exc

        records: List[Dict[str, Any]] = self._coerce_records(raw_dataset, split=split)
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

        if sample_size is not None:
            size = int(sample_size)
            normalized = [] if size <= 0 else normalized[:size]

        self._records = normalized
        return list(self._records)

    def format_prompt(self, item: Mapping[str, Any]) -> str:
        """將單一題目組裝為結構化多選一 Prompt。

        Args:
            item: 單筆題目資料，預期可包含 ``question``、``choices``、``subject``。

        Returns:
            符合評測範本的 Prompt 字串。

        Raises:
            ValueError: 當選項無法安全擷取為恰好 4 個非 None 字串時。
        """
        subject: str = str(item.get("subject") or "unknown")
        question: str = str(item.get("question") or "")
        choice_a, choice_b, choice_c, choice_d = self._extract_choices(item)
        return PROMPT_TEMPLATE.format(
            subject=subject,
            question=question,
            choice_A=choice_a,
            choice_B=choice_b,
            choice_C=choice_c,
            choice_D=choice_d,
        )

    def get_samples(self) -> List[Dict[str, Any]]:
        """回傳處理後可供評測使用的樣本清單。

        若尚未呼叫 ``load_data``，會依設定檔自動載入資料。

        Returns:
            每筆包含 ``question_id``、``formatted_prompt``、``target_letter`` 與
            ``subject`` 的字典清單。
        """
        if not self._records:
            dataset_cfg: Dict[str, Any] = dict(self.config.get("dataset") or {})
            self.load_data(
                subject=dataset_cfg.get("subject"),
                split=str(dataset_cfg.get("split") or "test"),
                sample_size=dataset_cfg.get("sample_size"),
            )

        samples: List[Dict[str, Any]] = []
        for item in self._records:
            question_id = item.get("question_id")
            answer_letter = item.get("answer_letter")
            subject = item.get("subject")
            if question_id is None or answer_letter is None or subject is None:
                continue
            samples.append(
                {
                    "question_id": str(question_id),
                    "formatted_prompt": self.format_prompt(item),
                    "target_letter": str(answer_letter),
                    "subject": str(subject),
                }
            )
        return samples

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
        subject: str = str(row.get("subject") or fallback_subject).strip() or fallback_subject
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
            raise ValueError(f"Choices must be a sequence or mapping, got {type(raw_choices)!r}.")

        if len(values) != EXPECTED_CHOICE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_CHOICE_COUNT} choices, got {len(values)}."
            )
        if any(value is None for value in values):
            raise ValueError(f"Choice list contains None: {values!r}.")

        return str(values[0]), str(values[1]), str(values[2]), str(values[3])
