"""MMLU 資料集載入、標籤映射與 Prompt 組裝模組。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]

LABEL_TO_LETTER: Mapping[int, str] = {0: "A", 1: "B", 2: "C", 3: "D"}
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

    def load_data(
        self,
        subject: Optional[str] = None,
        split: str = "test",
        sample_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """自 Hugging Face Hub 載入 MMLU 子集並正規化資料列。

        使用 ``datasets.load_dataset`` 讀取資料，將數字標籤映射為 A-D，
        並略過無法安全正規化的列以達成優雅降級。

        Args:
            subject: MMLU 科目名稱。若為 ``None``，改用設定檔中的科目。
            split: 資料切分名稱，預設為 ``test``。
            sample_size: 最多保留的筆數。``None`` 表示不額外截斷。

        Returns:
            正規化後的資料列清單。空資料集會回傳空清單。

        Raises:
            ImportError: 當 ``datasets`` 套件不可用時。
            ValueError: 當資料集載入失敗或必要設定缺失時。
        """
        dataset_cfg: Dict[str, Any] = dict(self.config.get("dataset") or {})
        dataset_name: str = str(dataset_cfg.get("name") or "cais/mmlu")
        resolved_subject: str = str(subject or dataset_cfg.get("subject") or "")
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

        records: List[Dict[str, Any]] = self._coerce_records(raw_dataset)
        if not records:
            self._records = []
            return []

        normalized: List[Dict[str, Any]] = []
        for index, row in enumerate(records):
            item: Optional[Dict[str, Any]] = self._normalize_row(
                row=row,
                fallback_subject=resolved_subject,
                index=index,
            )
            if item is not None:
                normalized.append(item)

        if sample_size is not None:
            if sample_size <= 0:
                normalized = []
            else:
                normalized = normalized[:sample_size]

        self._records = normalized
        return list(self._records)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        """將單一題目組裝為結構化多選一 Prompt。

        缺失的問題、科目或選項會以降級後的空字串填入，避免拋出 KeyError。

        Args:
            item: 單筆題目資料，預期可包含 ``question``、``choices``、``subject``。

        Returns:
            符合評測範本的 Prompt 字串。
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
            samples.append(
                {
                    "question_id": str(item.get("question_id")),
                    "formatted_prompt": self.format_prompt(item),
                    "target_letter": str(item.get("answer_letter")),
                    "subject": str(item.get("subject")),
                }
            )
        return samples

    @staticmethod
    def map_numeric_to_letter(label: Any) -> str:
        """將數字標籤精確映射為對應字母。

        Args:
            label: 整數標籤，合法值為 0、1、2、3。字串數字亦會嘗試轉換。

        Returns:
            對應的選項字母 ``A``、``B``、``C`` 或 ``D``。

        Raises:
            ValueError: 當標籤無法映射為 A-D 時。
        """
        if isinstance(label, str) and label.strip() in LABEL_TO_LETTER.values():
            return label.strip()
        try:
            numeric_label: int = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Answer label cannot be mapped: {label!r}.") from exc
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

    def _coerce_records(self, raw_dataset: Any) -> List[Dict[str, Any]]:
        """將 Hugging Face Dataset 或序列轉為字典清單。

        Args:
            raw_dataset: ``load_dataset`` 回傳物件，或可迭代的列資料。

        Returns:
            以字典表示的資料列。無法轉譯時回傳空清單。
        """
        if raw_dataset is None:
            return []
        if isinstance(raw_dataset, dict):
            split_name: str = str((self.config.get("dataset") or {}).get("split") or "test")
            split_data: Any = raw_dataset.get(split_name) or next(iter(raw_dataset.values()), [])
            return self._coerce_records(split_data)
        try:
            rows: List[Any] = list(raw_dataset)
        except TypeError:
            return []
        records: List[Dict[str, Any]] = []
        for row in rows:
            if isinstance(row, Mapping):
                records.append(dict(row))
        return records

    def _normalize_row(
        self,
        row: Mapping[str, Any],
        fallback_subject: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """正規化單列資料；欄位缺失時進行優雅降級。

        Args:
            row: 原始資料列。
            fallback_subject: 當列中缺少 ``subject`` 時使用的科目名稱。
            index: 列索引，用於產生 ``question_id``。

        Returns:
            正規化後的資料字典；若答案無法映射則回傳 ``None``。
        """
        subject: str = str(row.get("subject") or fallback_subject)
        question: str = str(row.get("question") or "")
        choices: List[str] = list(self._extract_choices(row))
        try:
            answer_letter: str = self.map_numeric_to_letter(row.get("answer"))
        except ValueError:
            return None
        return {
            "question_id": f"{subject}_{index}",
            "question": question,
            "choices": choices,
            "answer": row.get("answer"),
            "answer_letter": answer_letter,
            "subject": subject,
        }

    def _extract_choices(self, item: Mapping[str, Any]) -> Sequence[str]:
        """擷取 A-D 四個選項，缺失時以空字串補齊。

        Args:
            item: 單筆題目資料。

        Returns:
            長度固定為 4 的選項序列。
        """
        raw_choices: Any = item.get("choices")
        if isinstance(raw_choices, Mapping):
            return (
                str(raw_choices.get("A") or raw_choices.get(0) or ""),
                str(raw_choices.get("B") or raw_choices.get(1) or ""),
                str(raw_choices.get("C") or raw_choices.get(2) or ""),
                str(raw_choices.get("D") or raw_choices.get(3) or ""),
            )
        if isinstance(raw_choices, (list, tuple)):
            padded: List[str] = [str(choice) for choice in raw_choices]
        else:
            padded = [
                str(item.get("choice_A") or ""),
                str(item.get("choice_B") or ""),
                str(item.get("choice_C") or ""),
                str(item.get("choice_D") or ""),
            ]
        while len(padded) < 4:
            padded.append("")
        return padded[0], padded[1], padded[2], padded[3]
