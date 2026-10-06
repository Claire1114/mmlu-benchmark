"""MMLU 答案解析與評估指標計算模組。

純計算模組：負責對模型文字生成結果做三層解析，並計算 benchmark 指標。
不執行任何網路呼叫、不載入模型；所有輸入皆為 DataLoader 與 Runner
提供的記憶體資料結構。

三層答案解析（由嚴到寬，依序執行）：
    Tier 1 (Strict):   以設定正則（YAML ``evaluation.answer_regex``）匹配
                       標準格式 "The correct answer is (X)"。
    Tier 2 (Fallback): Tier 1 失配時，以 ``\\b([A-D])\\b`` 匹配**最後一個**
                       獨立選項字母（大小寫不敏感，統一正規化為大寫）。
    Tier 3 (Safe Guard): 全層失配或內部例外時回傳 ``None``（INVALID），
                       僅記錄日誌，絕不上拋例外。

指標分母慣例（Benchmark 公平性鐵律）：
    - ``overall_accuracy`` = correct ÷ **total**（INVALID 一律判定為答錯）。
    - ``valid_accuracy`` = correct ÷ valid（排除無法解析後的輔助診斷視角）。
    - ``invalid_parsing_rate`` = invalid ÷ total。
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

#: MMLU 固定四個有效選項字母（目標答案驗證與選項 Recall 迭代用）。
VALID_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")

#: Tier 1 嚴格範式來源：僅匹配 Prompt 標準格式
#: "The correct answer is (X)"（大小寫不敏感、括號可選、字邊界守衛）。
DEFAULT_STRICT_ANSWER_REGEX: str = r"(?i)\bthe correct answer is\s*\(?\b([A-D])\b\)?"

#: Tier 2 fallback 範式：任一獨立選項字母；採「最後一次匹配」語意。
#: 作為增強回落的最終後備（不帶錨點時使用），避免 quotidian 詞彙如 Bye!/Do 導致誤判。
FALLBACK_LETTER_REGEX: re.Pattern[str] = re.compile(r"\b([A-D])\b", re.IGNORECASE)

#: Tier 2 強化回落正則——弱宣告錨點。
#    匹配：answer/choice/option/選/答案 之後接選項字母（=、空格/冒號後可選括號）
#    大小寫不敏感的錨點關鍵字，僅擷取 [A-D]，若完全失配則為空（由 extract_answer 迴歸 None）。
ENHANCED_FALLBACK_LETTER_REGEX: re.Pattern[str] = re.compile(
    r"(?i:\b(?:answer|choice|option|選|答案)\b)[\s:=]*\(?([A-D])\)?",
)


def safe_mean(values: Sequence[float]) -> float:
    """安全計算算術平均值。

    Args:
        values: 數值序列（呼叫端應確保各值為有限數）。

    Returns:
        算術平均；空序列回傳 ``0.0``。
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_option_recalls(
    pairs: Sequence[Tuple[str, Optional[str]]],
) -> Dict[str, float]:
    """由 (target, predicted) 配對計算各選項 Recall。

    Args:
        pairs: 每筆為 ``(目標字母, 預測字母)``；任一端可為空字串或
            ``None``（目標缺失或無法解析）。

    Returns:
        選項字母 → Recall（該字母答對題數 ÷ 答案為該字母的題目總數）。
        僅包含有題目覆蓋的字母；答錯與無法解析的預測皆計入分母
        （即視為「答錯」）。
    """
    target_counts: Dict[str, int] = {}
    correct_counts: Dict[str, int] = {}
    for target, predicted in pairs:
        if target not in VALID_LETTERS:
            continue
        target_counts[target] = target_counts.get(target, 0) + 1
        if predicted == target:
            correct_counts[target] = correct_counts.get(target, 0) + 1
    return {
        letter: correct_counts.get(letter, 0) / target_counts[letter]
        for letter in VALID_LETTERS
        if letter in target_counts
    }


def compute_recall_std(recalls: Mapping[str, float]) -> float:
    """計算 RStd（Recall Standard Deviation），用於偵測 Option Bias。

    Args:
        recalls: 選項字母 → Recall（通常為 :func:`compute_option_recalls` 輸出）。

    Returns:
        有覆蓋選項 Recall 值的總體標準差（``statistics.pstdev``）；
        覆蓋字母少於 2 個時回傳 ``0.0``（無比較基準）。
    """
    values: List[float] = [
        recalls[letter] for letter in VALID_LETTERS if letter in recalls
    ]
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _safe_divide(numerator: int, denominator: int) -> float:
    """安全除法；分母為零或負數時回傳 ``0.0``（絕不拋出例外）。"""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _empty_counters() -> Dict[str, int]:
    """回傳零初始化的群組聚合計數器（total/valid/correct/invalid）。"""
    return {"total": 0, "valid": 0, "correct": 0, "invalid": 0}


def _bump_group(counters: Dict[str, int], is_valid: bool, is_correct: bool) -> None:
    """以單筆記錄更新群組計數器（``is_correct`` 必為 ``is_valid`` 的子集）。"""
    counters["total"] += 1
    if is_valid:
        counters["valid"] += 1
    else:
        counters["invalid"] += 1
    if is_correct:
        counters["correct"] += 1


def _coerce_question_id(value: object) -> str:
    """將 question_id 轉為字串以對齊；``None`` 映射為空字串。"""
    if value is None:
        return ""
    return str(value)


def _coerce_raw_output(value: object) -> str:
    """安全轉換模型原始輸出為字串。

    ``None``／非字串型別一律映射為空字串（後續解析將判定為 INVALID）；
    若 ``str()`` 轉換本身失敗，同樣回退空字串。
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:  # pragma: no cover - 防範怪異 __str__ 實作
        LOGGER.warning(
            "Failed to stringify raw_output of type %s; using empty string.",
            type(value).__name__,
        )
        return ""


def _coerce_latency(value: object) -> float:
    """安全轉換延遲值為非負有限浮點數。

    缺失值（``None``）靜默轉為 ``0.0``（缺欄為正常情況，非錯誤）；
    存在但非法的值（非數值、布林、NaN／Inf、負數）轉為 ``0.0`` 並記錄
    警告；合法的非負數值原樣保留。
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        LOGGER.warning(
            "Boolean latency %r is not a valid duration; treated as 0.0 seconds.", value
        )
        return 0.0
    if isinstance(value, (int, float)):
        latency: float = float(value)
    elif isinstance(value, str):
        try:
            latency = float(value)
        except ValueError:
            LOGGER.warning("Non-numeric latency %r; treated as 0.0 seconds.", value)
            return 0.0
    else:
        LOGGER.warning(
            "Non-numeric latency of type %s (value %r); treated as 0.0 seconds.",
            type(value).__name__,
            value,
        )
        return 0.0
    if not math.isfinite(latency) or latency < 0.0:
        LOGGER.warning("Invalid latency %s; treated as 0.0 seconds.", latency)
        return 0.0
    return latency


def _as_label(value: object, fallback: str) -> str:
    """轉換樣本欄位為非空字串；``None`` 或純空白時回退 fallback。"""
    if value is None:
        return fallback
    text: str = str(value).strip()
    return text or fallback


def _normalize_target(value: object) -> str:
    """正規化目標答案為大寫 A–D 字母。

    Args:
        value: 原始 ``target_letter`` 欄位值。

    Returns:
        合法大寫字母；無法解析時回傳空字串（後續計算將該樣本判定為
        INVALID）。
    """
    if value is None:
        return ""
    text: str = str(value).strip().upper()
    return text if text in VALID_LETTERS else ""


@dataclass(frozen=True)
class ParsedPrediction:
    """單筆模型輸出的解析結果。

    Attributes:
        question_id: 題目識別碼（已轉為字串，與樣本對齊）。
        raw_output: 模型原始輸出文字（非字串型別映射為空字串）。
        predicted: 解析出的選項字母 A–D；``None`` 表示 INVALID（無法解析）。
        is_valid: ``predicted`` 是否非空（即可解析）。
        latency: 該題推理延遲（秒）；非法值映射為 0.0。
    """

    question_id: str
    raw_output: str
    predicted: Optional[str]
    is_valid: bool
    latency: float


@dataclass(frozen=True)
class SubjectScore:
    """單一代表性科目（MMLU 子集）的指標聚合。

    Attributes:
        subject: MMLU 科目名稱。
        category: 該科目所屬領域。
        total: 該科目題目總數（accuracy 分母）。
        valid: 可解析題目數。
        correct: 答對題數。
        invalid: 無法解析（INVALID）題數。
        accuracy: 科目準確率 = correct ÷ total（INVALID 判定為答錯）。
    """

    subject: str
    category: str
    total: int
    valid: int
    correct: int
    invalid: int
    accuracy: float


@dataclass(frozen=True)
class CategoryScore:
    """單一個知識領域的指標聚合。

    Attributes:
        category: 領域名稱（STEM / Humanities / Social_Sciences / Other）。
        total: 該領域題目總數（accuracy 分母）。
        valid: 可解析題目數。
        correct: 答對題數。
        invalid: 無法解析（INVALID）題數。
        accuracy: 領域準確率 = correct ÷ total（領域內加權聚合，
            INVALID 判定為答錯）。
    """

    category: str
    total: int
    valid: int
    correct: int
    invalid: int
    accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    """單一模型評測跑的完整指標結果。

    Attributes:
        model_name: 模型名稱（來自設定檔 ``models[].name``）。
        total: 評測樣本總數（overall_accuracy 與所有比率的分母）。
        valid: 可解析樣本數（total - invalid）。
        correct: 答對樣本數。
        invalid: 無法解析（INVALID）樣本數（含缺失預測）。
        overall_accuracy: 總體準確率 = correct ÷ total（INVALID 判定為答錯）。
        valid_accuracy: 輔助診斷 = correct ÷ valid（valid 為 0 時 0.0）。
        invalid_parsing_rate: 無效解析率 = invalid ÷ total。
        domain_accuracy: 領域 → 準確率（領域內加權聚合，分母為該領域 total）。
        domain_accuracy_mean: 各領域準確率的算術平均（無領域時 0.0）。
        subject_accuracy: 科目 → 準確率。
        option_recall: 選項字母 → Recall（僅含有題目覆蓋的字母）。
        recall_std: RStd，各選項 Recall 的 pstdev（覆蓋字母 < 2 時 0.0）。
        average_latency: 平均推理延遲（秒／題；無樣本時 0.0）。
        category_scores: 各領域聚合明細（依首次出現順序）。
        subject_scores: 各科目聚合明細（依首次出現順序）。
    """

    model_name: str
    total: int
    valid: int
    correct: int
    invalid: int
    overall_accuracy: float
    valid_accuracy: float
    invalid_parsing_rate: float
    domain_accuracy: Dict[str, float]
    domain_accuracy_mean: float
    subject_accuracy: Dict[str, float]
    option_recall: Dict[str, float]
    recall_std: float
    average_latency: float
    category_scores: Tuple[CategoryScore, ...]
    subject_scores: Tuple[SubjectScore, ...]
    missing_predictions: int = 0

    def to_dict(self) -> Dict[str, object]:
        """轉換為 JSON 可序列化的報告字典。

        Returns:
            巢狀結構：總體指標 + 領域／科目雙層級準確率明細，
            供寫入 ``metrics_summary.json`` 或 CSV 後處理。
        """
        return {
            "model": self.model_name,
            "total_samples": self.total,
            "valid_samples": self.valid,
            "correct_samples": self.correct,
            "invalid_samples": self.invalid,
            "overall_accuracy": self.overall_accuracy,
            "valid_accuracy": self.valid_accuracy,
            "invalid_parsing_rate": self.invalid_parsing_rate,
            "domain_accuracy": dict(self.domain_accuracy),
            "domain_accuracy_mean": self.domain_accuracy_mean,
            "subject_accuracy": dict(self.subject_accuracy),
            "option_recall": dict(self.option_recall),
            "recall_std": self.recall_std,
            "average_latency_seconds": self.average_latency,
            "missing_predictions": self.missing_predictions,
            "category_scores": {
                score.category: {
                    "total": score.total,
                    "valid": score.valid,
                    "correct": score.correct,
                    "invalid": score.invalid,
                    "accuracy": score.accuracy,
                }
                for score in self.category_scores
            },
            "subject_scores": {
                score.subject: {
                    "category": score.category,
                    "total": score.total,
                    "valid": score.valid,
                    "correct": score.correct,
                    "invalid": score.invalid,
                    "accuracy": score.accuracy,
                }
                for score in self.subject_scores
            },
        }


class Evaluator:
    """答案解析器與評估指標計算器（純計算、無副作用）。

    Attributes:
        strict_pattern: 已編譯的 Tier 1 嚴格正則。

    Example:
        >>> evaluator = Evaluator.from_config(config)          # doctest: +SKIP
        >>> result = evaluator.evaluate(                       # doctest: +SKIP
        ...     samples, predictions, model_name="qwen2.5-0.5b"
        ... )
        >>> result.to_dict()                                   # doctest: +SKIP
    """

    def __init__(self, strict_pattern: str = DEFAULT_STRICT_ANSWER_REGEX) -> None:
        """初始化並編譯 Tier 1 嚴格正則。

        Args:
            strict_pattern: Tier 1 正則來源（預設為內建標準格式範式）；
                編譯失敗時記錄警告並回退 ``DEFAULT_STRICT_ANSWER_REGEX``，
                絕不拋出例外。
        """
        self.strict_pattern: re.Pattern[str] = self._compile_strict_pattern(
            strict_pattern
        )

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> Evaluator:
        """由設定頂層對應表建立 Evaluator。

        讀取 ``evaluation.answer_regex``（Tier 1 正則的單一來源）；缺失、
        非字串或空字串時回退內建預設範式並記錄警告。

        Args:
            config: 完整設定對應表（與 ``MMLUDatasetLoader.config`` 同源）。

        Returns:
            Evaluator 實例（絕不拋出例外）。
        """
        evaluation_cfg: object = config.get("evaluation")
        raw_pattern: object = (
            evaluation_cfg.get("answer_regex")
            if isinstance(evaluation_cfg, Mapping)
            else None
        )
        if not isinstance(raw_pattern, str) or not raw_pattern.strip():
            LOGGER.warning(
                "evaluation.answer_regex is missing or not a non-empty string; "
                "using the built-in default strict pattern."
            )
            return cls()
        return cls(raw_pattern)

    @staticmethod
    def _compile_strict_pattern(pattern: str) -> re.Pattern[str]:
        """編譯 Tier 1 範式；編譯失敗時警告並回退內建預設。

        Args:
            pattern: 正則來源字串。

        Returns:
            編譯後的正則；絕不拋出例外。
        """
        try:
            return re.compile(pattern)
        except re.error as exc:
            LOGGER.warning(
                "Invalid strict answer regex %r (%s); falling back to the built-in "
                "default.",
                pattern,
                exc,
            )
            return re.compile(DEFAULT_STRICT_ANSWER_REGEX)

    def extract_answer(self, raw_output: object) -> Optional[str]:
        """以三層解析自模型輸出擷取選項字母 A–D。

        - Tier 1 (Strict)：以設定正則匹配標準格式
          "The correct answer is (X)"；取第一次匹配。
        - Tier 2 (Fallback)：Tier 1 失配時，以 ``\\b([A-D])\\b`` 匹配
          **最後一個**獨立選項字母（大小寫不敏感）。
        - Tier 3 (Safe Guard)：全層失配或內部例外時回傳 ``None``
          （INVALID）並記錄日誌；絕不上拋例外。

        Args:
            raw_output: 模型原始輸出；接受任意型別（非字串直接判定
                INVALID）。

        Returns:
            大寫選項字母 ``A``–``D``；無法解析時為 ``None``。
        """
        if not isinstance(raw_output, str):
            LOGGER.debug(
                "raw_output is not a string (type %s); marked INVALID.",
                type(raw_output).__name__,
            )
            return None
        try:
            strict_match: Optional[re.Match[str]] = self.strict_pattern.search(
                raw_output
            )
            if strict_match is not None:
                return strict_match.group(1).upper()
            # Tier 2 enhanced: 弱宣告錨點（answer/choice/option/選/答案）
            # 只在文字確實包含該類關鍵字時才匹配，防止 "Bye!"、"Do you agree?" 等誤判
            has_anchor: bool = bool(
                re.search(r"(?i)\b(?:answer|choice|option|選|答案)\b", raw_output)
            )
            if has_anchor:
                fallback_matches: List[str] = ENHANCED_FALLBACK_LETTER_REGEX.findall(raw_output)
                if fallback_matches:
                    return fallback_matches[-1].upper()
            # Tier 2 plain: 獨立選項字母（作為無錨點文字的最後後備）
            plain_matches: List[str] = FALLBACK_LETTER_REGEX.findall(raw_output)
            if plain_matches:
                return plain_matches[-1].upper()
        except Exception as exc:  # Tier 3 Safe Guard：任何非預期例外
            LOGGER.warning(
                "Answer extraction raised unexpectedly; marked INVALID: %s", exc
            )
            return None
        return None

    def parse_predictions(
        self, predictions: Sequence[Mapping[str, object]]
    ) -> List[ParsedPrediction]:
        """批次解析 Runner 預測並強制轉譯欄位型別（不與樣本對齊）。

        髒記錄不拋出例外：缺失／非法欄位皆轉為安全形式（空 question_id、
        空 raw_output、0.0 latency）。

        Args:
            predictions: Runner 輸出；每筆預期含 ``question_id`` /
                ``raw_output`` / ``latency``。

        Returns:
            解析結果清單，與輸入順序對齊。
        """
        parsed: List[ParsedPrediction] = []
        for record in predictions:
            question_id: str = _coerce_question_id(record.get("question_id"))
            raw_output: str = _coerce_raw_output(record.get("raw_output"))
            predicted: Optional[str] = self.extract_answer(raw_output)
            parsed.append(
                ParsedPrediction(
                    question_id=question_id,
                    raw_output=raw_output,
                    predicted=predicted,
                    is_valid=predicted is not None,
                    latency=_coerce_latency(record.get("latency")),
                )
            )
        return parsed

    def evaluate(
        self,
        samples: Sequence[Mapping[str, object]],
        predictions: Sequence[Mapping[str, object]],
        model_name: str = "",
    ) -> EvaluationResult:
        """對齊樣本與預測並計算全部評估指標。

        對齊與防禦規則：
            - 以 ``question_id`` 對齊（雙邊強制轉為 ``str``）。
            - 雙邊重複記錄：第一筆生效，重複者記錄警告。
            - 樣本無對應預測：判定為 INVALID、latency 記 0.0，並記錄警告。
            - 孤兒預測（無對應樣本）：排除於指標外並記錄警告。
            - 總體／領域／科目準確率分母固定為 total 題數，INVALID
              一律判定為答錯；分母為零時回傳 0.0。

        Args:
            samples: ``MMLUDatasetLoader.get_samples()`` 輸出；每筆含
                ``question_id`` / ``target_letter`` / ``subject`` /
                ``category``。
            predictions: Runner 輸出；每筆含 ``question_id`` /
                ``raw_output`` / ``latency``。
            model_name: 模型名稱（用於報告顯示，預設空字串）。

        Returns:
            含全部指標的 ``EvaluationResult``；任何髒輸入皆不拋出例外。
        """
        sample_records: Dict[str, Mapping[str, object]] = {}
        sample_order: List[str] = []
        for record in samples:
            question_id: str = _coerce_question_id(record.get("question_id"))
            if question_id in sample_records:
                LOGGER.warning(
                    "Duplicate sample question_id %r; keeping the first record.",
                    question_id,
                )
                continue
            sample_records[question_id] = record
            sample_order.append(question_id)

        predictions_by_id: Dict[str, ParsedPrediction] = {}
        for record in predictions:
            question_id = _coerce_question_id(record.get("question_id"))
            if question_id not in sample_records:
                LOGGER.warning(
                    "Prediction question_id %r matches no sample; excluded from "
                    "metrics.",
                    question_id,
                )
                continue
            if question_id in predictions_by_id:
                LOGGER.warning(
                    "Duplicate prediction question_id %r; keeping the first record.",
                    question_id,
                )
                continue
            raw_output: str = _coerce_raw_output(record.get("raw_output"))
            predicted: Optional[str] = self.extract_answer(raw_output)
            predictions_by_id[question_id] = ParsedPrediction(
                question_id=question_id,
                raw_output=raw_output,
                predicted=predicted,
                is_valid=predicted is not None,
                latency=_coerce_latency(record.get("latency")),
            )

        subject_counts: Dict[str, Dict[str, int]] = {}
        category_counts: Dict[str, Dict[str, int]] = {}
        subject_category: Dict[str, str] = {}
        pairs: List[Tuple[str, Optional[str]]] = []
        latencies: List[float] = []
        total = 0
        valid = 0
        correct = 0
        invalid = 0
        missing_count = 0

        for question_id in sample_order:
            record = sample_records[question_id]
            total += 1
            subject: str = _as_label(record.get("subject"), "unknown")
            category: str = _as_label(record.get("category"), "unknown")
            subject_category.setdefault(subject, category)
            target: str = _normalize_target(record.get("target_letter"))
            prediction: Optional[ParsedPrediction] = predictions_by_id.get(question_id)
            if prediction is None:
                missing_count += 1
                predicted = None
            else:
                predicted = prediction.predicted
                latencies.append(prediction.latency)
            pairs.append((target, predicted))

            is_correct: bool = (
                target != "" and predicted is not None and predicted == target
            )
            is_valid_sample: bool = target != "" and predicted is not None
            if is_valid_sample:
                valid += 1
            else:
                invalid += 1
            if is_correct:
                correct += 1
            _bump_group(
                subject_counts.setdefault(subject, _empty_counters()),
                is_valid_sample,
                is_correct,
            )
            _bump_group(
                category_counts.setdefault(category, _empty_counters()),
                is_valid_sample,
                is_correct,
            )

        if missing_count:
            LOGGER.warning(
                "%d sample(s) had no matching prediction; counted as INVALID.",
                missing_count,
            )

        option_recall: Dict[str, float] = compute_option_recalls(pairs)

        subject_scores: List[SubjectScore] = []
        subject_accuracy: Dict[str, float] = {}
        for subject, counters in subject_counts.items():
            accuracy: float = _safe_divide(counters["correct"], counters["total"])
            subject_scores.append(
                SubjectScore(
                    subject=subject,
                    category=subject_category[subject],
                    total=counters["total"],
                    valid=counters["valid"],
                    correct=counters["correct"],
                    invalid=counters["invalid"],
                    accuracy=accuracy,
                )
            )
            subject_accuracy[subject] = accuracy

        category_scores: List[CategoryScore] = []
        domain_accuracy: Dict[str, float] = {}
        for category, counters in category_counts.items():
            accuracy = _safe_divide(counters["correct"], counters["total"])
            category_scores.append(
                CategoryScore(
                    category=category,
                    total=counters["total"],
                    valid=counters["valid"],
                    correct=counters["correct"],
                    invalid=counters["invalid"],
                    accuracy=accuracy,
                )
            )
            domain_accuracy[category] = accuracy

        return EvaluationResult(
            model_name=model_name,
            total=total,
            valid=valid,
            correct=correct,
            invalid=invalid,
            overall_accuracy=_safe_divide(correct, total),
            valid_accuracy=_safe_divide(correct, valid),
            invalid_parsing_rate=_safe_divide(invalid, total),
            domain_accuracy=domain_accuracy,
            domain_accuracy_mean=safe_mean(list(domain_accuracy.values())),
            subject_accuracy=subject_accuracy,
            option_recall=option_recall,
            recall_std=compute_recall_std(option_recall),
            average_latency=safe_mean(latencies),
            missing_predictions=missing_count,
            category_scores=tuple(category_scores),
            subject_scores=tuple(subject_scores),
        )
