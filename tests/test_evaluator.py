"""Evaluator 離線單元測試：三層答案解析與評估指標計算。

所有測試案例皆為純記憶體計算（合成 samples/predictions），零網路呼叫、
零模型權重載入；設定檔相關測試僅讀取本機真實
``configs/eval_config.yaml``（不觸發任何下載）。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml

from src.evaluator import (
    DEFAULT_STRICT_ANSWER_REGEX,
    VALID_LETTERS,
    Evaluator,
    compute_option_recalls,
    compute_recall_std,
    safe_mean,
)

REAL_CONFIG_PATH: str = str(
    Path(__file__).resolve().parents[1] / "configs" / "eval_config.yaml"
)

CATEGORY_SUBJECTS: Dict[str, Tuple[str, str]] = {
    "STEM": ("college_computer_science", "high_school_mathematics"),
    "Humanities": ("philosophy", "world_religions"),
    "Social_Sciences": ("econometrics", "high_school_psychology"),
    "Other": ("clinical_knowledge", "professional_law"),
}

_MISSING = object()


def _sample(
    question_id: Any,
    target: Any,
    subject: str = "philosophy",
    category: str = "Humanities",
) -> Dict[str, Any]:
    """建立單筆可評測樣本（對齊 get_samples() 輸出結構）。"""
    return {
        "question_id": question_id,
        "formatted_prompt": f"prompt for {question_id}",
        "target_letter": target,
        "subject": subject,
        "category": category,
    }


def _prediction(
    question_id: Any, raw_output: Any, latency: Any = _MISSING
) -> Dict[str, Any]:
    """建立單筆 Runner 預測記錄（缺省时省略 ``latency`` 鍵）。"""
    record: Dict[str, Any] = {"question_id": question_id, "raw_output": raw_output}
    if latency is not _MISSING:
        record["latency"] = latency
    return record


def _build_samples(n_per_subject: int = 3) -> List[Dict[str, Any]]:
    """建立 8 科目 × n_per_subject 樣本；target 依序輪換 A/B/C/D。"""
    samples: List[Dict[str, Any]] = []
    index = 0
    for category, subjects in CATEGORY_SUBJECTS.items():
        for subject in subjects:
            for _ in range(n_per_subject):
                question_id = f"{subject}__{index:06d}__aaaaaa00"
                samples.append(
                    _sample(question_id, VALID_LETTERS[index % 4], subject, category)
                )
                index += 1
    return samples


class TestTierOneStrictExtraction:
    """Tier 1 (Strict)：標準格式 "The correct answer is (X)"。"""

    def test_canonical_parenthesized(self) -> None:
        assert Evaluator().extract_answer("The correct answer is (B)") == "B"

    def test_canonical_bare_letter(self) -> None:
        assert Evaluator().extract_answer("The correct answer is B") == "B"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The correct answer is (B).", "B"),
            ("The correct answer is B.", "B"),
            ("THE CORRECT ANSWER IS (C)", "C"),
            ("the correct answer is (d)", "D"),
            ("The  correct   answer  is  (A)", "A"),
        ],
    )
    def test_variants_and_case_insensitivity(self, raw: str, expected: str) -> None:
        assert Evaluator().extract_answer(raw) == expected

    def test_phrase_after_reasoning_preamble(self) -> None:
        raw = (
            "Let me think. Option A looks plausible but is wrong. "
            "The correct answer is (D)."
        )
        assert Evaluator().extract_answer(raw) == "D"

    def test_embedded_word_not_captured_by_tier1(self) -> None:
        # "about" 的 'a' 不得被 Tier 1 誤擷取；應落 Tier 2 取得 B。
        assert Evaluator().extract_answer("The correct answer is about B") == "B"

    def test_first_occurrence_wins(self) -> None:
        raw = "The correct answer is A. Hmm, The correct answer is B."
        assert Evaluator().extract_answer(raw) == "A"


class TestTierTwoFallbackExtraction:
    """Tier 2 (Fallback)：\\b([A-D])\\b 最後一個獨立選項字母。"""

    def test_last_standalone_letter_wins(self) -> None:
        assert Evaluator().extract_answer("I think it's B, not A") == "A"

    def test_lowercase_fallback_normalized_to_upper(self) -> None:
        assert Evaluator().extract_answer("the answer is b") == "B"

    def test_cjk_text_with_parenthesized_letter(self) -> None:
        assert Evaluator().extract_answer("選項是(C)") == "C"

    def test_bare_single_letter(self) -> None:
        assert Evaluator().extract_answer("D") == "D"

    def test_parenthesized_standalone(self) -> None:
        assert Evaluator().extract_answer("(C)") == "C"

    def test_reasoning_chain_last_wins(self) -> None:
        assert Evaluator().extract_answer("A is unlikely. B is possible. So C.") == "C"

    def test_guess_prefix(self) -> None:
        assert Evaluator().extract_answer("My guess: B") == "B"


class TestTierThreeSafeGuard:
    """Tier 3 (Safe Guard)：無法解析／非法輸入一律回傳 None 且不拋出。"""

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "\n\t ",
            "I don't know",
            "E",
            "F",
            "abcd",
            "ABC",
            "abcdefg",
            "答案：E",
            "hello world 123",
            "the options are attractive but no letter",
        ],
    )
    def test_invalid_text_returns_none(self, raw: str) -> None:
        assert Evaluator().extract_answer(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [None, 123, 4.5, True, ["A", "B"], {"answer": "B"}, b"A"],
    )
    def test_non_string_input_never_raises(self, raw: Any) -> None:
        assert Evaluator().extract_answer(raw) is None

    def test_very_long_gibberish_returns_none(self) -> None:
        raw = ("lorem ipsum dolor sit amet " * 5000).strip()
        assert Evaluator().extract_answer(raw) is None

    def test_very_long_text_with_trailing_answer(self) -> None:
        raw = "padding text " * 20000 + "The correct answer is (B)"
        assert Evaluator().extract_answer(raw) == "B"


class TestParsePredictions:
    """parse_predictions：批次解析與欄位型別強制轉譯。"""

    def test_batch_parsing_aligned_order(self) -> None:
        evaluator = Evaluator()
        records = [
            _prediction("q1", "The correct answer is (A)"),
            _prediction("q2", "The correct answer is B"),
            _prediction("q3", "no idea"),
        ]
        parsed = evaluator.parse_predictions(records)
        assert [p.question_id for p in parsed] == ["q1", "q2", "q3"]
        assert [p.predicted for p in parsed] == ["A", "B", None]
        assert [p.is_valid for p in parsed] == [True, True, False]

    def test_missing_latency_defaults_to_zero(self) -> None:
        parsed = Evaluator().parse_predictions([_prediction("q1", "B")])
        assert parsed[0].latency == 0.0

    def test_missing_latency_emits_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 缺 latency 鍵屬正常情況，不應該刷出 WARNING 日誌。
        with caplog.at_level(logging.WARNING, logger="src.evaluator"):
            Evaluator().parse_predictions([_prediction("q1", "B")])
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.parametrize(
        "latency",
        [-3.0, float("nan"), float("inf"), "abc", None],
    )
    def test_invalid_latency_coerced_to_zero(self, latency: Any) -> None:
        parsed = Evaluator().parse_predictions([_prediction("q1", "B", latency)])
        assert parsed[0].latency == 0.0

    def test_none_raw_output_becomes_empty_and_invalid(self) -> None:
        parsed = Evaluator().parse_predictions([_prediction("q1", None)])
        assert parsed[0].raw_output == ""
        assert parsed[0].predicted is None
        assert parsed[0].is_valid is False

    def test_missing_question_id_maps_to_empty_string(self) -> None:
        parsed = Evaluator().parse_predictions([{"raw_output": "B", "latency": 1.0}])
        assert parsed[0].question_id == ""
        assert parsed[0].predicted == "B"


class TestEvaluateAlignment:
    """evaluate：question_id join 與對齊防禦。"""

    def test_unordered_predictions_align_by_id(self) -> None:
        samples = [_sample("q1", "A"), _sample("q2", "B")]
        predictions = [
            _prediction("q2", "The correct answer is (B)"),
            _prediction("q1", "The correct answer is (A)"),
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.total == 2
        assert result.correct == 2
        assert result.overall_accuracy == 1.0

    def test_duplicate_predictions_first_wins(self) -> None:
        samples = [_sample("q1", "A")]
        predictions = [
            _prediction("q1", "The correct answer is (A)"),
            _prediction("q1", "The correct answer is (B)"),
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.total == 1
        assert result.correct == 1

    def test_orphan_prediction_excluded(self) -> None:
        samples = [_sample("q1", "A")]
        predictions = [
            _prediction("q1", "The correct answer is (A)"),
            _prediction("ghost", "The correct answer is (B)"),
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.total == 1
        assert result.correct == 1
        assert result.overall_accuracy == 1.0

    def test_missing_prediction_counted_invalid(self) -> None:
        samples = [_sample("q1", "A"), _sample("q2", "B")]
        predictions = [_prediction("q1", "The correct answer is (A)")]
        result = Evaluator().evaluate(samples, predictions)
        assert result.total == 2
        assert result.invalid == 1
        assert result.correct == 1
        # 分母固定為 total：q2 判定答錯 → 1/2；valid 視角為 1/1。
        assert result.overall_accuracy == 0.5
        assert result.valid_accuracy == 1.0
        assert result.invalid_parsing_rate == 0.5

    def test_question_id_type_coercion_join(self) -> None:
        samples = [_sample(1, "A")]  # 樣本側 int
        predictions = [_prediction("1", "The correct answer is (A)")]  # 預測側 str
        result = Evaluator().evaluate(samples, predictions)
        assert result.correct == 1

    def test_extra_prediction_fields_ignored(self) -> None:
        samples = [_sample("q1", "A")]
        predictions = [
            {
                "question_id": "q1",
                "raw_output": "The correct answer is (A)",
                "latency": 1.0,
                "model_name": "extra-field",
                "tokens": 12,
            }
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.correct == 1


class TestMetricsComputation:
    """指標：分母慣例（correct ÷ total）與領域／科目聚合。"""

    def _mixed_case(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """8 題：3 答對、3 有效答錯、2 無法解析（INVALID）。"""
        samples = [
            _sample("q1", "A"),
            _sample("q2", "B"),
            _sample("q3", "C"),
            _sample("q4", "D"),
            _sample("q5", "A"),
            _sample("q6", "B"),
            _sample("q7", "C"),
            _sample("q8", "D"),
        ]
        predictions = [
            _prediction("q1", "The correct answer is (A)", 0.5),
            _prediction("q2", "The correct answer is (B)", 1.5),
            _prediction("q3", "The correct answer is (A)"),
            _prediction("q4", "The correct answer is (B)"),
            _prediction("q5", "I refuse to answer"),
            _prediction("q6", ""),
            _prediction("q7", "The correct answer is (C)"),
            _prediction("q8", "The correct answer is (A)"),
        ]
        return samples, predictions

    def test_denominator_is_total_with_invalid_as_wrong(self) -> None:
        samples, predictions = self._mixed_case()
        result = Evaluator().evaluate(samples, predictions)
        assert result.total == 8
        assert result.correct == 3
        assert result.invalid == 2
        assert result.valid == 6
        assert result.overall_accuracy == pytest.approx(3 / 8)
        assert result.valid_accuracy == pytest.approx(0.5)
        assert result.invalid_parsing_rate == pytest.approx(0.25)

    def test_all_invalid_no_zero_division(self) -> None:
        samples = [_sample(f"q{i}", "A") for i in range(3)]
        predictions = [_prediction(f"q{i}", "hmm") for i in range(3)]
        result = Evaluator().evaluate(samples, predictions)
        assert result.overall_accuracy == 0.0
        assert result.valid_accuracy == 0.0
        assert result.invalid_parsing_rate == 1.0
        # 僅 A 有覆蓋 → 少於 2 個字母 → RStd = 0.0
        assert result.recall_std == 0.0

    def test_all_correct_perfect_scores(self) -> None:
        letters = "ABCD"
        samples = [_sample(f"q{i}", letters[i % 4]) for i in range(8)]
        predictions = [
            _prediction(f"q{i}", f"The correct answer is ({letters[i % 4]})", 0.25)
            for i in range(8)
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.overall_accuracy == 1.0
        assert result.invalid == 0
        assert result.recall_std == 0.0
        assert result.average_latency == 0.25

    def test_empty_inputs_produce_zero_metrics(self) -> None:
        result = Evaluator().evaluate([], [])
        assert result.total == 0
        assert result.overall_accuracy == 0.0
        assert result.invalid_parsing_rate == 0.0
        assert result.average_latency == 0.0
        assert result.option_recall == {}
        assert result.recall_std == 0.0

    def test_domain_accuracy_is_weighted_aggregate(self) -> None:
        samples = [
            _sample("q1", "A", "college_computer_science", "STEM"),
            _sample("q2", "B", "college_computer_science", "STEM"),
            _sample("q3", "C", "high_school_mathematics", "STEM"),
            _sample("q4", "D", "clinical_knowledge", "Other"),
            _sample("q5", "A", "clinical_knowledge", "Other"),
        ]
        predictions = [
            _prediction("q1", "The correct answer is (A)"),
            _prediction("q2", "The correct answer is (A)"),
            _prediction("q3", "gibberish"),
            _prediction("q4", "The correct answer is (D)"),
            _prediction("q5", "The correct answer is (A)"),
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.domain_accuracy == pytest.approx({"STEM": 1 / 3, "Other": 1.0})
        assert result.domain_accuracy_mean == pytest.approx((1 / 3 + 1.0) / 2)
        assert result.subject_accuracy == pytest.approx(
            {
                "college_computer_science": 0.5,
                "high_school_mathematics": 0.0,
                "clinical_knowledge": 1.0,
            }
        )
        assert result.overall_accuracy == pytest.approx(0.6)

    def test_average_latency_includes_missing_prediction_zero(self) -> None:
        samples = [_sample("q1", "A"), _sample("q2", "B")]
        predictions = [_prediction("q1", "The correct answer is (A)", 2.0)]
        result = Evaluator().evaluate(samples, predictions)
        # q2 缺 prediction → 不計入平均延遲，僅 q1 的 2.0s 計入平均
        # 期望 average_latency == 2.0 (only the 1 prediction's latency), NOT 1.0
        assert result.average_latency == 2.0
        # missing_predictions 應為 1（q2 無預測）
        assert result.missing_predictions == 1

    def test_invalid_target_letter_treated_invalid(self) -> None:
        samples = [_sample("q1", "E"), _sample("q2", "A")]
        predictions = [
            _prediction("q1", "The correct answer is (E)"),
            _prediction("q2", "The correct answer is (A)"),
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.invalid == 1
        assert result.correct == 1
        assert result.overall_accuracy == pytest.approx(0.5)


class TestOptionBiasDetection:
    """Option Bias（RStd）偵測。"""

    def test_all_a_preference_yields_high_rstd(self) -> None:
        samples = [_sample(f"q{i}", letter) for i, letter in enumerate("ABCD", start=1)]
        predictions = [
            _prediction(f"q{i}", "The correct answer is (A)") for i in range(1, 5)
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.option_recall == {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
        assert result.recall_std == pytest.approx(
            statistics.pstdev([1.0, 0.0, 0.0, 0.0])
        )
        assert result.recall_std == pytest.approx(0.4330127, abs=1e-5)

    def test_single_letter_targets_rstd_zero(self) -> None:
        samples = [_sample("q1", "A"), _sample("q2", "A")]
        predictions = [
            _prediction("q1", "The correct answer is (A)"),
            _prediction("q2", "B"),
        ]
        result = Evaluator().evaluate(samples, predictions)
        assert result.option_recall == {"A": 0.5}
        assert result.recall_std == 0.0


class TestPureHelpers:
    """純函式邊界。"""

    def test_compute_option_recalls_mixed(self) -> None:
        pairs: List[Tuple[str, Optional[str]]] = [
            ("A", "A"),
            ("A", "B"),
            ("B", None),
            ("C", "C"),
            ("D", "A"),
            ("", "A"),
        ]
        assert compute_option_recalls(pairs) == {
            "A": 0.5,
            "B": 0.0,
            "C": 1.0,
            "D": 0.0,
        }

    def test_compute_option_recalls_ignores_unknown_letters(self) -> None:
        assert compute_option_recalls([("E", "E"), ("X", None)]) == {}

    def test_compute_recall_std_less_than_two_letters_is_zero(self) -> None:
        assert compute_recall_std({}) == 0.0
        assert compute_recall_std({"A": 1.0}) == 0.0

    def test_compute_recall_std_population_formula(self) -> None:
        recalls = {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.5}
        assert compute_recall_std(recalls) == pytest.approx(
            statistics.pstdev([1.0, 0.5, 0.0, 0.5])
        )

    def test_safe_mean_empty_and_basic(self) -> None:
        assert safe_mean([]) == 0.0
        assert safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


class TestConfigDrivenConstruction:
    """設定驅動（evaluation.answer_regex 單一來源）。"""

    def test_from_config_custom_regex_honored(self) -> None:
        config = {"evaluation": {"answer_regex": r"(?i)final answer: \(?([A-D])\)?"}}
        evaluator = Evaluator.from_config(config)
        assert evaluator.extract_answer("final answer: (d)") == "D"
        # 非自訂標準格式 → 落 Tier 2 fallback 處理。
        assert evaluator.extract_answer("I pick B") == "B"

    def test_from_config_missing_evaluation_uses_default(self) -> None:
        assert (
            Evaluator.from_config({}).extract_answer("The correct answer is (B)") == "B"
        )

    def test_from_config_non_string_regex_uses_default(self) -> None:
        evaluator = Evaluator.from_config({"evaluation": {"answer_regex": 123}})
        assert evaluator.extract_answer("The correct answer is (B)") == "B"

    def test_from_config_invalid_regex_falls_back_without_crash(self) -> None:
        evaluator = Evaluator.from_config({"evaluation": {"answer_regex": "(("}})
        assert evaluator.extract_answer("The correct answer is (B)") == "B"

    def test_constructor_invalid_regex_falls_back(self) -> None:
        evaluator = Evaluator(strict_pattern="(?P<broken(")
        assert evaluator.extract_answer("The correct answer is (C)") == "C"

    def test_real_config_yaml_regex_behavior(self) -> None:
        with open(REAL_CONFIG_PATH, encoding="utf-8") as handle:
            config: Dict[str, Any] = yaml.safe_load(handle)
        evaluator = Evaluator.from_config(config)
        # 標準格式由 Tier 1 直接命中。
        assert evaluator.extract_answer("The correct answer is (B)") == "B"
        # 字邊界修正後，非標準措辭不再被 Tier 1 誤擷取；落 Tier 2 得正確字母。
        assert evaluator.extract_answer("The answer is B") == "B"
        # 純胡言亂語 → INVALID。
        assert evaluator.extract_answer("I don't know") is None
        assert evaluator.extract_answer("abcd") is None

    def test_default_regex_matches_yaml_value(self) -> None:
        """內建預設範式應與現行 YAML 設定等值（防回归）。"""
        with open(REAL_CONFIG_PATH, encoding="utf-8") as handle:
            config: Dict[str, Any] = yaml.safe_load(handle)
        yaml_pattern: Any = config["evaluation"]["answer_regex"]
        assert DEFAULT_STRICT_ANSWER_REGEX == yaml_pattern


class TestResultSerialization:
    """EvaluationResult 報告序列化與結構。"""

    def test_to_dict_is_json_serializable(self) -> None:
        samples = _build_samples(n_per_subject=1)
        predictions = [
            _prediction(
                str(sample["question_id"]),
                f"The correct answer is ({sample['target_letter']})",
            )
            for sample in samples
        ]
        result = Evaluator().evaluate(samples, predictions, model_name="test-model")
        payload = json.dumps(result.to_dict(), sort_keys=True)
        assert '"test-model"' in payload

    def test_to_dict_contains_all_reported_metrics(self) -> None:
        result = Evaluator().evaluate([], [], model_name="m")
        data = result.to_dict()
        for key in (
            "model",
            "total_samples",
            "valid_samples",
            "correct_samples",
            "invalid_samples",
            "overall_accuracy",
            "valid_accuracy",
            "invalid_parsing_rate",
            "domain_accuracy",
            "domain_accuracy_mean",
            "subject_accuracy",
            "option_recall",
            "recall_std",
            "average_latency_seconds",
            "category_scores",
            "subject_scores",
        ):
            assert key in data

    def test_model_name_default_empty(self) -> None:
        result = Evaluator().evaluate([], [])
        assert result.model_name == ""

    def test_dataclasses_are_frozen(self) -> None:
        result = Evaluator().evaluate([], [])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.total = 5
