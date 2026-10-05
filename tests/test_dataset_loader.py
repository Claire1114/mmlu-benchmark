"""MMLUDatasetLoader 離線單元測試。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.dataset_loader import DEFAULT_PROMPT_TEMPLATE, MMLUDatasetLoader

MOCK_SUBJECT: str = "global_facts"
REAL_CONFIG_PATH: str = str(
    Path(__file__).resolve().parents[1] / "configs" / "eval_config.yaml"
)


def _write_config(
    tmp_path: Path,
    sample_size: int = 5,
    seed: Optional[int] = 42,
    active_mode: Optional[str] = "smoke_test",
    prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE,
    categories: Optional[Dict[str, Any]] = None,
    include_cache_dir: bool = False,
) -> str:
    """寫入新版格式（categories / modes / prompt_template）的測試用 YAML 設定檔。

    Args:
        tmp_path: Pytest 暫存目錄。
        sample_size: ``smoke_test`` 模式的 ``sample_size_per_subject``。
        seed: ``project.seed``；``None`` 表示不寫入該鍵。
        active_mode: ``dataset.active_mode``；``None`` 表示不寫入該鍵。
        prompt_template: ``dataset.prompt_template``；``None`` 表示不寫入該鍵。
        categories: 自訂領域／子集結構；預設為單領域含 ``MOCK_SUBJECT``。
        include_cache_dir: 是否寫入 ``dataset.cache_dir``。

    Returns:
        設定檔字串路徑。
    """
    if categories is None:
        categories = {
            "Humanities": {
                "description": "test category",
                "subjects": [{"name": MOCK_SUBJECT, "focus": "test focus"}],
            }
        }
    dataset: Dict[str, Any] = {
        "name": "cais/mmlu",
        "language": "en",
        "split": "test",
        "modes": {
            "smoke_test": {
                "sample_size_per_subject": sample_size,
                "purpose": "smoke test",
            },
            "demo": {"sample_size_per_subject": sample_size + 25, "purpose": "demo"},
        },
        "categories": categories,
    }
    if active_mode is not None:
        dataset["active_mode"] = active_mode
    if prompt_template is not None:
        dataset["prompt_template"] = prompt_template
    if include_cache_dir:
        dataset["cache_dir"] = str(tmp_path / "hf_cache")
    project: Dict[str, Any] = {"name": "test-project"}
    if seed is not None:
        project["seed"] = seed
    config: Dict[str, Any] = {"project": project, "dataset": dataset}
    config_path: Path = tmp_path / "eval_config.yaml"
    # sort_keys=False：維持 Python dict 插入順序，使領域／模式順序可測試。
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(config_path)


def _mock_mmlu_rows(subject: Optional[str] = MOCK_SUBJECT) -> List[Dict[str, Any]]:
    """建立四筆完整的模擬 MMLU 資料列。

    Args:
        subject: 寫入資料列的科目名；``None`` 時列中不含 ``subject`` 欄
            （供載入端以 fallback subject 填補）。

    Returns:
        含題幹、四選項與數字標籤的資料清單。
    """
    rows: List[Dict[str, Any]] = [
        {
            "question": "Which ocean is the largest?",
            "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
            "answer": 3,
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "choices": ["Earth", "Mars", "Venus", "Jupiter"],
            "answer": 1,
        },
        {
            "question": "What is the capital of France?",
            "choices": ["Paris", "Lyon", "Nice", "Lille"],
            "answer": 0,
        },
        {
            "question": "How many continents are there?",
            "choices": ["Five", "Six", "Seven", "Eight"],
            "answer": 2,
        },
    ]
    if subject is not None:
        for row in rows:
            row["subject"] = subject
    return rows


def _mock_mmlu_pool(count: int) -> List[Dict[str, Any]]:
    """建立連續編號的模擬 MMLU 列池（count 列），題幹含列序號。

    Args:
        count: 要產生的列數。

    Returns:
        含題幹、四選項與數字標籤的資料清單。
    """
    return [
        {
            "question": f"Sampling pool question {index}?",
            "subject": MOCK_SUBJECT,
            "choices": ["A", "B", "C", "D"],
            "answer": index % 4,
        }
        for index in range(count)
    ]


@pytest.fixture
def loader(tmp_path: Path) -> MMLUDatasetLoader:
    """提供綁定暫存設定檔的載入器實例。

    Args:
        tmp_path: Pytest 暫存目錄。

    Returns:
        初始化完成的 ``MMLUDatasetLoader``。
    """
    return MMLUDatasetLoader(config_path=_write_config(tmp_path))


def test_numeric_to_letter_mapping(loader: MMLUDatasetLoader) -> None:
    """驗證純函式：整數索引 0-3 可精確映射為 A-D（不依賴資料集載入）。"""
    assert loader.map_numeric_to_letter(0) == "A"
    assert loader.map_numeric_to_letter(1) == "B"
    assert loader.map_numeric_to_letter(2) == "C"
    assert loader.map_numeric_to_letter(3) == "D"


@patch("src.dataset_loader.load_dataset")
def test_load_data_maps_answer_letters(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """驗證整合路徑：載入後各列 answer_letter 與原始數字標籤一致。"""
    mock_load_dataset.return_value = _mock_mmlu_rows()

    records: List[Dict[str, Any]] = loader.load_data(subject=MOCK_SUBJECT, split="test")
    mapped_letters: List[str] = [str(row["answer_letter"]) for row in records]
    assert mapped_letters == ["D", "B", "A", "C"]
    assert loader.skipped_rows == 0
    mock_load_dataset.assert_called_once_with("cais/mmlu", MOCK_SUBJECT, split="test")


def test_format_prompt_structure(loader: MMLUDatasetLoader) -> None:
    """確保格式化 Prompt 包含題幹、A-D 選項與嚴格輸出指令。"""
    item: Dict[str, Any] = _mock_mmlu_rows()[0]

    prompt: str = loader.format_prompt(item)

    assert (
        "The following are multiple choice questions (with answers) about global_facts."
        in prompt
    )
    assert "Question: Which ocean is the largest?" in prompt
    assert "A. Atlantic" in prompt
    assert "B. Indian" in prompt
    assert "C. Arctic" in prompt
    assert "D. Pacific" in prompt
    assert (
        "Format your output strictly as: 'The correct answer is (X)' "
        "where X is A, B, C, or D."
    ) in prompt
    assert prompt.strip().endswith("Answer:")


@patch("src.dataset_loader.load_dataset")
def test_sample_size_slicing(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """驗證 sample_size 能限制回傳筆數，且種子抽樣可重現。"""
    mock_load_dataset.return_value = _mock_mmlu_rows()

    sliced: List[Dict[str, Any]] = loader.load_data(
        subject=MOCK_SUBJECT,
        split="test",
        sample_size=2,
        seed=42,
    )
    assert len(sliced) == 2
    pool_questions = {row["question"] for row in _mock_mmlu_rows()}
    assert {row["question"] for row in sliced} <= pool_questions

    again: List[Dict[str, Any]] = loader.load_data(
        subject=MOCK_SUBJECT,
        split="test",
        sample_size=2,
        seed=42,
    )
    assert [row["question"] for row in sliced] == [row["question"] for row in again]

    samples: List[Dict[str, Any]] = loader.get_samples()
    assert len(samples) == 4  # active_mode=smoke_test sample_size=5 > 列池 4 → 全數保留
    assert set(samples[0].keys()) == {
        "question_id",
        "formatted_prompt",
        "target_letter",
        "subject",
        "category",
    }
    assert [sample["target_letter"] for sample in samples] == ["D", "B", "A", "C"]
    assert samples[0]["subject"] == MOCK_SUBJECT
    assert samples[0]["category"] == "Humanities"


@patch("src.dataset_loader.load_dataset")
def test_missing_field_handling(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """確保髒資料（缺答案、選項不足、空列）被嚴格略過，不中斷整體載入。"""
    mock_load_dataset.return_value = [
        {
            "question": "Valid question with complete fields?",
            "subject": MOCK_SUBJECT,
            "choices": ["Yes", "No", "Maybe", "Unknown"],
            "answer": 0,
        },
        {
            # 選項數量不等於 4：嚴格拒絕，整列略過。
            "subject": MOCK_SUBJECT,
            "choices": ["Only-A"],
            "answer": 1,
        },
        {
            # 缺少 answer：整列略過。
            "question": "Missing answer should be skipped?",
            "choices": ["A1", "B1", "C1", "D1"],
        },
        {},
    ]

    records: List[Dict[str, Any]] = loader.load_data(subject=MOCK_SUBJECT, split="test")
    assert len(records) == 1
    assert records[0]["answer_letter"] == "A"
    assert records[0]["question"] == "Valid question with complete fields?"
    assert records[0]["choices"] == ["Yes", "No", "Maybe", "Unknown"]
    assert loader.skipped_rows == 3

    # 缺 choices 的 item 不得組出含字面 "None" 的 Prompt，應直接拋錯。
    with pytest.raises(ValueError, match="Choices must be"):
        loader.format_prompt({"subject": MOCK_SUBJECT})

    mock_load_dataset.return_value = []
    assert loader.load_data(subject=MOCK_SUBJECT, split="test") == []
    assert loader.skipped_rows == 0


@patch("src.dataset_loader.load_dataset")
def test_answer_label_whitespace_and_letter_aliases(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """驗證空白數字字串與大小寫字母標籤皆可映射；非法別名應略過。"""
    mock_load_dataset.return_value = [
        {
            "question": "Whitespace zero?",
            "subject": MOCK_SUBJECT,
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": " 0 ",
        },
        {
            "question": "Lowercase letter?",
            "subject": MOCK_SUBJECT,
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": "c",
        },
        {
            "question": "Invalid alias should drop?",
            "subject": MOCK_SUBJECT,
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": "E",
        },
    ]

    records = loader.load_data(subject=MOCK_SUBJECT, split="test")
    assert [row["answer_letter"] for row in records] == ["A", "C"]
    assert loader.skipped_rows == 1


@patch("src.dataset_loader.load_dataset")
def test_reject_float_truncation_and_bool_labels(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """拒絕會造成靜默錯標的 float 截斷與 bool 標籤。"""
    mock_load_dataset.return_value = [
        {
            "question": "Float truncation trap?",
            "subject": MOCK_SUBJECT,
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": 1.7,
        },
        {
            "question": "Bool trap?",
            "subject": MOCK_SUBJECT,
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": True,
        },
        {
            "question": "Exact integer float OK?",
            "subject": MOCK_SUBJECT,
            "choices": ["A1", "B1", "C1", "D1"],
            "answer": 2.0,
        },
    ]

    records = loader.load_data(subject=MOCK_SUBJECT, split="test")
    assert len(records) == 1
    assert records[0]["answer_letter"] == "C"
    assert loader.skipped_rows == 2
    with pytest.raises(ValueError):
        loader.map_numeric_to_letter(1.7)
    with pytest.raises(ValueError):
        loader.map_numeric_to_letter(True)


@patch("src.dataset_loader.load_dataset")
def test_choices_none_element_is_skipped(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """choices 含 None 時不得變成字面 'None'，整列應被略過。"""
    mock_load_dataset.return_value = [
        {
            "question": "Poisoned None choice?",
            "subject": MOCK_SUBJECT,
            "choices": ["Alpha", None, "Gamma", "Delta"],
            "answer": 0,
        },
        {
            "question": "Healthy row?",
            "subject": MOCK_SUBJECT,
            "choices": ["Alpha", "Beta", "Gamma", "Delta"],
            "answer": 1,
        },
    ]

    records = loader.load_data(subject=MOCK_SUBJECT, split="test")
    assert len(records) == 1
    assert records[0]["answer_letter"] == "B"
    assert loader.skipped_rows == 1
    assert "None" not in loader.format_prompt(records[0])


@patch("src.dataset_loader.load_dataset")
def test_choices_length_anomaly_rejected(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """選項少於或多於 4 個時應拒絕該列，避免靜默 pad/truncate。"""
    mock_load_dataset.return_value = [
        {
            "question": "Too few?",
            "subject": MOCK_SUBJECT,
            "choices": ["Only-A", "Only-B"],
            "answer": 0,
        },
        {
            "question": "Too many?",
            "subject": MOCK_SUBJECT,
            "choices": ["A", "B", "C", "D", "E"],
            "answer": 0,
        },
        {
            "question": "Exact four?",
            "subject": MOCK_SUBJECT,
            "choices": ["A", "B", "C", "D"],
            "answer": 3,
        },
    ]

    records = loader.load_data(subject=MOCK_SUBJECT, split="test")
    assert len(records) == 1
    assert records[0]["answer_letter"] == "D"
    assert loader.skipped_rows == 2


@patch("src.dataset_loader.load_dataset")
def test_question_id_stable_and_collision_resistant(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """相同內容應產生穩定 ID；不同題幹即使同 index 語意也應可區分。"""
    row_a = {
        "question": "Same index different text A?",
        "subject": MOCK_SUBJECT,
        "choices": ["A1", "B1", "C1", "D1"],
        "answer": 0,
    }
    row_b = {
        "question": "Same index different text B?",
        "subject": MOCK_SUBJECT,
        "choices": ["A1", "B1", "C1", "D1"],
        "answer": 0,
    }
    mock_load_dataset.return_value = [row_a]
    first = loader.load_data(subject=MOCK_SUBJECT, split="test")[0]["question_id"]

    mock_load_dataset.return_value = [row_a]
    first_again = loader.load_data(subject=MOCK_SUBJECT, split="test")[0]["question_id"]
    assert first == first_again

    mock_load_dataset.return_value = [row_b]
    second = loader.load_data(subject=MOCK_SUBJECT, split="test")[0]["question_id"]
    assert first != second
    assert first.startswith(f"{MOCK_SUBJECT}__000000__")


def test_datasetdict_missing_split_raises(loader: MMLUDatasetLoader) -> None:
    """DatasetDict 缺少指定 split 時必須失敗，禁止默默改載其他 split。"""
    dataset_dict = {
        "validation": [
            {
                "question": "Should never be used for test split?",
                "subject": MOCK_SUBJECT,
                "choices": ["A", "B", "C", "D"],
                "answer": 0,
            }
        ]
    }

    with pytest.raises(ValueError, match="missing required split"):
        loader._coerce_records(dataset_dict, split="test")


def test_iter_subjects_order_and_category_mapping(tmp_path: Path) -> None:
    """iter_subjects 應維持 YAML 宣告順序並攜帶領域對應。"""
    categories = {
        "STEM": {
            "description": "sci",
            "subjects": [
                {"name": "alpha_subject", "focus": "f1"},
                {"name": "beta_subject"},
            ],
        },
        "Other": {
            "description": "appl",
            "subjects": [{"name": "gamma_subject", "focus": "f2"}],
        },
    }
    loader = MMLUDatasetLoader(
        config_path=_write_config(tmp_path, categories=categories)
    )
    assert loader.iter_subjects() == [
        {"category": "STEM", "subject": "alpha_subject", "focus": "f1"},
        {"category": "STEM", "subject": "beta_subject", "focus": ""},
        {"category": "Other", "subject": "gamma_subject", "focus": "f2"},
    ]


def test_real_config_exposes_eight_subjects_and_smoke_mode() -> None:
    """對真實生產設定檔的迴歸：四領域 × 2 科目，active_mode=smoke_test（10 題）。"""
    loader = MMLUDatasetLoader(config_path=REAL_CONFIG_PATH)
    subjects = loader.iter_subjects()
    assert [entry["subject"] for entry in subjects] == [
        "college_computer_science",
        "high_school_mathematics",
        "philosophy",
        "world_religions",
        "econometrics",
        "high_school_psychology",
        "clinical_knowledge",
        "professional_law",
    ]
    assert {entry["subject"]: entry["category"] for entry in subjects} == {
        "college_computer_science": "STEM",
        "high_school_mathematics": "STEM",
        "philosophy": "Humanities",
        "world_religions": "Humanities",
        "econometrics": "Social_Sciences",
        "high_school_psychology": "Social_Sciences",
        "clinical_knowledge": "Other",
        "professional_law": "Other",
    }
    assert loader.resolve_sample_size() == 10


@pytest.mark.parametrize(
    "bad_categories",
    [
        {"STEM": {"subjects": []}},
        {"STEM": {"subjects": [{"name": ""}]}},
        {"STEM": {"subjects": "not-a-list"}},
        {
            "STEM": {"subjects": [{"name": "duplicated"}]},
            "Other": {"subjects": [{"name": "duplicated"}]},
        },
    ],
)
def test_invalid_categories_rejected(
    tmp_path: Path, bad_categories: Dict[str, Any]
) -> None:
    """結構異常（空清單／缺 name／非清單／跨領域重名）必須被拒絕。"""
    with pytest.raises(ValueError):
        MMLUDatasetLoader(
            config_path=_write_config(tmp_path, categories=bad_categories)
        )


def test_missing_categories_rejected(tmp_path: Path) -> None:
    config: Dict[str, Any] = {
        "project": {"seed": 1},
        "dataset": {"name": "cais/mmlu", "split": "test"},
    }
    config_path: Path = tmp_path / "eval_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="categories"):
        MMLUDatasetLoader(config_path=str(config_path))


def test_resolve_sample_size_follows_active_mode(tmp_path: Path) -> None:
    """active_mode=smoke_test 取 smoke_test 筆數；切換 demo 取對應值。"""
    loader = MMLUDatasetLoader(config_path=_write_config(tmp_path, sample_size=7))
    assert loader.resolve_sample_size() == 7
    loader.config["dataset"]["active_mode"] = "demo"
    assert loader.resolve_sample_size() == 32


@pytest.mark.parametrize("active_mode", [None, "full_benchmark"])
def test_unknown_active_mode_rejected(
    tmp_path: Path, active_mode: Optional[str]
) -> None:
    """active_mode 缺失或未知模式名必須被拒絕並列出可用模式。"""
    config_path = _write_config(tmp_path, active_mode=active_mode)
    with pytest.raises(ValueError, match="active_mode"):
        MMLUDatasetLoader(config_path=config_path)


def test_seed_missing_falls_back_to_default(tmp_path: Path) -> None:
    """project.seed 缺失時應回退預設值 42。"""
    loader = MMLUDatasetLoader(config_path=_write_config(tmp_path, seed=None))
    assert loader._seed == 42


def test_non_integer_seed_rejected(tmp_path: Path) -> None:
    """project.seed 非整數時應被拒絕。"""
    with pytest.raises(ValueError, match="project.seed"):
        MMLUDatasetLoader(config_path=_write_config(tmp_path, seed="not-an-int"))


@patch("src.dataset_loader.load_dataset")
def test_seeded_sampling_is_deterministic_and_preserves_order(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """同種子＋同科目應完全可重現；抽樣列保留原始索引順序。"""
    mock_load_dataset.return_value = _mock_mmlu_pool(30)

    def index_of(question: str) -> int:
        return int(question.rsplit(" ", 1)[-1].rstrip("?"))

    first = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=42)
    ]
    second = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=42)
    ]
    assert len(first) == 10
    assert first == second
    assert [index_of(q) for q in first] == sorted(index_of(q) for q in first)


@patch("src.dataset_loader.load_dataset")
def test_seeded_sampling_differs_across_seeds_and_subjects(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """不同種子、或同種子不同科目，應產生相互獨立的抽樣結果。"""
    mock_load_dataset.return_value = _mock_mmlu_pool(30)
    base = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=42)
    ]
    other_seed = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=99)
    ]
    other_subject = [
        row["question"]
        for row in loader.load_data(subject="another_subject", sample_size=10, seed=42)
    ]
    assert other_seed != base
    assert other_subject != base


@patch("src.dataset_loader.load_dataset")
def test_seed_none_falls_back_to_config_seed(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """``seed=None``（文件化預設值）應使用設定檔 ``project.seed`` 抽樣。

    對齊 ``load_data`` 契約：``None`` 表示使用設定檔 ``project.seed``；
    與顯式傳入相同種子必須產生完全相同的確定性子集。
    """
    loader = MMLUDatasetLoader(config_path=_write_config(tmp_path, seed=7))
    mock_load_dataset.return_value = _mock_mmlu_pool(30)
    implicit = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10)
    ]
    explicit = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=7)
    ]
    assert len(implicit) == 10
    assert implicit == explicit


@patch("src.dataset_loader.load_dataset")
def test_seed_none_with_missing_config_seed_uses_default(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """``project.seed`` 缺失時，``seed=None`` 回落預設種子 42。"""
    loader = MMLUDatasetLoader(config_path=_write_config(tmp_path, seed=None))
    mock_load_dataset.return_value = _mock_mmlu_pool(30)
    implicit = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10)
    ]
    explicit = [
        row["question"]
        for row in loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=42)
    ]
    assert len(implicit) == 10
    assert implicit == explicit


@patch("src.dataset_loader.load_dataset")
def test_explicit_non_integer_seed_still_rejected(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """顯式傳入非整數 seed（字串／bool）仍必須拒絕，fallback 不吞掉型別錯誤。"""
    mock_load_dataset.return_value = _mock_mmlu_pool(30)
    with pytest.raises(ValueError, match="seed must be an integer"):
        loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed="42")
    with pytest.raises(ValueError, match="seed must be an integer"):
        loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=True)


@patch("src.dataset_loader.load_dataset")
def test_sample_size_larger_than_pool_returns_all_rows(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """有效列池小於抽樣筆數時全數保留，不拋例外。"""
    mock_load_dataset.return_value = _mock_mmlu_rows()
    records = loader.load_data(subject=MOCK_SUBJECT, sample_size=10, seed=42)
    assert len(records) == 4
    assert loader.skipped_rows == 0


@patch("src.dataset_loader.load_dataset")
def test_non_integer_sample_size_rejected(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """sample_size 非整數時應被拒絕。"""
    with pytest.raises(ValueError, match="sample_size"):
        loader.load_data(subject=MOCK_SUBJECT, sample_size="ten")


def test_custom_prompt_template_used(tmp_path: Path) -> None:
    """設定檔自訂 prompt_template 應優於內建預設被採用。"""
    template = "SUBJECT={subject}|Q={question}|A={choice_A}|B={choice_B}|C={choice_C}|D={choice_D}"
    loader = MMLUDatasetLoader(
        config_path=_write_config(tmp_path, prompt_template=template)
    )
    prompt = loader.format_prompt(_mock_mmlu_rows()[0])
    assert prompt == (
        f"SUBJECT={MOCK_SUBJECT}|Q=Which ocean is the largest?"
        "|A=Atlantic|B=Indian|C=Arctic|D=Pacific"
    )


def test_missing_prompt_template_falls_back_to_default(tmp_path: Path) -> None:
    """dataset.prompt_template 缺失時應回退內建範本而不失敗。"""
    loader = MMLUDatasetLoader(
        config_path=_write_config(tmp_path, prompt_template=None)
    )
    prompt = loader.format_prompt(_mock_mmlu_rows()[0])
    assert (
        "The following are multiple choice questions (with answers) about global_facts."
        in prompt
    )
    assert prompt.strip().endswith("Answer:")


@pytest.mark.parametrize("bad_template", ["{choice_E}", "Oops {", "{0}"])
def test_invalid_prompt_template_rejected_at_init(
    tmp_path: Path,
    bad_template: str,
) -> None:
    """佔位符無效應於初始化即失敗，而非逐題格式化時才爆炸。"""
    with pytest.raises(ValueError, match="prompt_template"):
        MMLUDatasetLoader(
            config_path=_write_config(tmp_path, prompt_template=bad_template)
        )


@patch("src.dataset_loader.load_dataset")
def test_get_samples_covers_all_subjects_with_category(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """get_samples 應遍歷全部設定科目，且樣本附領域（category）欄位。"""
    categories = {
        "STEM": {
            "description": "sci",
            "subjects": [
                {"name": "alpha_subset", "focus": "f"},
                {"name": "beta_subset", "focus": "f"},
            ],
        },
        "Other": {
            "description": "appl",
            "subjects": [
                {"name": "gamma_subset", "focus": "f"},
                {"name": "delta_subset", "focus": "f"},
            ],
        },
    }
    loader = MMLUDatasetLoader(
        config_path=_write_config(tmp_path, sample_size=5, categories=categories)
    )
    mock_load_dataset.return_value = _mock_mmlu_rows(subject=None)

    samples = loader.get_samples()

    assert len(samples) == 16  # 4 科目 × 4 列（列池 < 5，全數保留）
    assert set(samples[0].keys()) == {
        "question_id",
        "formatted_prompt",
        "target_letter",
        "subject",
        "category",
    }
    assert {sample["subject"] for sample in samples} == {
        "alpha_subset",
        "beta_subset",
        "gamma_subset",
        "delta_subset",
    }
    mapping = {sample["subject"]: sample["category"] for sample in samples}
    assert mapping == {
        "alpha_subset": "STEM",
        "beta_subset": "STEM",
        "gamma_subset": "Other",
        "delta_subset": "Other",
    }
    assert [sample["target_letter"] for sample in samples] == ["D", "B", "A", "C"] * 4


@patch("src.dataset_loader.load_dataset")
def test_cache_dir_sets_hf_environment(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """dataset.cache_dir 設定時應預設寫入 HF_DATASETS_CACHE 環境變數。"""
    cache_path = str(tmp_path / "hf_cache")
    loader = MMLUDatasetLoader(
        config_path=_write_config(tmp_path, include_cache_dir=True)
    )
    old = os.environ.pop("HF_DATASETS_CACHE", None)
    try:
        mock_load_dataset.return_value = _mock_mmlu_rows()
        loader.load_data(subject=MOCK_SUBJECT)
        assert os.environ.get("HF_DATASETS_CACHE") == cache_path
    finally:
        os.environ.pop("HF_DATASETS_CACHE", None)
        if old is not None:
            os.environ["HF_DATASETS_CACHE"] = old
