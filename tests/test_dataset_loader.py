"""MMLUDatasetLoader 離線單元測試。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.dataset_loader import MMLUDatasetLoader

MOCK_SUBJECT: str = "global_facts"


def _write_config(tmp_path: Path, sample_size: int = 5) -> str:
    """寫入測試用 YAML 設定檔。

    Args:
        tmp_path: Pytest 暫存目錄。
        sample_size: 設定檔中的抽樣筆數。

    Returns:
        設定檔字串路徑。
    """
    config: Dict[str, Any] = {
        "dataset": {
            "name": "cais/mmlu",
            "subject": MOCK_SUBJECT,
            "split": "test",
            "sample_size": sample_size,
        }
    }
    config_path: Path = tmp_path / "eval_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(config_path)


def _mock_mmlu_rows() -> List[Dict[str, Any]]:
    """建立四筆完整的模擬 MMLU 資料列。

    Returns:
        含題幹、四選項與數字標籤的資料清單。
    """
    return [
        {
            "question": "Which ocean is the largest?",
            "subject": MOCK_SUBJECT,
            "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
            "answer": 3,
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "subject": MOCK_SUBJECT,
            "choices": ["Earth", "Mars", "Venus", "Jupiter"],
            "answer": 1,
        },
        {
            "question": "What is the capital of France?",
            "subject": MOCK_SUBJECT,
            "choices": ["Paris", "Lyon", "Nice", "Lille"],
            "answer": 0,
        },
        {
            "question": "How many continents are there?",
            "subject": MOCK_SUBJECT,
            "choices": ["Five", "Six", "Seven", "Eight"],
            "answer": 2,
        },
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
    """驗證 sample_size 能正確限制回傳筆數。"""
    mock_load_dataset.return_value = _mock_mmlu_rows()

    sliced: List[Dict[str, Any]] = loader.load_data(
        subject=MOCK_SUBJECT,
        split="test",
        sample_size=2,
    )
    assert len(sliced) == 2
    assert sliced[0]["question"] == "Which ocean is the largest?"
    assert sliced[1]["question"] == "Which planet is known as the Red Planet?"

    samples: List[Dict[str, Any]] = loader.get_samples()
    assert len(samples) == 2
    assert set(samples[0].keys()) == {
        "question_id",
        "formatted_prompt",
        "target_letter",
        "subject",
    }
    assert samples[0]["target_letter"] == "D"
    assert samples[1]["target_letter"] == "B"


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
