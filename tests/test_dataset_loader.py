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


@patch("src.dataset_loader.load_dataset")
def test_numeric_to_letter_mapping(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """驗證數字索引可精確映射為 A-D 字母。"""
    mock_load_dataset.return_value = _mock_mmlu_rows()

    assert loader.map_numeric_to_letter(0) == "A"
    assert loader.map_numeric_to_letter(1) == "B"
    assert loader.map_numeric_to_letter(2) == "C"
    assert loader.map_numeric_to_letter(3) == "D"

    records: List[Dict[str, Any]] = loader.load_data(subject=MOCK_SUBJECT, split="test")
    mapped_letters: List[str] = [str(row["answer_letter"]) for row in records]
    assert mapped_letters == ["D", "B", "A", "C"]
    mock_load_dataset.assert_called_once()


@patch("src.dataset_loader.load_dataset")
def test_format_prompt_structure(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """確保格式化 Prompt 包含題幹、A-D 選項與嚴格輸出指令。"""
    mock_load_dataset.return_value = _mock_mmlu_rows()
    item: Dict[str, Any] = _mock_mmlu_rows()[0]

    prompt: str = loader.format_prompt(item)

    assert "The following are multiple choice questions (with answers) about global_facts." in prompt
    assert "Question: Which ocean is the largest?" in prompt
    assert "A. Atlantic" in prompt
    assert "B. Indian" in prompt
    assert "C. Arctic" in prompt
    assert "D. Pacific" in prompt
    assert "Format your output strictly as: 'The correct answer is (X)' where X is A, B, C, or D." in prompt
    assert prompt.strip().endswith("Answer:")
    mock_load_dataset.assert_not_called()


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


@patch("src.dataset_loader.load_dataset")
def test_missing_field_handling(
    mock_load_dataset: MagicMock,
    loader: MMLUDatasetLoader,
) -> None:
    """確保資料欄位缺失時能優雅處理，不中斷整體載入流程。"""
    mock_load_dataset.return_value = [
        {
            "question": "Valid question with complete fields?",
            "subject": MOCK_SUBJECT,
            "choices": ["Yes", "No", "Maybe", "Unknown"],
            "answer": 0,
        },
        {
            # 缺少 question 與部分 choices，應以降級空字串組 Prompt。
            "subject": MOCK_SUBJECT,
            "choices": ["Only-A"],
            "answer": 1,
        },
        {
            # 缺少 answer，應略過此列。
            "question": "Missing answer should be skipped?",
            "choices": ["A1", "B1", "C1", "D1"],
        },
        {},
    ]

    records: List[Dict[str, Any]] = loader.load_data(subject=MOCK_SUBJECT, split="test")
    assert len(records) == 2
    assert records[0]["answer_letter"] == "A"
    assert records[1]["question"] == ""
    assert records[1]["choices"] == ["Only-A", "", "", ""]
    assert records[1]["answer_letter"] == "B"

    degraded_prompt: str = loader.format_prompt({"subject": MOCK_SUBJECT})
    assert "Question: " in degraded_prompt
    assert "A. " in degraded_prompt
    assert "B. " in degraded_prompt
    assert "C. " in degraded_prompt
    assert "D. " in degraded_prompt

    empty_mock: MagicMock = mock_load_dataset
    empty_mock.return_value = []
    assert loader.load_data(subject=MOCK_SUBJECT, split="test") == []
