# Step 3／4 對接規格：Runner 與 Evaluator

本文只定義 DataLoader 對 Runner／Evaluator 的最小對接契約，依據 `src/dataset_loader.py` 及 `configs/eval_config.yaml`。

## 1. Runner 呼叫方式

推薦一次取得完整樣本：

```python
from src.dataset_loader import MMLUDatasetLoader

loader = MMLUDatasetLoader("configs/eval_config.yaml")
loader.resolve_sample_size()  # 啟動時先驗證 active_mode 的抽樣筆數
samples = loader.get_samples()

for sample in samples:
    raw_output = runner.predict(sample["formatted_prompt"])
    # 將 raw_output 與 sample 一起保存／交給答案正規化及 Evaluator。
```

`runner.predict()` 是 Runner 自己的模型呼叫介面，不是 DataLoader 方法。模型輸入只使用 `formatted_prompt`；不得把正確答案 `target_letter` 餵給模型。

需要逐科處理時，`iter_subjects()` 只提供科目 metadata；搭配 `load_data()` 載入題目，再以 `format_prompt()` 組 Prompt：

```python
for subject_info in loader.iter_subjects():
    for row in loader.load_data(subject=subject_info["subject"]):
        raw_output = runner.predict(loader.format_prompt(row))
        # row["answer_letter"] 是標準答案；subject_info["category"] 是領域。
```

`load_data()` 未指定 `split`、`sample_size`、`seed` 時，使用設定檔 split、active mode 每科樣本上限與全域 seed。

## 2. `samples` 資料規格

`get_samples()` 回傳 `List[Dict[str, Any]]`；每個 Dict 固定有下列五個 key，且值均為 `str`：

| Key | Type | 說明 | 範例 |
|---|---|---|---|
| `question_id` | `str` | 題目識別碼；格式 `{subject}__{來源列索引:06d}__{題幹及選項 SHA-1 前 8 碼}`。請原樣保留，不要自行重建。 | `college_computer_science__000123__a1b2c3d4` |
| `formatted_prompt` | `str` | 依 YAML 的 `dataset.prompt_template` 填入科目、題幹與 A–D 選項後的完整模型輸入。 | `The following are multiple choice questions (with answers) about college_computer_science.\n\nQuestion: What is 1 + 1?\nA. 1\nB. 2\nC. 3\nD. 4\n\nFormat your output strictly as: 'The correct answer is (X)' where X is A, B, C, or D.\nAnswer:` |
| `target_letter` | `str` | 正規化的標準答案，只會是大寫 `A`、`B`、`C`、`D`；只供評估，不能放進模型輸入。 | `B` |
| `subject` | `str` | MMLU 子集／科目，用於科目層級分組。 | `college_computer_science` |
| `category` | `str` | 所屬大領域，用於領域層級分組。 | `STEM` |

樣本示意：

```python
{
    "question_id": "college_computer_science__000123__a1b2c3d4",
    "formatted_prompt": "The following are multiple choice questions (with answers) about college_computer_science.\n\nQuestion: What is 1 + 1?\nA. 1\nB. 2\nC. 3\nD. 4\n\nFormat your output strictly as: 'The correct answer is (X)' where X is A, B, C, or D.\nAnswer:",
    "target_letter": "B",
    "subject": "college_computer_science",
    "category": "STEM",
}
```

## 3. 模型輸入與評估流程／鐵律

1. DataLoader 產生樣本：`formatted_prompt` 是模型輸入；`target_letter`、`subject`、`category`、`question_id` 是評估與追蹤 metadata。
2. Runner 將 `formatted_prompt` 傳給模型，保存原始輸出，並將原始輸出依評估設定（`evaluation.answer_regex`）正規化為 A–D 預測答案。
3. Evaluator 以正規化預測答案比對 `target_letter` 計分，並保留 `question_id` 關聯及 `subject`／`category` 分組資料。

**不可違反的評估規則：**

- 標準答案不得進入 Prompt、模型請求或任何會影響生成的內容，避免答案洩漏。
- 只對可正規化為 A–D 的預測計分；無法解析的輸出應標記為無效／未答，並依 Evaluator 的明確政策處理，不可自行猜答案。
- 同時保留 `subject` 與 `category`：前者支援子集指標，後者支援大領域指標；不得只存一層。
- 保留原始模型輸出及 `question_id`，讓結果可追溯、可稽核；不要將無效題目或無效預測靜默丟棄。

本設定以 `active_mode` 決定每科抽樣上限（smoke test 10 題、demo 30 題）。相同 seed、科目及資料列順序可重現抽樣；各科以字串 `"{seed}::{subject}"` 派生獨立隨機流。有效題數不足時會保留全部有效題目，不會補題。

> **待確認的 Evaluator 政策：**無法解析的模型輸出是否計為答錯（分母仍納入），或標記為無效並排除分母？建議在 Step 4 明確選定並固定此政策，避免不同模型結果不可比較。
