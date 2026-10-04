# Step 3／4 對接規格：模型介面、Runner 與 Evaluator

本文定義 DataLoader／Model Interface（Runner）／Evaluator 之間的最小對接契約，
依據 `src/dataset_loader.py`、`src/models/`、`src/evaluator.py` 及 `configs/eval_config.yaml`。

## 1. Runner 呼叫方式與職責邊界

依循單一職責原則（Single Responsibility），**Runner 僅負責模型推論，不參與任何答案解析、正則過濾或指標計算**。

推薦全流程標準介面（以批次或列表方式呼叫，方便本地批次加速或雲端 API 併發）：

```python
from src.dataset_loader import MMLUDatasetLoader
from src.models import build_model_interface

# 1. 統一載入評測樣本
loader = MMLUDatasetLoader("configs/eval_config.yaml")
loader.resolve_sample_size()  # 驗證 active_mode 的抽樣筆數
samples = loader.get_samples()

# 2. 模型批次推論（批次大小或執行緒數依 configs/eval_config.yaml 控制）
# Runner 僅接收包含 formatted_prompt 的清單，不得接收 target_letter
model = build_model_interface(model_cfg, evaluation_cfg)  # 逐一依 models[] 建構
predictions = model.predict_batch(samples)

# predictions 輸出規格：List[Dict[str, object]]
# 每筆固定包含：
# - "question_id": str (原樣保留對齊)
# - "raw_output": str (模型生成的未處理原始文字，或 "ERROR: ..." 哨兵字串)
# - "latency": float (該題推理耗時，單位：秒，含重試等待時間)
```

## 2. 模型介面規格（`src/models/`）

### 2.1 統一介面與工廠
- `BaseModelInterface(ABC)`（`src/models/base.py`）：抽象方法 `predict(prompt: str) -> str`；
  具體 `predict_batch(samples)` 循序批次實作（不使用執行緒池，防止 VRAM OOM 或雲端限流放大）。
- `ModelProtocol`（`src/models/interfaces.py`）：結構化協議，暴露 `model_name` 與
  `predict()`，通過 `isinstance` 檢查之任一包裝類皆可作為 Runner。
- `build_model_interface(model_cfg, evaluation_cfg)` 工廠：依 `models[].type` 建構具體介面；
  第一階段支援 `mock` / `openai_compatible` / `groq` / `ollama` / `openrouter`，
  `huggingface` / `gemini` 排程於 Step 4 後續階段（現行拋 `NotImplementedError`）。

### 2.2 輸入／輸出資料格式
| 方向 | 欄位 | 型別 | 說明 |
| :--- | :--- | :--- | :--- |
| 輸入 | `question_id` | `str` | 題目識別碼，輸出原樣保留供對齊 |
| 輸入 | `formatted_prompt` | `str` | 唯一推論輸入（由 DataLoader 依 `dataset.prompt_template` 組裝） |
| 輸入 | `target_letter` / `subject` / `category` | `str` | **模型層禁止讀取**（防止答案鍵滲漏進推論） |
| 輸出 | `question_id` | `str` | 原樣保留（非字串型別以 `str()` 強制轉換） |
| 輸出 | `raw_output` | `str` | 模型原始文字，或第一層防護哨兵 `"ERROR: <ExceptionType>: <message>"` |
| 輸出 | `latency` | `float` | 秒；逐題量測（含重試等待時間）；錯誤紀錄為 `0.0` |

### 2.3 雙層例外防護機制
| 層級 | 位置 | 觸發條件 | 行為 |
| :--- | :--- | :--- | :--- |
| 第一層（Model 內建） | `OpenAICompatibleInterface` | 429 限流／5xx／逾時／連線異常（可重試） | tenacity 指數退避重試，最多 `evaluation.retry.max_retries`（預設 3）次，第 n 次重試等待 `backoff_factor^(n-1)` 秒（2.0 → 1s／2s／4s）；重試耗盡或不可重試錯誤（401／400／404 等）一律內部 catch 並回傳 `"ERROR: <ExceptionType>: <message>"` —— **單次推論不拋出未捕獲例外** |
| 第二層（Evaluator Safe Guard） | `src/evaluator.py` | 輸出為 `ERROR:`／空白／無法解析為 A–D | Tier 3 Safe Guard 判定 INVALID，`is_correct = False`，分母固定為 Total Evaluated Samples，管線不中斷 |

`predict_batch` 末梢防禦：`predict` 拋出的任何意外例外（含新驅動遺漏第一層防護之情況）
一律轉為 `ERROR:` 錯誤紀錄，確保 Runner 輸出結構恒穩。`MockModelInterface` 本質離線、永不拋出例外。

### 2.4 API Key 安全性與端點解析
- 金鑰一律經 `os.getenv(...)` 讀取，禁止寫死於程式碼或設定檔；
  可依模型以 `models[].api_key_env_var` 覆寫環境變數名稱。
- 預設環境變數：Groq → `GROQ_API_KEY`；OpenRouter → `OPENROUTER_API_KEY`；
  Ollama（本地）→ `OLLAMA_API_KEY`（選填）；通用 OpenAI 相容 → `OPENAI_API_KEY`。
- 本地端點（如 Ollama `http://localhost:11434/v1`）缺金鑰環境變數時允許預設佔位字串
  `"ollama"`；雲端 provider 缺金鑰時於建構期 fail-fast（`ValueError`）。
- OpenAI SDK 內建重試關閉（`max_retries=0`），tenacity 為唯一重試層，避免隱藏性重試倍乘。

### 2.5 循序執行流程（對齊 Pipeline 生命週期）
1. 依 `models[]` 清單逐一模型執行（`for model in models`），防止 GPU OOM 與限流。
2. 模型內部 `predict_batch` 逐題循序（`evaluation.batch_size: 1`，逐題量測延遲）。
3. 每模型結束後釋放 VRAM／資源（`torch.cuda.empty_cache()`）再載入下一模型。

## 3. 指標計算與分母定案規範
- **分母絕對約束**：所有準確率指標之分母均固定為總評測樣本數（`Total Evaluated Samples`），嚴禁動態排除無效輸出或解析失敗樣本。
- **異常輸出判定**：凡模型輸出為空、超時、拒答、或正則無法解析為 A–D 者，一律判定為無效並視同答錯（`is_correct = False`）。
- **診斷指標輔助**：系統額外提供 `valid_accuracy = correct / valid` 與 `invalid_parsing_rate = invalid / total` 作為模型格式遵循能力之輔助診斷。