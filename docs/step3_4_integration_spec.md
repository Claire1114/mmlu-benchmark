# Step 3／4 對接規格：Runner 與 Evaluator

本文定義 DataLoader 對 Runner／Evaluator 的最小對接契約，依據 `src/dataset_loader.py` 及 `configs/eval_config.yaml`。

## 1. Runner 呼叫方式與職責邊界

依循單一職責原則（Single Responsibility），**Runner 僅負責模型推論，不參與任何答案解析、正則過濾或指標計算**。

推薦全流程標準介面（以批次或列表方式呼叫，方便本地批次加速或雲端 API 併發）：

```python
from src.dataset_loader import MMLUDatasetLoader

# 1. 統一載入評測樣本
loader = MMLUDatasetLoader("configs/eval_config.yaml")
loader.resolve_sample_size()  # 驗證 active_mode 的抽樣筆數
samples = loader.get_samples()

# 2. 模型批次推論（批次大小或執行緒數依 configs/eval_config.yaml 控制）
# Runner 僅接收包含 formatted_prompt 的清單，不得接收 target_letter
predictions = runner.predict_batch(samples)

# predictions 輸出規格：List[Dict[str, Any]]
# 每筆固定包含：
# - "question_id": str (原樣保留對齊)
# - "raw_output": str (模型生成的未處理原始文字)
# - "latency": float (該題推理耗時，單位：秒)