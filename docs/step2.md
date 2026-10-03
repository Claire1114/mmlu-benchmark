# Step 2：dataset_loader 設定檔改版重構（多科目 × 抽樣模式 × Prompt 解耦）

## 1. 任務與使用的 AI 工具
- **任務**：將 `src/dataset_loader.py` 對齊新版 `configs/eval_config.yaml`（4 領域 × 2 科目、active_mode 兩階段抽樣、Prompt 模板外掛、seed 可重現）
- **AI 工具**：Cline（VS Code AI coding agent，免費方案）
- **協作模式**：Plan（唯讀盤點 + 重構計畫）→ Act（實作 + pytest 自動驗證）；人工僅於 5 個決策點拍板（模板 fallback、抽樣順序、預設 seed、category 欄位、README 更新）

## 2. 核心重構項目摘要
- **結構解析**：`iter_subjects()` 依 YAML 順序扁平化 `dataset.categories.<領域>.subjects`；跨領域重名／結構異常 init 即拒絕
- **抽樣模式**：`resolve_sample_size()` 讀取 `modes[active_mode].sample_size_per_subject`（smoke_test=10／demo=30）；無效模式 fail-fast
- **可重現抽樣**：`random.Random(f"{seed}::{subject}")` 取代前 N 直取；`project.seed`（缺省 42）、科目隨機流解耦、保留原題序
- **Prompt 解耦**：`format_prompt()` 改讀 `dataset.prompt_template`；哨兵值預驗證佔位符，缺失回退 `DEFAULT_PROMPT_TEMPLATE`
- **樣本擴充**：`get_samples()` 遍歷 8 科目並新增 `category` 欄位（支援雙層級正確率）
- **連帶**：`cache_dir` 快取支援（`HF_DATASETS_CACHE` + 版本相容 kwargs）

## 3. 單元測試實作與覆蓋結果
- 全數 `unittest.mock` 隔離，零網路、零權重下載
- **結果：34 passed, 0 failed**（基線 11 → 34，新增 23 筆）
- 新增覆蓋：真實設定檔 8 科目迴歸、領域對應與順序、重名/非法結構拒絕、模式切換與未知模式、seed 回退與型別守衛、抽樣確定性（同 seed 重現／跨 seed 與跨科目獨立）、池不足全量保留、Prompt 自訂/缺失/壞佔位符、`get_samples` category 映射、cache_dir 環境變數
- 測試另揪出並修正 2 個實作隱患：YAML 寫檔 `sort_keys` 亂序、`active_mode` 延遲驗證（改為 init fail-fast）

## 4. 其他連帶更新
- `README.md`：多科目/mode 驅動抽樣描述、`dataset.sample_size` 改為 `modes[active_mode].sample_size_per_subject`、檔案樹補列 `dataset_loader.py`
- `configs/eval_config.yaml` **未修改**：8 科目名皆為 `cais/mmlu` 有效子集，與現行 `load_dataset` 介面相容

## 5. AI 賦能效益總結
- **閉環提速**：設定落差分析→計畫→重構→34 筆測試全綠→冒煙回歸，單次 Plan→Act 週期完成，人工介入僅限決策
- **測試自產自銷**：23 筆邊界測試由 AI 主動產出（確定性、fail-fast、髒資料），並以迴歸測試自我查缺補漏
- **規範落實**：Type Hints、Google Style Docstrings、模組職責隔離（loader 不碰模型 I/O 與指標計算）一次到位
