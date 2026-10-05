# Step 5: Sequential Inference Execution Module (Benchmark Runner)

> 命名註記：本檔依任務指示命名為 `docs/step5.md`；`.clinerules` §6 格式為 `docs/step<N>_<task_name>.md`，兩者取捨記錄於此（與 `step4_model_interface.md` 之命名註記一致）。

## 1. Step Objective
- **Task**: 實作 MMLU Benchmark Pipeline 之循序推論執行模組 `src/runner.py`（Step 5）：`BenchmarkRunner`（循序模型迴圈、逐題循序推論、`request_delay` 速率控制、fail-fast 模型層契約驗證、VRAM 釋放保證）與不可變 `ModelRunOutput` 交接結構；`configs/eval_config.yaml` 純新增 `evaluation.request_delay: 0.0`；交付 49 筆全離線 mock 回歸測試（commit `e884acd`）；同時收錄前置 Hugging Face Chat Template 相容性修復與測試同步（commit `c274518`，197 passed）及 Phase B 對戰式審查。
- **AI Tool**: Cline (VS Code AI coding agent)
- **Collaboration Pattern**: Plan (Analysis & Refactor Proposal) -> Act (Implementation & Auto-Verification)
- **AI Efficiency Gains**: 一次生成 49 筆測試矩陣（9 個測試類、4 組參數化契約違切邊界），全量回歸約 5 秒閉環；自動審查鎖定 6 項邊界缺口（模型契約驗證、速率參數型別、資源釋放保證、並行禁令鎖定、靜態型別門禁），經 2 輪自校正達 ruff / mypy 零問題交付，全程零網路、零權重下載。

### 動機
Step 1/3/4 已分別完成「資料層（`dataset_loader`）、評估層（`evaluator`）、推論層（`models`）」，但尚無編排層將 `.clinerules` §2 的單一方向管線生命週期落實為強制性行為：樣本重用、循序執行防 OOM、逐題速率控制、逐模型資源釋放與評估對接全數停留在「約定」狀態。Runner 層將這些不變量從「約定」升級為「程式碼 + 測試鎖定」。

### Pipeline Execution Lifecycle (Sequential Execution Flow)
| # | 階段 | 實作與不變量 |
| :--- | :--- | :--- |
| 1 | One-time Loading | 呼叫端以 `MMLUDatasetLoader.get_samples()` 一次性完成 8 子集之載入、正規化與抽樣；Runner 僅消費預載樣本、不重讀資料，同一份樣本跨模型全程重用 |
| 2 | Sequential Model Loop | `run_benchmark()` 依 `models[]` 設定順序迭代（`for model in models`）；嚴禁 async／多執行緒／多處理，防 VRAM OOM 與雲端限流放大；以原始碼檢查測試鎖定並行原始字串不存在 |
| 3 | Step per Model | `build_model_interface()` 載入模型 → 逐題 `model.predict_batch([sample])`（batch=1）＋三鍵契約驗證 → `release_vram()`（try/finally 保證，含異常中断路徑）→ `evaluator.evaluate()` 取得 `EvaluationResult` 對接 |
| 4 | Aggregation（預留） | `run_benchmark()` 回傳 `List[ModelRunOutput]` 予呼叫端；總結表、比較產物與持久化屬 Step 6 之後之職責，本步不預設實現 |

## 2. Issues Identified (AI Review)
| # | Issue Identified | Risk / Potential Defect |
| :--- | :--- | :--- |
| 1 | HF 驅動對 Instruct 模型（SmolLM2-1.7B-Instruct）直接餵 raw prompt、未套用 chat template，且以 `return_full_text=True` 對 raw prompt 前綴裁切輸出（前置缺陷，commit `c274518` 修復） | 無模板框定時 Instruct 模型生成漂移或不產出「The correct answer is (X)」格式；裁切區域以 raw prompt 計算、與實際 prepend 之模板前綴不符 → 誤裁錯區／空輸出 → Tier-1 哨兵誤發或系統性解析失敗，不公正拉低準確率 |
| 2 | Runner 層未驗證模型層契約（`predict_batch` 紀錄數、三鍵完整性、`latency` 有限性） | Evaluator join 依序對齊 `str(question_id)`；若模型層回傳 0/2 筆或缺鍵，缺失預測將被靜默計為 INVALID，模型層缺陷被誤報為「模型未作答」，破壞評測報告可信度（分母鐵律下錯誤歸因） |
| 3 | `request_delay` 無建構期驗證（bool／NaN／±Inf／負值／非數值） | Python 中 `True` 為合法 int 會被當作 1 秒延遲；`float("inf")` 令 `time.sleep` 永久掛起、管線凍結；NaN 使 `> 0.0` 比較失效、速率控制行為未定義——靜默失效雲端限流設計 |
| 4 | VRAM 釋放在異常路徑（逐題迴圈例外中斷）未保證 | 例外中斷時 `torch.cuda.empty_cache()` 不被呼叫，下一個模型載入可能於迴圈中途 OOM，違反「Load Model → Predict → Free VRAM → Evaluate」生命週期條目 |
| 5 | 「單執行緒循序」契約無回歸鎖定 | 無測試鎖定時，後續重構可能混入 asyncio/threading/executor，破壞「循序推論防 OOM」與 GPU 排程假設；該缺陷僅於真實 GPU 負載下暴露，事後極難偵測 |
| 6 | `tqdm` 未提供型別 stub，`mypy src/ --strict` 報 missing stubs | 破壞靜態型別門禁（strict 0 錯誤交付承諾），迫使團隊在 strict 門禁上開例外孔洞 |

**審查後確認非問題（No-Action）項**：
- `evaluation.max_workers: 4` 僅適用雲端 API 模型（YAML 註記已明）；Runner 依設計嚴格循序且不透傳 `max_workers`——屬驅動類型職責分層，非缺陷。
- Runner 不讀取 `target_letter`（防答案鍵滲漏進推論路徑）；答案解析與指標計算全數委託 `Evaluator`——經原始碼審查確認無滲漏路徑。
- 單題 `latency` 以模型層為唯一量測來源，Runner 不重複量測（避免雙重量測偏差）——契約於 docstring 明列並以測試鎖定。

## 3. Refactoring & Implementation
- **Core Modifications**:
  - `src/models/huggingface.py`（前置修復，commit `c274518`）：新增 `_format_prompt()`——pipeline tokenizer 具備 `chat_template` 時以 `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` 動態套用對話模板；套用失敗、結果非字串或為空白時記錄警告並回退原始 prompt。`_generate()` 改以 `return_full_text=False` 請求，並對意外殘留前綴以「套用模板後之 `formatted_prompt`」裁切；`predict()` 對非 str prompt 加 `TypeError` fail-fast 型別守衛。
  - `src/runner.py`（新增，293 行）：`BenchmarkRunner`——建構期 fail-fast（`evaluator` 型別 + `request_delay` 有限非負數值）；`run_model()` 逐題 `predict_batch([sample])` ＋三鍵契約驗證（`PREDICTION_RECORD_KEYS = {question_id, raw_output, latency}`）＋`latency` 有限性驗證；`run_benchmark()` 循序模型迴圈（建構失敗原樣上拋即中止、不跳過）；`release_vram()` 防禦性 CUDA 快取釋放（torch 未安裝／無 CUDA／釋放失敗三層兜底，絕不中斷管線）；`ModelRunOutput` frozen dataclass 交接。
  - `configs/eval_config.yaml`：純新增 `evaluation.request_delay: 0.0` 並附註記（本機 HF 模型維持 0.0，雲端模型可設小正數避免 429）；執行參數維持「YAML 單一來源」，程式碼零硬編碼。
  - `tests/test_runner.py`（新增，497 行）：49 筆全離線案例——`StubModel`（`BaseModelInterface` 子類，記錄呼叫順序並模擬 4 類契約違規）、真實 `Evaluator` 實體（純計算、非外部端點）、`patch("src.runner.time.sleep")` 零真實延遲、`patch("torch.cuda.empty_cache")` 無真實 GPU、`TQDM_DISABLE=1` 鎖定非 TTY 行為。
- **Edge Cases Handled**:
  - `request_delay`：`True`／`"0.5"`／`float("nan")`／`±inf`／負值／非數值 → 建構期 `ValueError`（`bool ⊂ int`，先排除 bool）；`0.0` 完全不呼叫 `time.sleep`；單一樣本不休眠；最後一題之後不休眠（4 題 → 精確 3 次 sleep，以呼叫次數＋事件序列鎖定）。
  - 契約違規（參數化）：`predict_batch` 回傳 0／2 筆紀錄；缺 `question_id`／`raw_output`／`latency` ×3；`latency` = `"fast"`／`NaN`／`True`／`1.5e400`(=inf) ×4 → 一律 `RuntimeError` fail-fast。
  - 空輸入：0 樣本 → 空預測＋`total=0` 評估、零 sleep；0 模型設定 → 回傳 `[]`、工廠零呼叫。
  - 模型建構失敗（3 個模型之第 2 個）→ 原始例外上拋即中止，第 3 個不被嘗試（fail-fast、防評測報告靜默缺漏模型）。
  - VRAM 異常路徑：torch 未安裝 → debug 日誌；CUDA 不可用 → 跳過；`empty_cache` 拋 `RuntimeError` → warning（含 `exc_info`）、管線不中斷；try/finally 保證例外中断路徑仍被呼叫。
  - 進度條：`TQDM_DISABLE=1`（非 TTY／CI）正常執行；`mininterval=0.5` 降低日誌雜訊。
  - 欄位對齊：`question_id` 非 str（int）原樣透傳、Evaluator 雙邊 `str()` 對齊不受影響（`total` 不變）；輸入樣本不被修改（深拷貝比較鎖定）。
- **Key Design Decisions**:
  - 逐題推論以 `predict_batch([sample])`（batch=1）而非整批傳入模型層：對齊 YAML `evaluation.batch_size: 1`，取得逐題 latency 並縮小失敗重試範圍；循序執行仍由模型層 `predict_batch` 契約負責（Runner 是編排者、模型層是執行者，職責不重疊）。
  - fail-fast 與哨兵分層：模型層「執行期 API 異常」由 `ERROR:` 哨兵兜底（單題不中斷管線、Evaluator Tier-3 計 INVALID）；「契約違規」（紀錄結構／欄位型別）屬程式錯誤 → `RuntimeError` 立即中斷——兩類錯誤互不污染，與 Step 4 雙層防護一貫。
  - `release_vram()` 置於 `try/finally`（迴圈結束、評估前）：異常中断路徑亦釋放資源，但不吞掉例外、保持 fail-fast 語意。
  - `ModelRunOutput` 採 frozen dataclass：對下游持久化層（Step 6）不可變交接；`total_seconds` 僅計推論迴圈牆鐘（不含模型建構、VRAM 釋放與評估時間），跨模型吞吐比較公平。
  - Runner 不讀 `target_letter`、不執行答案解析：防答案鍵滲漏進推論路徑，維持「資料／編排／推論／評估」職責分層（對齊 step3_4 整合規格）。

### 三段式資料流對齊表（dataset_loader -> runner -> model -> evaluator）
| 階段 | 模組／入口 | 輸入 | 輸出 | 職責與不變量 |
| :--- | :--- | :--- | :--- | :--- |
| ① 資料載入 | `src/dataset_loader.py` · `MMLUDatasetLoader.get_samples()` | YAML `dataset:` 設定（categories/modes/prompt_template） | `List[Dict]` 樣本紀錄（`question_id`、`formatted_prompt`、`subject`、`category`、`target_letter`） | 一次性載入＋正規化＋抽樣；唯一資料來源，下游不重讀 |
| ② 編排（本步） | `src/runner.py` · `BenchmarkRunner.run_benchmark()` / `run_model()` | 預載樣本＋`models[]` 設定＋`evaluation` 設定 | `List[ModelRunOutput]`（predictions＋evaluation＋total_seconds） | 循序模型迴圈；樣本重用；速率控制；契約驗證；VRAM 釋放；零並行原始 |
| ③ 推論 | `src/models/*` · `BaseModelInterface.predict_batch([sample])`（工廠 `build_model_interface`） | 單一樣本紀錄（batch=1；模型層不使用 `target_letter`） | `[{question_id, raw_output, latency}]` 三鍵紀錄 | latency 唯一量測源；`ERROR:` 哨兵第一層防禦；建構期 fail-fast |
| ④ 評估 | `src/evaluator.py` · `Evaluator.evaluate(samples, predictions, model_name)` | 原樣本＋三鍵預測紀錄（雙邊 `str(question_id)` join） | `EvaluationResult`（total/valid/correct/invalid、整體＋4 領域＋8 科目準確率、延遲、選項 recall/RStd） | 純計算、無網路；三層答案解析；分母固定＝總評測樣本數 |

## 4. Verification & Before/After Comparison
- **Pytest Output Summary**: 246 passed in 5.43s（`pytest tests/ -v -o timeout=10`；Step 5 Runner 層 49 筆 ＋ Step 4 模型層（含 HF chat template 修復後同步之 197 筆基線）＋ Step 1 資料層 ＋ Step 3 Evaluator；測試覆蓋率：not measured（未配置 coverage 工具））。
- **Ruff & Mypy Output Status**: `ruff check` 0 違規；`ruff format --check` 15 檔全部格式化通過；`mypy src/ --strict` 10 個原始碼檔案 0 錯誤（含本步新增 `src/runner.py`）。

### Before / After Comparison Matrix
| Dimension | Before Refactor | After Refactor |
| :--- | :--- | :--- |
| Architecture / Design | 無編排層：推論呼叫、速率控制、資源釋放、評估對接全數散落於呼叫端；HF 驅動缺 chat template（Instruct 模型輸出格式漂移、前綴誤裁） | `BenchmarkRunner` 落實四階段生命週期（One-time Loading → Sequential Model Loop → Step per Model → Aggregation 預留）；`ModelRunOutput` 不可變交接；`request_delay` 設定驅動；HF 驅動動態模板＋回退＋正確裁切 |
| Robustness / Edge Cases | 模型契約無驗證（缺鍵/多筆→靜默 INVALID 污染）；`request_delay` 無驗證（bool/NaN/Inf 可致掛起或失效）；VRAM 異常路徑無保證；循序契約無鎖定 | 三鍵＋有限 latency `RuntimeError` fail-fast；建構期 `ValueError`（bool/NaN/±Inf/負值/非數值）；try/finally `release_vram()`＋三層例外兜底；原始碼檢查測試鎖定零並行原始 |
| Test Coverage | 197 passed（HF chat template 修復與測試同步後） | 246 passed（+49：建構 fail-fast、循序流轉、速率控制、latency 透傳、欄位對齊、邊界/空值、VRAM 釋放、循序編排、進度日誌 9 類，含 4 組參數化邊界） |
| Static Gates | ruff / mypy strict 全綠（9 source files） | ruff 0 違規（15 files）；mypy strict 0 錯誤（10 source files；`tqdm` 匯入以 `# type: ignore[import-untyped]` 直屬封裝層抑噪） |

## 5. Verification Artifacts & Logs
- **Terminal Output Log**（Phase B 於 2026-10-05 本機重驗，commit `e884acd` 狀態）:
  ```text
  $ python -m pytest tests/ -v -o timeout=10
  ============================= 246 passed in 5.43s ==============================

  $ ruff check src/ tests/
  All checks passed!

  $ ruff format --check src/ tests/
  15 files already formatted

  $ mypy src/ --strict
  Success: no issues found in 10 source files
  ```

### AI Tool Usage Log
- **Prompt Strategy**:
  - **Phase A（Builder）**：以 `.clinerules` §2 管線生命週期為硬約束（One-time Loading、Sequential Model Loop、Step per Model、零並行原始），並附子句化驗收標準（sleep 次數＝n−1 且末題後零 sleep、delay=0 零 sleep 呼叫、三鍵契約違規→`RuntimeError`、latency 有限性驗證、VRAM 異常路徑釋放、空輸入回傳空結果）與「全離線 mock、無網路、無權重下載」約束——AI 一次生成 `runner.py`（293 行）＋49 筆測試矩陣，邊界案例（bool⊂int、NaN/Inf、空序列、建構失敗中止）由 AI 主動枚舉，免去人工補齊。
  - **Phase B（Reviewer）**：Prompt 明確切換角色為 Principal Architect，禁止假定 Phase A 決策正確；要求逐條對照驗收子句做對戰式審查、產出 §6 強制五段式里程碑文件，並給出精確 `git add`/`git commit` 建議（不代為 commit）。審查聚焦：契約驗證完整性、型別門禁、資源釋放保證、並行禁令鎖定。
- **2 輪自校正歷程**:
  1. **Round 1（ruff format 空白修正）**：`ruff format --check` 對 `src/runner.py` 與 `tests/test_runner.py` 報行寬/空白不符 → 以 `ruff format` 自動重排（零邏輯變更），複驗格式門綠燈。
  2. **Round 2（浮點邊界 + 靜態型別門）**：修正 `latency` 有限性驗證之浮點邊界測試與 `math.isfinite` 實作契約的斷言不符（覆蓋 `NaN`、`True`、`1.5e400`(=inf)）；同輪 `mypy --strict` 對 `tqdm` 匯入報 missing stubs → 於匯入行加 `# type: ignore[import-untyped]`（與 `huggingface.py` 之 `transformers.pipelines.Pipeline` 處理先例一致，抑噪範圍限直屬封裝層）。全程未弱化斷言、未刪減測試案例。
- **自動執行指令細節**（每輪修正後閉環，依序執行）:
  | # | 指令 | 結果 |
  | :--- | :--- | :--- |
  | 1 | `python -m pytest tests/ -v -o timeout=10` | 246 passed in 5.43s（全離線：StubModel＋patch `time.sleep`＋patch `torch.cuda`＋`TQDM_DISABLE=1`，無網路、無權重下載） |
  | 2 | `ruff check src/ tests/` | All checks passed! |
  | 3 | `ruff format --check src/ tests/` | 15 files already formatted |
  | 4 | `mypy src/ --strict` | Success: no issues found in 10 source files |
- **本步相關 Commit 紀錄**:
  | Commit | Type | 內容 |
  | :--- | :--- | :--- |
  | `c274518` | `fix(models)` | chat-template-safe HF driver and type-guard `_format_prompt`（測試同步後 197 passed） |
  | `e884acd` | `feat(runner)` | implement sequential BenchmarkRunner with offline tests（3 files changed, 795 insertions） |
