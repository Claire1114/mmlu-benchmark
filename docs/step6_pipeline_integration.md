# Step 6: Pipeline 整合與測試（Phase B - 測試修復與審查）

## 1. Step Objective
- **Task**: 修復 Phase A 交付的根目錄 CLI 進入點 `main.py` 與 `tests/test_main.py`（44 測試）之間的全部 27 個失敗；完成敵對性審計（敵對審計：Schema 吻合、日誌降級、I/O 容錯、單模／單題雙層隔離、即時落盤），確認管線生命週期對齊 .clinerules §2 後產出本里程碑文檔。
- **AI Tool**: Cline (VS Code AI coding agent)
- **Collaboration Pattern**: Plan (Analysis & Refactor Proposal) -> Act (Implementation & Auto-Verification)
- **AI Efficiency Gains**: 單行根因定位（27 個失敗收斂為 1 個 `NameError`＋1 處格式偏離，而非 27 處散落修補）；跨模組 Schema 交叉比對（`main.py` ↔ `tests/test_main.py` ↔ `src/evaluator.py` ↔ `src/runner.py`）自動完成；全離線 290 測試回歸 3.7s 內閉環驗證。

## 2. Issues Identified (AI Review)

| # | Issue Identified | Risk / Potential Defect |
| :--- | :--- | :--- |
| 1 | `main.py:978` 於 `run_pipeline` 的 `with journal_file:` 區塊內呼叫 `execute_model(..., journal=journal)`，但區域變數名為 `journal_file`（line 955 開啟），`journal` 從未定義 | **致命**：每個走到模型迴圈的執行皆拋 `NameError: name 'journal' is not defined`；直接導致 27/44 測試全數失敗（含 Artifact、Throttling、FaultIsolation、ImmediatePersistence、OutputFormats、IOFaultHandling 全部類別），端到端管線完全不可執行 |
| 2 | `main.py:957` 的 `LOGGER.error(...)` 多行換行不符合 ruff format 風格（Phase A 交付時門禁僅覆蓋 `src/`、`tests/`，未納入根目錄 `main.py`，故未被攔截） | 格式不淨；若後續門禁納入 `main.py` 會持續報紅，且暗示門禁覆蓋範圍與檔案邊界不一致 |

### 審計後確認無需變更之項目（Before/After 決策依據）
- **Schema 吻合**：JSONL 8 欄（`model/question_id/subject/category/prompt/target_letter/raw_output/latency`）、CSV 核心欄序＋領域／科目動態欄、`metadata` 欄位、`models[]` 條目五鍵、`evaluator.to_dict()` 指標鍵集，與 `tests/test_main.py` 斷言逐項一致——**不修改測試、不修改 `src/evaluator.py`**。
- **`skipped_rows` 語意**：`MMLUDatasetLoader.skipped_rows` 於每次 `load_data` 內**重置**（`dataset_loader.py:183` 為賦值非累加），故 `_build_samples` 依科目 `+=` 累加為正確總計，無雙重計數。
- **`Evaluator.from_config` 健壯性**：文件契約「絕不拋出例外」（缺失／非字串／無效正則皆回退內建預設＋警告），`run_pipeline` 無需額外加 try/except。
- **`release_vram` 防禦性**：torch 未安裝／無 GPU／釋放失敗最多警告，絕不中斷管線，與測試斷言（每模型恰好一次、含失敗模型）吻合。
- **同秒雙跑**：`test_latest_alias_tracks_most_recent_run` 明示允許同秒覆寫（1–2 個時間戳記檔），為已文件化之設計取捨，非缺陷。

## 3. Refactoring & Implementation
- **Core Modifications**:
  1. `main.py:978`：`journal=journal` → `journal=journal_file`（將 `open_journal` 開啟之 `IO[str]` 物件正確交接至 `execute_model`，恢復逐題 `write()`＋`flush()` 即時落盤路徑）。
  2. `main.py:957`：`ruff format` 收合 `LOGGER.error("Failed to open predictions journal %s: %s", predictions_path, exc)` 為單行。
- **Edge Cases Handled**: 本次未新增邊界；既有 44 測試已覆蓋並全部通過——I/O 故障（日誌檔建立失敗降級 terminal-only、journal 開檔失敗 exit 1、磁碟寫入中途崩潰保留已 flush 紀錄、指標寫入失敗 exit 1）、無效 log 層級回落 INFO、單模建構／評估階段隔離、單題三鍵契約違規／例外／非有限 latency 哨兵、條件節流（本機 0.0、API 題間節流、末題不睡）、VRAM 每模釋放。
- **Key Design Decisions**: 採取「最小外科式修復」——根因為單一變數綁定錯誤，未重構 `run_pipeline` 結構；測試端 Schema 斷言審計為真後，**不改任何測試**（遵守「不為求綠而削弱斷言」原則，27 個失敗全數因實作缺陷而非測試錯誤）。

## 4. Verification & Before/After Comparison
- **Pytest Output Summary**: `tests/test_main.py` 44 passed（4.58s）；全數套件 `tests/` 290 passed（3.68s），零失敗零跳過。Coverage 狀態：**not measured**（未設定 pytest-cov）。
- **Ruff & Mypy Output Status**: `ruff check src/ tests/` → All checks passed!；`ruff format --check src/ tests/ main.py` → 17 files already formatted；`mypy src/ --strict` → Success: no issues found in 10 source files。

### Before / After Comparison Matrix

| Dimension | Before Refactor | After Refactor |
| :--- | :--- | :--- |
| Architecture / Design | 管線結構正確（One-time Loading → Sequential Model Loop → Step per Model → Aggregation），但 journal 物件未進入 `execute_model`，模型迴圈為死路 | 結構不變；journal 正確交接，`with journal_file:` 上下文管理即時落盤與關閉行為生效 |
| Robustness / Edge Cases| 任何端到端執行皆於 `execute_model(journal=journal)` 拋 `NameError`（exit 未定義、無產出） | 故障隔離／即時落盤／日誌降級／exit code 語意（0/1/2）全數依測試驗證生效；magical I/O 故障皆不遺失已寫入資料 |
| Test Coverage | `tests/test_main.py` 27 failed / 17 passed；全數套件 263 passed / 27 failed | `tests/test_main.py` 44 passed；全數套件 290 passed（無新增測試，僅修復實使既有 27 個邊界測試實際跑到） |
| Lint / Format / Types | ruff check 通過但 `main.py` 未納入 format 門禁（`Would reformat`） | `main.py` 納入 `ruff format --check` 全綠；mypy strict 全綠 |

## 5. Verification Artifacts & Logs
- **Terminal Output Log**:
  ```text
  === [1] ruff check src/ tests/ ===
  All checks passed!

  === [2] ruff format --check src/ tests/ main.py ===
  17 files already formatted

  === [3] mypy src/ --strict ===
  Success: no issues found in 10 source files

  === [4] pytest tests/test_main.py (44 items) ===
  tests/test_main.py::TestIOFaultHandling::test_invalid_log_level_falls_back_to_info
  -------------------------------- live log call ---------------------------------
  WARNING  main:main.py:387 Unknown logging level 'BOGUS'; falling back to INFO.
  PASSED                                                                   [100%]

  ============================== 44 passed in 4.58s =============================

  === [5] pytest tests/ full suite ===
  INFO     src.runner:runner.py:189 Model stub-model inference complete: samples=2 accuracy=0.5000 avg_latency=0.5000s total_wall=0.00s
  PASSED                                                                   [100%]

  ============================= 290 passed in 3.68s =============================
  ```
