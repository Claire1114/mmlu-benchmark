# Step 7: Few-Shot In-Context Learning Extension & CLI Override

## 1. Step Objective
- **Task**: 擴充 MMLU Benchmark 管線以支援 Few-Shot In-Context Learning，允許透過命令列參數（`--shots`）動態覆寫，並嚴格遵循 `CLI > YAML > 0-shot (default)` 階層化覆寫原則，同時確保 0-shot 向後相容性與 Mock 單元測試穩定性。
- **AI Tool**: Cline (VS Code AI coding agent)
- **Collaboration Pattern**: Plan (Analysis & Refactor Proposal) -> Act (Implementation & Auto-Verification)
- **AI Efficiency Gains**: 藉由 AI 代理人即時偵測參數斷裂（CLI wiring gap）與 Mock 介面簽名不匹配，於 10 分鐘內完成全專案 293 項單元測試回歸與端到端推論驗證，省去手動排查 Mock 與 Prompt 拼接除錯的時間。

## 2. Issues Identified (AI Review)

| # | Issue Identified | Risk / Potential Defect |
| :--- | :--- | :--- |
| 1 | `main.py` 的 `run_pipeline` 實例化 DataLoader 時漏傳 `num_shots=options.shots` | CLI `--shots` 參數形同虛設，系統退回 YAML 預設值，永遠只能執行 0-shot。 |
| 2 | `tests/test_main.py` 的 `StubLoader.__init__` 未支援 `num_shots` 與 `**kwargs` | 導致 28 項依賴 StubLoader 的單元測試全面噴出 `TypeError: unexpected keyword argument 'num_shots'`。 |
| 3 | `format_prompt` 內部對 dev 範例套用包含指令的 `_prompt_template` 並重複拼接 Instruction | 造成 Few-Shot Prompt 中每個範例皆出現格式提示，且目標題目末端出現雙重 `Answer:` 指令，干擾模型生成與答案抽取。 |
| 4 | 產出的 `metrics_summary_*.json` 的 `metadata` 未追蹤本次執行的 shots 設定 | 缺乏實驗可追溯性，無法從報表直接辨識該次評測為 0-shot 或是特定 Few-shot。 |

## 3. Refactoring & Implementation
- **Core Modifications**:
  - `src/dataset_loader.py`: 
    - 擴充 `MMLUDatasetLoader.__init__` 接收 `num_shots: Optional[int] = None`，實作 `CLI > YAML > 0` 優先順序。
    - 新增 `_load_dev_exemplars` 自 `dev` split 確定性抽取標準範例（包含正確答案）。
    - 重構 `format_prompt`：dev 範例僅組裝問題、選項與標準答案；目標題目維持標準範本格式，杜絕重複指令。
  - `main.py`:
    - `parse_args` 註冊 `--shots` 參數，`CliOptions` 補齊 `shots` 欄位。
    - `run_pipeline` 注入 `num_shots=options.shots` 至 DataLoader。
    - `metadata` 字典補上 `"num_shots"` 與 `"shots_source"` 欄位以供追蹤。
  - `tests/test_main.py`:
    - 更新 `StubLoader.__init__` 與相關 Stub 類別，加入 `num_shots: Optional[int] = None` 與 `**kwargs: Any`。
- **Edge Cases Handled**:
  - `dev` split 可用題目小於指定 `k` 時，優雅降級為全數可用題目而不拋錯。
  - `--shots 0` 時維持乾淨的 Zero-Shot Prompt，完全相容於原始輸出格式。
  - CLI 未給予 `--shots` 時解析為 `None`，平順回退至 YAML 設定檔。
- **Key Design Decisions**:
  - **資料隔離（Data Leakage Prevention）**：嚴格限制 Few-Shot 範例僅能抽取自 `dev` split，禁止從 `test` split 取樣。
  - **隨機流可重現性**：利用 `(seed, subject)` 衍生專屬隨機流，確保同輸入下跨執行的範例選取完全一致。

## 4. Verification & Before/After Comparison
- **Pytest Output Summary**: 293 passed in 4.91s（全套測試全綠通過，涵蓋參數解析、容錯隔離、格式驗證與端到端流程）。
- **Ruff & Mypy Output Status**: Passed（無 linting 違規，型別宣告皆符合 static type 檢驗）。

### Before / After Comparison Matrix

| Dimension | Before Refactor | After Refactor |
| :--- | :--- | :--- |
| Architecture / Design | 僅支援靜態 Zero-Shot Prompting，CLI 與 YAML 無法動態指定範例數 | 支援標準 In-Context Learning，階層化覆寫原則（`CLI > YAML > 0-shot`）完全落實 |
| Robustness / Edge Cases | 測試 Stub 簽名硬編碼導致新增參數時全面崩潰（28 failed）；Prompt 重複拼接雙重指令 | Stub 支援動態參數吸收；Prompt 組裝標準化，杜絕 Instruction 重複疊加 |
| Test Coverage | 265 passed（因 StubLoader 簽名衝突導致 28 項 main 測試失敗） | 293 passed in 4.91s（全數修復並保持 100% 通過率） |

## 5. Verification Artifacts & Logs
- **Terminal Output Log**:
  ```text
  ======================================== test session starts ========================================
  platform darwin -- Python 3.13.5, pytest-8.3.4, pluggy-1.5.0 -- /opt/anaconda3/bin/python
  cachedir: .pytest_cache
  rootdir: /Users/claireweng/mmlu-benchmark
  configfile: pytest.ini
  plugins: langsmith-0.4.29, dash-4.1.0, anyio-4.15.1
  collected 293 items                                                                                  

  tests/test_dataset_loader.py ................................................................ [ 21%]
  tests/test_evaluator.py ..................................................................... [ 45%]
  tests/test_main.py ............................................                               [ 60%]
  tests/test_models.py ........................................................................ [ 85%]
  tests/test_runner.py ...........................................                              [100%]

  ======================================== 293 passed in 4.91s ========================================
