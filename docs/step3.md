# Step 3: Evaluator Answer Parsing & Hierarchical Metrics Module

## 1. Step Objective
- **Task**: 實作 MMLU Benchmark Pipeline 之 Evaluator 答案解析模組，建立具備三層防禦性容錯架構（Tier 1 Strict、Tier 2 Fallback、Tier 3 Safe Guard）的提取邏輯，並依據核准之全分母鐵律（Total Evaluated Samples）精確計算階層化指標（整體準確率、4 大領域、8 科目、選項 Recall 及 RStd），確保模組為純計算、無網路相依且永不崩潰。
- **AI Tool**: Cline (VS Code AI coding agent)
- **Collaboration Pattern**: Plan (Analysis & Refactor Proposal) -> Act (Implementation & Auto-Verification)
- **AI Efficiency Gains**: 自動化生成 79 筆極端邊界測試案例，閉環驗證覆蓋率 100%，以 0.11 秒完成極速回歸，靜態型別與格式化達到 0 警告、0 錯誤交付。

## 2. Issues Identified (AI Review)
| # | Issue Identified | Risk / Potential Defect |
| :--- | :--- | :--- |
| 1 | 正則表達式缺乏單詞邊界守衛（`\b`） | 原正則會誤將包含字母的單詞拆解擷取（例如 `"The answer is B"` 誤取單詞中的 `a`、`"I don't know"` 誤取 `d`），導致 Tier 2/Tier 3 機制失效，系統性污染準確率。 |
| 2 | Accuracy 計算分母定義歧義（`valid` vs `total`） | 若動態將解析失敗、超時或拒答的樣本排除於分母之外，將使輸出格式嚴重崩潰的模型獲得虛高分數，破壞 Benchmark 的公正性。 |
| 3 | Option Recall 標準差（RStd）未明確統計母體定義 | 在模型僅預測部分選項（如覆蓋字母 < 2）或全猜單一選項時，若使用樣本標準差除以 $N-1$ 會引發除以零或統計偏差。 |
| 4 | `src/dataset_loader.py` 與全庫環境存在靜態型別 stub 缺失與多餘 ignore | `mypy src/ --strict` 檢測到 `types-PyYAML` 缺失與未觸發的 `unused "type: ignore"`，破壞 CI/CD 嚴格型別門禁。 |

## 3. Refactoring & Implementation
- **Core Modifications**:
  - `configs/eval_config.yaml`：更新 `answer_regex` 為 `(?i)\bthe correct answer is\s*\(?\b([A-D])\b\)?`，落實字邊界守衛並標註三層分工說明。
  - `src/evaluator.py`：新增 681 行純計算模組，包含 `extract_answer()` 三層解析與 `evaluate()` 階層化聚合指標計算。
  - `docs/step3_4_integration_spec.md`：補充第 3 節條文，定案分母固定為總評測樣本數（Total Evaluated Samples）。
  - `src/dataset_loader.py`：精確配置 `# type: ignore[import-untyped]` 並移除多餘的 ignore 註解。
- **Edge Cases Handled**:
  - 支援無效輸入防禦：處理 `None`、非字串型別、空白字串、混亂標點符號及多選項同時出現情境，安全回傳 `None` 不拋出例外。
  - 支援異常指標處理：單一選項覆蓋時 RStd 安全降級回傳 `0.0`；缺失或負數 latency 自動發出警告並修正為 `0.0`。
  - Join 防護機制：雙邊強制 `str(question_id)` 轉換，重複預測自動取第一筆，孤兒預測安全排除，缺失預測計為 INVALID。
- **Key Design Decisions**:
  - 分母絕對約束：主指標 `overall_accuracy = correct / total`，未成功解析者直接判定為錯誤（`is_correct = False`），另設 `valid_accuracy` 作為輔助診斷。
  - 偏差檢驗：一律採用總體標準差（`statistics.pstdev`）計算選項召回率標準差，標準化反應模型對特定選項的偏好度。

## 4. Verification & Before/After Comparison
- **Pytest Output Summary**: 113 passed in 0.11s（包含 Step 1 回歸測試 34 筆，Step 3 Evaluator 新增測試 79 筆；測試涵蓋率：not measured）。
- **Ruff & Mypy Output Status**: Ruff 檢查與格式化全數通過（0 error, 0 warning）；Mypy 4 個核心原始碼檔案 strict 模式全數通過（Success: no issues found）。

### Before / After Comparison Matrix
| Dimension | Before Refactor | After Refactor |
| :--- | :--- | :--- |
| Architecture / Design | 缺乏獨立答案解析模組，正則無容錯層級，無標準指標聚合結構 | 建立解耦之三層解析架構（Tier 1 Strict -> Tier 2 Fallback -> Tier 3 Safe Guard），具備獨立資料類別與序列化支援 |
| Robustness / Edge Cases | 正則會誤抓單詞內部字母；分母定義不明；對空值或異常資料無兜底防護 | 字邊界守衛防誤抓；分母固定為 Total；外層全面防禦例外，完全保證純計算無中斷 |
| Test Coverage | 34 passed（僅 Step 1 DataLoader 單元測試） | 113 passed（新增 79 筆全離線邊界與指標運算測試，耗時 0.11s） |

## 5. Verification Artifacts & Logs
- **Terminal Output Log**:
  ```text
  $ ruff check src/ tests/
  All checks passed!

  $ ruff format --check src/evaluator.py tests/test_evaluator.py
  2 files already formatted

  $ pytest tests/ -o timeout=10
  ============================= 113 passed in 0.11s =============================

  $ mypy src/ --strict
  Success: no issues found in 4 source files
