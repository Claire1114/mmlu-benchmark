# 🤖 人機協同與工程治理歷程紀錄

> MMLU Benchmark Pipeline（`mmlu-benchmark`）端到端治理日誌。全文採雙層結構：**全局協同架構僅定義一次**；各模組以標準化 Step 日誌追加。

---

## 第一部分：全局協同架構

## 1. 三方協同角色定義
- **Primary Agent (Cursor Composer / Cline)**：實作端，負責依規範產出程式碼、Type Hints 與初版測試。
- **Adversarial QA / Reviewer (獨立 LLM / 對抗審查)**：審查端，負責挑剔 Edge Cases、揪出型別漏洞與過度 Mock 假性覆蓋。
- **Human Engineer (主導工程師)**：決策與驗收端，負責技術決策拍板、本機終端機獨立驗收與 Git Commit 控管。

## 2. Human-in-the-Loop 研發閉環
```text
Prompt 指令下達
  → Agent 初稿實作
  → Reviewer 對抗審查與揪錯
  → 人工決策拍板（方案 A/B 評估）
  → 本機 Terminal 驗證（pytest / mypy）
  → Git Commit 固化版本

---

## 第二部分：模組治理紀錄

# Step 1: Dataset Loader 模組實作與防禦性重構

## 1. Step Objective
- 實作 Hugging Face `cais/mmlu` 資料載入器（`src/dataset_loader.py`）。
- 建立標籤映射機制（`0–3 → A–D`）與多選一 Prompt 組裝。
- 透過 `unittest.mock` 建立完全隔離外部網路與權重的單元測試（`tests/test_dataset_loader.py`）。

## 2. Issues Identified (AI Review)
- **標籤型別相容性漏洞**：浮點數截斷（`1.7 → 1` 誤判為 `B`）、布林型別污染（`True → 1` 誤判）、大小寫未正規化（`"a"` 需轉大寫）。
- **選項品質與邊界異常**：選項長度不等於 4，或選項內容包含 `None`（字串化變成 `"None"` 導致模型誤判）。
- **Question ID 碰撞風險**：多 split 合併或過濾取樣時，若純依 `index` 易產生鍵值碰撞。
- **資料集 Split 靜默錯置**：`DatasetDict` 缺失指定 split 時若靜默使用第一筆，易造成 train/val/test 洩漏。
- **測試套件假性覆蓋**：純函式過度使用 Mock，缺乏真實資料雜訊與邊界輸入的嚴格測試。

## 3. Refactoring & Implementation
- **嚴格型別守衛（Type Guard）**：拒絕 bool 與非整數 float；加入 `.strip().upper()` 正規化，確保 Ground Truth 正確性。
- **異常選項嚴格拒絕（Reject Policy）**：捨棄暴力填充，遇到異常列直接 Skip 並累計 `skipped_rows` 與輸出 WARNING。
- **穩定複合 ID 機制**：實作 `{subject}__{index:06d}__{sha1[:8]}` 複合鍵，保證冪等性與抗碰撞能力。
- **Split 存在性強制檢驗**：嚴格比對 split 名稱，若不存在立即拋出 `ValueError`。
- **測試解耦與邊界補強**：解耦純函式與整合測試，補足 6 支真實異常情境測試案例。

## 4. Verification & Before/After Comparison
- **Pytest Output Summary**: 11 passed in 0.84s（100% 通過，完全離線零外部呼叫）。
- **Mypy Output Status**: 型別提示完整，通過靜態檢查。
- **Key Improvements**:
  - **資料強健性**：杜絕靜默錯標與資料洩漏。
  - **防禦攔截可觀測**：成功觸發 `WARNING` 追蹤被跳過的異常資料列，流程不中斷。
  - **執行速度**：從初版冷啟動 28.15s 大幅優化至 0.84s 極速回歸。

## 5. Verification Artifacts & Logs
- **Terminal Output Log**:
  ```text
  ============================= test session starts ==============================
  collected 11 items

  tests/test_dataset_loader.py ...........                                 [100%]
  WARNING:src.dataset_loader:Skipped 2/10 MMLU rows due to invalid format.
  ============================== 11 passed in 0.84s ==============================
