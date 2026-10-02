# 🤖 人機協同與工程治理歷程紀錄

> MMLU Benchmark Pipeline（`mmlu-benchmark`）端到端治理日誌。全文採雙層結構：**全局協同架構僅定義一次**；各模組以標準化 Step 日誌追加。

---

## 第一部分：全局協同架構

### 三方角色

| 角色 | 定位 | 核心職責 |
| :--- | :--- | :--- |
| **Primary Agent**（Cursor Composer） | 實作端 | 依 `.cursorrules` 產出功能碼、Type Hints、繁中 Google Style Docstrings、初版單元測試 |
| **Adversarial QA / Reviewer**（獨立 LLM） | 對抗審查端 | 挖掘 Edge Cases、暴露 Self-Verification Bias、挑戰防禦性邏輯 |
| **Human Engineer**（主導工程師） | 決策與驗收端 | 架構拍板、本機 Terminal 獨立驗收、Git Commit 版本控管 |

### Human-in-the-Loop 閉環

```text
Prompt 指令下達
  → Agent 初稿
  → Reviewer 審查挑戰
  → 人工決策與重構
  → Terminal 本機驗收
  → Git Commit 提交
```

---

## 第二部分：模組治理紀錄

### Step 2: Dataset Loader 模組實作與防禦性重構

#### (A) 初始建構與協作流程

| 項目 | 內容 |
| :--- | :--- |
| **目標檔案** | `src/dataset_loader.py`、`tests/test_dataset_loader.py` |
| **核心職責** | Hugging Face `cais/mmlu` 載入；標籤映射 `0–3 → A–D`；多選一 Prompt 組裝 |
| **初版手段** | `datasets.load_dataset` + `unittest.mock.patch` 離線隔離 |
| **初版耗時** | 冷啟動 **28.15s** → 二次 **0.73s** |
| **已知缺口** | 缺真實 HF 雜訊資料之強健防守（標籤型別、選項結構、split 錯置） |

#### (B) 審查盲點、重構與驗收成效

**Reviewer 盲點 → 重構對照**

| 發現盲點 / 潛在缺陷 | 隱患情境（HF Real-World Edge Case） | 具體修改與防禦性重構寫法 | 改善成效與防禦等級 |
| :--- | :--- | :--- | :--- |
| **標籤相容性漏洞** | 浮點截斷（`1.7 → 1` 誤判 `B`）、布林污染（`True → 1`）、大小寫不一致（`"a"`） | 型別守衛拒 bool／非整數 float；`.strip().upper()` 正規化字母；僅接受整數語義 | 杜絕靜默錯標，Ground Truth 正確性可保證 |
| **選項資料品質** | 選項數 ≠ 4，或元素含 `None`（變字面 `"None"`） | 捨棄填充；採嚴格 **Reject**：異常列略過並累計 `skipped_rows` + WARNING | 消除 Evaluation Bias，避免模型猜測空選項 |
| **Question ID 碰撞** | 跨 split／過濾合併時純依 `index` 易鍵值碰撞 | 複合穩定鍵：`{subject}__{index:06d}__{sha1[:8]}` | 冪等、抗碰撞，利於結果關聯 |
| **資料集分割防護** | `DatasetDict` 缺指定 split 時靜默 `next(iter(...))` | 嚴格比對 split；不存在則拋 `ValueError` | 防評測集錯置與 train／val 洩漏 |
| **測試套件品質** | 純函式過度 Mock、職責混雜、同義反覆斷言 | 方案 A：解耦純函式／整合測試；擴充 6 支真實邊界測試 | 消除虛假覆蓋；**0.84s** 極速回歸、報錯訊號精準 |

**AI 對話與執行 Logs**

| 類型 | 摘要 |
| :--- | :--- |
| **決策拍板** | 方案 A（修正測試缺陷）；選項異常採嚴格 Reject（Safe Drop & Skip + 可追蹤 WARNING） |
| **驗收命令** | `pytest tests/test_dataset_loader.py -v` |
| **驗收結果** | `11 passed in 0.84s`（100% 通過） |
| **Live Warning** | 成功觸發 `WARNING src.dataset_loader ... Skipped X/Y MMLU rows` → 防禦攔截有效、流程不中斷 |
| **驗收截圖** | `![Step 2 Pytest 驗收截圖](images/step2_pytest_passed.png)` |

**相關 Git Commits**

- `feat(dataset): implement dataset_loader and unit tests with mock data`
- `refactor(dataset): apply defensive validation, stable question_id, and audit edge-case tests`

---

## 第三部分：後續步驟模板骨架

### Step 3: 統一模型推論驅動層 `[進行中 / 待追加]`

#### (A) 初始建構與協作流程

| 項目 | 內容 |
| :--- | :--- |
| **目標檔案** | _待填_ |
| **核心職責** | _待填_ |
| **初版手段** | _待填_ |
| **初版耗時** | _待填_ |
| **已知缺口** | _待填_ |

#### (B) 審查盲點、重構與驗收成效

| 發現盲點 / 潛在缺陷 | 隱患情境 | 具體修改與防禦性重構寫法 | 改善成效與防禦等級 |
| :--- | :--- | :--- | :--- |
| _待填_ | _待填_ | _待填_ | _待填_ |

**AI 對話與執行 Logs**

| 類型 | 摘要 |
| :--- | :--- |
| **決策拍板** | _待填_ |
| **驗收命令** | _待填_ |
| **驗收結果** | _待填_ |
| **Live Warning／觀測** | _待填_ |
| **驗收截圖** | `![Step 3 驗收截圖](images/step3_placeholder.png)` |

**相關 Git Commits**

- _待填_
