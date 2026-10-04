# Step 4: Model Inference Interface Layer (Model Interface)

## 1. Step Objective
- **Task**: 實作 MMLU Benchmark Pipeline 之模型推論介面層（`src/models/`，Step 4 第一階段），建立统一 `predict` / `predict_batch` 契約、雙層例外防護（Model 內建 tenacity 指數退避重試 + `ERROR:` 哨兵；Evaluator Tier-3 Safe Guard 兜底）、OpenAI 相容 API 驅動（Groq／Ollama／OpenRouter／通用端點）與離線 Mock 模型，並由設定工廠 `build_model_interface` 依 `configs/eval_config.yaml` 之 `models[]` 區塊建構；同時完成對戰式審查（Adversarial Audit）與治理檔案結構修正。
- **AI Tool**: Cline (VS Code AI coding agent)
- **Collaboration Pattern**: Plan (Analysis & Refactor Proposal) -> Act (Implementation & Auto-Verification)
- **AI Efficiency Gains**: 全離線 67 筆模型層測試（mock OpenAI client + 純記憶體例外實體）以約 1 秒完成回歸；自動審查定位 `.cursorrules` 結構重疊、建構期 fail-fast 缺口與 SDK 5xx 映射盲區，並以 6 筆新邊界測試鎖定行為；ruff／mypy 靜態門禁 0 警告交付。

### 實作盤點表
| 檔案 | 類型 | 職責與關鍵機制 |
| :--- | :--- | :--- |
| `src/models/base.py` | 新增 | `BaseModelInterface`（ABC）：抽象 `predict(prompt) -> str`；具體 `predict_batch()` 循序批次（不用執行緒池防 VRAM OOM），輸出固定三鍵 `question_id`/`raw_output`/`latency`；髒資料不呼叫模型、直接產出 `ERROR:` 錯誤紀錄；`predict` 意外例外由末梢防禦轉哨兵（`ERROR_PREFIX`） |
| `src/models/interfaces.py` | 新增 | `ModelProtocol`（`runtime_checkable` 結構化協議：`model_name` + `predict`）；`build_model_interface(model_cfg, evaluation_cfg)` 工廠（`SUPPORTED_TYPES`：mock/openai_compatible/groq/ollama/openrouter；`huggingface`/`gemini` 排程後期階段 → `NotImplementedError`）；防禦式配置 coercing（`_require_str`/`_optional_str`/`_coerce_int`/`_coerce_float`，非法值警告 + 回退預設） |
| `src/models/mock_model.py` | 新增 | `MockModelInterface`：`fixed`（預設 `"The correct answer is (A)"`）／`random`（seed 可重現 A–D）；零網路、零權重 |
| `src/models/openai_compatible.py` | 新增 | `OpenAICompatibleInterface`：`PROVIDER_PRESETS`（Groq/Ollama/OpenRouter/OpenAI 相容端點與金鑰策略）；金鑰僅經 `os.getenv`（雲端 fail-fast、Ollama 佔位字串）；tenacity 指數退避（`backoff_factor^(n-1)` 秒）重試 429/5xx/逾時/連線異常；不可重試或重試耗盡回傳 `ERROR: <Type>: <msg>`；SDK 層重試關閉（`max_retries=0`）確保重試層唯一 |
| `src/models/__init__.py` | 修改 | 統一匯出全部介面、協議、工廠與常數 |
| `tests/test_models.py` | 新增 | 67 筆全離線測試（本步新增 6 筆邊界／回歸鎖定）：mock OpenAI client（`MagicMock` + `SimpleNamespace`）、純記憶體 openai 例外（429/500/401/400/timeout/502/503/504）、patch tenacity sleep 零真實延遲、與 Evaluator 哨兵對接整合測試 |
| `requirements.txt` | 修改 | 新增 `openai>=1.0.0` 依賴 |
| `src/dataset_loader.py`、`tests/test_dataset_loader.py` | 修改 | 僅 ruff 行寬重排（無邏輯變更，維持 lint 門禁綠燈） |
| `docs/step3_4_integration_spec.md` | 修改 | 對接規格定案：Runner 契約（三鍵輸出、模型層禁讀 `target_letter`）、雙層例外防護表、分母鐵律 |
| `.clinerules` / `.cursorrules` / `.gitignore` | 修改 | 治理規則對齊（里程碑文件命名格式、agent 指引去重）、Jupyter 產物 ignore |

> 命名註記：本檔名 `step4_model_interface.md` 依任務指示命名；`.clinerules` §6 現行格式為 `docs/step<N>.md`，兩者取捨記錄於此。

## 2. Issues Identified (AI Review)
| # | Issue Identified | Risk / Potential Defect |
| :--- | :--- | :--- |
| 1 | `.cursorrules` 結構重疊：標題與 §1/§2 各出現兩次，§4/§5/§6 亦各重複一次（Phase A 混入的 governance 異動未受審查） | 該檔為 agent 治理規則來源；重複段落浪費 context、產生雙份可漂移的規則文本，任一處修改即與另一處脫節，破壞規則一致性 |
| 2 | `OpenAICompatibleInterface.__init__` 未對 `max_tokens`（負值／0／非整數）與 `temperature`/`top_p`（NaN/Inf/非數值）做 fail-fast 驗證 | 與 `model_id`/`timeout_seconds` 的建構期驗證風格不一致：非法 generation 參數會延遲到**推論時**才以 API 400 暴雷（虽有 `ERROR:` 哨兵兜底不崩潰，但靜默消耗評測樣本，且誤報原因難以定位） |
| 3 | `docs/step3.md` 檔尾缺換行（`\ No newline at end of file`） | 純外觀問題；該檔屬本次提交範圍，diff 會持續出現無意義的換行噪訊 |

**審查後確認非問題（No-Action）項**：
- 502/503/504 重試覆蓋：經 openai SDK 1.108.1 源碼核實（`_client.py`：`status_code >= 500` → `InternalServerError`，而 `InternalServerError` 已在 `_RETRYABLE_EXCEPTIONS`），無漏網；Step 4 另加參數化回歸測試鎖定此行為以防 SDK 升級漂移。
- 分母鐵律對接：`ERROR:` 哨兵 → `Evaluator.extract_answer` 回傳 `None`（Tier 3 INVALID）→ `overall_accuracy` 分母固定 total，整合測試已覆蓋。

## 3. Refactoring & Implementation
- **Core Modifications**:
  - `.cursorrules`：去除重複的標題／§1／§2／§4／§5／§6 區塊，收斂為單一乾淨的 §1–§6 結構，並清除行內尾隨空白。
  - `src/models/openai_compatible.py`：`__init__` 新增 fail-fast 驗證 —— `max_tokens` 必須為整數且 `>= 1`（排除 `bool` 與浮點）；`temperature`／`top_p` 必須為有限數值（`math.isfinite`）；同步更新 docstring 之 `Args`／`Raises`。
  - `docs/step3.md`：補上檔尾換行。
- **Edge Cases Handled**:
  - `max_tokens=0`／負值／`128.0`／`"128"`／`True` → 建構期 `ValueError`（不再延遲至推論期）。
  - `temperature`／`top_p` 傳 `NaN`／`±Inf` → 建構期 `ValueError`。
  - 502/503/504 网关狀態碼：鎖定「經 SDK 映射為 `InternalServerError` → 必經重試 → 耗盡後 `ERROR:` 哨兵」之全鏈路行為（首次 + 2 重試 = 3 次呼叫）。
- **Key Design Decisions**:
  - 建構期 fail-fast 與「單次推論不拋未捕獲例外」分層：配置錯誤（有意義、可預知）於建構期拒絕；運時 API 異常（不可預知）於第一層防護兜底為哨兵字串——兩類錯誤不互相污染。
  - `temperature`／`top_p` 僅驗證「有限數值」，不硬編碼 provider 特定範圍（各端點範圍不一），範圍合法性交由 API 端 4xx 回應處理（不可重試 → 立即哨兵）。
  - 回歸鎖定採用端到端行為斷言（`call_count == 3`）而非 white-box 檢查 tenacity predicate，使測試對 SDK 內部實作重構保持韌性。

## 4. Verification & Before/After Comparison
- **Pytest Output Summary**: 180 passed in 1.00s（Step 4 模型層 67 筆 + Step 1 資料層 + Step 3 Evaluator；本步重構新增 6 筆邊界／回歸測試；測試覆蓋率：not measured（未配置 coverage 工具））。
- **Ruff & Mypy Output Status**: `ruff check` 0 違規；`ruff format --check` 12 檔全部格式化通過；`mypy src/ --strict` 8 個原始碼檔案 0 錯誤。

### Before / After Comparison Matrix
| Dimension | Before Refactor | After Refactor |
| :--- | :--- | :--- |
| Architecture / Design | 模型層已模組化；`.cursorrules` 含重複標題與 §1/§2/§4/§5/§6 雙份文本 | 治理規則收斂為單一 §1–§6 結構；模型層架構不變（維持雙層防護與設定工廠） |
| Robustness / Edge Cases | `max_tokens=0`/`NaN temperature` 等非法 generation 參數建構期放行，延遲至推論期以 API 400 暴雷；5xx 重試覆蓋無行為鎖定測試 | 建構期 fail-fast（`ValueError`）；502/503/504 重試行為以參數化端到端測試鎖定，防 SDK 升級漂移 |
| Test Coverage | 174 passed | 180 passed（+6：max_tokens 正整數/型別、temperature/top_p 有限性、5xx 重試分類 × 3） |

## 5. Verification Artifacts & Logs
- **Terminal Output Log**:
  ```text
  $ ruff check src/ tests/
  All checks passed!

  $ ruff format --check src/ tests/
  12 files already formatted

  $ pytest tests/ -o timeout=10
  ============================= 180 passed in 1.00s =============================

  $ mypy src/ --strict
  Success: no issues found in 8 source files
  ```

## 6. HuggingFace 介面擴充說明（Step 4 後續階段）

- **新增模組**：`src/models/huggingface.py` — `HuggingFacePipelineInterface`（繼承 `BaseModelInterface`），以 `transformers.pipeline("text-generation")` 驅動 Qwen／Llama 等本地模型；`SUPPORTED_TYPES` 加入 `huggingface`／`hf_pipeline`（別名），`build_model_interface` 依 `configs/eval_config.yaml` 之 `models[]` 區塊建構（`device`／`max_new_tokens`／`temperature`／`torch_dtype` 自設定讀取）。
- **關鍵機制**：
  - **Prompt 前綴裁剪**：以 `return_full_text=True` 請求，`_generate()` 自動裁掉回傳文字中輸入 prompt 前綴，僅回傳新生成文字；輸出未以 prompt 起頭（chat 模板差異）時記錄警告並回傳去除首尾空白之全文。
  - **Token 安全性**：`HF_TOKEN` 僅經 `os.getenv("HF_TOKEN", None)` 讀取並傳入 pipeline，無寫死憑證。
  - **第一層例外防禦**：`predict()` 全推論路徑 try...except 兜底，失敗回傳 `ERROR: HuggingFace Inference Failed - <msg>` 哨兵（權重載入失敗、OOM、輸出結構異常），與 Evaluator Tier-3 兜底分層。
  - **建構期 fail-fast**：`model_id` 非空、`device` 型別（int/str/None）、`max_new_tokens` 正整數、`temperature` 有限非負值，非法值一律 `ValueError`。
- **型別修復**：`transformers.pipelines` 未顯式匯出 `Pipeline`，mypy strict 報 `attr-defined`，於 `src/models/huggingface.py` 匯入行加 `# type: ignore[attr-defined]` 抑噪（唯一直屬封裝層，不影響其餘 strict 檢查）。
- **測試（全離線）**：mock `src.models.huggingface.pipeline`，無真實權重下載／無網路：`TestHuggingFacePipeline`（建構參數驗證、prompt 前綴裁剪、空輸出、例外 → 哨兵、`TypeError` fail-fast）＋ `TestBuildHuggingFaceInterface`（`huggingface`／`hf_pipeline` 別名、device/generation 參數對接、缺 `model_id` fail-fast）。
- **全綠紀錄**：`pytest tests/` **194 passed in 4.46s**（模型層含 HF 介面測試全數通過；覆蓋率：not measured）。

  ```text
  $ mypy src/ --strict
  Success: no issues found in 9 source files

  $ ruff check src/ tests/
  All checks passed!

  $ ruff format --check src/ tests/
  13 files already formatted

  $ pytest tests/ -o timeout=10
  ============================= 194 passed in 4.46s ==============================
  ```
