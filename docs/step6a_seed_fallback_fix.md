# Step 6: DatasetLoader Seed 預設值修復（seed=None → project.seed）

## 1. Step Objective
- **Task**: 修復 `python main.py --mode smoke_test --limit 4` 真實執行時之崩潰：`main.py:_build_samples` 依 `load_data` 文件化契約以預設 `seed=None` 呼叫，`MMLUDatasetLoader.load_data` 卻在 fallback 到 `self._seed` 之前先對 `None` 執行整數驗證，拋出 `ValueError: seed must be an integer, got None.`；修復順序錯誤並補上 `seed=None` 迴歸測試網。
- **AI Tool**: Cline (VS Code AI coding agent)
- **Collaboration Pattern**: Plan (Analysis & Refactor Proposal) -> Act (Implementation & Auto-Verification)
- **AI Efficiency Gains**: 由 traceback 直接定位「先驗證後 fallback」之語序缺陷（非調參試錯）；行為等價矩陣（None／int／bool／str／float）事先推導確保不削弱既有驗證；3 個新邊界測試自動生成並 0.74s 內閉環驗證。

## 2. Issues Identified (AI Review)

| # | Issue Identified | Risk / Potential Defect |
| :--- | :--- | :--- |
| 1 | `src/dataset_loader.py:195-197`（修前）：抽樣分支先以 `isinstance(seed, int)` 驗證原始參數、其後才 `self._seed if seed is None else seed` fallback；`None` 永远命中驗證失敗 | **致命**：違反 `load_data` docstring 明載契約（「seed: None 表示使用設定檔 project.seed」）；凡 `len(normalized) > sample_size`（真實 MMLU 子集 3000+ 列對 smoke/demo 抽樣 10/30 題恒為真）即崩潰，`main.py` 一次性載入路徑（不顯式傳 seed）完全不可執行；`get_samples()` 因顯式傳 `seed=self._seed` 而倖免，造成「公開 API 正常、管線組合崩潰」之隱性陷阱 |
| 2 | 測試保真度缺口：`tests/test_dataset_loader.py` 既有 seed 測試全部顯式傳 `seed=42/99`，無一覆蓋文件化預設值 `seed=None`；`tests/test_main.py` 之 `StubLoader.load_data` 接受 `seed=None` 不驗證，比真實 loader 寬鬆 | 44/290 離線測試全綠仍無法攔截真實執行崩潰（test double 與被測契約漂移）；`seed=None` 成為唯一無防護之抽樣路徑 |

## 3. Refactoring & Implementation
- **Core Modifications**: `load_data` 抽樣分支改為**先 resolve 後驗證**：
  ```python
  resolved_seed: int = self._seed if seed is None else seed
  if isinstance(resolved_seed, bool) or not isinstance(resolved_seed, int):
      raise ValueError(f"seed must be an integer, got {seed!r}.")
  ```
  `self._seed` 已於 `__init__`（`_resolve_seed`）保證為整數，fallback 後仍保留對呼叫端顯式傳入之非整數／bool 嚴格拒絕（錯誤訊息維持 `seed!r` 指向呼叫端值）。`main.py` 無需變更（其呼叫方式符合文件化契約，缺陷在 loader 端）。
- **Edge Cases Handled**（新增 3 個迴歸測試）：
  1. `test_seed_none_falls_back_to_config_seed`：`seed=None` 與顯式 `seed=<project.seed>`（=7）產生**逐筆相同**之確定性抽樣；
  2. `test_seed_none_with_missing_config_seed_uses_default`：設定缺 `project.seed` 時 `seed=None` 回落 `DEFAULT_SEED=42`，與顯式 `seed=42` 一致；
  3. `test_explicit_non_integer_seed_still_rejected`：顯式 `seed="42"`／`seed=True` 仍拋 `ValueError`（fallback 不吞掉型別錯誤、不削弱驗證）。
- **Key Design Decisions**: 修復點選在 `src/dataset_loader.py`（契約所有權端）而非 `main.py` 補傳 `seed=loader._seed`（屬私有屬性的旁路取用，且重複 loader 之配置解析職責）；驗證式由「拒絕 None」改為「拒絕 resolve 後仍非整數」，對既有合法輸入（顯式 int）行為完全等價。

## 4. Verification & Before/After Comparison
- **Pytest Output Summary**: `tests/test_dataset_loader.py -k seed` 7 passed（含 3 新增，0.74s）；全數套件 `tests/` **293 passed**（4.79s，較修前 290 增加 3 筆邊界測試）。Coverage 狀態：**not measured**（未設定 pytest-cov）。
- **Ruff & Mypy Output Status**: `ruff check src/ tests/` → All checks passed!；`ruff format --check src/ tests/ main.py` → 17 files already formatted；`mypy src/ --strict` → Success: no issues found in 10 source files。

### Before / After Comparison Matrix

| Dimension | Before Refactor | After Refactor |
| :--- | :--- | :--- |
| Architecture / Design | `load_data` 抽樣分支「先驗證後 fallback」，與 docstring 契約（None→project.seed）矛盾；`main.py` 管線路徑對真實 loader 不可用 | 「先 resolve 後驗證」；`None` 依約回落設定檔種子（缺失時回落 42），`main.py` 與 `get_samples()` 兩條路徑皆可用且語意一致 |
| Robustness / Edge Cases| `seed=None`（唯一文件化預設值）必崩潰；顯式非整數拒絕行為存在 | `seed=None` 確定性抽樣正常；顯式 `"42"`／`True`／float 仍 fail-fast 拒絕；錯誤訊息精確指向呼叫端值 |
| Test Coverage | 290 passed，但 `seed=None` 零覆蓋；test double 寬於真實契約 | 293 passed（3 個新邊界測試：config seed 對齊、缺失回落 42、顯式非法型別仍拒絕） |

## 5. Verification Artifacts & Logs
- **Terminal Output Log**:
  ```text
  $ python -m pytest tests/test_dataset_loader.py -k "seed" -v -o timeout=10
  tests/test_dataset_loader.py::test_seed_missing_falls_back_to_default PASSED
  tests/test_dataset_loader.py::test_non_integer_seed_rejected PASSED
  tests/test_dataset_loader.py::test_seeded_sampling_is_deterministic_and_preserves_order PASSED
  tests/test_dataset_loader.py::test_seeded_sampling_differs_across_seeds_and_subjects PASSED
  tests/test_dataset_loader.py::test_seed_none_falls_back_to_config_seed PASSED
  tests/test_dataset_loader.py::test_seed_none_with_missing_config_seed_uses_default
  WARNING  src.dataset_loader:dataset_loader.py:550 project.seed is missing; falling back to default seed 42.
  PASSED
  tests/test_dataset_loader.py::test_explicit_non_integer_seed_still_rejected PASSED
  ======================= 7 passed, 30 deselected in 0.74s ======================

  $ python -m pytest tests/ -o timeout=10 -q
  ============================= 293 passed in 4.79s =============================

  $ ruff check src/ tests/
  All checks passed!

  $ ruff format --check src/ tests/ main.py
  17 files already formatted

  $ python -m mypy src/ --strict
  Success: no issues found in 10 source files
  ```
