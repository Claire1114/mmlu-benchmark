#!/usr/bin/env bash
# ==============================================================================
# 腳本名稱: auto_audit.sh
# 職責: 呼叫 Aider (Adversarial Reviewer & QA) 執行單模組代碼審查、重構、測試生成與驗證
# ==============================================================================

set -euo pipefail

# 參數檢查
if [ $# -lt 1 ]; then
    echo "❌ 錯誤: 缺少模組名稱參數！"
    echo "使用方式: $0 <MODULE_NAME> (例如: $0 dataset_loader)"
    exit 1
fi

MODULE_NAME="$1"
TARGET_FILE="src/${MODULE_NAME}.py"
TEST_FILE="tests/test_${MODULE_NAME}.py"
ARTIFACT_DIR="ai_artifacts"
LOG_FILE="${ARTIFACT_DIR}/${MODULE_NAME}_review_log.md"
MODEL="gemini/gemini-3.8-flash"

# 目錄與目標檔案檢查
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ 錯誤: 找不到目標原始碼檔案: $TARGET_FILE"
    exit 1
fi

mkdir -p "$ARTIFACT_DIR" tests

echo "========================================================================"
echo "🚀 開始執行單模組審查流程: ${MODULE_NAME} (模型: ${MODEL})"
echo "========================================================================"
echo "# ${MODULE_NAME} 審查與重構日誌 ($(date '+\%Y-\%m-\%d \%H:\%M:\%S'))" > "$LOG_FILE"

# ------------------------------------------------------------------------------
# 步驟 A: Code Review (以唯讀模式進行，產出建議並記錄至日誌)
# ------------------------------------------------------------------------------
echo "🔍 [步驟 A/D] 執行代碼審查 (Code Review)..."
echo -e "\n## 1. 代碼審查報告\n" >> "$LOG_FILE"

aider --model "$MODEL" \
      --no-show-model-warnings \
      --read "$TARGET_FILE" \
      --message "身為 Adversarial Reviewer，請仔細審查 ${TARGET_FILE}。重點檢查：1. 網路/API 請求重試機制 2. 快取防漏與邊界檢查 3. 異常捕捉完整性與型態標註。請條列具體缺陷與重構建議，不要修改檔案。" \
      --yes-always >> "$LOG_FILE" 2>&1 || true

echo "✅ 審查建議已沉澱至 ${LOG_FILE}"

# ------------------------------------------------------------------------------
# 步驟 B: Refactor (讀取審查建議進行重構，保持介面相容)
# ------------------------------------------------------------------------------
echo "🛠️  [步驟 B/D] 根據審查建議執行重構 (Refactor)..."
echo -e "\n## 2. 代碼重構 Git Diff\n" >> "$LOG_FILE"

aider --model "$MODEL" \
      --no-show-model-warnings \
      --file "$TARGET_FILE" \
      --message "請依據日誌 ${LOG_FILE} 中的審查建議，重構 ${TARGET_FILE}：強化邊界處理與防禦性邏輯，嚴格維持對外函數與類別介面相容。請直接修改該檔案。" \
      --yes-always

echo '```diff' >> "$LOG_FILE"
git diff "$TARGET_FILE" >> "$LOG_FILE" 2>&1 || true
echo '```' >> "$LOG_FILE"
echo "✅ 重構完成，Git Diff 已沉澱至 ${LOG_FILE}"

# ------------------------------------------------------------------------------
# 步驟 C: Unit Test Generation (生成單元測試，強調 Mock 與極限邊界)
# ------------------------------------------------------------------------------
echo "🧪 [步驟 C/D] 生成單元測試 (Unit Test Generation)..."
echo -e "\n## 3. 單元測試生成說明\n" >> "$LOG_FILE"

aider --model "$MODEL" \
      --no-show-model-warnings \
      --file "$TEST_FILE" \
      --read "$TARGET_FILE" \
      --message "請為 ${TARGET_FILE} 在 ${TEST_FILE} 撰寫完整的 pytest 單元測試：1. 強制使用 unittest.mock 隔離所有外部 HTTP/網路請求與磁碟 I/O。2. 包含 Happy Path、極限邊界值（空值、Malformed JSON、逾時異常）3. 確保測試具備高覆蓋率與自包含性。" \
      --yes-always >> "$LOG_FILE" 2>&1

echo "✅ 單元測試已生成於 ${TEST_FILE}"

# ------------------------------------------------------------------------------
# 步驟 D: Pytest Verify (執行測試並記錄結果，未通過立即中止)
# ------------------------------------------------------------------------------
echo "🚦 [步驟 D/D] 執行 Pytest 驗證..."
echo -e "\n## 4. Pytest 驗證日誌\n" >> "$LOG_FILE"
echo '```text' >> "$LOG_FILE"

if pytest "$TEST_FILE" -v 2>&1 | tee -a "$LOG_FILE"; then
    echo '```' >> "$LOG_FILE"
    echo -e "\n**驗證結果**: ✅ 測試全數通過！\n" >> "$LOG_FILE"
    echo "🎉 模組 ${MODULE_NAME} 審查、重構與測試驗證全數通過！"
else
    echo '```' >> "$LOG_FILE"
    echo -e "\n**驗證結果**: ❌ 單元測試失敗！\n" >> "$LOG_FILE"
    echo "❌ 模組 ${MODULE_NAME} 測試失敗，請檢查 ${LOG_FILE} 與 ${TEST_FILE}"
    exit 1
fi