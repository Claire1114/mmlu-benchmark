#!/usr/bin/env bash
# ==============================================================================
# 腳本名稱: run_full_pipeline.sh
# 職責: 依序執行多模組自動審查、主評測 Pipeline 與交付報表產出
# ==============================================================================

set -euo pipefail

REPORT_DIR="reports"
RESULTS_FILE="${REPORT_DIR}/eval_results.json"
CONFIG_FILE="configs/eval_config.yaml"
ARTIFACT_DIR="ai_artifacts"

mkdir -p "$REPORT_DIR" "$ARTIFACT_DIR"

echo "========================================================================"
echo "🎯 啟動 MMLU Benchmark 全流程自動化 Pipeline"
echo "========================================================================"

# ------------------------------------------------------------------------------
# 階段 1: 遍歷模組並自動審查 (Auto-Audit)
# ------------------------------------------------------------------------------
echo "📦 [階段 1/3] 掃描並審查核心模組..."
MODULES=("dataset_loader" "model_runner" "prompt_builder" "evaluator" "metrics")

for mod in "${MODULES[@]}"; do
    TARGET_PATH="src/${mod}.py"
    if [ -f "$TARGET_PATH" ]; then
        echo "➡️ 發現模組 ${TARGET_PATH}，啟動審查程序..."
        ./scripts/auto_audit.sh "$mod"
    else
        echo "⚠ 模組檔案 ${TARGET_PATH} 不存在，跳過審查。"
    fi
done

# ------------------------------------------------------------------------------
# 階段 2: 執行端到端評測 (End-to-End Evaluation)
# ------------------------------------------------------------------------------
echo "⚙️ [階段 2/3] 執行端到端評測主程式..."

ENTRY_SCRIPT=""
if [ -f "src/main.py" ]; then
    ENTRY_SCRIPT="src/main.py"
elif [ -f "src/main_pipeline.py" ]; then
    ENTRY_SCRIPT="src/main_pipeline.py"
else
    echo "❌ 錯誤: 找不到主評測程式 (src/main.py 或 src/main_pipeline.py)"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 錯誤: 找不到設定檔 $CONFIG_FILE"
    exit 1
fi

echo "🚀 執行: python $ENTRY_SCRIPT --config$CONFIG_FILE"
python "$ENTRY_SCRIPT" --config "$CONFIG_FILE" --output "$RESULTS_FILE" || python "$ENTRY_SCRIPT" --config "$CONFIG_FILE"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "⚠️ 提醒: 未偵測到 $RESULTS_FILE，請確認評測腳本是否有儲存評測數據。"
else
    echo "✅ 評測數據已存入 $RESULTS_FILE"
fi

# ------------------------------------------------------------------------------
# 階段 3: 自動化交付物產出 (Report Generation)
# ------------------------------------------------------------------------------
echo "📄 [階段 3/3] 編譯最終評測報表與展示簡報..."

PPTX_OUT="${REPORT_DIR}/benchmark_presentation.pptx"
DOCX_OUT="${REPORT_DIR}/technical_report.docx"
MD_REPORT="${REPORT_DIR}/technical_report.md"

if command -v office-cli &> /dev/null; then
    echo "🪄 偵測到 office-cli，開始轉換 PPTX 與 DOCX..."
    office-cli export --input "$ARTIFACT_DIR" --data "$RESULTS_FILE" --output "$DOCX_OUT" || true
    office-cli slides --data "$RESULTS_FILE" --output "$PPTX_OUT" || true
    echo "✅ Office 檔案生成完畢: $DOCX_OUT,$PPTX_OUT"
else
    echo "ℹ️  未安裝 office-cli，改以 Markdown 形式彙整交付報表..."
    cat << EOF > "$MD_REPORT"
# MMLU Benchmark 技術評測報告

- 執行時間: $(date '+%Y-%m-%d %H:%M:%S')
- 設定檔: ${CONFIG_FILE}
- 評測數據位置: ${RESULTS_FILE}

## 模組審查與測試驗證摘要
各模組的審查、重構歷程與 Git Diff 沉澱於 \`${ARTIFACT_DIR}/\` 目錄。

## 評測結果
請參閱 \`${RESULTS_FILE}\`。
EOF
    echo "✅ Markdown 彙整報告已產出至: $MD_REPORT"
    echo "💡 提示: 若需要產出 PPTX/DOCX，請確認全域已安裝 office-cli。"
fi

echo "========================================================================"
echo "🎉 全流程 Pipeline 順利執行完成！"
echo "========================================================================"