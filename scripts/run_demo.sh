#!/usr/bin/env bash
set -euo pipefail

# 切換至專案根目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "🚀 執行正式 Demo"
echo "=========================================="

# 1. 檢查並自動匯入 .env 環境變數
if [ -f ".env" ]; then
    echo "[INFO] 匯入 .env 環境變數..."
    set -a
    source .env
    set +a
else
    echo "[ERROR] 找不到 .env 檔案！" >&2
    exit 1
fi

# 2. 前置檢查確認 API 回應
echo "[INFO] 確認 Gemini API 連線狀態..."
python -c "from src.models.gemini_native import GeminiNativeInterface; m = GeminiNativeInterface('test', 'gemini-3.1-flash-lite'); print('API Result:', m.predict('Hi'))"

# 3. 正式啟動 0-Shot 評測
echo "[INFO] 開始執行 3 模型評測流程..."
python main.py \
    --mode demo \
    --models qwen2.5-0.5b smollm2-1.7b gemini-3.1-flash-lite \
    --shots 0 \
    --delay 8.0

echo "[SUCCESS] Demo 執行完畢，結果已輸出！"
