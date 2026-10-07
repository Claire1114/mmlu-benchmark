#!/usr/bin/env bash
set -euo pipefail

# 切換至專案根目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "⚡ 執行 Smoke Test"
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

# 2. 驗證 API 連線與金鑰狀態
echo "[INFO] 測試 Gemini API 連線..."
python -c "from src.models.gemini_native import GeminiNativeInterface; m = GeminiNativeInterface('test', 'gemini-3.1-flash-lite'); print('API Result:', m.predict('Hi'))"

# 3. 冒煙測試（單一輕量模型快速驗證）
echo "[INFO] 啟動管線測試..."
python main.py \
    --mode demo \
    --models smollm2-1.7b \
    --shots 0 \
    --delay 8.0

echo "[SUCCESS] Smoke Test 全部通過！"
