#!/bin/bash
# =============================================================================
# Models at Home Training Studio — Визуальное приложение
# =============================================================================
# Запуск:
#   ./scripts/run_studio.sh
#   ./scripts/run_studio.sh --port 8502
# =============================================================================

set -e

cd "$(dirname "$0")/.."

PORT="${PORT:-8501}"

echo "=============================================="
echo "  🏠 Models at Home Training Studio"
echo "=============================================="
echo "  Открой в браузере: http://localhost:$PORT"
echo "=============================================="

streamlit run homellm/app/main.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    "$@"

