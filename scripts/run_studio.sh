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

# Создаём cache-директории заранее, чтобы зависимости не падали на Permission/ENOENT.
mkdir -p \
  "${XDG_CACHE_HOME:-/tmp/.cache}" \
  "${HF_HOME:-/tmp/.cache/huggingface}" \
  "${HF_DATASETS_CACHE:-/tmp/.cache/huggingface/datasets}" \
  "${HUGGINGFACE_HUB_CACHE:-/tmp/.cache/huggingface/hub}" \
  "${TRITON_CACHE_DIR:-/tmp/.cache/triton}" \
  "${TORCH_HOME:-/tmp/.cache/torch}" \
  "${TORCHINDUCTOR_CACHE_DIR:-/tmp/.cache/torchinductor}" \
  "${CUDA_CACHE_PATH:-/tmp/.cache/nv}" \
  "${NUMBA_CACHE_DIR:-/tmp/.cache/numba}" \
  "${MPLCONFIGDIR:-/tmp/.cache/matplotlib}" \
  "${PIP_CACHE_DIR:-/tmp/.cache/pip}"

echo "=============================================="
echo "  🏠 Models at Home Training Studio"
echo "=============================================="
echo "  Открой в браузере: http://localhost:$PORT"
echo "=============================================="

streamlit run homellm/app/LLM.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    "$@"
