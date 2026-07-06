#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p \
  "${PROJECT_ROOT}/.cache/uv" \
  "${PROJECT_ROOT}/.cache/pip" \
  "${PROJECT_ROOT}/.cache/torch" \
  "${PROJECT_ROOT}/.cache/matplotlib" \
  "${PROJECT_ROOT}/.cache/pycache" \
  "${PROJECT_ROOT}/outputs"

export UV_CACHE_DIR="${PROJECT_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
export PYTHONPYCACHEPREFIX="${PROJECT_ROOT}/.cache/pycache"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

echo "Activated local environment at ${PROJECT_ROOT}/.venv"
echo "Caches are under ${PROJECT_ROOT}/.cache"
