#!/usr/bin/env bash
# =============================================================================
# ETAIQ Format — backend Python source (isort, black, ruff fix)
# Usage: ./scripts/format.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: Virtual environment not found. Run ./scripts/setup.sh first." >&2
  exit 1
fi

cd "${BACKEND_DIR}"

echo "==> Sorting imports (isort)"
"${VENV_DIR}/bin/isort" .

echo "==> Formatting code (black)"
"${VENV_DIR}/bin/black" .

echo "==> Applying safe lint fixes (ruff)"
"${VENV_DIR}/bin/ruff" check --fix .

echo "==> Format complete"
