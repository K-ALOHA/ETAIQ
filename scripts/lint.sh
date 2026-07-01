#!/usr/bin/env bash
# =============================================================================
# ETAIQ Lint — backend (ruff, mypy) and frontend (eslint)
# Usage: ./scripts/lint.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: Virtual environment not found. Run ./scripts/setup.sh first." >&2
  exit 1
fi

echo "==> Linting backend (ruff)"
cd "${BACKEND_DIR}"
"${VENV_DIR}/bin/ruff" check .

echo "==> Linting backend (mypy)"
"${VENV_DIR}/bin/mypy" app

echo "==> Linting frontend (eslint)"
cd "${FRONTEND_DIR}"
npm run lint

echo "==> Lint complete"
