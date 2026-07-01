#!/usr/bin/env bash
# =============================================================================
# ETAIQ Test — backend test suite with coverage
# Usage: ./scripts/test.sh [pytest args...]
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

echo "==> Running backend tests"
"${VENV_DIR}/bin/pytest" --cov=app --cov-report=term-missing -v "$@"

echo "==> Tests complete"
