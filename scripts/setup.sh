#!/usr/bin/env bash
# =============================================================================
# ETAIQ Development Environment Setup
# Requires: Python 3.11
# Usage: ./scripts/setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
VENV_DIR="${PROJECT_ROOT}/.venv"
REQUIREMENTS_DEV="${BACKEND_DIR}/requirements/dev.txt"

log() {
  printf '\n==> %s\n' "$1"
}

error() {
  printf '\nERROR: %s\n' "$1" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Python 3.11 check
# ---------------------------------------------------------------------------
if command -v python3.11 &>/dev/null; then
  PYTHON=python3.11
elif command -v python3 &>/dev/null; then
  PYTHON=python3
  PY_VERSION="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${PY_VERSION}" != "3.11" ]]; then
    error "Python 3.11 is required (found ${PY_VERSION}). Install python3.11 and re-run."
  fi
else
  error "Python 3.11 is required but was not found on PATH."
fi

log "Using interpreter: $("${PYTHON}" --version)"

# ---------------------------------------------------------------------------
# Virtual environment (single root-level venv)
# ---------------------------------------------------------------------------
if [[ -d "${BACKEND_DIR}/.venv" ]]; then
  log "Removing nested virtual environment at ${BACKEND_DIR}/.venv"
  rm -rf "${BACKEND_DIR}/.venv"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON}" -m venv "${VENV_DIR}"
else
  log "Virtual environment already exists at ${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# Bootstrap pip toolchain
# ---------------------------------------------------------------------------
log "Upgrading pip, wheel, and setuptools"
python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
log "Installing dependencies from ${REQUIREMENTS_DEV}"
pip install -r "${REQUIREMENTS_DEV}"

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
log "Verifying core package imports"
python - <<'PY'
import importlib
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version}")

packages = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
    "alembic",
    "psycopg",
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "scipy",
    "category_encoders",
    "shap",
    "lime",
    "matplotlib",
    "plotly",
    "seaborn",
    "loguru",
    "prometheus_client",
    "structlog",
    "dotenv",
    "joblib",
    "orjson",
    "rich",
    "passlib",
    "jose",
    "bcrypt",
    "pandera",
    "great_expectations",
    "pytest",
    "httpx",
    "black",
    "ruff",
    "isort",
    "mypy",
]

failed = []
for name in packages:
    try:
        importlib.import_module(name)
    except ImportError as exc:
        failed.append(f"{name}: {exc}")

if failed:
    print("Verification failed for:")
    for item in failed:
        print(f"  - {item}")
    raise SystemExit(1)

print("All core imports verified successfully.")
PY

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
log "Python version"
python --version

log "Installed packages"
pip list

log "Setup complete. Activate the environment with:"
printf '  source %s/bin/activate\n' "${VENV_DIR}"
