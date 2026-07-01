# =============================================================================
# ETAIQ Monorepo Makefile
# Python 3.11 backend + Next.js frontend
# =============================================================================

.PHONY: setup lint format test backend frontend clean docker help

BACKEND_DIR := backend
FRONTEND_DIR := frontend
VENV := .venv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Bootstrap Python venv and install all dependencies
	chmod +x scripts/*.sh
	./scripts/setup.sh

lint: ## Run backend and frontend linters
	./scripts/lint.sh

format: ## Auto-format backend Python source
	./scripts/format.sh

test: ## Run backend test suite with coverage
	./scripts/test.sh

backend: ## Start FastAPI dev server on port 8000
	@test -d $(VENV) || (echo "Run 'make setup' first." && exit 1)
	cd $(BACKEND_DIR) && ../$(VENV)/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend: ## Start Next.js dev server on port 3000
	cd $(FRONTEND_DIR) && npm run dev

docker: ## Start Docker Compose development stack
	./scripts/dev.sh

clean: ## Remove build artifacts, caches, and virtual environment
	rm -rf $(VENV)
	rm -rf $(BACKEND_DIR)/.venv
	rm -rf $(BACKEND_DIR)/.pytest_cache
	rm -rf $(BACKEND_DIR)/.ruff_cache
	rm -rf $(BACKEND_DIR)/.mypy_cache
	rm -rf $(BACKEND_DIR)/htmlcov
	rm -rf $(BACKEND_DIR)/coverage
	rm -f $(BACKEND_DIR)/.coverage
	find $(BACKEND_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/.next
	rm -rf $(FRONTEND_DIR)/node_modules
