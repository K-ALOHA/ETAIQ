#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Starting ETAIQ development environment..."
docker compose -f "${PROJECT_ROOT}/docker/docker-compose.yml" up --build "$@"
