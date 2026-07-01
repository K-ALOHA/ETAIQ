# ETAIQ � AI Powered ETA Intelligence Platform

ETAIQ is a production-grade monorepo for predicting delivery ETAs in quick-commerce using machine learning, explainability, and an AI assistant.

## Repository Structure

```
ETAIQ/
??? backend/          # FastAPI REST API
??? frontend/         # Next.js web application
??? ml/               # Machine learning pipelines
??? docs/             # Project documentation
??? diagrams/         # Architecture diagrams
??? scripts/          # Development utilities
??? docker/           # Container configuration
??? .github/          # CI/CD workflows
??? .cursor/          # AI development guidance
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and Docker Compose (optional)

### Environment Setup

```bash
cp .env.example .env
```

### Backend

```bash
make setup
make backend
```

Or manually:

```bash
./scripts/setup.sh
source .venv/bin/activate
cd backend && uvicorn app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

### Frontend

```bash
cd frontend && npm install && npm run dev
```

Or: `make frontend`

Application: `http://localhost:3000`

### Docker (Development)

```bash
chmod +x scripts/*.sh
./scripts/dev.sh
```

## Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_SPEC.md](docs/PROJECT_SPEC.md) | Product and technical specification |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture overview |
| [ROADMAP.md](docs/ROADMAP.md) | Development milestones |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Contribution guidelines |
| [CHANGELOG.md](docs/CHANGELOG.md) | Version history |

## License

Proprietary � All rights reserved.
