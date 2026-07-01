# ETAIQ Project Memory

Persistent context for AI-assisted development sessions.

## Project Identity

- **Name**: ETAIQ (ETA Intelligence)
- **Purpose**: ML-powered delivery ETA prediction for quick-commerce
- **Repository**: Monorepo at `ETAIQ/`
- **Current Version**: 0.1.0
- **Current Milestone**: M1 � Project Initialization (complete)

## Key Paths

| Path | Description |
|------|-------------|
| `backend/app/main.py` | FastAPI application entry point |
| `backend/app/api/health.py` | Health check endpoint |
| `backend/app/core/config.py` | Environment-based settings |
| `backend/app/core/logging.py` | Structlog configuration |
| `frontend/src/app/(app)/` | Main application pages |
| `frontend/src/components/layout/Sidebar.tsx` | Navigation sidebar |
| `ml/` | ML pipeline directories (scaffolded only) |
| `docker/docker-compose.yml` | Development orchestration |
| `.github/workflows/ci.yml` | CI pipeline |
| `docs/` | Project documentation |

## Environment Variables

Defined in `.env.example`:

- `DATABASE_URL` � PostgreSQL connection
- `JWT_SECRET` � Auth token signing
- `OPENAI_API_KEY` � AI assistant
- `GEMINI_API_KEY` � Alternative AI provider
- `MODEL_PATH` � ML artifact location
- `LOG_LEVEL` � Logging verbosity

## API Endpoints (Implemented)

| Method | Path | Response |
|--------|------|----------|
| GET | `/health` | `{"status": "healthy"}` |

## Frontend Pages (Placeholder)

- `/dashboard` � Operational overview
- `/prediction` � ETA prediction interface
- `/analytics` � Model performance analytics
- `/history` � Prediction history
- `/ai-assistant` � Conversational AI interface
- `/settings` � Configuration management

## Development Commands

```bash
# Setup
make setup                    # or ./scripts/setup.sh

# Backend / Frontend
make backend                  # FastAPI on :8000
make frontend                 # Next.js on :3000

# Quality
make lint                     # or ./scripts/lint.sh
make format                   # or ./scripts/format.sh
make test                     # or ./scripts/test.sh

# Docker
make docker                   # or ./scripts/dev.sh
```

## Conventions Established

- Python: type hints, docstrings, structlog, pydantic-settings
- TypeScript: strict mode, Tailwind, App Router route groups
- API prefix: `/api/v1` for versioned endpoints
- Health check at root: `/health` (not versioned)

## What NOT to Implement Yet

Per milestone 1 scope boundary:

- ML training or inference logic
- Prediction API endpoints
- Database models or migrations
- Authentication
- SHAP explainability
- AI assistant integration
- Production deployment configuration

## Next Milestone Focus

**M2 � Database & Core API**: PostgreSQL schema, Alembic migrations, SQLAlchemy models, prediction CRUD without ML inference.
