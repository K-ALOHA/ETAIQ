# ETAIQ Development Roadmap

## Milestone 1 – Project Initialization ?

**Status**: Complete

- [x] Monorepo structure
- [x] FastAPI backend with health endpoint
- [x] Next.js frontend with placeholder pages
- [x] ML directory scaffolding
- [x] Docker development environment
- [x] GitHub Actions CI pipeline
- [x] Documentation and Cursor guidance files

## Milestone 2 – Database & Core API

**Status**: Planned

- [ ] PostgreSQL schema design and migrations (Alembic)
- [ ] SQLAlchemy models and repository layer
- [ ] Database connection pooling and health checks
- [ ] Prediction request/response schemas (no ML yet)
- [ ] CRUD endpoints for prediction records
- [ ] API versioning and error handling standards

## Milestone 3 – Authentication & Authorization

**Status**: Planned

- [ ] JWT token issuance and validation
- [ ] User registration and login endpoints
- [ ] Role-based access control middleware
- [ ] Frontend auth flow (login, protected routes)
- [ ] Session management and token refresh

## Milestone 4 – ML Pipeline Foundation

**Status**: Planned

- [ ] Data ingestion and validation pipelines
- [ ] Feature engineering modules
- [ ] Baseline model training (e.g., gradient boosting)
- [ ] Model evaluation and metrics reporting
- [ ] Model registry and artifact versioning
- [ ] Prediction inference service integration

## Milestone 5 – Prediction API & Frontend

**Status**: Planned

- [ ] `POST /api/v1/predictions` endpoint
- [ ] Real-time ETA prediction from trained model
- [ ] Prediction form UI with input validation
- [ ] Results display with confidence intervals
- [ ] Prediction history list and detail views

## Milestone 6 – Explainability (SHAP)

**Status**: Planned

- [ ] SHAP value computation for predictions
- [ ] Explanation API endpoint
- [ ] Feature importance visualization in frontend
- [ ] Explanation caching for performance

## Milestone 7 – Analytics Dashboard

**Status**: Planned

- [ ] Model accuracy tracking (MAE, RMSE, MAPE)
- [ ] Prediction vs actual comparison charts
- [ ] Time-series performance trends
- [ ] Operational KPI dashboard

## Milestone 8 – AI Assistant

**Status**: Planned

- [ ] OpenAI/Gemini integration service
- [ ] Context-aware chat with prediction data
- [ ] Streaming response support
- [ ] AI Assistant UI with conversation history

## Milestone 9 – Monitoring & Production Readiness

**Status**: Planned

- [ ] Prometheus metrics and Grafana dashboards
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Production Docker configuration
- [ ] Kubernetes deployment manifests
- [ ] Load testing and performance benchmarks

## Milestone 10 – Deployment

**Status**: Planned

- [ ] Staging environment provisioning
- [ ] Production deployment pipeline
- [ ] Blue-green or canary deployment strategy
- [ ] Runbooks and incident response documentation
