# ETAIQ Project Specification

## Vision

ETAIQ delivers accurate, explainable delivery ETA predictions for quick-commerce platforms, enabling operations teams to optimize rider allocation, set customer expectations, and reduce late deliveries.

## Problem Statement

Quick-commerce delivery times are influenced by traffic, weather, order complexity, rider availability, and store load. Static ETA estimates lead to poor customer experience and operational inefficiency.

## Solution

A full-stack AI platform that:

1. Ingests delivery context features in real time
2. Predicts ETA using trained ML models
3. Explains predictions via SHAP feature attribution
4. Provides an AI assistant for operational insights
5. Tracks prediction accuracy over time

## System Components

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| Backend API | FastAPI | REST endpoints, auth, orchestration |
| Frontend | Next.js | Dashboard, predictions, analytics UI |
| ML Pipeline | Python | Training, evaluation, model registry |
| Database | PostgreSQL | Predictions, users, audit logs |
| AI Assistant | OpenAI / Gemini | Natural language insights |
| Explainability | SHAP | Feature importance visualization |
| Infrastructure | Docker, GitHub Actions | Dev environment, CI/CD |

## API Design (Planned)

### Milestone 1 (Current)

- `GET /health` – Service health check

### Future Endpoints

- `POST /api/v1/predictions` – Submit prediction request
- `GET /api/v1/predictions/{id}` – Retrieve prediction result
- `GET /api/v1/predictions/{id}/explain` – SHAP explanation
- `POST /api/v1/assistant/chat` – AI assistant conversation
- `GET /api/v1/analytics/accuracy` – Model accuracy metrics

## Data Model (Planned)

- **Predictions**: input features, predicted ETA, confidence, model version
- **Actuals**: actual delivery time for accuracy tracking
- **Users**: authentication and role-based access
- **Model Versions**: registry metadata, metrics, artifact paths

## Non-Functional Requirements

- **Latency**: Prediction API p99 < 200ms
- **Availability**: 99.9% uptime target
- **Observability**: Structured JSON logging, metrics, tracing
- **Security**: JWT authentication, secrets via environment variables
- **Scalability**: Stateless API, horizontal scaling ready

## Out of Scope (Milestone 1)

- ML model training and inference
- Database persistence
- Authentication
- SHAP explainability
- AI assistant integration
- Production deployment
