# ETAIQ Task Board

Active development tasks organized by milestone.

## Milestone 1 – Project Initialization

| ID | Task | Status | Owner |
|----|------|--------|-------|
| M1-01 | Create monorepo directory structure | Done | - |
| M1-02 | Initialize FastAPI backend with health endpoint | Done | - |
| M1-03 | Initialize Next.js frontend with placeholder pages | Done | - |
| M1-04 | Scaffold ML pipeline directories | Done | - |
| M1-05 | Create Docker development environment | Done | - |
| M1-06 | Set up GitHub Actions CI | Done | - |
| M1-07 | Write project documentation | Done | - |
| M1-08 | Create Cursor guidance files | Done | - |

## Milestone 2 – Database & Core API

| ID | Task | Status | Owner |
|----|------|--------|-------|
| M2-01 | Design PostgreSQL schema | Backlog | - |
| M2-02 | Set up Alembic migrations | Backlog | - |
| M2-03 | Implement SQLAlchemy models | Backlog | - |
| M2-04 | Create database session dependency | Backlog | - |
| M2-05 | Add prediction CRUD schemas | Backlog | - |
| M2-06 | Implement prediction service layer | Backlog | - |
| M2-07 | Add database health to /health | Backlog | - |

## Milestone 3 – Authentication

| ID | Task | Status | Owner |
|----|------|--------|-------|
| M3-01 | JWT token service | Backlog | - |
| M3-02 | User model and registration | Backlog | - |
| M3-03 | Login endpoint | Backlog | - |
| M3-04 | Auth middleware | Backlog | - |
| M3-05 | Frontend login page | Backlog | - |
| M3-06 | Protected route wrapper | Backlog | - |

## Milestone 4 – ML Pipeline

| ID | Task | Status | Owner |
|----|------|--------|-------|
| M4-01 | Data validation pipeline | Backlog | - |
| M4-02 | Feature engineering module | Backlog | - |
| M4-03 | Baseline model training script | Backlog | - |
| M4-04 | Model evaluation reports | Backlog | - |
| M4-05 | Model registry integration | Backlog | - |
| M4-06 | Inference service wrapper | Backlog | - |

## Milestone 5 – Prediction Feature

| ID | Task | Status | Owner |
|----|------|--------|-------|
| M5-01 | POST /api/v1/predictions endpoint | Backlog | - |
| M5-02 | Prediction form UI | Backlog | - |
| M5-03 | Results display component | Backlog | - |
| M5-04 | History list and detail views | Backlog | - |

## Blocked / Decisions Needed

| ID | Item | Notes |
|----|------|-------|
| D-01 | ML model selection | Gradient boosting vs neural network – decide in M4 |
| D-02 | AI provider default | OpenAI vs Gemini – may support both |
| D-03 | Deployment target | AWS vs GCP vs self-hosted – decide in M9 |
