# ETAIQ Constitution

This document defines the non-negotiable engineering principles for the ETAIQ platform.

## Mission

Build a reliable, explainable, and operable ETA prediction system for quick-commerce that engineering teams can trust in production.

## Core Principles

### 1. Correctness Over Speed

Predictions affect customer experience and operations. Prefer verified, tested implementations over fast hacks. Every API change must have corresponding tests.

### 2. Clean Architecture

Maintain strict layer boundaries:

- **API layer**: HTTP concerns only, no business logic
- **Service layer**: Business rules and orchestration
- **Data layer**: Persistence and external integrations
- **Schemas**: Validation and serialization at boundaries

### 3. Observability by Default

Every service operation must be loggable, traceable, and measurable. Use structured JSON logging. Never swallow exceptions silently.

### 4. Security First

- No secrets in source code
- Environment-based configuration
- Authentication required for all non-health endpoints (once implemented)
- Input validation on every API boundary

### 5. Explainability Matters

ML predictions must be interpretable. SHAP explanations are a first-class feature, not an afterthought.

### 6. Incremental Delivery

Ship working milestones. Each milestone must compile, test, and deploy independently. No half-implemented features in production branches.

## Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API Framework | FastAPI | Async, typed, auto-docs |
| Frontend | Next.js App Router | SSR, TypeScript, ecosystem |
| Database | PostgreSQL | Relational integrity, JSON support |
| ML | Python scikit-learn/XGBoost | Industry standard, SHAP compatible |
| Containerization | Docker | Reproducible environments |
| CI/CD | GitHub Actions | Native integration |

## Quality Gates

Before any PR merges:

1. All tests pass
2. Linters pass with zero errors
3. No hardcoded configuration values
4. Type hints on all Python functions
5. Docstrings on public APIs

## Anti-Patterns to Avoid

- God classes that mix API, business, and data concerns
- Untyped Python functions
- Frontend API calls without error handling
- Training models without evaluation metrics
- Deploying without health checks
