# Contributing to ETAIQ

Thank you for contributing to the ETAIQ platform. This guide covers development workflow, standards, and pull request expectations.

## Getting Started

1. Clone the repository
2. Copy `.env.example` to `.env` and configure values
3. Set up backend and frontend per [README.md](../README.md)
4. Read `.cursor/coding_rules.md` for code standards

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code |
| `develop` | Integration branch for features |
| `feature/*` | Individual feature branches |
| `fix/*` | Bug fix branches |

## Development Workflow

1. Create a feature branch from `develop`
2. Implement changes with tests
3. Run linters and tests locally
4. Open a pull request against `develop`
5. Address review feedback
6. Merge after CI passes and approval

## Code Standards

### Python (Backend)

- Python 3.11+ with full type hints
- Docstrings on all public functions and classes
- Structured logging via `get_logger(__name__)`
- No hardcoded secrets or configuration values
- Follow clean architecture layer boundaries

### TypeScript (Frontend)

- Strict TypeScript mode enabled
- Functional React components with typed props
- Tailwind CSS for styling
- Colocate components with their routes when page-specific

### General

- Keep commits focused and atomic
- Write descriptive commit messages (imperative mood)
- Update `CHANGELOG.md` for user-facing changes
- Do not commit `.env` files or secrets

## Running Tests

```bash
# Backend
make test                     # or ./scripts/test.sh

# Frontend
make lint
cd frontend && npm run build
```

## Pull Request Checklist

- [ ] Code follows project conventions
- [ ] Tests pass locally
- [ ] No secrets or credentials committed
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for notable changes

## Code Review

- All PRs require at least one approval
- CI must pass before merge
- Address all review comments or explain deferrals
- Prefer small, reviewable PRs over large batches
