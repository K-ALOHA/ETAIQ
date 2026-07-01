# ETAIQ Coding Rules

Follow these rules for all code contributions to ETAIQ.

## Python (Backend)

### Type Hints

Every function must have typed parameters and return types.

```python
async def get_prediction(prediction_id: str) -> PredictionResponse:
    ...
```

### Docstrings

All modules, classes, and public functions require Google-style docstrings.

```python
def calculate_eta(features: DeliveryFeatures) -> float:
    """Calculate estimated delivery time from input features.

    Args:
        features: Validated delivery context features.

    Returns:
        float: Predicted ETA in minutes.
    """
```

### Logging

Use structured logging via `get_logger(__name__)`. Never use bare `print()`.

```python
from app.core.logging import get_logger

logger = get_logger(__name__)
logger.info("prediction_created", prediction_id=prediction_id, eta_minutes=eta)
```

### Configuration

Load all configuration from `app.core.config.get_settings()`. Never hardcode URLs, secrets, or paths.

### Error Handling

- Use HTTPException in API layer with appropriate status codes
- Log exceptions with context before raising
- Return consistent error response schemas

### Imports

Order: standard library, third-party, local. Use absolute imports from `app.`.

## TypeScript (Frontend)

### Components

- Use functional components with explicit prop interfaces
- Prefer server components unless client interactivity is required
- Mark client components with `"use client"` directive

### Styling

- Use Tailwind utility classes
- Follow existing color palette (zinc scale)
- Support dark mode where applicable

### API Integration

- Centralize API calls in a dedicated service module (future milestone)
- Use environment variables for API base URL (`NEXT_PUBLIC_API_URL`)
- Handle loading, error, and empty states

## File Naming

| Type | Convention | Example |
|------|------------|---------|
| Python modules | snake_case | `health.py` |
| React components | PascalCase | `PagePlaceholder.tsx` |
| Route pages | lowercase | `page.tsx` |
| Tests | test_ prefix | `test_health.py` |

## Git Conventions

- Branch: `feature/short-description` or `fix/short-description`
- Commits: imperative mood, max 72 characters
- PRs: link to issue, include test plan

## Testing

- Backend: pytest with async support, test API endpoints via httpx
- Frontend: component tests added when logic complexity warrants them
- Minimum: health endpoint test, build verification in CI
