# ETAIQ Backend

FastAPI backend for the ETAIQ ETA Prediction Platform.

## Development

From the repository root:

```bash
make setup
make backend
```

Or manually:

```bash
./scripts/setup.sh
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Dependencies

Layered requirements live in `requirements/`:

| File | Purpose |
|------|---------|
| `requirements/base.txt` | API, database, security, observability |
| `requirements/ml.txt` | Machine learning stack |
| `requirements/prod.txt` | Production runtime (base + ml) |
| `requirements/dev.txt` | Development tooling (prod + test/lint) |

```bash
pip install -r requirements/dev.txt    # local development
pip install -r requirements/prod.txt   # production runtime
```

## Testing

```bash
make test
```

Or: `./scripts/test.sh`
