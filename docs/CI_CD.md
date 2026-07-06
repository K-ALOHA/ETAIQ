# CI/CD Pipeline

## Architecture

```
Developer Push / Pull Request
          │
          ▼
   GitHub Actions
    ┌────────────────────────────────────┐
    │                                    │
    ▼                                    ▼
Backend job                       Frontend job
  pip cache (dev.txt)               npm cache (package-lock.json)
  ruff check                        eslint
  pytest --tb=short -q              tsc --noEmit
                                    next build
                                    vitest --run
    │                                    │
    └──────────────┬─────────────────────┘
                   │ both jobs must pass
                   ▼
          Render auto-deploy          Vercel auto-deploy
          (push to main only)         (push to main only)
```

## Workflow file

`.github/workflows/ci.yml`

Triggers on every push and every pull request to any branch.

## Jobs

### Backend

| Step | Command | Purpose |
|---|---|---|
| Install | `pip install -r requirements/dev.txt` | Installs ruff, pytest, httpx, pytest-cov. ML deps are excluded intentionally. |
| Lint | `ruff check .` | Enforces code style and catches common errors. |
| Test | `pytest --tb=short -q` | Runs the full API test suite under `backend/tests/`. |

`dev.txt` is used as the sole install target. It transitively includes `prod.txt` → `base.txt`, but **not** `ml.txt`. This keeps the install under 30 seconds. The ML stack (xgboost, shap, catboost, lightgbm) is not needed to run the API tests.

### Frontend

| Step | Command | Purpose |
|---|---|---|
| Install | `npm ci` | Reproducible install from `package-lock.json`. |
| Lint | `npm run lint` | Runs `eslint` via `eslint.config.mjs`. |
| Typecheck | `npx tsc --noEmit` | Full TypeScript type check without emitting files. |
| Build | `npm run build` | Production Next.js build. Fails if any page has a build error. |
| Test | `npm run test -- --run` | Runs Vitest in non-watch (CI) mode. |

`NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_API_BASE` are both set during the build step so `next.config.ts` resolves the rewrite destination deterministically.

## Caching

| Cache | Key input | Saved path |
|---|---|---|
| pip | `backend/requirements/dev.txt` | `~/.cache/pip` (managed by `actions/setup-python`) |
| npm | `frontend/package-lock.json` | `~/.npm` (managed by `actions/setup-node`) |

Cache hits skip the install step entirely on repeat runs of the same dependency set.

## Quality gates

The pipeline fails if any of the following occur:

- `ruff check` reports a lint violation
- `pytest` reports a test failure or error
- `eslint` reports a lint error
- `tsc --noEmit` reports a type error
- `next build` fails
- `vitest --run` reports a test failure

Render and Vercel will not receive a deploy trigger from a broken commit because both platforms are configured to deploy on push to `main`, and a failed CI run does not block the push — but the broken state is immediately visible in the Actions tab. To enforce a hard gate, enable branch protection on `main` and require the CI status checks to pass before merging.

## Deployment

### Render (backend)

Render is configured as a Docker Web Service pointing at `docker/Dockerfile.backend`. It deploys automatically on push to `main`. The CI pipeline does not interact with Render directly. No Render-specific secrets are required in GitHub.

The Dockerfile installs `requirements/dev.txt` (which includes the full stack). This is intentional for the Docker image — the CI job deliberately avoids the ML deps to keep CI fast.

### Vercel (frontend)

Vercel is connected to the GitHub repository and deploys automatically on push to `main`. The CI pipeline does not interact with Vercel directly. Environment variables (`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_API_BASE`) must be set in the Vercel project settings dashboard — they are not sourced from GitHub Secrets.

## Required GitHub Secrets

No secrets are required for the CI pipeline to run. The pipeline performs lint, typecheck, build, and test only — it does not deploy, push images, or call external APIs.

If you add a deployment step in the future, the following secrets would be needed:

| Secret | Used for |
|---|---|
| `RENDER_API_KEY` | Triggering a Render deploy via the Render API |
| `VERCEL_TOKEN` | Triggering a Vercel deploy via the Vercel CLI |

## How to re-run a failed job

1. Open the repository on GitHub.
2. Click **Actions** in the top navigation.
3. Select the failed workflow run.
4. Click **Re-run failed jobs** (top right of the run summary page).

To re-run all jobs regardless of status, click **Re-run all jobs**.

## How to debug a failing step

1. Click the failed job in the workflow run summary.
2. Expand the failed step to read the full log output.
3. For `pytest` failures, the `--tb=short` flag prints a condensed traceback. Remove `-q` locally and add `-v` for verbose output.
4. For `tsc` failures, the error message includes the file path, line number, and type error description.
5. For `next build` failures, the build log includes the page route and the specific error.
6. To reproduce locally:

```bash
# Backend
cd backend
pip install -r requirements/dev.txt
ruff check .
pytest --tb=short -q

# Frontend
cd frontend
npm ci
npm run lint
npx tsc --noEmit
npm run build
npm run test -- --run
```
