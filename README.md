# Fraud Detection Engine

Real-time credit card fraud scoring. A scikit-learn pipeline is served behind a FastAPI REST API
that returns a fraud probability, a risk bucket, and the threshold the decision was taken at.

The point of this project is the engineering around the model rather than the model itself: what
it takes to make a scorer you could actually operate — configuration that fails loudly, an
artifact you can prove is the one you trained, a limit you have seen reject something, and a
decision you can trace back to the run that produced it.

[![CI](https://github.com/Isma-L154/Fraud-Detection-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Isma-L154/Fraud-Detection-Engine/actions/workflows/ci.yml)

---

## 🔌 API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/health` | Liveness, and whether a model is loaded |
| POST | `/api/v1/predict` | Score one transaction |
| POST | `/api/v1/predict/batch` | Score up to 100 in one request — ~100x cheaper per transaction |
| POST | `/api/v1/metrics` | Prometheus exposition (bearer token) |
| POST | `/api/v1/retrain` | Returns **501**: deliberately not implemented. Train offline |

All endpoints except `/health` require `X-API-Key`.

```jsonc
// POST /api/v1/predict
{"V1": -1.36, "V2": -0.07, /* ... V28 ... */, "Amount": 149.62}

// 200
{"is_fraud": false, "fraud_probability": 0.0123, "risk_level": "LOW",
 "model_version": "58d6b317487b465a916f7e2b02e8026f", "decision_threshold": 0.3}
```

`model_version` is the MLflow run id of the training that produced the loaded artifact, and
`decision_threshold` is the operating point the decision was taken at — together they make a
disputed decision reconstructable.

---

## 🧠 How it is put together

```text
fraud-detection-engine/
├── app/
│   ├── api/          routes, middleware (security headers, body limit, request id, metrics)
│   ├── core/         configuration, logging, risk rules, artifact integrity
│   ├── ml/           the scorer, its protocol, the artifact format
│   └── schemas/      the request and response contracts
├── scripts/          dataset fetch and a synthetic sample generator
├── notebooks/train.py
├── tests/            200 tests, 100% line coverage of app/
└── Dockerfile, docker-compose.yml, requirements*.txt
```

Handlers are thin: they validate, delegate and format. Inference lives in `app/ml/`, business
rules in `app/core/`, and the contract in `app/schemas/`. The scorer reaches handlers through a
FastAPI dependency against a `TransactionScorer` protocol, so a different estimator or a remote
scorer can replace it without touching a route.

**Stack:** FastAPI · scikit-learn · pydantic-settings · slowapi · prometheus-client · MLflow
(training only) · Docker

---

## 🔒 Security

Reviewed against seven baseline controls. Each is implemented or ruled out in writing.

| Control | State |
|---|---|
| Secrets in environment variables | Typed settings, validated at import; the app refuses to start on a missing or invalid value |
| CORS restricted to known origins | Allowlist from configuration, wildcard rejected, and a test proves a disallowed origin is refused |
| Backend validation | 29 typed fields with bounds and `extra: "forbid"`; `Amount` rejects rather than silently rounding |
| Input sanitisation | Mostly N/A — no SQL, shell, templating or uploads. The live surface is artifact deserialisation, below |
| Rate limiting | Per account, charged per transaction so batching cannot bypass it; `X-Forwarded-For` believed only from a configured proxy |
| Row Level Security | **N/A** — no datastore, nothing stored. Re-evaluate on the PR that adds one |
| Content Security Policy | Set on every response, verified with `curl` and in a browser |

**Artifact integrity.** `joblib.load` is `pickle`: loading a model executes its contents. The
service verifies a SHA-256 *before* unpickling and refuses on mismatch. The digest comes from
configuration, not from a file beside the artifact — a checksum an artifact-writer can also
rewrite verifies nothing.

**Authentication.** One API key per consumer, in `X-API-Key`, compared in constant time. Keys
come from configuration and the service refuses to start outside development without them, so it
cannot accidentally come up open. `/health` stays unauthenticated — a load balancer cannot
present a credential. Why keys rather than JWTs or mTLS is recorded in
[`docs/decisions/0002`](docs/decisions/0002-api-keys-for-authentication.md).

---

## 📊 Model performance

Measured on the 56,962-row holdout (98 frauds, 0.172%), at both thresholds — because the two are
not the same, and the historically published figures were the first row:

| threshold | precision | recall | F1 | false positives | missed frauds |
|---|---|---|---|---|---|
| 0.50 — sklearn's default | 0.96 | 0.76 | 0.85 | 3 | 24 |
| **0.30 — what the service uses** | **0.93** | **0.82** | **0.87** | 6 | **18** |

AUC-ROC 0.958. The service operates at 0.30: six more frauds caught for three more manual
reviews, which is the right side of that trade when a review is cheap and a missed fraud is not.

Reproduce with `python notebooks/train.py`, which prints both.

---

## ☁️ Deployment

**None.** There is no AWS infrastructure, no IaC and no pipeline — the Docker image builds in CI
and is never published. The container refuses to start without `ENV`, `CORS_ORIGINS`,
`MODEL_SHA256` and `METRICS_TOKEN`.

How the model artifact reaches a production image is undecided and needs the deployment shape
first — see [#4](../../issues/4).

---

## 📁 Decisions

Choices where the reasoning matters more than the diff are recorded in
[`docs/decisions/`](docs/decisions/) — starting with why there is no drift detection.

---

## 🛠️ Development

All commands run from `fraud-detection-engine/`, with the virtualenv active.

Dependencies are split three ways so the runtime image installs only what the service
imports: `requirements.txt` (runtime), `requirements-train.txt` (adds MLflow for
`notebooks/train.py`), `requirements-dev.txt` (adds linting, types and tests). Each includes
the one before it.

### From clone to a served prediction

No Kaggle account and no 150 MB download needed to check the pipeline runs:

```bash
pip install -r requirements-dev.txt
cp .env.example .env                        # fill in ENV and CORS_ORIGINS
python scripts/make_sample_dataset.py       # synthetic data, same schema
python notebooks/train.py                   # prints MODEL_VERSION and MODEL_SHA256
uvicorn app.api.main:app --port 8000
```

The model that produces is **not meaningful** — the sample is noise with a planted signal, and
the artifact records which dataset trained it so this is visible rather than assumed. For a real
model, fetch the real data first:

```bash
python scripts/fetch_dataset.py             # Kaggle CLI, or prints manual steps
```

### Commands

```bash
# Tests
pytest                                    # run the suite
pytest --cov                              # with a coverage report
pytest tests/test_api.py -v               # one file, verbose

# Lint, format, type check
ruff check app/ notebooks/ tests/         # lint
ruff format app/ notebooks/ tests/        # format
ruff format --check app/ notebooks/ tests/  # verify without writing
mypy                                      # type check app/

# Run the API locally
uvicorn app.api.main:app --reload --port 8000
```

No test loads the real model artifact — the pipeline is substituted with a fake, so the suite
runs on a fresh clone without `models/fraud_model.pkl` present.

### Pre-commit hooks

The same checks run before a commit exists, rather than after a pull request is rejected.
Install once per clone, from the repository root:

```bash
pre-commit install
pre-commit run --all-files   # first run, over the whole tree
```

The `mypy` hook resolves from `PATH`, so the virtualenv must be active when committing.

**Windows note.** If `pre-commit` fails while building a hook environment with
`[WinError 206] The filename or extension is too long`, it is hitting the 260-character path
limit — likely with a Microsoft Store Python, whose install path is already very long. Either
enable long paths system-wide, or point virtualenv's cache somewhere short:

```powershell
# One-off for the current shell
$env:VIRTUALENV_APP_DATA = "C:\vecache"

# Or permanently
[Environment]::SetEnvironmentVariable("VIRTUALENV_APP_DATA", "C:\vecache", "User")
```
