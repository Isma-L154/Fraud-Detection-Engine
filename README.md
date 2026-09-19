# Fraud Detection Engine

Production-style machine learning system for real-time credit card fraud detection.

This project focuses on designing, training, deploying, and monitoring an end-to-end ML system, going beyond a simple model into a production-oriented architecture.

---

## 🚀 Overview

The system exposes a REST API that evaluates financial transactions and returns a fraud probability in real time.

The goal of this project is not just prediction, but demonstrating how machine learning systems are built and maintained in real-world scenarios.

---

## 🧠 Architecture

The system is structured with clear separation of concerns:

- **API Layer** — Handles requests and validation (FastAPI)
- **ML Layer** — Feature transformation and model inference
- **Monitoring Layer** — Drift detection and experiment tracking
- **Infrastructure Layer** — Containerized deployment on AWS

### Stack

- API: FastAPI + Uvicorn  
- ML: scikit-learn (Pipeline with StandardScaler + RandomForest)  
- Tracking: MLflow  
- Monitoring: Evidently  
- Containerization: Docker  
- Cloud: AWS (EC2, S3, ECR)  

---

## 🔌 API Design

The API exposes three main endpoints:

| Method | Endpoint | Purpose |
|--------|----------|--------|
| GET | `/api/v1/health` | Check service and model status |
| POST | `/api/v1/predict` | Evaluate a transaction |
| POST | `/api/v1/retrain` | Trigger model retraining |

### Prediction Flow

1. A transaction is received as JSON  
2. Input is validated using Pydantic  
3. Features are transformed to match training format  
4. Model generates a fraud probability  
5. Response is returned  

---

## 📊 Model Performance

Evaluated on a holdout set from the Credit Card Fraud Detection dataset.

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.958 |
| Precision (fraud) | 0.96 |
| Recall (fraud) | 0.76 |
| F1 (fraud) | 0.85 |

The dataset is highly imbalanced (~0.17% fraud), handled using `class_weight="balanced"`.

---

## ☁️ Deployment

The system follows a simple cloud deployment flow:

- Model trained locally  
- Artifact stored in S3  
- Docker image pushed to ECR  
- EC2 instance pulls and runs the service  

This setup mirrors a lightweight production environment.

---

## 📁 Project Structure

```text
fraud-detection-engine/
├── app/
│   ├── api/
│   │   ├── main.py          # FastAPI app factory + lifespan
│   │   └── routes.py        # Endpoint handlers
│   ├── core/                # Reserved for business logic
│   ├── ml/
│   │   └── model.py         # Singleton model loader + inference
│   └── schemas/
│       └── transaction.py   # Pydantic request/response models
├── models/                  # Model artifacts (not in git)
├── notebooks/
│   └── train.py             # Training pipeline + MLflow logging
├── tests/
│   └── test_api.py          # API integration tests
├── Dockerfile               # Multi-stage build
├── docker-compose.yml       # Local development
└── requirements.txt
```

## 📚 What I Learned

- How to move from a notebook-based model to a production-ready API  
- The importance of input validation and schema design using Pydantic  
- How to structure an ML project with clean architecture principles  
- How to containerize applications using Docker  
- How experiment tracking (MLflow) helps manage model versions  
- Why monitoring (Evidently) is critical for detecting data drift  
- How to deploy an ML service on AWS using EC2, S3, and ECR  
- The challenges of working with imbalanced datasets  

---

## 🛠️ Development

All commands run from `fraud-detection-engine/`, with the virtualenv active.

Dependencies are split three ways so the runtime image installs only what the service
imports: `requirements.txt` (runtime), `requirements-train.txt` (adds MLflow for
`notebooks/train.py`), `requirements-dev.txt` (adds linting, types and tests). Each includes
the one before it.

```bash
# Install runtime + training + development tooling
pip install -r requirements-dev.txt

# Configure. ENV and CORS_ORIGINS are required — the app refuses to start
# without them rather than assuming a default.
cp .env.example .env    # then fill it in

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
