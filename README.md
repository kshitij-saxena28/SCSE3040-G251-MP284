# SCSE3040 MLOps Mini Project — G251

**Group ID:** G251  
**Problem ID:** MP284 — Tiffin Relay Delivery Time (Dehradun)  
**Faculty:** Upasna Joshi  
**Batch:** 76

## Members

| Enrolment Number | Member |
| --- | --- |
| S24CSEU1993 | Hridayansh Agrawal |
| S24CSEU0050 | Kshitij Saxena |
| S24CSEU1989 | Divey Grover |
| S24CSEU1976 | Pratyaksh Manav |

## Goal

Predict tiffin relay delivery time (`delivery_min`) using a reproducible end-to-end MLOps pipeline.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

### 1. Data Versioning (DVC)

`data/raw/train.csv` and `data/raw/later.csv` are git-ignored and tracked with DVC:

```bash
dvc init
dvc add data/raw/train.csv data/raw/later.csv
git add data/raw/*.dvc .gitignore
```

To pull tracked data in a fresh clone:

```bash
dvc pull
```

### 2. Data Audit

Run the programmatic audit across `train.csv`, `later.csv`, and `holdout_features.csv`:

```bash
python -m src.mp284.audit
```

### 3. Training

Run the training pipeline (logs experiments and registers the model in MLflow via `sqlite:///mlflow.db`):

```bash
python -m src.mp284.train
```

### 4. API

Serve predictions using FastAPI and Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Testing

Run the test suite with pytest:

```bash
pytest -v
```
