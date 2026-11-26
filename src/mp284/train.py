"""Model training and MLflow experiment tracking for MP284 Tiffin Relay Delivery Time."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mp284.clean import clean_dataframe

PARAMS_PATH = Path("params.yaml")
EXPECTED_TRAIN_SHA256 = (
    "68b13118b52f3abb1a1749e11daf4c742518f2e6728dfc0a9f0b8e59c3a4e104"
)
RANDOM_STATE = 42


def load_params(params_path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load pipeline parameters from YAML."""
    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def get_candidate_models() -> dict[str, tuple[Any, dict[str, Any]]]:
    """Return candidate regression estimators and their logged hyperparameters."""
    return {
        "ridge": (
            Ridge(alpha=1.0),
            {"alpha": 1.0},
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {"n_estimators": 200, "max_depth": 12, "random_state": RANDOM_STATE},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 4,
                "random_state": RANDOM_STATE,
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=RANDOM_STATE,
            ),
            {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": RANDOM_STATE,
            },
        ),
    }


def _log_sklearn_model(pipeline: Pipeline, input_example: pd.DataFrame) -> Any:
    """Log an sklearn Pipeline to MLflow, falling back to cloudpickle if needed."""
    try:
        return mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
        )
    except Exception:
        return mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
            input_example=input_example,
        )


def update_findings_tracking(
    findings_path: Path,
    experiment_name: str,
    best_run_id: str,
    registered_model_name: str,
    champion_version: int,
) -> None:
    """Update only the tracking section of reports/findings.yaml, preserving all else."""
    content = findings_path.read_text(encoding="utf-8")
    tracking_pattern = re.compile(
        r"(^tracking:\r?\n)(.*?)(?=^\S|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = tracking_pattern.search(content)
    if not match:
        raise ValueError(f"Could not find 'tracking:' section in {findings_path}")

    newline = "\r\n" if "\r\n" in content else "\n"
    new_tracking_body = (
        f'  experiment_name: "{experiment_name}"{newline}'
        f'  best_run_id: "{best_run_id}"        # the MLflow run id of your v1 model{newline}'
        f'  registered_model_name: "{registered_model_name}"{newline}'
        f"  champion_version: {champion_version}{newline}{newline}"
    )

    updated = (
        content[: match.start(2)]
        + new_tracking_body
        + content[match.end(2) :]
    )
    findings_path.write_text(updated, encoding="utf-8")


def train(params_path: Path = PARAMS_PATH) -> None:
    """Run training experiments, register the champion model, and save artifacts."""
    params = load_params(params_path)
    paths = params["paths"]

    raw_train_path = Path(paths["raw_train"])
    findings_path = Path(paths["findings"])
    model_path = Path(paths["model_path"])

    target_col: str = params["target"]
    numeric_features: list[str] = params["numeric_features"]
    categorical_features: list[str] = params["categorical_features"]
    feature_cols = numeric_features + categorical_features

    tracking_uri: str = params["tracking_uri"]
    experiment_name: str = params["experiment_name"]
    registered_model_name: str = params["registered_model_name"]

    data_sha256 = compute_sha256(raw_train_path)
    if data_sha256 != EXPECTED_TRAIN_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {raw_train_path}: {data_sha256} "
            f"(expected {EXPECTED_TRAIN_SHA256})"
        )

    raw_df = pd.read_csv(raw_train_path)
    cleaned_df = clean_dataframe(raw_df, is_train=True)

    X = cleaned_df[feature_cols]
    y = cleaned_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    best_run_id: str | None = None
    best_model_uri: str | None = None
    best_pipeline: Pipeline | None = None
    best_val_mae = float("inf")
    best_name = ""

    for run_name, (estimator, hyperparams) in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features),
                ),
                ("regressor", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        val_preds = pipeline.predict(X_val)
        val_mae = float(mean_absolute_error(y_val, val_preds))

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", type(estimator).__name__)
            mlflow.log_param("split_random_state", RANDOM_STATE)
            mlflow.log_param("val_size", 0.2)
            mlflow.log_params(hyperparams)
            mlflow.log_metric("val_mae", val_mae)

            model_info = _log_sklearn_model(pipeline, input_example=X_train.head(5))

            print(f"[{run_name}] run_id={run.info.run_id} val_mae={val_mae:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_run_id = run.info.run_id
                best_model_uri = model_info.model_uri
                best_pipeline = pipeline
                best_name = run_name

    if best_run_id is None or best_model_uri is None or best_pipeline is None:
        raise RuntimeError("No training runs succeeded.")

    mv = mlflow.register_model(best_model_uri, registered_model_name)
    client = MlflowClient()
    client.set_registered_model_alias(
        registered_model_name, "champion", str(mv.version)
    )

    final_pipeline = clone(best_pipeline)
    final_pipeline.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)

    update_findings_tracking(
        findings_path=findings_path,
        experiment_name=experiment_name,
        best_run_id=best_run_id,
        registered_model_name=registered_model_name,
        champion_version=int(mv.version),
    )

    print(
        f"Best model: {best_name} (run_id={best_run_id}, val_mae={best_val_mae:.4f}), "
        f"registered as {registered_model_name} v{mv.version} (@champion), "
        f"saved refitted pipeline to {model_path}."
    )


if __name__ == "__main__":
    train()
