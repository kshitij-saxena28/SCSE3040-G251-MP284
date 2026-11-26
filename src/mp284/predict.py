"""Batch prediction script for MP284 holdout evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mp284.clean import clean_dataframe

PARAMS_PATH = Path("params.yaml")


def load_params(params_path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load pipeline parameters from YAML."""
    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict(params_path: Path = PARAMS_PATH) -> Path:
    """Generate holdout predictions using the trained model artifact."""
    params = load_params(params_path)
    paths = params["paths"]

    holdout_path = Path(paths["holdout"])
    model_path = Path(paths["model_path"])
    predictions_path = Path(paths["predictions_path"])

    feature_cols: list[str] = (
        list(params["numeric_features"]) + list(params["categorical_features"])
    )

    model = joblib.load(model_path)
    raw_holdout = pd.read_csv(holdout_path)
    cleaned_holdout = clean_dataframe(raw_holdout, is_train=False)

    preds = model.predict(cleaned_holdout[feature_cols])

    output_df = pd.DataFrame(
        {
            "record_id": raw_holdout["record_id"],
            "prediction": preds,
        }
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(predictions_path, index=False)
    print(f"Saved {len(output_df)} predictions to {predictions_path}.")
    return predictions_path


if __name__ == "__main__":
    predict()
