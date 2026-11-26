"""FastAPI prediction service for MP284 Tiffin Relay Delivery Time."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_MODEL_PATH = "models/model.joblib"
PARAMS_PATH = Path("params.yaml")
FEATURE_COLUMNS: list[str] = [
    "distance_km",
    "tiffins",
    "handovers",
    "wait_at_station_min",
    "rider_age",
    "temp_c",
    "transport",
    "route_zone",
]


class JsonFormatter(logging.Formatter):
    """Format log records as single-line structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record into a JSON string without PII."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        if hasattr(record, "prediction"):
            payload["prediction"] = record.prediction
        if hasattr(record, "latency_ms"):
            payload["latency_ms"] = record.latency_ms
        return json.dumps(payload)


class _DynamicStdoutHandler(logging.StreamHandler):
    """StreamHandler that always emits to the active sys.stdout."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def _configure_logger() -> logging.Logger:
    """Configure standard logging to emit JSON lines exclusively to sys.stdout."""
    service_logger = logging.getLogger("mp284.service")
    service_logger.setLevel(logging.INFO)
    service_logger.propagate = False
    service_logger.handlers.clear()
    handler = _DynamicStdoutHandler()
    handler.setFormatter(JsonFormatter())
    service_logger.addHandler(handler)
    return service_logger


logger = _configure_logger()


class PredictionRequest(BaseModel):
    """Validated input schema for delivery time prediction."""

    distance_km: float
    tiffins: int = Field(..., le=95)
    handovers: int
    wait_at_station_min: float
    rider_age: float
    temp_c: float = Field(..., ge=5)
    transport: str
    route_zone: str


def load_model_path(params_path: Path = PARAMS_PATH) -> Path:
    """Load model_path from params.yaml, defaulting to models/model.joblib."""
    target_path = params_path
    if not target_path.exists():
        repo_params = Path(__file__).resolve().parents[1] / "params.yaml"
        if repo_params.exists():
            target_path = repo_params
        else:
            return Path(DEFAULT_MODEL_PATH)

    with target_path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    paths = params.get("paths")
    if isinstance(paths, dict) and paths.get("model_path"):
        return Path(paths["model_path"])
    if params.get("model_path"):
        return Path(params["model_path"])
    return Path(DEFAULT_MODEL_PATH)


def load_model(model_path: Path | None = None) -> Any:
    """Load the serialized model artifact with joblib if present on disk."""
    resolved = model_path if model_path is not None else load_model_path()
    if not resolved.exists() and not resolved.is_absolute():
        repo_relative = Path(__file__).resolve().parents[1] / resolved
        if repo_relative.exists():
            resolved = repo_relative
    if not resolved.exists():
        return None
    return joblib.load(resolved)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Load the model artifact at startup unless already injected."""
    if getattr(fastapi_app.state, "model", None) is None:
        fastapi_app.state.model = load_model()
    yield


app = FastAPI(title="MP284 Tiffin Relay Delivery Time API", lifespan=lifespan)
app.state.model = None


def get_model() -> Any:
    """Retrieve the active model from app state or load it on demand."""
    loaded = getattr(app.state, "model", None)
    if loaded is None:
        loaded = load_model()
        app.state.model = loaded
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    return loaded


@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    """Predict delivery_min from validated request features and log JSON telemetry."""
    start = time.perf_counter()
    model = get_model()

    features_df = pd.DataFrame([request.model_dump()], columns=FEATURE_COLUMNS)
    pred = float(model.predict(features_df)[0])
    latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

    logger.info(
        "prediction",
        extra={
            "event": "prediction",
            "prediction": pred,
            "latency_ms": latency_ms,
        },
    )
    return {"prediction": pred}
