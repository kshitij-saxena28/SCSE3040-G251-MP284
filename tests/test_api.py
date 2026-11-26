"""API tests for the MP284 FastAPI prediction service."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as app_main
from app.main import app, load_model_path


class _DummyRegressor:
    """Lightweight fallback regressor used when models/model.joblib is absent."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 42.5, dtype=float)


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "distance_km": 13.7,
        "tiffins": 56,
        "handovers": 2,
        "wait_at_station_min": 20.9,
        "rider_age": 37.0,
        "temp_c": 17.5,
        "transport": "scooter",
        "route_zone": "Central",
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def client() -> TestClient:
    """Provide a TestClient, injecting a dummy regressor if the model file is absent."""
    previous_model = getattr(app.state, "model", None)
    model_path = load_model_path()
    if not model_path.exists():
        app.state.model = _DummyRegressor()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.state.model = previous_model


def test_health(client: TestClient) -> None:
    """GET /health returns HTTP 200 and {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """POST /predict returns 200, numeric prediction, and logs one JSON line to stdout."""
    response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
    assert not isinstance(body["prediction"], bool)
    assert math.isfinite(body["prediction"])

    stdout_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    log_entry = json.loads(stdout_lines[0])
    assert {"timestamp", "level", "event", "prediction", "latency_ms"}.issubset(
        log_entry.keys()
    )
    assert log_entry["level"] == "INFO"
    assert log_entry["event"] == "prediction"
    assert log_entry["prediction"] == body["prediction"]
    assert isinstance(log_entry["latency_ms"], (int, float))
    assert "rider_age" not in log_entry


@pytest.mark.parametrize("tiffins,temp_c", [(95, 5.0)])
def test_predict_boundary_valid_values(
    client: TestClient, tiffins: int, temp_c: float
) -> None:
    """Boundary values tiffins=95 and temp_c=5.0 satisfy validation rules."""
    response = client.post(
        "/predict", json=_valid_payload(tiffins=tiffins, temp_c=temp_c)
    )
    assert response.status_code == 200


def test_predict_rejects_tiffins_above_limit(client: TestClient) -> None:
    """POST /predict with tiffins=96 fails Pydantic validation with HTTP 422."""
    response = client.post("/predict", json=_valid_payload(tiffins=96))
    assert response.status_code == 422


def test_predict_rejects_temp_below_minimum(client: TestClient) -> None:
    """POST /predict with temp_c=4.9 fails Pydantic validation with HTTP 422."""
    response = client.post("/predict", json=_valid_payload(temp_c=4.9))
    assert response.status_code == 422


def test_predict_with_dummy_injector_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Service still runs cleanly in tests with a dummy regressor when model file is missing."""
    previous_model = getattr(app.state, "model", None)
    monkeypatch.setattr(app_main, "load_model_path", lambda: tmp_path / "missing.joblib")
    app.state.model = _DummyRegressor()

    try:
        with TestClient(app) as test_client:
            response = test_client.post("/predict", json=_valid_payload())
            assert response.status_code == 200
            assert response.json() == {"prediction": 42.5}
    finally:
        app.state.model = previous_model
