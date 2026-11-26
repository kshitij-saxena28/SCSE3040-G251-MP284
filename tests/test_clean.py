"""Unit tests for data cleaning utilities in src.mp284.clean."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mp284.clean import clean_dataframe


def _build_synthetic_df() -> pd.DataFrame:
    """Build a small synthetic dataframe covering all audited cleaning rules."""
    return pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3", "R4"],
            "date": ["2026-03-01", "2026-04-05", "2026-05-01", "2026-05-02"],
            "distance_km": [13700.0, 12.5, 8400.0, 150.0],
            "tiffins": [40, 55, 25, 30],
            "handovers": [2, 3, 1, 2],
            "wait_at_station_min": [5.0, 0.8, 12.0, 7.5],
            "rider_age": [-999, 34, -999, 41],
            "temp_c": [22.0, 19.5, 25.0, 18.0],
            "transport": ["Bicycle", "bic.", "local train", "sco."],
            "route_zone": ["Central", "Central", "Central", "Suburb"],
            "delivered_min_from_receipt": [48.2, 39.0, 31.5, 28.0],
        }
    )


def test_clean_dataframe_drops_leaking_column() -> None:
    """delivered_min_from_receipt must be dropped during cleaning."""
    raw = _build_synthetic_df()
    cleaned = clean_dataframe(raw, is_train=False)

    assert "delivered_min_from_receipt" not in cleaned.columns
    assert "delivered_min_from_receipt" in raw.columns


def test_clean_dataframe_fixes_central_unit_mixup() -> None:
    """Central rows with distance_km > 100 are divided by 1000; others remain unchanged."""
    raw = _build_synthetic_df()
    cleaned = clean_dataframe(raw, is_train=False)

    assert np.isclose(cleaned.loc[0, "distance_km"], 13.7)
    assert np.isclose(cleaned.loc[1, "distance_km"], 12.5)
    assert np.isclose(cleaned.loc[2, "distance_km"], 8.4)
    assert np.isclose(cleaned.loc[3, "distance_km"], 150.0)


def test_clean_dataframe_replaces_sentinel_rider_age_with_nan() -> None:
    """rider_age sentinel value -999 becomes NaN while valid ages stay intact."""
    raw = _build_synthetic_df()
    cleaned = clean_dataframe(raw, is_train=False)

    assert pd.isna(cleaned.loc[0, "rider_age"])
    assert cleaned.loc[1, "rider_age"] == 34
    assert pd.isna(cleaned.loc[2, "rider_age"])
    assert cleaned.loc[3, "rider_age"] == 41


def test_clean_dataframe_normalizes_transport_variants() -> None:
    """Messy transport labels normalize to bicycle, local_train, and scooter."""
    raw = _build_synthetic_df()
    cleaned = clean_dataframe(raw, is_train=False)

    assert cleaned["transport"].tolist() == [
        "bicycle",
        "bicycle",
        "local_train",
        "scooter",
    ]
