"""Data cleaning utilities for MP284 Tiffin Relay Delivery Time."""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRANSPORT_PREFIX_MAP = (
    ("bic", "bicycle"),
    ("sco", "scooter"),
    ("loc", "local_train"),
)


def _normalize_transport_value(value: object) -> object:
    """Normalize a single transport string by prefix, preserving nulls."""
    if pd.isna(value):
        return value
    cleaned = str(value).strip().lower()
    for prefix, canonical in _TRANSPORT_PREFIX_MAP:
        if cleaned.startswith(prefix):
            return canonical
    return cleaned


def standardize_transport(series: pd.Series) -> pd.Series:
    """Standardize transport category strings to canonical levels."""
    return series.map(_normalize_transport_value)


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean a raw MP284 dataframe according to audited data-quality rules.

    Args:
        df: Input raw dataframe.
        is_train: Whether the dataframe comes from the historical training split
            where the March-April 2026 stuck sensor window applies.

    Returns:
        A cleaned copy of the dataframe.
    """
    cleaned = df.copy()

    if "delivered_min_from_receipt" in cleaned.columns:
        cleaned = cleaned.drop(columns=["delivered_min_from_receipt"])

    if {"route_zone", "distance_km"}.issubset(cleaned.columns):
        unit_mask = (cleaned["route_zone"] == "Central") & (cleaned["distance_km"] > 100)
        cleaned.loc[unit_mask, "distance_km"] = cleaned.loc[unit_mask, "distance_km"] / 1000.0

    if "rider_age" in cleaned.columns:
        cleaned["rider_age"] = cleaned["rider_age"].replace(-999, np.nan)

    if is_train and {"date", "wait_at_station_min"}.issubset(cleaned.columns):
        dates = pd.to_datetime(cleaned["date"], errors="coerce")
        stuck_mask = (
            (dates >= pd.Timestamp("2026-03-29"))
            & (dates <= pd.Timestamp("2026-04-20"))
            & (cleaned["wait_at_station_min"] == 0.8)
        )
        cleaned.loc[stuck_mask, "wait_at_station_min"] = np.nan

    if "transport" in cleaned.columns:
        cleaned["transport"] = standardize_transport(cleaned["transport"])

    return cleaned
