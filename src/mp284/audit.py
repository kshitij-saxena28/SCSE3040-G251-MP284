"""Programmatic data quality audit for MP284 Tiffin Relay Delivery Time."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mp284.clean import standardize_transport

PARAMS_PATH = Path("params.yaml")


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


def _check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    extra_cols = sorted(set(train_df.columns) - set(holdout_df.columns) - {target})
    if not extra_cols:
        return {"present": False, "column": None, "correlation": None}

    best_col = extra_cols[0]
    best_corr = None
    if target in train_df.columns:
        corrs = {
            col: float(train_df[col].corr(train_df[target]))
            for col in extra_cols
            if pd.api.types.is_numeric_dtype(train_df[col])
        }
        if corrs:
            best_col = max(corrs, key=lambda c: abs(corrs[c]))
            best_corr = corrs[best_col]
    return {"present": True, "column": best_col, "correlation": best_corr}


def _check_unit_mixup(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    if "route_zone" not in categorical_features or "distance_km" not in numeric_features:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    train_med = train_df.groupby("route_zone")["distance_km"].median()
    later_med = later_df.groupby("route_zone")["distance_km"].median()
    holdout_med = holdout_df.groupby("route_zone")["distance_km"].median()
    ref_med = (later_med + holdout_med) / 2.0

    for zone, val in train_med.items():
        if zone in ref_med and ref_med[zone] > 0:
            ratio = val / ref_med[zone]
            if ratio > 100:
                factor = int(10 ** round(math.log10(ratio)))
                affected = int((train_df["route_zone"] == zone).sum())
                return {
                    "present": True,
                    "column": "distance_km",
                    "group_value": str(zone),
                    "factor": factor,
                    "affected_rows": affected,
                    "train_median": float(val),
                    "later_median": float(later_med[zone]),
                }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def _check_sentinel_value(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    numeric_features: list[str],
) -> dict[str, Any]:
    for col in numeric_features:
        neg_mask = train_df[col] < 0
        if neg_mask.any():
            sentinel_val = train_df.loc[neg_mask, col].mode().iloc[0]
            count = int((train_df[col] == sentinel_val).sum())
            later_count = int((later_df[col] == sentinel_val).sum())
            holdout_count = int((holdout_df[col] == sentinel_val).sum())
            return {
                "present": True,
                "column": col,
                "value": int(sentinel_val) if float(sentinel_val).is_integer() else float(sentinel_val),
                "count": count,
                "later_count": later_count,
                "holdout_count": holdout_count,
            }
    return {"present": False, "column": None, "value": None, "count": None}


def _check_duplicate_rows(train_df: pd.DataFrame) -> dict[str, Any]:
    dup_exact = int(train_df.duplicated().sum())
    feature_cols = [c for c in train_df.columns if c != "record_id"]
    dup_without_id = int(train_df.duplicated(subset=feature_cols).sum())
    return {
        "present": dup_exact > 0 or dup_without_id > 0,
        "count": dup_exact,
        "count_excluding_id": dup_without_id,
    }


def _check_stuck_sensor(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    numeric_features: list[str],
) -> dict[str, Any]:
    ordered = train_df.sort_values(["date", "record_id"]).reset_index(drop=True)
    for col in numeric_features:
        series = ordered[col]
        run_id = (series != series.shift()).cumsum()
        runs = (
            ordered.assign(_run=run_id)
            .groupby("_run")
            .agg(
                value=(col, "first"),
                count=(col, "size"),
                start_date=("date", "min"),
                end_date=("date", "max"),
                n_dates=("date", "nunique"),
            )
        )
        stuck_runs = runs[(runs["count"] >= 20) & (runs["n_dates"] >= 5)]
        if not stuck_runs.empty:
            top = stuck_runs.sort_values("count", ascending=False).iloc[0]
            later_same = int((later_df[col] == top["value"]).sum())
            return {
                "present": True,
                "column": col,
                "value": float(top["value"]),
                "start_date": str(top["start_date"]),
                "end_date": str(top["end_date"]),
                "count": int(top["count"]),
                "later_occurrences": later_same,
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def _check_target_typos(train_df: pd.DataFrame, later_df: pd.DataFrame, target: str) -> dict[str, Any]:
    median_val = float(train_df[target].median())
    typo_mask = train_df[target] > (5.0 * median_val)
    count = int(typo_mask.sum())
    return {
        "present": count > 0,
        "count": count if count > 0 else None,
        "train_min": float(train_df[target].min()),
        "train_max": float(train_df[target].max()),
        "later_max": float(later_df[target].max()),
    }


def _check_spelling_variants(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, Any]:
    for col in categorical_features:
        raw_levels = train_df[col].nunique(dropna=True)
        later_levels = later_df[col].nunique(dropna=True)
        holdout_levels = holdout_df[col].nunique(dropna=True)
        if col == "transport":
            cleaned_levels = sorted(standardize_transport(train_df[col]).dropna().unique().tolist())
        else:
            cleaned_levels = sorted(train_df[col].astype(str).str.strip().str.lower().unique().tolist())
        if raw_levels > len(cleaned_levels) or raw_levels > later_levels:
            return {
                "present": True,
                "column": col,
                "raw_unique_count": int(raw_levels),
                "later_unique_count": int(later_levels),
                "holdout_unique_count": int(holdout_levels),
                "true_levels": len(cleaned_levels),
                "canonical_values": cleaned_levels,
            }
    return {"present": False, "column": None, "true_levels": None}


def run_audit(params_path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Run all 7 data-quality checks on raw datasets and return structured results."""
    params = load_params(params_path)
    paths = params["paths"]
    target = params["target"]
    numeric_features = params["numeric_features"]
    categorical_features = params["categorical_features"]

    train_path = Path(paths["raw_train"])
    later_path = Path(paths["raw_later"])
    holdout_path = Path(paths["holdout"])

    train_sha256 = compute_sha256(train_path)
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)
    holdout_df = pd.read_csv(holdout_path)

    checks = {
        "leaking_column": _check_leaking_column(train_df, holdout_df, target),
        "unit_mixup": _check_unit_mixup(
            train_df, later_df, holdout_df, numeric_features, categorical_features
        ),
        "sentinel_value": _check_sentinel_value(train_df, later_df, holdout_df, numeric_features),
        "duplicate_rows": _check_duplicate_rows(train_df),
        "stuck_sensor": _check_stuck_sensor(train_df, later_df, numeric_features),
        "target_typos": _check_target_typos(train_df, later_df, target),
        "spelling_variants": _check_spelling_variants(
            train_df, later_df, holdout_df, categorical_features
        ),
    }

    return {
        "train_path": str(train_path),
        "train_sha256": train_sha256,
        "shapes": {
            "train": train_df.shape,
            "later": later_df.shape,
            "holdout": holdout_df.shape,
        },
        "checks": checks,
    }


def main() -> None:
    """Execute the audit and print a formatted summary."""
    report = run_audit()
    checks = report["checks"]

    print(f"SHA-256 ({report['train_path']}): {report['train_sha256']}")
    print(
        f"Shapes -> train: {report['shapes']['train']}, "
        f"later: {report['shapes']['later']}, "
        f"holdout: {report['shapes']['holdout']}"
    )
    print("\n=== MP284 Data Quality Audit Summary (7 Checks) ===")

    c1 = checks["leaking_column"]
    print(
        f"1) leaking_column    : present={c1['present']}, column={c1['column']}, "
        f"corr_with_target={c1['correlation']:.6f}" if c1["correlation"] is not None else
        f"1) leaking_column    : present={c1['present']}, column={c1['column']}"
    )

    c2 = checks["unit_mixup"]
    print(
        f"2) unit_mixup        : present={c2['present']}, column={c2['column']}, "
        f"group_value={c2['group_value']}, factor={c2['factor']} "
        f"(affected_rows={c2.get('affected_rows')})"
    )

    c3 = checks["sentinel_value"]
    print(
        f"3) sentinel_value    : present={c3['present']}, column={c3['column']}, "
        f"value={c3['value']}, count={c3['count']}"
    )

    c4 = checks["duplicate_rows"]
    print(
        f"4) duplicate_rows    : present={c4['present']}, count={c4['count']} "
        f"(excluding_id={c4['count_excluding_id']})"
    )

    c5 = checks["stuck_sensor"]
    print(
        f"5) stuck_sensor      : present={c5['present']}, column={c5['column']}, "
        f"value={c5.get('value')}, window={c5['start_date']}..{c5['end_date']} "
        f"(rows={c5.get('count')})"
    )

    c6 = checks["target_typos"]
    print(
        f"6) target_typos      : present={c6['present']}, count={c6['count']} "
        f"(train_range=[{c6['train_min']}, {c6['train_max']}], later_max={c6['later_max']})"
    )

    c7 = checks["spelling_variants"]
    print(
        f"7) spelling_variants : present={c7['present']}, column={c7['column']}, "
        f"raw_levels={c7.get('raw_unique_count')} -> true_levels={c7['true_levels']} "
        f"{c7.get('canonical_values')}"
    )


if __name__ == "__main__":
    main()
