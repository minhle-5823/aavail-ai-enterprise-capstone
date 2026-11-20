"""Audited, deterministic ingestion for the public AAVAIL churn dataset.

The target is churn == 1 when is_subscriber == 0/False.
No personal names or customer IDs enter model features.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["country", "age", "subscriber_type", "num_streams"]
REQUIRED = FEATURES + ["is_subscriber"]


def normalize_country(value):
    if pd.isna(value):
        return np.nan
    country = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return {"usa": "united_states", "us": "united_states", "united_states_of_america": "united_states",
            "sg": "singapore"}.get(country, country) or np.nan


def _to_subscribed(value):
    if pd.isna(value):
        return np.nan
    mapping = {"1": 1, "1.0": 1, "true": 1, "yes": 1, "0": 0, "0.0": 0,
               "false": 0, "no": 0}
    return mapping.get(str(value).strip().lower(), np.nan)


def _mode_or(series, fallback):
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else fallback


def _median_or(series, fallback):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if not s.empty else float(fallback)


def ingest_frame(frame: pd.DataFrame, *, impute: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return validated, imputed customer frame and an auditable ingestion report."""
    df = frame.copy()
    df = df.rename(columns={"country_name": "country", "customer_age": "age", "is_subscribed": "is_subscriber"})
    if any(col not in df.columns for col in REQUIRED):
        raise ValueError(f"Missing required columns: {sorted(set(REQUIRED)-set(df.columns))}")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    original_n = len(df)
    missing_as_read = {col: int(df[col].isna().sum()) for col in df.columns}
    df["country"] = df["country"].map(normalize_country)
    df["subscriber_type"] = df["subscriber_type"].astype("string").str.lower().str.strip()
    df["subscriber_type"] = df["subscriber_type"].replace({"nan": pd.NA, "none": pd.NA, "<na>": pd.NA})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["num_streams"] = pd.to_numeric(df["num_streams"], errors="coerce")
    df["is_subscriber"] = df["is_subscriber"].map(_to_subscribed)
    invalid_ages = int(((df["age"] < 0) | (df["age"] > 110)).sum())
    invalid_streams = int((df["num_streams"] < 0).sum())
    df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan
    df.loc[df["num_streams"] < 0, "num_streams"] = np.nan
    invalid_labels = int(df["is_subscriber"].isna().sum())
    missing_countries = int(df["country"].isna().sum())
    df = df.dropna(subset=["country", "is_subscriber"]).copy()
    duplicate_ids = 0
    if "customer_id" in df.columns:
        duplicate_ids = int(df["customer_id"].notna().sum() - df["customer_id"].dropna().nunique())
        df = df.drop_duplicates(subset=["customer_id"], keep="last")
    missing_before_imputation = {col: int(df[col].isna().sum()) for col in FEATURES}
    country_imputation = {}
    if impute and len(df):
        global_defaults = {
            "age": _median_or(df["age"], 30),
            "num_streams": _median_or(df["num_streams"], 0),
            "subscriber_type": str(_mode_or(df["subscriber_type"], "unknown")),
        }
        for country, group in df.groupby("country"):
            stats = {
                "age": _median_or(group["age"], global_defaults["age"]),
                "num_streams": _median_or(group["num_streams"], global_defaults["num_streams"]),
                "subscriber_type": str(_mode_or(group["subscriber_type"], global_defaults["subscriber_type"])),
            }
            country_imputation[str(country)] = stats
            mask = df["country"].eq(country)
            for col, replacement in stats.items():
                df.loc[mask, col] = df.loc[mask, col].fillna(replacement)
    df["is_subscriber"] = df["is_subscriber"].astype(int)
    df["churn"] = 1 - df["is_subscriber"]
    report = {
        "rows_read": original_n,
        "rows_retained": len(df),
        "rows_dropped": original_n-len(df),
        "invalid_age_values_converted_to_missing": invalid_ages,
        "negative_stream_values_converted_to_missing": invalid_streams,
        "invalid_status_rows": invalid_labels,
        "missing_country_rows": missing_countries,
        "duplicate_customer_ids_removed": duplicate_ids,
        "missing_as_read": missing_as_read,
        "missing_after_validation_before_imputation": missing_before_imputation,
        "missing_after_imputation": {col: int(df[col].isna().sum()) for col in FEATURES},
        "imputation_by_country": country_imputation,
        "model_features": FEATURES,
        "excluded_from_model": [c for c in ["customer_id", "customer_name"] if c in df.columns],
        "target_definition": "churn = 1 when is_subscriber is 0/False",
    }
    return df.reset_index(drop=True), report


def ingest_csv(path: str | Path, *, impute: bool = True) -> tuple[pd.DataFrame, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AAVAIL data not found: {p}. See README for the dataset download command.")
    return ingest_frame(pd.read_csv(p), impute=impute)


def write_ingestion_audit(report: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
