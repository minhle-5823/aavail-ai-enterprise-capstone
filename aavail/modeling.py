"""Reproducible model comparison, artifact production, and inference."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .ingestion import FEATURES, ingest_csv, write_ingestion_audit

VERSION = "1.0.0"


def make_preprocessor():
    numerics = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categories = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numerical", numerics, ["age", "num_streams"]),
                              ("category", categories, ["country", "subscriber_type"])])


def make_candidates(seed=42):
    return {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([("prep", make_preprocessor()),
                                         ("classifier", LogisticRegression(max_iter=600, random_state=seed, class_weight="balanced"))]),
        "random_forest": Pipeline([("prep", make_preprocessor()),
                                    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=9,
                                                                           min_samples_leaf=3, random_state=seed,
                                                                           n_jobs=1))]),
    }


def scores(y_true, predictions, probabilities):
    return {"accuracy": round(float(accuracy_score(y_true, predictions)), 5),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 5),
            "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 5),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 5),
            "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 5),
            "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 5) if len(set(y_true)) > 1 else None}


def fit_and_compare(frame: pd.DataFrame, *, output_dir: str | Path, seed: int = 42,
                    data_mode: str = "official", audit: dict | None = None):
    """Train/test split, compare three models, persist winner and stakeholder visuals."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if len(frame) < 30 or frame["churn"].nunique() < 2:
        raise ValueError("Need >=30 rows with both churn classes for training and evaluation")
    X, y = frame[FEATURES], frame["churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=seed)
    metrics, fitted = {}, {}
    for label, model in make_candidates(seed).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        metrics[label] = scores(y_test, pred, proba)
        fitted[label] = model
    # Compare non-baseline candidates and prioritize discriminative ability.
    best = max((label for label in metrics if label != "baseline_prior"),
               key=lambda label: (metrics[label]["roc_auc"] or -1, metrics[label]["f1"]))
    joblib.dump(fitted[best], dest / "model.joblib")
    report = {"model_version": VERSION, "data_mode": data_mode, "n_rows": len(frame),
              "split": {"train": len(X_train), "test": len(X_test), "seed": seed},
              "selected_model": best, "metrics": metrics,
              "test_by_country": {}, "model_inputs": FEATURES,
              "note": "Demo results are not factual AAVAIL business measurements" if data_mode == "demo" else
                      "Results based on supplied AAVAIL training dataset; observational evaluation only"}
    for country in sorted(X_test["country"].unique()):
        mask = (X_test["country"] == country)
        actual = y_test.loc[mask]
        if len(actual) < 2:
            continue
        model = fitted[best]
        pred = model.predict(X_test.loc[mask])
        proba = model.predict_proba(X_test.loc[mask])[:, list(model.classes_).index(1)]
        report["test_by_country"][country] = scores(actual, pred, proba)
        report["test_by_country"][country]["n_test"] = int(mask.sum())
    market = []
    for country, group in frame.groupby("country"):
        market.append({"country": country, "n_customers": len(group),
                       "churned": int(group["churn"].sum()), "churn_rate": round(float(group["churn"].mean()), 5),
                       "mean_streams": round(float(group["num_streams"].mean()), 4)})
    report["market_summary"] = market
    (dest / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if audit is not None:
        write_ingestion_audit(audit, dest / "ingestion_audit.json")
    try:
        from .visualize import create_report_plots, create_driver_plots
        create_report_plots(report, audit, dest)
        create_driver_plots(frame, dest, data_mode=data_mode)
    except Exception as exc:
        # A plotting failure must not silently pass in the command-line workflow.
        raise RuntimeError(f"Cannot create evaluation plots: {exc}") from exc
    return report


def train_from_csv(data_path, output_dir, *, data_mode="official"):
    frame, audit = ingest_csv(data_path, impute=True)
    return fit_and_compare(frame, output_dir=output_dir, data_mode=data_mode, audit=audit)


def load_artifacts(model_dir):
    path = Path(model_dir)
    if not (path / "model.joblib").is_file():
        raise FileNotFoundError(f"No trained model at {path}; run python -m scripts.train --data ...")
    return joblib.load(path / "model.joblib"), json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def predict_rows(rows: list[dict], model):
    if not rows:
        return []
    for index, row in enumerate(rows):
        missing = sorted(set(FEATURES)-set(row))
        if missing:
            raise ValueError(f"Row {index} is missing fields: {missing}")
    frame = pd.DataFrame(rows, columns=FEATURES)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["num_streams"] = pd.to_numeric(frame["num_streams"], errors="coerce")
    frame.loc[(frame["age"] < 0) | (frame["age"] > 110), "age"] = np.nan
    frame.loc[frame["num_streams"] < 0, "num_streams"] = np.nan
    probas = model.predict_proba(frame)
    idx = list(model.classes_).index(1)
    probas = probas[:, idx]
    preds = model.predict(frame)
    return [{"country": str(country), "predicted_churn": int(pred),
             "churn_probability": round(float(prob), 5)}
            for country, pred, prob in zip(frame["country"], preds, probas)]
