"""FastAPI churn service with market-specific and combined inference."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from .auditlog import append_event, new_prediction_id, read_events
from .ingestion import normalize_country
from .modeling import VERSION, load_artifacts, predict_rows
from .monitoring import monitor

app = FastAPI(title="AAVAIL Churn API", version=VERSION,
              description="Churn-risk inference with auditable logs and feedback monitoring")
app.state.model_dir = Path("models")
app.state.log_dir = Path("logs")


class Customer(BaseModel):
    country: str
    age: float | None
    subscriber_type: str | None
    num_streams: float | None


class PredictionRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(min_length=8, max_length=128)
    actual_churn: int = Field(ge=0, le=1)


@app.get("/")
def root():
    return {"application": "AAVAIL Customer Churn API", "documentation": "/docs", "version": VERSION}


@app.get("/health")
def health():
    exists = (Path(app.state.model_dir) / "model.joblib").exists()
    return {"status": "ready" if exists else "model_not_trained", "model_version": VERSION,
            "model_available": exists}


@app.post("/predict")
def predict(body: PredictionRequest, country: str = Query("all")):
    market = normalize_country(country) if country.strip().lower() != "all" else "all"
    if market not in {"all", "united_states", "singapore"}:
        raise HTTPException(status_code=422, detail="country must be all, united_states, or singapore")
    input_rows = [entry.model_dump() for entry in body.customers]
    for row in input_rows:
        row["country"] = normalize_country(row["country"])
        if row["country"] not in {"united_states", "singapore"}:
            raise HTTPException(status_code=422, detail="Unknown customer country")
    subset = [row for row in input_rows if market == "all" or row["country"] == market]
    if not subset:
        raise HTTPException(status_code=422, detail="No customers for the selected country")
    try:
        model, model_info = load_artifacts(app.state.model_dir)
        outputs = predict_rows(subset, model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for result in outputs:
        result["prediction_id"] = new_prediction_id()
        append_event(app.state.log_dir, "predictions",
                     {"prediction_id": result["prediction_id"], "country": result["country"],
                      "predicted_churn": result["predicted_churn"],
                      "churn_probability": result["churn_probability"],
                      "model_version": model_info["model_version"],
                      "data_mode": model_info["data_mode"]})
    return {"country_filter": market, "n_predictions": len(outputs),
            "model_version": model_info["model_version"], "data_mode": model_info["data_mode"],
            "predictions": outputs}


@app.get("/markets/summary")
def market_summary(country: str = Query("all")):
    market = normalize_country(country) if country.strip().lower() != "all" else "all"
    if market not in {"all", "united_states", "singapore"}:
        raise HTTPException(status_code=422, detail="Unsupported country")
    p = Path(app.state.model_dir) / "metrics.json"
    if not p.exists():
        raise HTTPException(status_code=503, detail="Model metrics not yet available")
    report = json.loads(p.read_text(encoding="utf-8"))
    rows = [r for r in report["market_summary"] if market == "all" or r["country"] == market]
    return {"country_filter": market, "data_mode": report["data_mode"], "markets": rows}


@app.post("/feedback")
def feedback(body: FeedbackRequest):
    known = {e.get("prediction_id") for e in read_events(app.state.log_dir, "predictions")}
    if body.prediction_id not in known:
        raise HTTPException(status_code=404, detail="Unknown prediction_id")
    append_event(app.state.log_dir, "feedback", body.model_dump())
    return {"accepted": True, "prediction_id": body.prediction_id}


@app.get("/monitor")
def monitoring():
    return monitor(app.state.log_dir)
