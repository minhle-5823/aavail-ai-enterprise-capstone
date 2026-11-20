"""Feedback-based performance monitor and operational volume signals."""
from collections import defaultdict
from .auditlog import read_events
from .modeling import scores


def monitor(log_dir, *, min_labeled: int = 5):
    predictions = read_events(log_dir, "predictions")
    feedback = read_events(log_dir, "feedback")
    pred_map = {e["prediction_id"]: e for e in predictions if "prediction_id" in e}
    # Resolve latest feedback on a prediction by id, preserving the log as append-only.
    feedback_map = {e["prediction_id"]: e for e in feedback if "prediction_id" in e}
    by_country = defaultdict(list)
    for key, f in feedback_map.items():
        p = pred_map.get(key)
        if p:
            by_country[p.get("country", "unknown")].append((p, f))
    labeled = [pair for rows in by_country.values() for pair in rows]
    def aggregate(rows):
        if len(rows) < min_labeled:
            return {"status": "insufficient_labeled_outcomes", "labeled_count": len(rows),
                    "min_required": min_labeled}
        y_true = [int(f["actual_churn"]) for p, f in rows]
        y_pred = [int(p["predicted_churn"]) for p, f in rows]
        p_score = [float(p["churn_probability"]) for p, f in rows]
        return {"status": "evaluated", "labeled_count": len(rows), **scores(y_true, y_pred, p_score)}
    return {"total_prediction_events": len(predictions), "total_feedback_events": len(feedback),
            "global": aggregate(labeled),
            "by_country": {country: aggregate(rows) for country, rows in by_country.items()},
            "note": "Performance reflects only labeled feedback. Small or selective samples can be misleading."}
