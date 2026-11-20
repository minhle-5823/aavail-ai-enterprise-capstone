from aavail.auditlog import append_event
from aavail.monitoring import monitor


def test_monitor_shows_insufficient_status_without_feedback(tmp_path):
    x=monitor(tmp_path)
    assert x["global"]["status"]=="insufficient_labeled_outcomes"
    assert x["total_prediction_events"]==0


def test_monitor_computes_performance_after_five_labels(tmp_path):
    for i in range(7):
        prediction_id=f"unit-test-{i:02}"
        append_event(tmp_path,"predictions",{"prediction_id":prediction_id,"country":"singapore",
                     "predicted_churn":i%2,"churn_probability":.75 if i%2 else .25})
        append_event(tmp_path,"feedback",{"prediction_id":prediction_id,"actual_churn":i%2})
    result=monitor(tmp_path,min_labeled=5)
    assert result["global"]["status"]=="evaluated"
    assert result["global"]["accuracy"]==1
    assert result["by_country"]["singapore"]["f1"]==1
