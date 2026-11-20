from aavail.auditlog import append_event,read_events,new_prediction_id


def test_separate_append_only_logs_with_isolated_test_directory(tmp_path):
    test_dir=tmp_path/"logs"  # NEVER production ./logs
    e1=append_event(test_dir,"predictions",{"prediction_id":new_prediction_id(),"country":"singapore"})
    append_event(test_dir,"feedback",{"prediction_id":e1["prediction_id"],"actual_churn":1})
    e2=append_event(test_dir,"predictions",{"prediction_id":new_prediction_id(),"country":"united_states"})
    assert len(read_events(test_dir,"predictions"))==2
    assert len(read_events(test_dir,"feedback"))==1
    assert all("timestamp_utc" in x for x in read_events(test_dir,"predictions"))
    assert not (test_dir/"training.jsonl").exists()
    assert e1["prediction_id"] != e2["prediction_id"]


def test_logging_category_allowlist(tmp_path):
    try:append_event(tmp_path,"../../production",{"data":"unsafe"})
    except ValueError:pass
    else:raise AssertionError("unexpected category should fail")


def test_empty_log_is_graceful(tmp_path):
    assert read_events(tmp_path,"predictions")==[]
