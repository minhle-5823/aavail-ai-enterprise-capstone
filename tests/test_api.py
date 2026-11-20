
PAYLOAD={"customers":[
    {"country":"singapore","age":29,"subscriber_type":"aavail_basic","num_streams":13},
    {"country":"united_states","age":32,"subscriber_type":"aavail_premium","num_streams":23},
    {"country":"singapore","age":41,"subscriber_type":"aavail_unlimited","num_streams":8}
]}


def test_health_and_market_summary(api_client):
    assert api_client.get("/health").json()["status"]=="ready"
    all_markets=api_client.get("/markets/summary").json()
    assert len(all_markets["markets"])==2
    singapore=api_client.get("/markets/summary?country=singapore").json()
    assert len(singapore["markets"])==1
    assert singapore["markets"][0]["country"]=="singapore"
    assert singapore["data_mode"]=="demo"


def test_predict_all_countries_and_filter_singapore(api_client):
    both=api_client.post("/predict?country=all",json=PAYLOAD)
    assert both.status_code==200, both.text
    assert both.json()["n_predictions"]==3
    assert {x["country"] for x in both.json()["predictions"]}=={"singapore","united_states"}
    only=api_client.post("/predict?country=singapore",json=PAYLOAD)
    assert only.status_code==200, only.text
    assert only.json()["n_predictions"]==2
    assert all(x["country"]=="singapore" for x in only.json()["predictions"])
    assert len({x["prediction_id"] for x in only.json()["predictions"]})==2


def test_predict_errors_are_explicit(api_client):
    bad=api_client.post("/predict?country=mars",json=PAYLOAD)
    assert bad.status_code==422
    bad2=api_client.post("/predict",json={"customers":[{"country":"singapore","age":23}]})
    assert bad2.status_code==422
    bad3=api_client.post("/predict?country=united_states",json={"customers":[PAYLOAD["customers"][0]]})
    assert bad3.status_code==422


def test_feedback_and_monitor_end_to_end(api_client):
    prediction=api_client.post("/predict",json=PAYLOAD).json()["predictions"][0]
    x=api_client.post("/feedback",json={"prediction_id":prediction["prediction_id"],"actual_churn":1})
    assert x.status_code==200
    assert api_client.get("/monitor").json()["total_feedback_events"]==1
    missing=api_client.post("/feedback",json={"prediction_id":"does-not-exist","actual_churn":0})
    assert missing.status_code==404
