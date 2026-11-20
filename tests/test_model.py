import json
from aavail.modeling import load_artifacts, predict_rows, FEATURES


def test_multiple_models_compared_and_champion_saved(trained_workspace):
    model, metrics=load_artifacts(trained_workspace["models"])
    assert set(metrics["metrics"])=={"baseline_prior","logistic_regression","random_forest"}
    assert metrics["selected_model"] in {"logistic_regression","random_forest"}
    assert metrics["data_mode"]=="demo"
    assert 0 <= metrics["metrics"]["baseline_prior"]["roc_auc"] <= 1
    assert (trained_workspace["models"] / "model_vs_baseline.png").exists()
    assert (trained_workspace["models"] / "missing_values.png").exists()
    assert (trained_workspace["models"] / "market_churn.png").exists()
    assert (trained_workspace["models"] / "eda_churn_factors.png").exists()


def test_model_predicts_for_both_markets(trained_workspace):
    model, info = load_artifacts(trained_workspace["models"])
    inputs=[{"country":"singapore","age":31,"subscriber_type":"aavail_unlimited","num_streams":12},
            {"country":"united_states","age":42,"subscriber_type":"aavail_basic","num_streams":25}]
    result=predict_rows(inputs,model)
    assert len(result)==2
    assert {r["country"] for r in result}=={"united_states","singapore"}
    assert all(0<=r["churn_probability"]<=1 for r in result)
    assert all(r["predicted_churn"] in [0,1] for r in result)


def test_inference_rejects_missing_required_input(trained_workspace):
    model, _=load_artifacts(trained_workspace["models"])
    try:predict_rows([{"country":"singapore"}],model)
    except ValueError as exc:assert "missing fields" in str(exc)
    else:raise AssertionError("invalid input should not be accepted")
