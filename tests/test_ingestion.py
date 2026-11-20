import pandas as pd
import numpy as np
from aavail.ingestion import ingest_frame, ingest_csv, FEATURES


def test_missing_values_are_audited_and_imputed():
    frame = pd.DataFrame({"customer_id":[1,2,3,4],
        "country":["singapore","singapore","united_states","united_states"],
        "age":[25,-35,39,33], "subscriber_type":["aavail_basic",None,"aavail_premium",None],
        "num_streams":[12,None,18,22],"is_subscriber":[False,True,True,False]})
    out, report = ingest_frame(frame)
    assert out[FEATURES].isna().sum().sum() == 0
    assert report["invalid_age_values_converted_to_missing"] == 1
    assert report["missing_after_validation_before_imputation"]["subscriber_type"] == 2
    assert report["missing_after_imputation"]["num_streams"] == 0
    assert list(out["churn"]) == [1,0,0,1]
    assert out.loc[1,"num_streams"] == 12


def test_duplicate_identifiers_and_invalid_labels_drop_rows():
    frame=pd.DataFrame({"customer_id":[1,1,3], "country":["US","US","Singapore"],
        "age":[20,25,34],"subscriber_type":["aavail_basic"]*3,
        "num_streams":[9,10,11],"is_subscriber":[1,0,"???"]})
    out, audit=ingest_frame(frame)
    assert len(out)==1
    assert audit["rows_dropped"]==2
    assert audit["duplicate_customer_ids_removed"]==1


def test_required_columns_checked():
    try:ingest_frame(pd.DataFrame({"age":[20]}))
    except ValueError as exc:assert "Missing required" in str(exc)
    else:raise AssertionError("must reject invalid data")


def test_training_fixture_has_no_remaining_missing_fields(trained_workspace):
    out, audit=ingest_csv(trained_workspace["data"])
    assert len(out)==480
    assert audit["invalid_age_values_converted_to_missing"]>0
    assert sum(audit["missing_after_imputation"].values())==0
    assert len(out["churn"].unique())==2
