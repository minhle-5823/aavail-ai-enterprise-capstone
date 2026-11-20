"""Synthetic demo fixture for offline tests and Docker smoke tests only.

These values are randomly generated, NOT historical AAVAIL records.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo_csv(path, n=600, seed=12345):
    rng = np.random.default_rng(seed)
    country = rng.choice(["united_states", "singapore"], n, p=[.68,.32])
    age = rng.integers(18, 64, size=n).astype(float)
    plan = rng.choice(["aavail_basic", "aavail_premium", "aavail_unlimited"], n, p=[.4,.32,.28]).astype(object)
    streams = np.maximum(0, rng.normal(19, 6, n)).round().astype(float)
    risk = -.9 + 1.1*(country=="singapore") + .65*(plan=="aavail_unlimited") + .3*(plan=="aavail_premium") - .045*(streams-15)
    probs = 1/(1+np.exp(-risk))
    churn = rng.binomial(1, probs)
    missing_plan_idx = rng.choice(n, max(3,n//16), replace=False)
    missing_stream_idx = rng.choice(n, max(3,n//20), replace=False)
    plan[missing_plan_idx] = None
    streams[missing_stream_idx] = np.nan
    age[rng.choice(n, max(2, n//40), replace=False)] = -20
    df = pd.DataFrame({"customer_id":np.arange(1,n+1),"is_subscriber":1-churn,
                       "country":country,"age":age,"customer_name":["DEMO_RECORD"]*n,
                       "subscriber_type":plan,"num_streams":streams})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False)
    return path


if __name__=="__main__":
    print(make_demo_csv("data/demo_synthetic.csv"))
