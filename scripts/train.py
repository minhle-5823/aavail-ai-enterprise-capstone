"""Usage: python -m scripts.train --data data/aavail-target.csv --output models"""
import argparse
import json
from pathlib import Path
from aavail.modeling import train_from_csv
from aavail.auditlog import append_event


def main():
    parser=argparse.ArgumentParser(description="Train and evaluate AAVAIL churn candidates")
    parser.add_argument("--data",default="data/aavail-target.csv",help="CSV path")
    parser.add_argument("--output",default="models",help="Artifact directory")
    parser.add_argument("--demo",action="store_true",help="Data is synthetic: reports will be visibly labeled demo")
    args=parser.parse_args()
    data=Path(args.data)
    if not data.exists():
        parser.error(f"Dataset not found at {data}. Download the course data or run python -m scripts.generate_demo")
    report=train_from_csv(data,args.output,data_mode="demo" if args.demo else "official")
    append_event(Path(args.output).parent/"logs", "training",
                 {"selected_model":report["selected_model"],"data_mode":report["data_mode"],
                  "n_rows":report["n_rows"],"model_version":report["model_version"]})
    print(json.dumps({"trained":True,"selected_model":report["selected_model"],"metrics":report["metrics"]},indent=2))


if __name__=="__main__":main()
