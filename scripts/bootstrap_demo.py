from pathlib import Path
from scripts.generate_demo import make_demo_csv
from aavail.modeling import train_from_csv


def main():
    output=Path("models")
    if (output/"model.joblib").exists():
        print("Existing model; bootstrap not required")
        return
    authentic=Path("data/aavail-target.csv")
    if authentic.exists():
        print("Training on local AAVAIL data")
        train_from_csv(authentic,output,data_mode="official")
    else:
        print("No real dataset found. Building an explicitly labeled SYNTHETIC DEMO model.")
        p=make_demo_csv("data/demo_synthetic.csv")
        train_from_csv(p,output,data_mode="demo")


if __name__=="__main__":main()
