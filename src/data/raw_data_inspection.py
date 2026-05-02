import pandas as pd
from pathlib import Path
from inspection_summary import inspect_summary

project_root = Path(__file__).resolve().parents[2]
raw_data_path = project_root / "data" / "raw" 
processed_data_path = project_root / "data" / "processed" 

def main():
    df = pd.read_csv(raw_data_path / "credit_risk_dataset.csv")

    print(inspect_summary(df))
    inspect_summary(df).to_csv(processed_data_path / "credit_risk_raw_summary.csv", index=False)

    return

if __name__ == "__main__":
    main()