#Data cleaning

import pandas as pd
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[2]
raw_data_path = project_root / "data" / "raw" 
processed_data_path = project_root / "data" / "processed" 

# functions: 
# remove duplicate rows
# remove invalid rows


def main():
    df = pd.read_csv(raw_data_path / "credit_risk_dataset.csv")

    # remove duplicate rows
    df = df.drop_duplicates()

    # remove invalid rows
    df = df[(df['person_age'] <= 100) & (df['person_emp_length'] <= 80)] 

    # 'cb_person_default_on_file' from  'Y'/'N' to 1/0
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    df.to_csv(processed_data_path / "credit_risk_dataset_cleaned.csv", index=False)

    return

if __name__ == "__main__":
    main()