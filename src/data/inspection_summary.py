import pandas as pd
import numpy as np

def inspect_summary(df: pd.DataFrame) -> pd.DataFrame:

    summary = pd.DataFrame({
        "data_type": df.dtypes,
        "num_unique": df.nunique(),
        "num_missing": df.isnull().sum(),
        #"percent_missing": df.isnull().mean() * 100
        "highest_freq": df.apply(lambda x: x.value_counts().idxmax()),
        "highest_val": [df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else np.nan for col in df.columns],
        "lowest_val": [df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else np.nan for col in df.columns],
        "mean": [df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else np.nan for col in df.columns],
        "std_dev": [df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else np.nan for col in df.columns]
    })

    return summary.reset_index().rename(columns={"index": "column"})
