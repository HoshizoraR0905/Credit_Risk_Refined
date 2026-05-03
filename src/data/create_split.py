import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_DATA_PATH, 
    TRAIN_INDEX_PATH, 
    VALD_INDEX_PATH, 
    TEST_INDEX_PATH, 
    TARGET,
    TEST_SIZE,
    VALD_SIZE,
    RANDOM_STATE
    )

def main():
    df = pd.read_csv(CLEAN_DATA_PATH)

    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=VALD_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    vald_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_SIZE/(VALD_SIZE + TEST_SIZE), 
        random_state=RANDOM_STATE,
        stratify=df.loc[temp_idx, TARGET]
    )

    pd.Series(train_idx, name="index").to_csv(TRAIN_INDEX_PATH, index=False)
    pd.Series(vald_idx, name="index").to_csv(VALD_INDEX_PATH, index=False)
    pd.Series(test_idx, name="index").to_csv(TEST_INDEX_PATH, index=False)

    print(f"Saved train index to: {TRAIN_INDEX_PATH}")
    print(f"Saved validation index to: {VALD_INDEX_PATH}")
    print(f"Saved test index to: {TEST_INDEX_PATH}")


if __name__ == "__main__":
    main()