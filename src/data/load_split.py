import pandas as pd

from src.config import CLEAN_DATA_PATH, TRAIN_INDEX_PATH, VALD_INDEX_PATH, TEST_INDEX_PATH


def load_train_test_split():
    df = pd.read_csv(CLEAN_DATA_PATH)

    train_idx = pd.read_csv(TRAIN_INDEX_PATH)["index"]
    vald_idx = pd.read_csv(VALD_INDEX_PATH)["index"]
    test_idx = pd.read_csv(TEST_INDEX_PATH)["index"]

    train_df = df.loc[train_idx].copy()
    vald_df = df.loc[vald_idx].copy()
    test_df = df.loc[test_idx].copy()

    return train_df, vald_df, test_df