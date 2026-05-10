from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "credit_risk_dataset.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "credit_risk_dataset_cleaned.csv"

RAW_SUMMARY_PATH = PROCESSED_DATA_DIR / "credit_risk_raw_summary.csv"
PROCESSED_SUMMARY_PATH = PROCESSED_DATA_DIR / "credit_risk_processed_summary.csv"

TRAIN_INDEX_PATH = PROCESSED_DATA_DIR / "train_index.csv"
VALD_INDEX_PATH = PROCESSED_DATA_DIR / "vald_index.csv"
TEST_INDEX_PATH = PROCESSED_DATA_DIR / "test_index.csv"

TARGET = "loan_status"

TEST_SIZE = 0.2
VALD_SIZE = 0.2
RANDOM_STATE = 42