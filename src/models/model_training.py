import pandas as pd
import joblib

from src.config import MODELS_DIR, TARGET
from src.data.load_split import load_train_test_split
from src.models.model_collection import build_model_pipelines


def main():
    train_df, _, _ = load_train_test_split()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    models = build_model_pipelines(numeric_features, categorical_features)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_pipeline in models.items():
        print(f"Training {model_name}...")

        model_pipeline.fit(X_train, y_train)

        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model_pipeline, model_path)

        print(f"{model_name} training completed.")
        print(f"Saved {model_name} to {model_path}")
    
    return

if __name__ == "__main__":
    main()