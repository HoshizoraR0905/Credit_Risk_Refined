import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss, # mean squared error for probabilistic predictions 1/n * sum((pi - yi)^2)
    confusion_matrix,
)

from src.config import MODELS_DIR, REPORTS_DIR, TARGET
from src.data.load_split import load_train_test_split


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return metrics

def main():
    _, _, test_df = load_train_test_split()

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    results = []

    for model_file in MODELS_DIR.glob("*.joblib"):
        model_name = model_file.stem

        print(f"Evaluating {model_name}...")

        model = joblib.load(model_file)

        metrics = evaluate_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=0.5,
        )

        metrics["model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No model files found. Run training first.")
        return

    results_df = results_df[
        [
            "model",
            "roc_auc",
            "average_precision",
            "brier_score",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]

    results_df = results_df.sort_values("roc_auc", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nModel comparison:")
    print(results_df)

    print(f"\nSaved model comparison to: {output_path}")


if __name__ == "__main__":
    main()