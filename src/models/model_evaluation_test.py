import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import MODELS_DIR, REPORTS_DIR, TARGET
from src.data.load_split import load_train_test_split


def evaluate_model_on_test(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
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
        "mean_predicted_pd": y_prob.mean(),
        "actual_default_rate": y_test.mean(),
    }


def main():
    _, _, test_df = load_train_test_split()

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    model_dir = MODELS_DIR / "tuned"
    if not model_dir.exists():
        model_dir = MODELS_DIR

    results = []

    for model_file in model_dir.glob("*.joblib"):
        model_name = model_file.stem
        print(f"Evaluating {model_name} on test set...")

        model = joblib.load(model_file)

        metrics = evaluate_model_on_test(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=0.5,
        )

        metrics["model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No model files found.")
        return

    results_df = results_df[
        [
            "model",
            "threshold",
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
            "mean_predicted_pd",
            "actual_default_rate",
        ]
    ].sort_values("roc_auc", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "test_model_comparison.csv"
    results_df.to_csv(output_path, index=False, float_format="%.6f")

    print("\nTest-set model comparison:")
    print(results_df)

    print(f"\nSaved test model comparison to: {output_path}")


if __name__ == "__main__":
    main()