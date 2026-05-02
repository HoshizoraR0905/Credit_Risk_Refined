import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import MODELS_DIR, REPORTS_DIR, TARGET
from src.data.load_split import load_train_test_split


def search_thresholds(y_true, y_prob, model_name):
    results = []

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        approval_rate = (y_pred == 0).mean()
        rejection_rate = (y_pred == 1).mean()

        approved_mask = y_pred == 0
        if approved_mask.sum() > 0:
            default_rate_among_approved = y_true[approved_mask].mean()
        else:
            default_rate_among_approved = np.nan

        results.append({
            "model": model_name,
            "threshold": round(threshold, 2),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "approval_rate": approval_rate,
            "rejection_rate": rejection_rate,
            "default_rate_among_approved": default_rate_among_approved,
        })

    return pd.DataFrame(results)


def main():
    _, vald_df, _ = load_train_test_split()

    X_vald = vald_df.drop(columns=[TARGET])
    y_vald = vald_df[TARGET]

    model_dir = MODELS_DIR / "tuned"
    if not model_dir.exists():
        model_dir = MODELS_DIR

    all_results = []

    for model_file in model_dir.glob("*.joblib"):
        model_name = model_file.stem
        print(f"Searching thresholds for {model_name}...")

        model = joblib.load(model_file)
        y_prob = model.predict_proba(X_vald)[:, 1]

        threshold_df = search_thresholds(
            y_true=y_vald,
            y_prob=y_prob,
            model_name=model_name,
        )

        all_results.append(threshold_df)

    if not all_results:
        print("No model files found.")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    #output_path = REPORTS_DIR / "threshold_search_results.csv"
    #results_df.to_csv(output_path, index=False)

    #print(f"Saved threshold search results to: {output_path}")

    # Quick summary: best threshold by F1 for each model
    best_by_f1 = (
        results_df
        .sort_values(["model", "f1"], ascending=[True, False])
        .groupby("model")
        .head(1)
    )

    print("\nBest threshold by F1:")
    print(best_by_f1[
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "approval_rate",
            "default_rate_among_approved",
            "fp",
            "fn",
        ]
    ])

    # save best thresholds summary
    best_summary_path = REPORTS_DIR / "best_thresholds_prob_summary.csv"
    best_by_f1[[
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "approval_rate",
            "default_rate_among_approved",
            "fp",
            "fn",
        ]].to_csv(best_summary_path, index=False)


if __name__ == "__main__":
    main()