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


def search_thresholds_f1(y_true, y_prob, X_loan_amount, X_interest_rate, model_name):
    thresholds = np.sort(np.unique(np.asarray(y_prob)))
    print(f"Evaluating {len(thresholds)} thresholds for {model_name}...")
    best_threshold = None
    best_result = None
    best_f1 = -1
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    threshold = best_threshold
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    approval_rate = (y_pred == 0).mean()
    rejection_rate = (y_pred == 1).mean()

    approved_mask = y_pred == 0
    if approved_mask.sum() > 0:
        default_rate_among_approved = y_true[approved_mask].mean()
    else:
        default_rate_among_approved = np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)

    realized_loss = (y_true * X_loan_amount)[approved_mask].sum() - ((1 - y_true) * X_interest_rate * X_loan_amount)[approved_mask].sum()
    best_result = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "default_rate_among_approved": default_rate_among_approved,
        "realized_loss": realized_loss,
    }

    return pd.DataFrame([best_result])

def search_thresholds_EL(y_true, y_prob, X_loan_amount, X_interest_rate, model_name):
    thresholds = np.sort(np.unique(np.asarray(y_prob)))
    expected_loss = y_prob * X_loan_amount - (1 - y_prob) * X_interest_rate * X_loan_amount
    print(f"Evaluating {len(thresholds)} thresholds for {model_name}...")
    best_threshold = None
    best_result = None
    best_realized_loss = np.inf
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        approved_mask = y_pred == 0
        realized_loss = (y_true * X_loan_amount)[approved_mask].sum() - ((1 - y_true) * X_interest_rate * X_loan_amount)[approved_mask].sum()

        if realized_loss < best_realized_loss:
            best_realized_loss = realized_loss
            best_threshold = threshold


    threshold = best_threshold
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    approval_rate = (y_pred == 0).mean()
    rejection_rate = (y_pred == 1).mean()

    approved_mask = y_pred == 0
    if approved_mask.sum() > 0:
        default_rate_among_approved = y_true[approved_mask].mean()
    else:
        default_rate_among_approved = np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)

    realized_loss = (y_true * X_loan_amount)[approved_mask].sum() - ((1 - y_true) * X_interest_rate * X_loan_amount)[approved_mask].sum()
    best_result = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "default_rate_among_approved": default_rate_among_approved,
        "realized_loss": realized_loss,
    }

    return pd.DataFrame([best_result])

def default_threshold(y_true, y_prob, X_loan_amount, X_interest_rate, model_name):
    print(f"Evaluating default threshold (0.5) for {model_name}...")

    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    approval_rate = (y_pred == 0).mean()
    rejection_rate = (y_pred == 1).mean()

    approved_mask = y_pred == 0
    if approved_mask.sum() > 0:
        default_rate_among_approved = y_true[approved_mask].mean()
    else:
        default_rate_among_approved = np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)

    realized_loss = (y_true * X_loan_amount)[approved_mask].sum() - ((1 - y_true) * X_interest_rate * X_loan_amount)[approved_mask].sum()
    default_result = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "default_rate_among_approved": default_rate_among_approved,
        "realized_loss": realized_loss,
    }

    return pd.DataFrame([default_result])


def main():
    _, vald_df, _ = load_train_test_split()

    X_vald = vald_df.drop(columns=[TARGET])
    y_vald = vald_df[TARGET]

    loan_amount = X_vald["loan_amnt"]
    interest_rate = X_vald["loan_int_rate"].fillna(X_vald["loan_int_rate"].median()) / 100

    model_dir = MODELS_DIR / "tuned"
    if not model_dir.exists():
        model_dir = MODELS_DIR

    all_results_f1 = []
    all_results_EL = []
    default_threshold_results = []

    for model_file in model_dir.glob("*.joblib"):
        model_name = model_file.stem
        print(f"Searching thresholds for {model_name}...")

        model = joblib.load(model_file)
        y_prob = model.predict_proba(X_vald)[:, 1]
        print(model_name)
        print("mean PD:", y_prob.mean())
        print("actual default rate:", y_vald.mean())
        print("score summary:")
        print(pd.Series(y_prob * X_vald["loan_amnt"] - (1 - y_prob) * X_vald["loan_int_rate"]/100 * X_vald["loan_amnt"]).describe())
        threshold_f1_df = search_thresholds_f1(
            y_true=y_vald,
            y_prob=y_prob,
            X_loan_amount=loan_amount,
            X_interest_rate=interest_rate,
            model_name=model_name,
        )
        threshold_EL_df = search_thresholds_EL(
            y_true=y_vald,
            y_prob=y_prob,
            X_loan_amount=loan_amount,
            X_interest_rate=interest_rate,
            model_name=model_name,
        )
        default_threshold_df = default_threshold(
            y_true=y_vald,
            y_prob=y_prob,
            X_loan_amount=loan_amount,
            X_interest_rate=interest_rate,
            model_name=model_name,
        )
        all_results_f1.append(threshold_f1_df)
        all_results_EL.append(threshold_EL_df)
        default_threshold_results.append(default_threshold_df)

    if not all_results_f1 and not all_results_EL and not default_threshold_results:
        print("No model files found.")
        return

    results_f1_df = pd.concat(all_results_f1, ignore_index=True)
    results_EL_df = pd.concat(all_results_EL, ignore_index=True)
    results_default_df = pd.concat(default_threshold_results, ignore_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    #output_path = REPORTS_DIR / "threshold_search_results.csv"
    #results_df.to_csv(output_path, index=False)

    #print(f"Saved threshold search results to: {output_path}")

    # Quick summary: best threshold by F1 for each model
    #best_by_f1 = (
    #    results_df
    #    .sort_values(["model", "f1"], ascending=[True, False])
    #    .groupby("model")
    #    .head(1)
    #)
    pd.set_option("display.float_format", "{:.4f}".format)
    results_f1_df["realized_loss_1e5"] = results_f1_df["realized_loss"] / 1e5
    results_EL_df["realized_loss_1e5"] = results_EL_df["realized_loss"] / 1e5
    results_default_df["realized_loss_1e5"] = results_default_df["realized_loss"] / 1e5

    print("\nBest probability threshold by F1:")
    print(results_f1_df[
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
            "realized_loss_1e5",
        ]
    ])

    # save best thresholds summary
    best_summary_path_f1 = REPORTS_DIR / "best_thresholds_f1_prob_summary.csv"
    results_f1_df[[
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "approval_rate",
            "default_rate_among_approved",
            "fp",
            "fn",
            "realized_loss_1e5",
        ]].to_csv(best_summary_path_f1, index=False, float_format="%.6f")
    
    print("\nBest probability threshold by EL:")
    print(results_EL_df[
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
            "realized_loss_1e5",
        ]
    ])

    # save best thresholds summary
    best_summary_path_EL = REPORTS_DIR / "best_thresholds_EL_prob_summary.csv"
    results_EL_df[[
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "approval_rate",
            "default_rate_among_approved",
            "fp",
            "fn",
            "realized_loss_1e5",
        ]].to_csv(best_summary_path_EL, index=False, float_format="%.6f")
    
    print("\nDefault threshold (0.5) results:")
    print(results_default_df[
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
            "realized_loss_1e5",
        ]
    ])

    # save best thresholds summary
    default_summary_path = REPORTS_DIR / "default_thresholds_prob_summary.csv"
    results_default_df[[
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "approval_rate",
            "default_rate_among_approved",
            "fp",
            "fn",
            "realized_loss_1e5",
        ]].to_csv(default_summary_path, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()