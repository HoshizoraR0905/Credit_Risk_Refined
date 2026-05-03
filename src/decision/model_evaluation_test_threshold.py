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


def evaluate_decision(
    y_true,
    y_prob,
    loan_amount,
    interest_rate,
    threshold,
    threshold_type,
    model_name,
    policy_name,
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    loan_amount = np.asarray(loan_amount)
    interest_rate = np.asarray(interest_rate)

    # If interest rate is accidentally passed as percent, convert to decimal
    if np.nanmax(interest_rate) > 1:
        interest_rate = interest_rate / 100

    if threshold_type == "prob":
        # y_pred = 1 means reject / predicted default
        y_pred = (y_prob >= threshold).astype(int)

    elif threshold_type == "EL":
        expected_cost_score = (
            y_prob * loan_amount
            - (1 - y_prob) * loan_amount * interest_rate
        )
        y_pred = (expected_cost_score >= threshold).astype(int)

    else:
        raise ValueError("threshold_type must be either 'prob' or 'EL'.")

    approved_mask = y_pred == 0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    interest_income = (
        ((1 - y_true) * loan_amount * interest_rate)[approved_mask].sum()
    )

    default_loss = (
        (y_true * loan_amount)[approved_mask].sum()
    )

    realized_net_cost = default_loss - interest_income
    realized_profit = -realized_net_cost

    return {
        "policy": policy_name,
        "model": model_name,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "approval_rate": approved_mask.mean(),
        "rejection_rate": 1 - approved_mask.mean(),
        "default_rate_among_approved": (
            y_true[approved_mask].mean() if approved_mask.sum() > 0 else np.nan
        ),
        "interest_income": interest_income,
        "default_loss": default_loss,
        "realized_net_cost": realized_net_cost,
        "realized_profit": realized_profit,
        "realized_net_cost_1e5": realized_net_cost / 1e5,
        "realized_profit_1e5": realized_profit / 1e5,
    }


def load_threshold_report(path, policy_name, threshold_type):
    df = pd.read_csv(path)
    df["policy"] = policy_name
    df["threshold_type"] = threshold_type
    return df


def main():
    _, _, test_df = load_train_test_split()

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    loan_amount = X_test["loan_amnt"]
    interest_rate = X_test["loan_int_rate"].fillna(X_test["loan_int_rate"].median()) / 100

    threshold_reports = [
        load_threshold_report(
            REPORTS_DIR / "best_thresholds_f1_prob_summary.csv",
            policy_name="f1_prob_threshold",
            threshold_type="prob",
        ),
        load_threshold_report(
            REPORTS_DIR / "best_thresholds_EL_prob_summary.csv",
            policy_name="EL_prob_threshold",
            threshold_type="prob",
        ),
        load_threshold_report(
            REPORTS_DIR / "best_thresholds_EL_EL_summary.csv",
            policy_name="EL_EL_threshold",
            threshold_type="EL",
        ),
    ]

    thresholds_df = pd.concat(threshold_reports, ignore_index=True)

    model_dir = MODELS_DIR / "tuned"
    if not model_dir.exists():
        model_dir = MODELS_DIR

    results = []

    for model_file in model_dir.glob("*.joblib"):
        model_name = model_file.stem
        print(f"Applying decision thresholds for {model_name}...")

        model = joblib.load(model_file)
        y_prob = model.predict_proba(X_test)[:, 1]

        model_thresholds = thresholds_df[thresholds_df["model"] == model_name]

        if model_thresholds.empty:
            print(f"No threshold found for {model_name}. Skipping.")
            continue

        for _, row in model_thresholds.iterrows():
            result = evaluate_decision(
                y_true=y_test,
                y_prob=y_prob,
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                threshold=row["threshold"],
                threshold_type=row["threshold_type"],
                model_name=model_name,
                policy_name=row["policy"],
            )

            results.append(result)

        # Also include default 0.5 threshold as baseline
        results.append(
            evaluate_decision(
                y_true=y_test,
                y_prob=y_prob,
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                threshold=0.5,
                threshold_type="prob",
                model_name=model_name,
                policy_name="default_prob_threshold_0.5",
            )
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No decision results generated.")
        return

    display_cols = [
        "policy",
        "model",
        "threshold_type",
        "threshold",
        "precision",
        "recall",
        "f1",
        "approval_rate",
        "default_rate_among_approved",
        "fp",
        "fn",
        #"realized_net_cost_1e5",
        "realized_profit_1e5",
    ]

    results_df = results_df.sort_values(
        ["model", "realized_profit"],
        ascending=[True, False],
    )

    output_path = REPORTS_DIR / "test_decision_comparison.csv"
    results_df.to_csv(output_path, index=False, float_format="%.6f")

    print("\nTest-set decision comparison:")
    print(results_df[display_cols])

    print(f"\nSaved test decision comparison to: {output_path}")


if __name__ == "__main__":
    main()