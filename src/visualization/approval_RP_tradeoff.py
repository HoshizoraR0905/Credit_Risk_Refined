# expected loss and approval rate tradeoff

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import MODELS_DIR, REPORTS_DIR, TARGET, FIGURES_DIR
from src.data.load_split import load_train_test_split

def approval_RP_tradeoff(y_true, y_prob, X_loan_amount, X_interest_rate, model_name, weight=1.0):

    thresholds = np.sort(np.unique(np.asarray(y_prob)))

    print(f"Evaluating {len(thresholds)} thresholds for {model_name}...")
    
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        approved_mask = y_pred == 0
        approval_rate = approved_mask.mean()

        realized_loss = (y_true * X_loan_amount)[approved_mask].sum() - ((1 - y_true) * X_interest_rate * X_loan_amount)[approved_mask].sum()

        results.append({
            "model": model_name,
            "threshold": threshold,
            "approval_rate": approval_rate,
            "realized_profit_1e6": -realized_loss/1e6,
        })

    return pd.DataFrame(results)


def main():
    train_df, _, test_df = load_train_test_split()

    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    loan_amount = X_test["loan_amnt"]
    interest_rate = X_test["loan_int_rate"].fillna(X_train["loan_int_rate"].median()) / 100

    model_dir = MODELS_DIR / "tuned"
    if not model_dir.exists():
        model_dir = MODELS_DIR
    model_files = sorted(model_dir.glob("*_tuned.joblib"))
    #all_results = []

    for model_file in model_files:
        model_name = model_file.stem
        print(f"Evaluating approval rate - realized profit tradeoff for {model_name}...")

        model = joblib.load(model_file)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        tradeoff_df = approval_RP_tradeoff(
            y_true=y_test,
            y_prob=y_prob,
            X_loan_amount=loan_amount,
            X_interest_rate=interest_rate,
            model_name=model_name,
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(
            tradeoff_df["approval_rate"],
            tradeoff_df["realized_profit_1e6"],
            s=12,
            alpha=0.6,
            label="Probability thresholds",
        )
        plt.title(f"Approval Rate vs Realized Profit for {model_name}")
        plt.xlabel("Approval Rate")
        plt.ylabel("Realized Profit (in millions)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = FIGURES_DIR / f"approval_realized_profit_tradeoff_{model_name}.svg"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        #plt.show()
        plt.close()

        print(f"Saved figure to: {output_path}")
        #all_results.append(tradeoff_df)

if __name__ == "__main__":
    main()