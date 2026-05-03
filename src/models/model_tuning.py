import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from src.config import MODELS_DIR, REPORTS_DIR, TARGET
from src.data.load_split import load_train_test_split
from src.models.model_collection import build_model_pipelines

def main():
    
    param_grid = {}

    # logistic regression
    param_grid['logistic_regression'] = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"],
    }

    # random forest
    param_grid['random_forest'] = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [10, 20],
        "model__min_samples_split": [20, 50],
    }

    # gradient boosting
    param_grid['gradient_boosting'] = {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 4],
        "model__min_samples_leaf": [10, 20, 50],
        "model__subsample": [0.8, 1.0],
    }

    # xgboost
    param_grid['xgboost'] = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 4],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0],
    }

    train_df, _, _ = load_train_test_split()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.drop(columns=numeric_features).columns.tolist()

    models = build_model_pipelines(numeric_features, categorical_features)

    TUNED_MODELS_DIR = MODELS_DIR / "tuned"
    TUNED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    #MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tuning_results = []

    for model_name, model_pipeline in models.items():
        print(f"\nTuning {model_name}...")

        search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid[model_name],
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=2,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_score = search.best_score_
        best_params = search.best_params_

        model_path = TUNED_MODELS_DIR / f"{model_name}_tuned.joblib"
        joblib.dump(best_model, model_path)

        tuning_results.append({
            "model": model_name,
            "best_cv_roc_auc": best_score,
            "best_params": best_params,
            "model_path": str(model_path),
        })

        cv_results_path = REPORTS_DIR / f"{model_name}_tuning_cv_results.csv"
        pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)

        print(f"{model_name} tuning completed.")
        print(f"Best CV ROC-AUC: {best_score:.4f}")
        print(f"Best params: {best_params}")
        print(f"Saved tuned model to: {model_path}")
        print(f"Saved CV results to: {cv_results_path}")

    tuning_summary = pd.DataFrame(tuning_results)
    tuning_summary_path = REPORTS_DIR / "tuning_summary.csv"
    tuning_summary.to_csv(tuning_summary_path, index=False)

    print(f"\nSaved tuning summary to: {tuning_summary_path}")

if __name__ == "__main__":
    main()