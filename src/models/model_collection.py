from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

def build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def build_model_pipelines(numeric_features, categorical_features):
    models = {}

    models["logistic_regression"] = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features,
                scale_numeric=True,
            ),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
            ),
        ),
    ])

    models["random_forest"] = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    models["gradient_boosting"] = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    models["xgboost"] = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features,
                scale_numeric=False,
            ),
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )
        ),
    ])

    return models