# ============================================================
# Train LightGBM + compute SHAP beeswarm plots on train/test
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier
import shap


# ============================================================
# Config
# ============================================================

TRAIN_PATH = "./generated/competition_train.csv"
TEST_PATH = "./generated/competition_test.csv"

TARGET_COL = "default"
DROP_COLS = ["row_id", "application_month", "etl_batch_id", "schema_version"]

RANDOM_STATE = 1233
ARTIFACT_DIR = "./lightgbm_shap_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# To keep SHAP plots readable / faster
MAX_TRAIN_SHAP_ROWS = 3000
MAX_TEST_SHAP_ROWS = 3000
MAX_DISPLAY = 25


# ============================================================
# Cleaning
# ============================================================

def clean_education(dfin, assumption=True):
    df = dfin.copy()
    df["education"] = (
        df["education"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    text_map = {
        "uni": "university",
        "university": "university",
        "grad school": "graduate school",
        "graduate school": "graduate school",
        "high school": "high school",
        "hs": "high school",
        "other": "other",
    }
    df["education"] = df["education"].replace(text_map)

    if assumption:
        numeric_map = {
            "1": "high school",
            "2": "university",
            "3": "graduate school",
            "5": "other",
            "0": "other",
            "4": "other",
            "6": "other",
        }
        df["education"] = df["education"].replace(numeric_map)

    return df


def clean_age(dfin):
    df = dfin.copy()

    str_to_value_dict = {
        "--": np.nan,
        "na": np.nan,
        "unknown": np.nan,
        "missing": np.nan,
        "fifty-two": 52,
        "thirty-five": 35,
        "twenty-nine": 29,
        "forty": 40,
    }

    def transform_age(x):
        try:
            return int(x)
        except Exception:
            return str_to_value_dict.get(str(x).strip().lower(), np.nan)

    df["age"] = df["age"].apply(transform_age)
    return df


def clean_data(dfin, assumption=True):
    df = dfin.copy()
    df = clean_education(df, assumption=assumption)
    df = clean_age(df)
    return df


# ============================================================
# Feature engineering
# ============================================================

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def add_engineered_features_no_time(dfin):
    df = dfin.copy()

    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]
    delin_cols = ["pay_0", "pay_3", "pay_6"]

    limit_safe = df["limit_bal"].replace(0, np.nan)
    age_safe = df["age"].replace(0, np.nan)
    cards_safe = df["num_credit_cards"].replace(0, np.nan)

    # Bills
    df["bill_sum"] = df[bill_cols].sum(axis=1)
    df["bill_mean"] = df[bill_cols].mean(axis=1)
    df["bill_std"] = df[bill_cols].std(axis=1)
    df["bill_min"] = df[bill_cols].min(axis=1)
    df["bill_max"] = df[bill_cols].max(axis=1)
    df["bill_range"] = df["bill_max"] - df["bill_min"]

    abs_bill = df[bill_cols].abs()
    df["bill_abs_sum"] = abs_bill.sum(axis=1)
    df["bill_abs_mean"] = abs_bill.mean(axis=1)
    df["bill_abs_std"] = abs_bill.std(axis=1)
    df["bill_abs_max"] = abs_bill.max(axis=1)

    df["bill_neg_count"] = (df[bill_cols] < 0).sum(axis=1)
    df["bill_any_negative"] = (df["bill_neg_count"] > 0).astype(int)

    df["bill_pos_sum"] = df[bill_cols].clip(lower=0).sum(axis=1)
    df["bill_neg_sum"] = df[bill_cols].clip(upper=0).sum(axis=1)
    df["bill_neg_abs_sum"] = df[bill_cols].clip(upper=0).abs().sum(axis=1)

    # Payments
    df["pay_sum"] = df[pay_cols].sum(axis=1)
    df["pay_mean"] = df[pay_cols].mean(axis=1)
    df["pay_std"] = df[pay_cols].std(axis=1)
    df["pay_min"] = df[pay_cols].min(axis=1)
    df["pay_max"] = df[pay_cols].max(axis=1)

    # Utilization-like features
    for i in range(1, 7):
        df[f"util_signed_{i}"] = df[f"bill_amt{i}"] / limit_safe
        df[f"util_abs_{i}"] = df[f"bill_amt{i}"].abs() / limit_safe
        df[f"util_pos_{i}"] = df[f"bill_amt{i}"].clip(lower=0) / limit_safe

    util_signed_cols = [f"util_signed_{i}" for i in range(1, 7)]
    util_abs_cols = [f"util_abs_{i}" for i in range(1, 7)]
    util_pos_cols = [f"util_pos_{i}" for i in range(1, 7)]

    df["util_signed_mean"] = df[util_signed_cols].mean(axis=1)
    df["util_signed_min"] = df[util_signed_cols].min(axis=1)
    df["util_signed_max"] = df[util_signed_cols].max(axis=1)
    df["util_abs_mean"] = df[util_abs_cols].mean(axis=1)
    df["util_abs_max"] = df[util_abs_cols].max(axis=1)
    df["util_pos_mean"] = df[util_pos_cols].mean(axis=1)
    df["util_pos_max"] = df[util_pos_cols].max(axis=1)

    # Payment vs bill
    for i in range(1, 7):
        bill = df[f"bill_amt{i}"]
        pay = df[f"pay_amt{i}"]

        df[f"pay_to_abs_bill_{i}"] = pay / (bill.abs() + 1.0)
        df[f"pay_to_pos_bill_{i}"] = pay / (bill.clip(lower=0) + 1.0)
        df[f"pay_minus_bill_{i}"] = pay - bill
        df[f"pay_gt_abs_bill_{i}"] = (pay > bill.abs()).astype(int)

    pay_to_abs_bill_cols = [f"pay_to_abs_bill_{i}" for i in range(1, 7)]
    pay_to_pos_bill_cols = [f"pay_to_pos_bill_{i}" for i in range(1, 7)]
    pay_minus_bill_cols = [f"pay_minus_bill_{i}" for i in range(1, 7)]
    pay_gt_cols = [f"pay_gt_abs_bill_{i}" for i in range(1, 7)]

    df["pay_to_abs_bill_mean"] = df[pay_to_abs_bill_cols].mean(axis=1)
    df["pay_to_abs_bill_max"] = df[pay_to_abs_bill_cols].max(axis=1)
    df["pay_to_pos_bill_mean"] = df[pay_to_pos_bill_cols].mean(axis=1)
    df["pay_minus_bill_mean"] = df[pay_minus_bill_cols].mean(axis=1)
    df["pay_minus_bill_sum"] = df[pay_minus_bill_cols].sum(axis=1)
    df["pay_gt_abs_bill_frac"] = df[pay_gt_cols].mean(axis=1)

    # Trends
    df["bill_change_1_6"] = df["bill_amt1"] - df["bill_amt6"]
    df["pay_change_1_6"] = df["pay_amt1"] - df["pay_amt6"]

    for i in range(1, 6):
        df[f"bill_diff_{i}_{i+1}"] = df[f"bill_amt{i}"] - df[f"bill_amt{i+1}"]
        df[f"pay_diff_{i}_{i+1}"] = df[f"pay_amt{i}"] - df[f"pay_amt{i+1}"]

    bill_diff_cols = [f"bill_diff_{i}_{i+1}" for i in range(1, 6)]
    pay_diff_cols = [f"pay_diff_{i}_{i+1}" for i in range(1, 6)]

    df["bill_diff_mean"] = df[bill_diff_cols].mean(axis=1)
    df["bill_diff_std"] = df[bill_diff_cols].std(axis=1)
    df["pay_diff_mean"] = df[pay_diff_cols].mean(axis=1)
    df["pay_diff_std"] = df[pay_diff_cols].std(axis=1)

    # Delinquency
    df["late_mean"] = df[delin_cols].mean(axis=1)
    df["late_std"] = df[delin_cols].std(axis=1)
    df["late_min"] = df[delin_cols].min(axis=1)
    df["late_max"] = df[delin_cols].max(axis=1)
    df["late_any"] = (df[delin_cols].gt(0).any(axis=1)).astype(int)
    df["late_count"] = df[delin_cols].gt(0).sum(axis=1)
    df["late_severe_count"] = df[delin_cols].ge(2).sum(axis=1)

    # Structure
    df["limit_per_card"] = df["limit_bal"] / cards_safe
    df["months_per_card"] = df["months_since_first_credit"] / cards_safe
    df["limit_to_age"] = df["limit_bal"] / age_safe
    df["credit_history_per_age"] = df["months_since_first_credit"] / age_safe

    # Global
    df["total_cash_flow"] = df["pay_sum"] - df["bill_sum"]
    df["total_cash_flow_absbill"] = df["pay_sum"] - df["bill_abs_sum"]

    # Transforms
    for col in bill_cols:
        df[f"{col}_signed_log"] = signed_log1p(df[col])

    safe_log_cols = [
        "limit_bal", "months_since_first_credit", "num_credit_cards",
        "bill_abs_sum", "bill_abs_mean", "bill_abs_max",
        "bill_neg_abs_sum", "bill_pos_sum",
        "pay_sum", "pay_mean", "pay_max"
    ]
    for col in safe_log_cols:
        df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


# ============================================================
# Preprocessing
# ============================================================

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe())
            ]), cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


# ============================================================
# SHAP helpers
# ============================================================

def sample_rows(X, max_rows, random_state=1233):
    if len(X) <= max_rows:
        return X.copy()
    return X.sample(n=max_rows, random_state=random_state).copy()


def normalize_binary_shap_output(shap_values):
    """
    SHAP output can differ by version/model.
    Return a 2D array [n_samples, n_features] for the positive class.
    """
    if isinstance(shap_values, list):
        # older behavior for binary classifiers: [class0, class1]
        if len(shap_values) == 2:
            return shap_values[1]
        return shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        # possible shape: (n_samples, n_features, n_classes)
        if shap_values.shape[2] == 2:
            return shap_values[:, :, 1]
        return shap_values[:, :, 0]

    return shap_values


def save_beeswarm_plot(explanation, out_path, title, max_display=25):
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    # Load
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Clean
    train = clean_data(train, assumption=True)
    test = clean_data(test, assumption=True)

    # Engineer
    train_fe = add_engineered_features_no_time(train)
    test_fe = add_engineered_features_no_time(test)

    # Build X / y
    X_train = train_fe.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    y_train = train_fe[TARGET_COL].astype(int)
    X_test = test_fe.drop(columns=DROP_COLS, errors="ignore")

    # Align columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # Preprocess + model
    preprocessor = make_preprocessor(X_train)

    model = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    # Fit
    pipe.fit(X_train, y_train)

    # Save fitted pipeline
    joblib.dump(pipe, os.path.join(ARTIFACT_DIR, "lightgbm_pipeline.joblib"), compress=3)

    # Transform train/test
    X_train_trans = pipe.named_steps["prep"].transform(X_train)
    X_test_trans = pipe.named_steps["prep"].transform(X_test)
    feature_names = pipe.named_steps["prep"].get_feature_names_out()

    # Convert to DataFrame for SHAP readability
    X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # Sample rows for plotting speed/readability
    X_train_plot = sample_rows(X_train_trans_df, MAX_TRAIN_SHAP_ROWS, RANDOM_STATE)
    X_test_plot = sample_rows(X_test_trans_df, MAX_TEST_SHAP_ROWS, RANDOM_STATE)

    # SHAP explainer for tree model
    booster_model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(booster_model)

    # Train SHAP
    shap_values_train = explainer.shap_values(X_train_plot)
    shap_values_train = normalize_binary_shap_output(shap_values_train)

    train_explanation = shap.Explanation(
        values=shap_values_train,
        data=X_train_plot.values,
        feature_names=X_train_plot.columns.tolist()
    )

    # Test SHAP
    shap_values_test = explainer.shap_values(X_test_plot)
    shap_values_test = normalize_binary_shap_output(shap_values_test)

    test_explanation = shap.Explanation(
        values=shap_values_test,
        data=X_test_plot.values,
        feature_names=X_test_plot.columns.tolist()
    )

    # Save beeswarm plots
    save_beeswarm_plot(
        train_explanation,
        os.path.join(ARTIFACT_DIR, "shap_beeswarm_train.png"),
        title="SHAP Beeswarm - Train",
        max_display=MAX_DISPLAY
    )

    save_beeswarm_plot(
        test_explanation,
        os.path.join(ARTIFACT_DIR, "shap_beeswarm_test.png"),
        title="SHAP Beeswarm - Test",
        max_display=MAX_DISPLAY
    )

    # Save mean absolute SHAP importance
    shap_importance = pd.DataFrame({
        "feature": X_train_plot.columns,
        "mean_abs_shap_train": np.abs(shap_values_train).mean(axis=0),
        "mean_abs_shap_test": np.abs(shap_values_test).mean(axis=0),
    }).sort_values("mean_abs_shap_train", ascending=False)

    shap_importance.to_csv(
        os.path.join(ARTIFACT_DIR, "shap_feature_importance.csv"),
        index=False
    )

    # Save transformed feature names
    pd.Series(feature_names, name="feature_name").to_csv(
        os.path.join(ARTIFACT_DIR, "transformed_feature_names.csv"),
        index=False
    )

    print("\nSaved files:")
    print("-", os.path.join(ARTIFACT_DIR, "lightgbm_pipeline.joblib"))
    print("-", os.path.join(ARTIFACT_DIR, "shap_beeswarm_train.png"))
    print("-", os.path.join(ARTIFACT_DIR, "shap_beeswarm_test.png"))
    print("-", os.path.join(ARTIFACT_DIR, "shap_feature_importance.csv"))
    print("-", os.path.join(ARTIFACT_DIR, "transformed_feature_names.csv"))


if __name__ == "__main__":
    main()