# ============================================================
# End-to-end tabular default prediction script
# - drops application_month entirely
# - no time-derived features
# - compares CatBoost / LightGBM / XGBoost
# - evaluates equal-weight and weighted ensembles
# - saves best single model or ensemble + test predictions
# ============================================================

import os
import json
import warnings
import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------
# Optional libraries
# ---------------------------
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
    print("catboost not installed -> CatBoost will be skipped.")

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    print("lightgbm not installed -> LightGBM will be skipped.")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("xgboost not installed -> XGBoost will be skipped.")


# ============================================================
# Config
# ============================================================

TRAIN_PATH = "./generated/competition_train.csv"
TEST_PATH = "./generated/competition_test.csv"

TARGET_COL = "default"
ID_COL = "row_id"
TIME_COL = "application_month"

RANDOM_STATE = 1233
N_SPLITS = 5
ARTIFACT_DIR = "./artifacts_no_time"

os.makedirs(ARTIFACT_DIR, exist_ok=True)


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
    else:
        known_labels = {"high school", "university", "graduate school", "other"}
        df["education"] = df["education"].apply(
            lambda x: x if x in known_labels else f"code_{x}" if str(x).isdigit() else x
        )
        counts = df["education"].value_counts()
        rare_labels = counts[counts < 250].index
        df["education"] = df["education"].replace(rare_labels, "other")

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

def impute_numeric_features(X_train, X_test, strategy="median"):
    """
    Impute numeric columns only, fit on train and apply to train/test.
    
    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    strategy : str
        One of: "none", "median", "knn"
    
    Returns
    -------
    X_train_out, X_test_out : pd.DataFrame
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    if strategy == "none":
        return X_train, X_test

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) == 0:
        return X_train, X_test

    if strategy == "median":
        imputer = SimpleImputer(strategy="median")
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=5, weights="distance")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    return X_train, X_test

# ============================================================
# Feature engineering
# IMPORTANT:
# - application_month is NOT used
# - no derived time features are created
# - bill_amt can be negative, so use sign-aware / abs-safe features
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

    # -------------------------
    # Bill aggregates
    # -------------------------
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

    # -------------------------
    # Payment aggregates
    # -------------------------
    df["pay_sum"] = df[pay_cols].sum(axis=1)
    df["pay_mean"] = df[pay_cols].mean(axis=1)
    df["pay_std"] = df[pay_cols].std(axis=1)
    df["pay_min"] = df[pay_cols].min(axis=1)
    df["pay_max"] = df[pay_cols].max(axis=1)

    # -------------------------
    # Utilization-like features
    # -------------------------
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

    # -------------------------
    # Payment vs bill features
    # -------------------------
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

    # -------------------------
    # Trends
    # -------------------------
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

    # -------------------------
    # Delinquency features
    # -------------------------
    df["late_mean"] = df[delin_cols].mean(axis=1)
    df["late_std"] = df[delin_cols].std(axis=1)
    df["late_min"] = df[delin_cols].min(axis=1)
    df["late_max"] = df[delin_cols].max(axis=1)
    df["late_any"] = (df[delin_cols].gt(0).any(axis=1)).astype(int)
    df["late_count"] = df[delin_cols].gt(0).sum(axis=1)
    df["late_severe_count"] = df[delin_cols].ge(2).sum(axis=1)

    # -------------------------
    # Customer structure features
    # -------------------------
    df["limit_per_card"] = df["limit_bal"] / cards_safe
    df["months_per_card"] = df["months_since_first_credit"] / cards_safe
    df["limit_to_age"] = df["limit_bal"] / age_safe
    df["credit_history_per_age"] = df["months_since_first_credit"] / age_safe

    # -------------------------
    # Cross-feature aggregates
    # -------------------------
    df["total_cash_flow"] = df["pay_sum"] - df["bill_sum"]
    df["total_cash_flow_absbill"] = df["pay_sum"] - df["bill_abs_sum"]

    # -------------------------
    # Safe transforms
    # -------------------------
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
# Feature set builder
# - always remove application_month
# ============================================================

def drop_constant_and_all_missing(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    nunique = X_train.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()

    missing_rate = X_train.isna().mean()
    all_missing_cols = missing_rate[missing_rate >= 1.0].index.tolist()

    to_drop = sorted(set(constant_cols + all_missing_cols))

    X_train = X_train.drop(columns=to_drop, errors="ignore")
    X_test = X_test.drop(columns=to_drop, errors="ignore")

    return X_train, X_test, to_drop


def build_feature_sets(train_df, test_df, target_col=TARGET_COL):
    train_raw = train_df.copy()
    test_raw = test_df.copy()

    train_eng = add_engineered_features_no_time(train_df)
    test_eng = add_engineered_features_no_time(test_df)

    feature_sets = {}

    drop_model_cols = [target_col, TIME_COL, ID_COL, "etl_batch_id", "schema_version"]
    drop_test_cols = [TIME_COL, ID_COL, "etl_batch_id", "schema_version"]

    # -------------------------
    # Base raw features
    # -------------------------
    X_train_raw = train_raw.drop(columns=drop_model_cols, errors="ignore")
    X_test_raw = test_raw.drop(columns=drop_test_cols, errors="ignore")
    X_train_raw, X_test_raw, dropped_raw = drop_constant_and_all_missing(X_train_raw, X_test_raw)

    feature_sets["raw_no_time"] = {
        "X_train": X_train_raw.copy(),
        "X_test": X_test_raw.copy(),
        "dropped_columns": dropped_raw + [c for c in drop_model_cols if c != target_col],
        "imputation": "none",
    }

    X_train_raw_med, X_test_raw_med = impute_numeric_features(X_train_raw, X_test_raw, strategy="median")
    feature_sets["raw_no_time_median"] = {
        "X_train": X_train_raw_med,
        "X_test": X_test_raw_med,
        "dropped_columns": dropped_raw + [c for c in drop_model_cols if c != target_col],
        "imputation": "median",
    }

    X_train_raw_knn, X_test_raw_knn = impute_numeric_features(X_train_raw, X_test_raw, strategy="knn")
    feature_sets["raw_no_time_knn"] = {
        "X_train": X_train_raw_knn,
        "X_test": X_test_raw_knn,
        "dropped_columns": dropped_raw + [c for c in drop_model_cols if c != target_col],
        "imputation": "knn",
    }

    # -------------------------
    # Base engineered features
    # -------------------------
    X_train_eng = train_eng.drop(columns=drop_model_cols, errors="ignore")
    X_test_eng = test_eng.drop(columns=drop_test_cols, errors="ignore")
    X_train_eng, X_test_eng, dropped_eng = drop_constant_and_all_missing(X_train_eng, X_test_eng)

    feature_sets["eng_no_time"] = {
        "X_train": X_train_eng.copy(),
        "X_test": X_test_eng.copy(),
        "dropped_columns": dropped_eng + [c for c in drop_model_cols if c != target_col],
        "imputation": "none",
    }

    X_train_eng_med, X_test_eng_med = impute_numeric_features(X_train_eng, X_test_eng, strategy="median")
    feature_sets["eng_no_time_median"] = {
        "X_train": X_train_eng_med,
        "X_test": X_test_eng_med,
        "dropped_columns": dropped_eng + [c for c in drop_model_cols if c != target_col],
        "imputation": "median",
    }

    X_train_eng_knn, X_test_eng_knn = impute_numeric_features(X_train_eng, X_test_eng, strategy="knn")
    feature_sets["eng_no_time_knn"] = {
        "X_train": X_train_eng_knn,
        "X_test": X_test_eng_knn,
        "dropped_columns": dropped_eng + [c for c in drop_model_cols if c != target_col],
        "imputation": "knn",
    }

    print("Feature sets created:")
    for k, v in feature_sets.items():
        print(f"{k:20s} | train={v['X_train'].shape} | test={v['X_test'].shape} | imputation={v['imputation']}")

    return feature_sets


# ============================================================
# Preprocessing for LightGBM / XGBoost
# ============================================================

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_ohe_preprocessor(X, numeric_impute=True):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    if numeric_impute:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
    else:
        num_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe())
            ]), cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor, num_cols, cat_cols


# ============================================================
# CatBoost preparation
# ============================================================

def prepare_catboost_frames(X, extra_frames=None):
    if extra_frames is None:
        extra_frames = []

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    all_frames = [X.copy()] + [f.copy() for f in extra_frames]

    prepared = []
    for df in all_frames:
        for c in cat_cols:
            df[c] = df[c].fillna("missing").astype(str)
        prepared.append(df)

    return prepared, cat_cols


# ============================================================
# Model factories
# ============================================================

def make_lgbm_model():
    return LGBMClassifier(
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


def make_xgb_model(y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    return XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def make_catboost_model():
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False
    )


# ============================================================
# CV runners
# ============================================================

def cv_pipeline_model(model_name, X, y, X_test, numeric_already_imputed=False):
    if model_name == "lightgbm":
        model = make_lgbm_model()
    elif model_name == "xgboost":
        model = make_xgb_model(y)
    else:
        raise ValueError(f"Unsupported pipeline model: {model_name}")

    preprocessor, num_cols, cat_cols = make_ohe_preprocessor(
        X,
        numeric_impute=not numeric_already_imputed
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        est = clone(pipe)
        est.fit(X_tr, y_tr)

        va_pred = est.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = va_pred
        test_pred += est.predict_proba(X_test)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(fold_auc)

    return {
        "model_type": model_name,
        "feature_columns": list(X.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "fold_scores": fold_scores,
        "mean_auc": float(np.mean(fold_scores)),
        "std_auc": float(np.std(fold_scores)),
        "oof_auc": float(roc_auc_score(y, oof_pred)),
        "oof_pred": oof_pred,
        "test_pred": test_pred,
    }

def cv_catboost_model(X, y, X_test):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        prepared, cat_cols = prepare_catboost_frames(X_tr, [X_va, X_test])
        X_tr_cb, X_va_cb, X_test_cb = prepared

        model = make_catboost_model()
        model.fit(X_tr_cb, y_tr, cat_features=cat_cols)

        va_pred = model.predict_proba(X_va_cb)[:, 1]
        oof_pred[va_idx] = va_pred
        test_pred += model.predict_proba(X_test_cb)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(fold_auc)

    return {
        "model_type": "catboost",
        "feature_columns": list(X.columns),
        "cat_cols": cat_cols,
        "fold_scores": fold_scores,
        "mean_auc": float(np.mean(fold_scores)),
        "std_auc": float(np.std(fold_scores)),
        "oof_auc": float(roc_auc_score(y, oof_pred)),
        "oof_pred": oof_pred,
        "test_pred": test_pred,
    }


# ============================================================
# Full-data fitting for final selected winner
# ============================================================

def fit_full_single_model(model_name, X, y, X_test, numeric_already_imputed=False):
    if model_name == "catboost":
        prepared, cat_cols = prepare_catboost_frames(X, [X_test])
        X_cb, X_test_cb = prepared

        model = make_catboost_model()
        model.fit(X_cb, y, cat_features=cat_cols)

        test_pred = model.predict_proba(X_test_cb)[:, 1]

        artifact = {
            "artifact_type": "single_model",
            "model_type": "catboost",
            "feature_columns": list(X.columns),
            "cat_cols": cat_cols,
            "fitted_model": model,
        }
        return artifact, test_pred

    elif model_name in {"lightgbm", "xgboost"}:
        if model_name == "lightgbm":
            model = make_lgbm_model()
        else:
            model = make_xgb_model(y)

        preprocessor, num_cols, cat_cols = make_ohe_preprocessor(
            X,
            numeric_impute=not numeric_already_imputed
        )

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        pipe.fit(X, y)

        test_pred = pipe.predict_proba(X_test)[:, 1]

        artifact = {
            "artifact_type": "single_model",
            "model_type": model_name,
            "feature_columns": list(X.columns),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "fitted_model": pipe,
        }
        return artifact, test_pred

    else:
        raise ValueError(f"Unknown model_name={model_name}")

# ============================================================
# Ensemble search
# Uses the best single run per model family
# ============================================================

def evaluate_equal_weight_ensembles(selected_runs, y):
    rows = []

    keys = list(selected_runs.keys())
    if len(keys) < 2:
        return pd.DataFrame(rows)

    for r in range(2, len(keys) + 1):
        for combo in itertools.combinations(keys, r):
            weight = 1.0 / len(combo)
            oof_pred = np.zeros(len(y))
            test_pred = np.zeros_like(selected_runs[combo[0]]["test_pred"])

            for key in combo:
                oof_pred += weight * selected_runs[key]["oof_pred"]
                test_pred += weight * selected_runs[key]["test_pred"]

            auc = roc_auc_score(y, oof_pred)
            rows.append({
                "ensemble_name": "equal__" + "__".join(combo),
                "members": list(combo),
                "weights": [weight] * len(combo),
                "oof_auc": float(auc),
                "mean_auc_proxy": float(auc),
                "test_pred": test_pred,
                "oof_pred": oof_pred,
            })

    return pd.DataFrame(rows)


def simplex_weights(n_models, step=0.05):
    vals = np.arange(0, 1 + 1e-9, step)

    if n_models == 2:
        for w1 in vals:
            w2 = 1 - w1
            if w2 >= -1e-9:
                yield np.array([w1, w2])

    elif n_models == 3:
        for w1 in vals:
            for w2 in vals:
                w3 = 1 - w1 - w2
                if w3 >= -1e-9:
                    yield np.array([w1, w2, max(0.0, w3)])


def evaluate_weighted_ensembles(selected_runs, y, step=0.05):
    rows = []
    keys = list(selected_runs.keys())

    if len(keys) < 2:
        return pd.DataFrame(rows)

    if len(keys) > 3:
        raise ValueError("This helper is written for up to 3 selected models.")

    for weights in simplex_weights(len(keys), step=step):
        active = np.sum(weights > 0)
        if active < 2:
            continue

        oof_pred = np.zeros(len(y))
        test_pred = np.zeros_like(selected_runs[keys[0]]["test_pred"])

        for w, key in zip(weights, keys):
            if w == 0:
                continue
            oof_pred += w * selected_runs[key]["oof_pred"]
            test_pred += w * selected_runs[key]["test_pred"]

        auc = roc_auc_score(y, oof_pred)

        rows.append({
            "ensemble_name": "weighted__" + "__".join(
                [f"{k}={w:.2f}" for k, w in zip(keys, weights)]
            ),
            "members": keys,
            "weights": weights.tolist(),
            "oof_auc": float(auc),
            "mean_auc_proxy": float(auc),
            "test_pred": test_pred,
            "oof_pred": oof_pred,
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    # ---------------------------
    # Load and clean
    # ---------------------------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = clean_data(train, assumption=True)
    test = clean_data(test, assumption=True)

    y = train[TARGET_COL].astype(int)

    # ---------------------------
    # Feature sets
    # ---------------------------
    feature_sets = build_feature_sets(train, test, target_col=TARGET_COL)

    print("Feature set summary:")
    for name, obj in feature_sets.items():
        print(
            f"{name:15s} | train={obj['X_train'].shape} | test={obj['X_test'].shape}"
        )

    # ---------------------------
    # Train single models
    # ---------------------------
    base_rows = []
    base_store = {}
    feature_set_names = [
        "raw_no_time",
        "raw_no_time_median",
        "raw_no_time_knn",
        "eng_no_time",
        "eng_no_time_median",
        "eng_no_time_knn",
        ]

    for fs_name in feature_set_names:
        X = feature_sets[fs_name]["X_train"]
        X_test = feature_sets[fs_name]["X_test"]

        if CatBoostClassifier is not None:
            print(f"\nRunning CatBoost on {fs_name} ...")
            res = cv_catboost_model(X, y, X_test)
            run_key = f"catboost__{fs_name}"
            base_store[run_key] = {
                "run_key": run_key,
                "model_type": "catboost",
                "feature_set": fs_name,
                **res
            }
            base_rows.append({
                "run_key": run_key,
                "model_type": "catboost",
                "feature_set": fs_name,
                "mean_auc": res["mean_auc"],
                "std_auc": res["std_auc"],
                "oof_auc": res["oof_auc"],
                "n_features": X.shape[1],
            })
            print(f"OOF AUC: {res['oof_auc']:.6f}")

        numeric_already_imputed = feature_sets[fs_name]["imputation"] in {"median", "knn"}

        if LGBMClassifier is not None:
            print(f"\nRunning LightGBM on {fs_name} ...")
            res = cv_pipeline_model("lightgbm", X, y, X_test, numeric_already_imputed=numeric_already_imputed)
            run_key = f"lightgbm__{fs_name}"
            base_store[run_key] = {
                "run_key": run_key,
                "model_type": "lightgbm",
                "feature_set": fs_name,
                **res
            }
            base_rows.append({
                "run_key": run_key,
                "model_type": "lightgbm",
                "feature_set": fs_name,
                "mean_auc": res["mean_auc"],
                "std_auc": res["std_auc"],
                "oof_auc": res["oof_auc"],
                "n_features": X.shape[1],
            })
            print(f"OOF AUC: {res['oof_auc']:.6f}")

        if XGBClassifier is not None:
            print(f"\nRunning XGBoost on {fs_name} ...")
            res = cv_pipeline_model("xgboost", X, y, X_test, numeric_already_imputed=numeric_already_imputed)
            run_key = f"xgboost__{fs_name}"
            base_store[run_key] = {
                "run_key": run_key,
                "model_type": "xgboost",
                "feature_set": fs_name,
                **res
            }
            base_rows.append({
                "run_key": run_key,
                "model_type": "xgboost",
                "feature_set": fs_name,
                "mean_auc": res["mean_auc"],
                "std_auc": res["std_auc"],
                "oof_auc": res["oof_auc"],
                "n_features": X.shape[1],
            })
            print(f"OOF AUC: {res['oof_auc']:.6f}")

    base_results_df = pd.DataFrame(base_rows).sort_values(
        by=["oof_auc", "mean_auc"], ascending=False
    ).reset_index(drop=True)

    print("\n=== Single-model leaderboard ===")
    print(base_results_df.to_string(index=False))
    base_results_df.to_csv(os.path.join(ARTIFACT_DIR, "base_model_results.csv"), index=False)

    # ---------------------------
    # Best single per model family
    # ---------------------------
    best_single_per_family = {}
    if len(base_results_df) > 0:
        for family in base_results_df["model_type"].unique():
            top_row = (
                base_results_df[base_results_df["model_type"] == family]
                .sort_values(["oof_auc", "mean_auc"], ascending=False)
                .iloc[0]
            )
            key = top_row["run_key"]
            best_single_per_family[key] = base_store[key]

    print("\nSelected runs for ensemble search:")
    for k, v in best_single_per_family.items():
        print(f"{k} -> OOF AUC={v['oof_auc']:.6f}")

    # ---------------------------
    # Evaluate ensembles
    # ---------------------------
    ensemble_frames = []

    equal_df = evaluate_equal_weight_ensembles(best_single_per_family, y)
    if len(equal_df) > 0:
        ensemble_frames.append(equal_df)

    weighted_df = evaluate_weighted_ensembles(best_single_per_family, y, step=0.05)
    if len(weighted_df) > 0:
        ensemble_frames.append(weighted_df)

    if len(ensemble_frames) > 0:
        ensemble_results_df = pd.concat(ensemble_frames, ignore_index=True)
        ensemble_results_df = ensemble_results_df.sort_values("oof_auc", ascending=False).reset_index(drop=True)

        ensemble_results_df[["ensemble_name", "members", "weights", "oof_auc"]].to_csv(
            os.path.join(ARTIFACT_DIR, "ensemble_results.csv"), index=False
        )

        print("\n=== Ensemble leaderboard ===")
        print(
            ensemble_results_df[["ensemble_name", "members", "weights", "oof_auc"]]
            .head(20)
            .to_string(index=False)
        )
    else:
        ensemble_results_df = pd.DataFrame()

    # ---------------------------
    # Choose overall winner
    # ---------------------------
    best_single_row = base_results_df.iloc[0].to_dict()

    if len(ensemble_results_df) > 0 and ensemble_results_df.iloc[0]["oof_auc"] > best_single_row["oof_auc"]:
        winner_type = "ensemble"
        winner_info = ensemble_results_df.iloc[0].to_dict()
    else:
        winner_type = "single"
        winner_info = best_single_row

    print("\n=== BEST OVERALL ===")
    if winner_type == "single":
        print("Winner type : single model")
        print(f"Run key      : {winner_info['run_key']}")
        print(f"Model        : {winner_info['model_type']}")
        print(f"Feature set  : {winner_info['feature_set']}")
        print(f"OOF AUC      : {winner_info['oof_auc']:.6f}")
    else:
        print("Winner type : ensemble")
        print(f"Name         : {winner_info['ensemble_name']}")
        print(f"Members      : {winner_info['members']}")
        print(f"Weights      : {winner_info['weights']}")
        print(f"OOF AUC      : {winner_info['oof_auc']:.6f}")

    # ---------------------------
    # Refit winner on all training data
    # ---------------------------
    if winner_type == "single":
        run_key = winner_info["run_key"]
        run_obj = base_store[run_key]
        fs_name = run_obj["feature_set"]
        model_type = run_obj["model_type"]

        X_full = feature_sets[fs_name]["X_train"]
        X_test_full = feature_sets[fs_name]["X_test"]

        numeric_already_imputed = feature_sets[fs_name]["imputation"] in {"median", "knn"}

        best_artifact, final_test_pred = fit_full_single_model(
            model_name=model_type,
            X=X_full,
            y=y,
            X_test=X_test_full,
            numeric_already_imputed=numeric_already_imputed
        )
        meta = {
            "winner_type": "single",
            "run_key": run_key,
            "model_type": model_type,
            "feature_set": fs_name,
            "oof_auc": float(run_obj["oof_auc"]),
        }

    else:
        members = winner_info["members"]
        weights = winner_info["weights"]

        component_artifacts = []
        final_test_pred = np.zeros(len(test))

        for member_key, weight in zip(members, weights):
            if weight == 0:
                continue

            run_obj = best_single_per_family[member_key]
            fs_name = run_obj["feature_set"]
            model_type = run_obj["model_type"]

            X_full = feature_sets[fs_name]["X_train"]
            X_test_full = feature_sets[fs_name]["X_test"]

            comp_artifact, comp_test_pred = fit_full_single_model(
                model_name=model_type,
                X=X_full,
                y=y,
                X_test=X_test_full
            )

            final_test_pred += weight * comp_test_pred

            component_artifacts.append({
                "member_key": member_key,
                "weight": float(weight),
                "model_type": model_type,
                "feature_set": fs_name,
                "artifact": comp_artifact,
                "oof_auc": float(run_obj["oof_auc"]),
            })

        best_artifact = {
            "artifact_type": "ensemble",
            "components": component_artifacts,
        }

        meta = {
            "winner_type": "ensemble",
            "ensemble_name": winner_info["ensemble_name"],
            "members": members,
            "weights": weights,
            "oof_auc": float(winner_info["oof_auc"]),
        }

    # ---------------------------
    # Save best artifact
    # ---------------------------
    artifact_path = os.path.join(ARTIFACT_DIR, "best_artifact.joblib")
    joblib.dump(best_artifact, artifact_path, compress=3)

    with open(os.path.join(ARTIFACT_DIR, "best_artifact_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------------------------
    # Save predictions
    # ---------------------------
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: final_test_pred
    })
    submission_path = os.path.join(ARTIFACT_DIR, "best_test_predictions.csv")
    submission.to_csv(submission_path, index=False)

    # save best OOF too
    if winner_type == "single":
        best_oof_pred = base_store[winner_info["run_key"]]["oof_pred"]
    else:
        best_oof_pred = winner_info["oof_pred"]

    oof_df = pd.DataFrame({
        ID_COL: train[ID_COL],
        TARGET_COL: y,
        "oof_pred": best_oof_pred
    })
    oof_df.to_csv(os.path.join(ARTIFACT_DIR, "best_oof_predictions.csv"), index=False)

    print("\nSaved files:")
    print(f"- {os.path.join(ARTIFACT_DIR, 'base_model_results.csv')}")
    if len(ensemble_results_df) > 0:
        print(f"- {os.path.join(ARTIFACT_DIR, 'ensemble_results.csv')}")
    print(f"- {artifact_path}")
    print(f"- {os.path.join(ARTIFACT_DIR, 'best_artifact_meta.json')}")
    print(f"- {submission_path}")
    print(f"- {os.path.join(ARTIFACT_DIR, 'best_oof_predictions.csv')}")


if __name__ == "__main__":
    main()
