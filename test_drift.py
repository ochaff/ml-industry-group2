# ============================================================
# Train-test drift check script
# - application_month is dropped entirely
# - numeric drift: KS statistic
# - categorical drift: total variation distance (optional)
# - adversarial validation: predict train vs test
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# Optional adversarial models
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ============================================================
# Config
# ============================================================

TRAIN_PATH = "./generated/competition_train.csv"
TEST_PATH = "./generated/competition_test.csv"

TARGET_COL = "default"
ID_COL = "row_id"
TIME_COL = "application_month"   # dropped completely

# You can optionally drop other metadata-like columns here
EXTRA_DROP_COLS = []  # e.g. ["etl_batch_id", "schema_version"]

N_SPLITS = 5
RANDOM_STATE = 42

SAVE_DIR = "./drift_report"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Cleaning functions
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


# ============================================================
# Build comparable train/test feature matrices
# ============================================================

def build_feature_frames(train_df, test_df):
    train_df = clean_data(train_df, assumption=True)
    test_df = clean_data(test_df, assumption=True)

    drop_cols = {TARGET_COL, ID_COL, TIME_COL} | set(EXTRA_DROP_COLS)

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

    # Keep only common columns and same order
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    return X_train, X_test


# ============================================================
# Numeric drift: KS report
# ============================================================

def ks_drift_report(X_train, X_test):
    num_cols_train = X_train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_test = X_test.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = sorted(set(num_cols_train) & set(num_cols_test))

    rows = []
    for col in numeric_cols:
        tr = X_train[col]
        te = X_test[col]

        tr_non_null = tr.dropna()
        te_non_null = te.dropna()

        if len(tr_non_null) == 0 or len(te_non_null) == 0:
            rows.append({
                "feature": col,
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
                "train_mean": tr.mean(),
                "test_mean": te.mean(),
                "train_std": tr.std(),
                "test_std": te.std(),
                "train_missing_rate": tr.isna().mean(),
                "test_missing_rate": te.isna().mean(),
                "missing_rate_diff": te.isna().mean() - tr.isna().mean(),
                "train_q05": tr.quantile(0.05),
                "test_q05": te.quantile(0.05),
                "train_q50": tr.quantile(0.50),
                "test_q50": te.quantile(0.50),
                "train_q95": tr.quantile(0.95),
                "test_q95": te.quantile(0.95),
            })
            continue

        ks = ks_2samp(tr_non_null.values, te_non_null.values)

        rows.append({
            "feature": col,
            "ks_stat": ks.statistic,
            "ks_pvalue": ks.pvalue,
            "train_mean": tr_non_null.mean(),
            "test_mean": te_non_null.mean(),
            "train_std": tr_non_null.std(),
            "test_std": te_non_null.std(),
            "train_missing_rate": tr.isna().mean(),
            "test_missing_rate": te.isna().mean(),
            "missing_rate_diff": te.isna().mean() - tr.isna().mean(),
            "train_q05": tr_non_null.quantile(0.05),
            "test_q05": te_non_null.quantile(0.05),
            "train_q50": tr_non_null.quantile(0.50),
            "test_q50": te_non_null.quantile(0.50),
            "train_q95": tr_non_null.quantile(0.95),
            "test_q95": te_non_null.quantile(0.95),
        })

    out = pd.DataFrame(rows).sort_values(
        ["ks_stat", "missing_rate_diff"], ascending=[False, False]
    ).reset_index(drop=True)

    return out


# ============================================================
# Optional categorical drift report
# KS is only for numeric columns; this helps on categoricals
# ============================================================

def categorical_drift_report(X_train, X_test):
    cat_cols_train = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols_test = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = sorted(set(cat_cols_train) & set(cat_cols_test))

    rows = []
    for col in cat_cols:
        tr = X_train[col].fillna("<<MISSING>>").astype(str)
        te = X_test[col].fillna("<<MISSING>>").astype(str)

        p_tr = tr.value_counts(normalize=True)
        p_te = te.value_counts(normalize=True)

        levels = sorted(set(p_tr.index) | set(p_te.index))
        p_tr = p_tr.reindex(levels, fill_value=0.0)
        p_te = p_te.reindex(levels, fill_value=0.0)

        tv_distance = 0.5 * np.abs(p_tr - p_te).sum()

        train_only_levels = [x for x in levels if (p_tr[x] > 0 and p_te[x] == 0)]
        test_only_levels = [x for x in levels if (p_te[x] > 0 and p_tr[x] == 0)]

        rows.append({
            "feature": col,
            "tv_distance": tv_distance,
            "n_levels_train": tr.nunique(dropna=False),
            "n_levels_test": te.nunique(dropna=False),
            "train_missing_rate": X_train[col].isna().mean(),
            "test_missing_rate": X_test[col].isna().mean(),
            "missing_rate_diff": X_test[col].isna().mean() - X_train[col].isna().mean(),
            "n_train_only_levels": len(train_only_levels),
            "n_test_only_levels": len(test_only_levels),
            "train_top_1": p_tr.idxmax() if len(p_tr) else None,
            "train_top_1_freq": p_tr.max() if len(p_tr) else None,
            "test_top_1": p_te.idxmax() if len(p_te) else None,
            "test_top_1_freq": p_te.max() if len(p_te) else None,
        })

    out = pd.DataFrame(rows).sort_values(
        ["tv_distance", "missing_rate_diff"], ascending=[False, False]
    ).reset_index(drop=True)

    return out


# ============================================================
# Adversarial validation
# ============================================================

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


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
    return preprocessor, num_cols, cat_cols


def make_adv_model():
    if LGBMClassifier is not None:
        return "lightgbm", LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
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

    if XGBClassifier is not None:
        return "xgboost", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ImportError("Please install lightgbm or xgboost for adversarial validation.")


def aggregate_feature_importance(preprocessor, model, num_cols, cat_cols):
    if not hasattr(model, "feature_importances_"):
        return None

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    rows = []
    for fname, imp in zip(feature_names, importances):
        if fname.startswith("num__"):
            raw_feature = fname.replace("num__", "", 1)
        elif fname.startswith("cat__"):
            encoded = fname.replace("cat__", "", 1)

            # map one-hot feature back to original categorical column
            matches = [c for c in cat_cols if encoded == c or encoded.startswith(c + "_")]
            if len(matches) == 0:
                raw_feature = encoded
            else:
                raw_feature = max(matches, key=len)
        else:
            raw_feature = fname

        rows.append({"raw_feature": raw_feature, "importance_piece": imp})

    imp_df = pd.DataFrame(rows)
    imp_df = (
        imp_df.groupby("raw_feature", as_index=False)["importance_piece"]
        .sum()
        .rename(columns={"importance_piece": "importance"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp_df


def run_adversarial_validation(X_train, X_test):
    # label original train as 0, original test as 1
    X_adv = pd.concat([X_train.copy(), X_test.copy()], axis=0, ignore_index=True)
    y_adv = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]

    preprocessor, num_cols, cat_cols = make_preprocessor(X_adv)
    model_name, model = make_adv_model()

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros(len(X_adv))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_adv, y_adv), start=1):
        X_tr, X_va = X_adv.iloc[tr_idx], X_adv.iloc[va_idx]
        y_tr, y_va = y_adv[tr_idx], y_adv[va_idx]

        est = clone(pipe)
        est.fit(X_tr, y_tr)

        va_pred = est.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = va_pred

        fold_auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold}: adversarial AUC = {fold_auc:.6f}")

    oof_auc = roc_auc_score(y_adv, oof_pred)

    # fit on full adversarial data for importances
    pipe.fit(X_adv, y_adv)
    imp_df = aggregate_feature_importance(
        preprocessor=pipe.named_steps["prep"],
        model=pipe.named_steps["model"],
        num_cols=num_cols,
        cat_cols=cat_cols
    )

    return {
        "model_name": model_name,
        "fold_scores": fold_scores,
        "mean_auc": float(np.mean(fold_scores)),
        "std_auc": float(np.std(fold_scores)),
        "oof_auc": float(oof_auc),
        "oof_pred": oof_pred,
        "y_adv": y_adv,
        "feature_importance": imp_df,
    }


# ============================================================
# Interpretation helper
# ============================================================

def interpret_adv_auc(auc):
    if auc < 0.55:
        return "Low detectable drift"
    elif auc < 0.62:
        return "Mild drift"
    elif auc < 0.70:
        return "Moderate drift"
    elif auc < 0.80:
        return "Strong drift"
    else:
        return "Very strong drift / train and test are easily separable"


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Building aligned feature frames...")
    X_train, X_test = build_feature_frames(train, test)

    print("\nFinal feature frame shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\nDropped columns:")
    print([ID_COL, TIME_COL] + EXTRA_DROP_COLS)

    # --------------------------------------------------------
    # KS numeric drift
    # --------------------------------------------------------
    print("\nRunning KS drift report on numeric features...")
    ks_df = ks_drift_report(X_train, X_test)
    ks_path = os.path.join(SAVE_DIR, "numeric_ks_drift.csv")
    ks_df.to_csv(ks_path, index=False)

    print("\nTop numeric drift features by KS:")
    print(ks_df.head(20).to_string(index=False))

    # --------------------------------------------------------
    # Categorical drift
    # --------------------------------------------------------
    print("\nRunning categorical drift report...")
    cat_df = categorical_drift_report(X_train, X_test)
    cat_path = os.path.join(SAVE_DIR, "categorical_drift.csv")
    cat_df.to_csv(cat_path, index=False)

    if len(cat_df) > 0:
        print("\nTop categorical drift features by TV distance:")
        print(cat_df.head(20).to_string(index=False))
    else:
        print("No categorical columns found.")

    # --------------------------------------------------------
    # Adversarial validation
    # --------------------------------------------------------
    print("\nRunning adversarial validation...")
    adv = run_adversarial_validation(X_train, X_test)

    adv_summary = {
        "model_name": adv["model_name"],
        "fold_scores": adv["fold_scores"],
        "mean_auc": adv["mean_auc"],
        "std_auc": adv["std_auc"],
        "oof_auc": adv["oof_auc"],
        "interpretation": interpret_adv_auc(adv["oof_auc"]),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "dropped_columns": [ID_COL, TIME_COL] + EXTRA_DROP_COLS,
    }

    print("\nAdversarial validation summary:")
    print(json.dumps(adv_summary, indent=2))

    with open(os.path.join(SAVE_DIR, "adversarial_summary.json"), "w") as f:
        json.dump(adv_summary, f, indent=2)

    adv_oof_df = pd.DataFrame({
        "is_test": adv["y_adv"],
        "adv_pred": adv["oof_pred"]
    })
    adv_oof_path = os.path.join(SAVE_DIR, "adversarial_oof_predictions.csv")
    adv_oof_df.to_csv(adv_oof_path, index=False)

    if adv["feature_importance"] is not None:
        adv_imp_path = os.path.join(SAVE_DIR, "adversarial_feature_importance.csv")
        adv["feature_importance"].to_csv(adv_imp_path, index=False)

        print("\nTop adversarial importance features:")
        print(adv["feature_importance"].head(20).to_string(index=False))
    else:
        print("\nFeature importances unavailable for the adversarial model.")

    print("\nSaved files:")
    print("-", ks_path)
    print("-", cat_path)
    print("-", os.path.join(SAVE_DIR, "adversarial_summary.json"))
    print("-", adv_oof_path)
    if adv["feature_importance"] is not None:
        print("-", os.path.join(SAVE_DIR, "adversarial_feature_importance.csv"))


if __name__ == "__main__":
    main()