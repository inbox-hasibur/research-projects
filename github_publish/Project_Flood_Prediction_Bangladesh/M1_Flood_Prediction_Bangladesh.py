# Flood Prediction in Bangladesh (Kaggle Script Starter)
# ======================================================
# Research Project Style:
# - Single runnable .py file (Kaggle-friendly)
# - All steps in one pipeline: load -> preprocess -> train -> evaluate -> save

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Flood Prediction in Bangladesh - starter running")

# Kaggle-style path defaults (override with env vars if needed)
DATA_PATH = os.environ.get("FLOOD_DATA_PATH", "/kaggle/input/flood-prediction-bangladesh/flood.csv")
TARGET_COL = os.environ.get("FLOOD_TARGET_COL", "flood")


def build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    feature_cols = [c for c in df.columns if c != target_col]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )


def main():
    data_file = Path(DATA_PATH)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_file}. Set FLOOD_DATA_PATH env variable to your CSV path."
        )

    df = pd.read_csv(data_file)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y if y.nunique() > 1 else None,
    )

    preprocessor = build_preprocessor(df, TARGET_COL)
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        eval_metric="logloss",
    )

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("\n=== Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Macro F1 : {f1_score(y_test, preds, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    if y.nunique() == 2:
        probs = clf.predict_proba(X_test)[:, 1]
        print(f"ROC AUC  : {roc_auc_score(y_test, probs):.4f}")

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "flood_predictions.csv"
    pd.DataFrame({"y_true": y_test.values, "y_pred": preds}).to_csv(pred_path, index=False)
    print(f"\nSaved predictions: {pred_path}")


if __name__ == "__main__":
    main()
