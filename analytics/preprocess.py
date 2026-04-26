from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    y_col = "conversion"
    ignore = {"customer_id", y_col}
    features = [c for c in df.columns if c not in ignore]
    numeric = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    categorical = [c for c in features if c not in numeric]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )
    return pre, numeric, categorical


def build_preprocessor_dense(df: pd.DataFrame) -> ColumnTransformer:
    y_col = "conversion"
    ignore = {"customer_id", y_col}
    features = [c for c in df.columns if c not in ignore]
    numeric = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    categorical = [c for c in features if c not in numeric]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        sparse_threshold=0.0,
    )


def build_causal_preprocessor(base_features: list[str]) -> ColumnTransformer:
    num_cols = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    cat_cols = [c for c in base_features if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return pre


def build_outcome_preprocessor(base_features: list[str]) -> ColumnTransformer:
    num_cols = ["recency", "history", "used_discount", "used_bogo", "is_referral"]
    cat_cols = [c for c in base_features if c not in num_cols]
    pre_out = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols + ["treatment"],
            ),
        ]
    )
    return pre_out
