from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import Filters

def safe_1d(y: Any) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y

def infer_task_type(y_raw: Any) -> str:
    y = safe_1d(y_raw)
    if not np.issubdtype(y.dtype, np.number):
        return "classification"
    y_nonan = y[~np.isnan(y)] if y.dtype.kind == "f" else y
    uniq = np.unique(y_nonan)
    k = len(uniq)
    n = len(y)
    if k >= 20 and (k / max(1, n)) > 0.05:
        return "regression"
    return "classification"

def make_preprocessor(Xdf: pd.DataFrame, filters: Filters) -> ColumnTransformer:
    cat_cols = Xdf.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in Xdf.columns if c not in cat_cols]
    transformers = []
    if len(num_cols):
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if len(cat_cols):
        transformers.append(
            ("cat",
             Pipeline([
                 ("imp", SimpleImputer(strategy="most_frequent")),
                 ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=filters.sparse_output)),
             ]),
             cat_cols)
        )
    if not transformers:
        raise ValueError("no_usable_columns")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=filters.sparse_threshold)

def apply_filters_and_preprocess(
    Xdf: pd.DataFrame,
    y_raw: Any,
    filters: Filters,
) -> Tuple[bool, str | dict, Any, Any, str, Optional[int]]:
    n = len(Xdf)
    if n < filters.min_rows:
        return False, "too_few_rows", None, None, "", None

    if filters.max_rows is not None and n > filters.max_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=filters.max_rows, replace=False)
        Xdf = Xdf.iloc[idx].reset_index(drop=True)
        y_raw = safe_1d(y_raw)[idx]
        n = len(Xdf)

    task_type = infer_task_type(y_raw)

    if task_type == "classification":
        y = safe_1d(y_raw)
        y_codes, uniques = pd.factorize(y)
        k = int(len(uniques))
        if k < 2:
            return False, "degenerate_labels", None, None, task_type, None
        y_enc = y_codes.astype(np.int32)
        n_classes: Optional[int] = k
    else:
        y_enc = safe_1d(y_raw).astype(np.float32)
        n_classes = None

    if filters.preprocess:
        pre = make_preprocessor(Xdf, filters)
        X = pre.fit_transform(Xdf)
        d = int(X.shape[1])
    else:
        X = Xdf.to_numpy()
        d = int(X.shape[1])

    if d > filters.max_features:
        return False, "too_many_features_after_preprocess", None, None, task_type, n_classes

    if int(n) * int(d) > int(filters.max_dense_elements):
        return False, "dense_cost_too_high", None, None, task_type, n_classes

    info = {"n_rows": int(n), "n_features": int(d), "n_classes": (n_classes if n_classes is not None else np.nan)}
    return True, info, X, y_enc, task_type, n_classes
