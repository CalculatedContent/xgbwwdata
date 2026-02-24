from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import Filters
from .sources.base import DataSource

# ---- LIBSVM special loader (numeric sparse)
import os, re, requests
from sklearn.datasets import load_svmlight_file

logger = logging.getLogger(__name__)

_LIBSVM_PAGES = [
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html",
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html",
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html",
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html",
]
_LIBSVM_EXT = (".bz2",".gz",".xz",".zip",".scale",".t",".train",".test",".val",".txt",".data",".svm")

def _libsvm_is_data_url(u: str) -> bool:
    if "#" in u: return False
    if u.lower().endswith(".html"): return False
    if "libsvmtools/datasets" not in u: return False
    leaf = u.split("/")[-1]
    if any(leaf.lower().endswith(ext) for ext in _LIBSVM_EXT): return True
    if leaf and ("." not in leaf): return True
    return False

def _libsvm_fetch_links(url: str) -> List[str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', r.text)
    out = []
    for h in hrefs:
        if h.startswith("#"):
            continue
        out.append(requests.compat.urljoin(url, h))
    return sorted(set(out))

def _libsvm_remap_labels(y):
    y = np.asarray(y)
    uniq = np.unique(y)
    mapping = {v: i for i, v in enumerate(uniq.tolist())}
    y2 = np.array([mapping[v] for v in y], dtype=np.int32)
    return y2, len(uniq)

class LibSVMIndex:
    def __init__(self):
        links = []
        for page in _LIBSVM_PAGES:
            try:
                links.extend(_libsvm_fetch_links(page))
            except Exception:
                continue
        self.urls = [u for u in sorted(set(links)) if _libsvm_is_data_url(u)]

    def iter_uids(self):
        for u in self.urls:
            yield "libsvm:" + u.split("/")[-1]

def _libsvm_locate_url(leaf: str) -> Optional[str]:
    # Try known base dirs first
    bases = [
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/",
    ]
    for b in bases:
        cand = b + leaf
        try:
            h = requests.head(cand, timeout=10, allow_redirects=True)
            if h.status_code == 200:
                return cand
        except Exception:
            continue
    return None

def _libsvm_load(dataset_uid: str, filters: Filters) -> Tuple[Any, Any, Dict[str, Any]]:
    leaf = dataset_uid.split(":", 1)[1]
    url = _libsvm_locate_url(leaf)
    if url is None:
        raise ValueError(f"Could not locate LIBSVM URL for {dataset_uid}")

    os.makedirs("libsvm_cache", exist_ok=True)
    path = os.path.join("libsvm_cache", leaf.replace("/", "_"))
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    X, y = load_svmlight_file(path)
    n, d = X.shape

    if n < filters.min_rows:
        raise RuntimeError("too_few_rows")
    if d > filters.max_features:
        raise RuntimeError("too_many_features")

    if filters.max_rows is not None and n > filters.max_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=filters.max_rows, replace=False)
        X = X[idx]
        y = y[idx]
        n = filters.max_rows

    if int(n) * int(d) > int(filters.max_dense_elements):
        raise RuntimeError("dense_cost_too_high")

    y2, k = _libsvm_remap_labels(y)
    meta = {"n_rows": int(n), "n_features": int(d), "n_classes": float(k), "task_type": "classification", "name": leaf}
    return X.tocsr(), y2, meta

def _smoke_train_1round(X: Any, y: Any, task_type: str, n_classes: Optional[int], seed: int = 0) -> bool:
    try:
        dtrain = xgb.DMatrix(X, label=y)
        params: Dict[str, Any] = dict(
            tree_method="hist",
            max_depth=2,
            eta=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=int(seed),
            verbosity=0,
        )
        if task_type == "classification":
            if int(n_classes) == 2:
                params["objective"] = "binary:logistic"
                params["eval_metric"] = "logloss"
            else:
                params["objective"] = "multi:softprob"
                params["num_class"] = int(n_classes)
                params["eval_metric"] = "mlogloss"
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"

        _ = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        return True
    except Exception:
        return False

def _get_sources(names: Optional[Sequence[str]]) -> List[str]:
    return list(names) if names is not None else ["openml","pmlb","keel","libsvm","amlb"]

def scan_datasets(
    sources: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    filters: Optional[Filters] = None,
    smoke_train: bool = True,
    random_state: int = 0,
    log_every: int = 10,
) -> pd.DataFrame:
    filters = filters or Filters()
    names = [s.lower() for s in _get_sources(sources)]
    records: List[Dict[str, Any]] = []
    logger.info("Starting dataset scan for sources=%s, limit=%s", names, limit)

    # Instantiate sources lazily (optional deps)
    src_objs: List[Any] = []
    if "openml" in names:
        from .sources.openml import OpenMLSource
        src_objs.append(OpenMLSource())
    if "pmlb" in names:
        from .sources.pmlb import PMLBSource
        src_objs.append(PMLBSource(include_regression=True))
    if "keel" in names:
        from .sources.keel import KEELSource
        src_objs.append(KEELSource())
    if "amlb" in names:
        from .sources.amlb import AMLBSource
        src_objs.append(AMLBSource())
    libsvm_index = LibSVMIndex() if "libsvm" in names else None
    if libsvm_index is not None:
        logger.info("Prepared LIBSVM index with %d candidate datasets", len(libsvm_index.urls))

    passed = 0
    attempted = 0

    def _maybe_log_progress(source: str) -> None:
        if log_every > 0 and attempted % int(log_every) == 0:
            logger.info(
                "Progress [%s]: found=%d attempted=%d failed_or_filtered=%d",
                source,
                passed,
                attempted,
                attempted - passed,
            )

    # LIBSVM first (fast)
    if libsvm_index is not None:
        for uid in libsvm_index.iter_uids():
            attempted += 1
            _maybe_log_progress("libsvm")
            logger.info("Evaluating dataset %s", uid)
            try:
                X, y, meta = _libsvm_load(uid, filters)
                if smoke_train and (not _smoke_train_1round(X, y, "classification", int(meta["n_classes"]), seed=random_state)):
                    logger.info("Skipping %s: smoke train failed", uid)
                    continue
                records.append({
                    "source": "libsvm",
                    "dataset_uid": uid,
                    "name": meta["name"],
                    "task_type": "classification",
                    "n_rows": meta["n_rows"],
                    "n_features": meta["n_features"],
                    "n_classes": meta["n_classes"],
                })
                passed += 1
                logger.info("Accepted dataset %s (%d accepted total)", uid, passed)
                if limit is not None and passed >= int(limit):
                    logger.info("Reached dataset limit (%d), returning results", int(limit))
                    return pd.DataFrame(records)
            except Exception as exc:
                logger.warning("Skipping %s due to error: %s", uid, exc)
                continue

    # Other sources
    for src in src_objs:
        logger.info("Scanning source=%s", src.source_name)
        for uid in src.iter_ids():
            attempted += 1
            _maybe_log_progress(src.source_name)
            logger.info("Evaluating dataset %s", uid)
            try:
                ok, info, X, y, task_type, n_classes, name = src.validate_and_prepare(uid, filters)
                if not ok:
                    logger.info("Skipping %s: did not pass filters", uid)
                    continue
                if smoke_train and (not _smoke_train_1round(X, y, task_type, n_classes, seed=random_state)):
                    logger.info("Skipping %s: smoke train failed", uid)
                    continue
                info = info if isinstance(info, dict) else {}
                records.append({
                    "source": src.source_name,
                    "dataset_uid": uid,
                    "name": name,
                    "task_type": task_type,
                    "n_rows": int(info.get("n_rows", np.nan)),
                    "n_features": int(info.get("n_features", np.nan)),
                    "n_classes": info.get("n_classes", np.nan),
                })
                passed += 1
                logger.info("Accepted dataset %s (%d accepted total)", uid, passed)
                if limit is not None and passed >= int(limit):
                    logger.info("Reached dataset limit (%d), returning results", int(limit))
                    return pd.DataFrame(records)
            except Exception as exc:
                logger.warning("Skipping %s due to error: %s", uid, exc)
                continue

    logger.info("Completed dataset scan: attempted=%d passed=%d", attempted, passed)
    return pd.DataFrame(records)

def load_dataset(dataset_uid: str, filters: Optional[Filters] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    filters = filters or Filters()
    logger.info("Loading dataset %s", dataset_uid)

    if dataset_uid.startswith("libsvm:"):
        X, y, meta = _libsvm_load(dataset_uid, filters)
        meta["dataset_uid"] = dataset_uid
        logger.info("Loaded dataset %s with %s rows and %s features", dataset_uid, meta.get("n_rows"), meta.get("n_features"))
        return X, y, meta

    prefix = dataset_uid.split(":", 1)[0].lower()
    if prefix == "openml":
        from .sources.openml import OpenMLSource
        src = OpenMLSource()
    elif prefix == "pmlb":
        from .sources.pmlb import PMLBSource
        src = PMLBSource(include_regression=True)
    elif prefix == "keel":
        from .sources.keel import KEELSource
        src = KEELSource()
    elif prefix == "amlb_openml":
        from .sources.amlb import AMLBSource
        src = AMLBSource()
    else:
        raise ValueError(f"Unknown dataset_uid prefix: {prefix}")

    ok, info, X, y, task_type, n_classes, name = src.validate_and_prepare(dataset_uid, filters)
    if not ok:
        logger.warning("Dataset %s did not pass filters: %s", dataset_uid, info)
        raise RuntimeError(f"Dataset did not pass filters: {info}")
    meta = dict(info if isinstance(info, dict) else {})
    meta.update({"dataset_uid": dataset_uid, "name": name, "task_type": task_type, "n_classes": n_classes})
    logger.info("Loaded dataset %s (%s) with %s rows and %s features", dataset_uid, task_type, meta.get("n_rows"), meta.get("n_features"))
    return X, y, meta
