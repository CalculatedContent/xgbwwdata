# xgbwwdata

A small, extensible **dataset scanning + loading** library designed for large-scale
**XGBoost2WW / WeightWatcher** experiments.

## What it does

- Scan one or more dataset sources (default: all supported sources)
- Apply consistent **structural guards**
  - `min_rows`, `max_rows` (subsample)
  - `max_features` (after preprocessing)
  - `max_dense_elements` (`n_rows * n_features`)
  - optional preprocessing (median/mode impute + one-hot)
- Run a **1-round XGBoost smoke test** to confirm trainability
- Return a `pandas.DataFrame` registry of passing datasets with stable `dataset_uid`s
- Load a dataset later by `dataset_uid` with the same preprocessing pipeline

## Supported sources

- `openml`  → `openml:<did>`
- `pmlb`    → `pmlb:<name>`
- `keel`    → `keel:<name>` (via `keel-ds`)
- `libsvm`  → `libsvm:<filename>` (from LIBSVM dataset pages)
- `amlb`    → `amlb_openml:<did>` (OpenML suites 269/270/271 by default)

## Install

Minimal:

```bash
pip install -e .
```

With optional sources:

```bash
pip install -e ".[openml,pmlb,keel]"
```

## Quickstart

### Scan datasets

```python
from xgbwwdata import Filters, scan_datasets

filters = Filters(
    min_rows=200,
    max_rows=60000,
    max_features=50000,
    max_dense_elements=int(2e8),
    preprocess=True,
)

df = scan_datasets(
    sources=["openml", "pmlb", "keel", "libsvm", "amlb"],  # default is all
    limit=200,              # stop after N passing datasets (None = all)
    filters=filters,
    smoke_train=True,       # 1-round XGBoost check
    random_state=0,
)

print(df.head())
print(df["source"].value_counts())
```

### Load a dataset by id

```python
from xgbwwdata import load_dataset

row = df.iloc[0]
X, y, meta = load_dataset(row["dataset_uid"], filters=filters)

# X is ready for xgboost.DMatrix / xgb.train
```
