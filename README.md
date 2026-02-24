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

Local editable install:

```bash
pip install -e .
```

Install all runtime dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Colab / notebook fresh-clone install

If you want the notebook cell to explicitly install requirements before the editable package,
add `%pip install -r /content/repo_xgbwwdata/requirements.txt` before `%pip install -e ...`:

```python
# Fresh clone
!rm -rf /content/repo_xgbwwdata
!git clone https://github.com/CalculatedContent/xgbwwdata.git /content/repo_xgbwwdata

# MUST use %pip so install targets current kernel env
%pip install -U pip setuptools wheel
%pip install -r /content/repo_xgbwwdata/requirements.txt
%pip install -e /content/repo_xgbwwdata --no-build-isolation --no-deps

# Clear stale modules
import sys
for m in list(sys.modules):
    if m == "xgbwwdata" or m.startswith("xgbwwdata."):
        del sys.modules[m]

# Optional: ensure repo root isn't shadowing
sys.path = [p for p in sys.path if p not in ("/content/repo_xgbwwdata", "/content/xgbwwdata")]

import xgbwwdata
print("module:", xgbwwdata)
print("__file__:", getattr(xgbwwdata, "__file__", None))
print("__path__:", getattr(xgbwwdata, "__path__", None))
print("exports:", [x for x in dir(xgbwwdata) if not x.startswith("_")])

from xgbwwdata import Filters, scan_datasets, load_dataset
print("import OK")
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
