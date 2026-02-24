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

Use the included helper so install + module refresh happens in one step (no manual kernel restart dance):

```python
# Fresh clone
!rm -rf /content/repo_xgbwwdata
!git clone https://github.com/CalculatedContent/xgbwwdata.git /content/repo_xgbwwdata

# Installs requirements + editable package, clears stale xgbwwdata modules,
# removes common repo-shadowing paths, and verifies import.
%run /content/repo_xgbwwdata/scripts/colab_install.py --repo /content/repo_xgbwwdata

from xgbwwdata import Filters, scan_datasets, load_dataset
print("import OK")
```

If you prefer fully manual steps, the helper script is equivalent to:
- upgrading `pip/setuptools/wheel`
- installing `requirements.txt`
- installing `-e` with `--no-build-isolation --no-deps`
- clearing stale `xgbwwdata*` modules and cache paths in the current kernel.

## Quickstart

### Scan datasets

```python
from xgbwwdata import Filters, enable_logging, scan_datasets

enable_logging()  # show scan progress in stdout

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
    log_every=10,           # emit aggregate progress every N datasets
)

print(df.head())
print(df["source"].value_counts())
```

`enable_logging()` attaches a stream handler for the `xgbwwdata` logger so you can see per-dataset and periodic progress updates while scanning.

### Load a dataset by id

```python
from xgbwwdata import load_dataset

row = df.iloc[0]
X, y, meta = load_dataset(row["dataset_uid"], filters=filters)

# X is ready for xgboost.DMatrix / xgb.train
```
