from __future__ import annotations
from typing import Any, Iterable, Tuple

import pandas as pd
from .base import DataSource

class PMLBSource(DataSource):
    source_name = "pmlb"

    def __init__(self, include_regression: bool = True):
        from pmlb import classification_dataset_names, regression_dataset_names, fetch_data
        self._class_names = list(classification_dataset_names)
        self._reg_names = list(regression_dataset_names) if include_regression else []
        self._fetch = fetch_data

    def iter_ids(self) -> Iterable[str]:
        for n in self._class_names:
            yield f"pmlb:{n}"
        for n in self._reg_names:
            yield f"pmlb:{n}"

    def load_raw(self, dataset_uid: str) -> Tuple[pd.DataFrame, Any, str]:
        name = dataset_uid.split(":", 1)[1]
        df = self._fetch(name, return_X_y=False)
        if "target" in df.columns:
            y_raw = df["target"]
            Xdf = df.drop(columns=["target"])
        else:
            y_raw = df.iloc[:, -1]
            Xdf = df.iloc[:, :-1]
        return Xdf, y_raw, name
