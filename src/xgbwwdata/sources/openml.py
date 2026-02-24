from __future__ import annotations
from typing import Any, Iterable, Tuple

import pandas as pd
from .base import DataSource

class OpenMLSource(DataSource):
    source_name = "openml"

    def __init__(self):
        import openml
        self._openml = openml

    def iter_ids(self) -> Iterable[str]:
        df = self._openml.datasets.list_datasets(output_format="dataframe")
        did_col = "did" if "did" in df.columns else ("dataset_id" if "dataset_id" in df.columns else None)
        if did_col is None:
            return []
        for did in df[did_col].dropna().astype(int).tolist():
            yield f"openml:{int(did)}"

    def load_raw(self, dataset_uid: str) -> Tuple[pd.DataFrame, Any, str]:
        did = int(dataset_uid.split(":", 1)[1])
        ds = self._openml.datasets.get_dataset(did)
        target = ds.default_target_attribute
        Xdf, y_raw, _, _ = ds.get_data(dataset_format="dataframe", target=target)
        return Xdf, y_raw, str(ds.name)
