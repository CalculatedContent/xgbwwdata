from __future__ import annotations
from typing import Any, Iterable, Tuple, List

import pandas as pd
from .base import DataSource

DEFAULT_AMLB_SUITES = [269, 270, 271]

class AMLBSource(DataSource):
    source_name = "amlb"

    def __init__(self, suite_ids: List[int] | None = None):
        import openml
        self._openml = openml
        suite_ids = suite_ids or list(DEFAULT_AMLB_SUITES)
        dids = []
        for sid in suite_ids:
            try:
                suite = self._openml.study.get_suite(int(sid))
                dids.extend([int(d) for d in suite.data])
            except Exception:
                continue
        self._dids = sorted(set(dids))

    def iter_ids(self) -> Iterable[str]:
        for did in self._dids:
            yield f"amlb_openml:{int(did)}"

    def load_raw(self, dataset_uid: str) -> Tuple[pd.DataFrame, Any, str]:
        did = int(dataset_uid.split(":", 1)[1])
        ds = self._openml.datasets.get_dataset(did)
        target = ds.default_target_attribute
        Xdf, y_raw, _, _ = ds.get_data(dataset_format="dataframe", target=target)
        return Xdf, y_raw, str(ds.name)
