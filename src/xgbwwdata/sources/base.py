from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

import pandas as pd

from ..config import Filters

class DataSource(ABC):
    source_name: str

    @abstractmethod
    def iter_ids(self) -> Iterable[str]:
        ...

    @abstractmethod
    def load_raw(self, dataset_uid: str) -> Tuple[pd.DataFrame, Any, str]:
        """Return (Xdf, y_raw, name)."""
        ...

    def validate_and_prepare(self, dataset_uid: str, filters: Filters):
        from ..preprocess import apply_filters_and_preprocess
        Xdf, y_raw, name = self.load_raw(dataset_uid)
        ok, info, X, y, task_type, n_classes = apply_filters_and_preprocess(Xdf, y_raw, filters)
        return ok, info, X, y, task_type, n_classes, name
