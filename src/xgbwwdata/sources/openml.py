from __future__ import annotations
from typing import Any, Iterable, Tuple

import pandas as pd
from .base import DataSource
from ..config import Filters

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

    def iter_ids_filtered(self, filters: Filters, include_unknown_sizes: bool = False) -> Iterable[str]:
        """Yield OpenML IDs pre-filtered by metadata size columns when available.

        This helps avoid loading datasets that are very likely to violate downstream
        structural constraints and can destabilize long-running notebook sessions.
        """
        df = self._openml.datasets.list_datasets(output_format="dataframe")
        did_col = "did" if "did" in df.columns else ("dataset_id" if "dataset_id" in df.columns else None)
        if did_col is None:
            return []

        filtered = df
        rows_col = "NumberOfInstances" if "NumberOfInstances" in df.columns else None
        feats_col = "NumberOfFeatures" if "NumberOfFeatures" in df.columns else None

        if rows_col is not None:
            if include_unknown_sizes:
                row_mask = filtered[rows_col].isna() | (
                    (filtered[rows_col] >= int(filters.min_rows))
                    & (
                        filtered[rows_col] <= int(filters.max_rows)
                        if filters.max_rows is not None
                        else True
                    )
                )
            else:
                row_mask = filtered[rows_col].notna() & (filtered[rows_col] >= int(filters.min_rows))
                if filters.max_rows is not None:
                    row_mask = row_mask & (filtered[rows_col] <= int(filters.max_rows))
            filtered = filtered[row_mask]

        if feats_col is not None:
            if include_unknown_sizes:
                feat_mask = filtered[feats_col].isna() | (filtered[feats_col] <= int(filters.max_features))
            else:
                feat_mask = filtered[feats_col].notna() & (filtered[feats_col] <= int(filters.max_features))
            filtered = filtered[feat_mask]

        for did in filtered[did_col].dropna().astype(int).tolist():
            yield f"openml:{int(did)}"

    def load_raw(self, dataset_uid: str) -> Tuple[pd.DataFrame, Any, str]:
        did = int(dataset_uid.split(":", 1)[1])
        ds = self._openml.datasets.get_dataset(did)
        target = ds.default_target_attribute
        Xdf, y_raw, _, _ = ds.get_data(dataset_format="dataframe", target=target)
        return Xdf, y_raw, str(ds.name)
