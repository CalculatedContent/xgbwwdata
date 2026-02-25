import pandas as pd

from xgbwwdata import Filters
from xgbwwdata.sources.openml import OpenMLSource


class _FakeDatasetsAPI:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def list_datasets(self, output_format="dataframe"):
        assert output_format == "dataframe"
        return self._df


class _FakeOpenML:
    def __init__(self, df: pd.DataFrame):
        self.datasets = _FakeDatasetsAPI(df)


def _make_source(df: pd.DataFrame) -> OpenMLSource:
    src = OpenMLSource.__new__(OpenMLSource)
    src._openml = _FakeOpenML(df)
    return src


def test_iter_ids_filtered_excludes_unknown_metadata_by_default():
    df = pd.DataFrame(
        {
            "did": [1, 2, 3, 4],
            "NumberOfInstances": [500, None, 100_000, 400],
            "NumberOfFeatures": [50, 10, 30, None],
        }
    )
    src = _make_source(df)
    ids = list(src.iter_ids_filtered(Filters(max_rows=60_000, max_features=500), include_unknown_sizes=False))
    assert ids == ["openml:1"]


def test_iter_ids_filtered_can_include_unknown_metadata():
    df = pd.DataFrame(
        {
            "dataset_id": [11, 12, 13],
            "NumberOfInstances": [None, 500, 10],
            "NumberOfFeatures": [100, None, 5],
        }
    )
    src = _make_source(df)
    ids = list(src.iter_ids_filtered(Filters(min_rows=20, max_rows=1_000, max_features=200), include_unknown_sizes=True))
    assert ids == ["openml:11", "openml:12"]
