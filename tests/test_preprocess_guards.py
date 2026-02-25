import numpy as np
import pandas as pd

from xgbwwdata import Filters
from xgbwwdata import preprocess


def test_estimate_feature_count_after_preprocess_counts_numeric_and_categorical():
    Xdf = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["a", "b", "a"],
            "flag": [True, False, True],
        }
    )

    # num=1 + cat(unique=2) + flag(unique=2)
    assert preprocess.estimate_feature_count_after_preprocess(Xdf) == 5


def test_apply_filters_returns_early_when_estimated_features_exceed_limit(monkeypatch):
    Xdf = pd.DataFrame({"cat": [f"v{i}" for i in range(300)]})
    y = np.array([0, 1] * 150)
    filters = Filters(max_features=100)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("preprocessor should not be built for over-limit datasets")

    monkeypatch.setattr(preprocess, "make_preprocessor", _should_not_run)

    ok, info, X, y_enc, task_type, n_classes = preprocess.apply_filters_and_preprocess(Xdf, y, filters)

    assert ok is False
    assert info == "too_many_features_estimated_preprocess"
    assert X is None
    assert y_enc is None
    assert task_type == "classification"
    assert n_classes == 2
