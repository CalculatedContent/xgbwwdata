from __future__ import annotations
from typing import Any, Iterable, Tuple

import pandas as pd
from .base import DataSource

class KEELSource(DataSource):
    source_name = "keel"

    def __init__(self):
        import keel_ds
        self._keel = keel_ds

    def iter_ids(self) -> Iterable[str]:
        for n in list(self._keel.list_data()):
            yield f"keel:{n}"

    def load_raw(self, dataset_uid: str) -> Tuple[pd.DataFrame, Any, str]:
        name = dataset_uid.split(":", 1)[1]
        out = None
        last = None
        for fn_name in ("load_data", "load"):
            if hasattr(self._keel, fn_name):
                fn = getattr(self._keel, fn_name)
                try:
                    out = fn(name)  # positional
                    break
                except Exception as e:
                    last = e
        if out is None:
            raise RuntimeError(f"keel_ds could not load {name}: {last}")

        if isinstance(out, tuple) and len(out) == 2:
            X, y = out
            Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            return Xdf, y, name

        if isinstance(out, dict):
            if "X" in out and "y" in out:
                Xdf = out["X"] if isinstance(out["X"], pd.DataFrame) else pd.DataFrame(out["X"])
                return Xdf, out["y"], name
            if "data" in out and "target" in out:
                Xdf = out["data"] if isinstance(out["data"], pd.DataFrame) else pd.DataFrame(out["data"])
                return Xdf, out["target"], name

        if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], (list, tuple)):
            first = out[0]
            if len(first) >= 2:
                x_tr, y_tr = first[0], first[1]
                Xdf = x_tr if isinstance(x_tr, pd.DataFrame) else pd.DataFrame(x_tr)
                return Xdf, y_tr, name

        if isinstance(out, pd.DataFrame):
            if "target" in out.columns:
                return out.drop(columns=["target"]), out["target"], name
            return out.iloc[:, :-1], out.iloc[:, -1], name

        raise ValueError(f"Unsupported keel_ds return type for {name}: {type(out)}")
