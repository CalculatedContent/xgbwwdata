from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass(frozen=True)
class Filters:
    """Structural guards + preprocessing settings."""
    min_rows: int = 200
    max_rows: int = 60000
    max_features: int = 50_000
    max_dense_elements: int = int(2e8)
    preprocess: bool = True
    sparse_threshold: float = 0.3
    sparse_output: bool = True

@dataclass(frozen=True)
class ScanOptions:
    """Options controlling scan behavior."""
    sources: Optional[Sequence[str]] = None
    limit: Optional[int] = None
    smoke_train: bool = True
    random_state: int = 0
    log_every: int = 25
    checkpoint_csv: Optional[str] = None
    checkpoint_done_json: Optional[str] = None
    checkpoint_skips_json: Optional[str] = None
