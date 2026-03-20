from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean-like environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def keel_enabled_by_default() -> bool:
    """Return whether KEEL integrations should be enabled automatically."""
    return env_flag("XGBWW_ENABLE_KEEL", default=False)

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
    log_every: int = 10
    checkpoint_csv: Optional[str] = None
    checkpoint_done_json: Optional[str] = None
    checkpoint_skips_json: Optional[str] = None
