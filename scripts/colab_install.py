#!/usr/bin/env python3
"""Notebook bootstrap helper for editable installs without kernel restarts.

Usage in Colab/Jupyter:

    %run /content/repo_xgbwwdata/scripts/colab_install.py --repo /content/repo_xgbwwdata
"""

from __future__ import annotations

import argparse
import importlib
import site
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def _pip_install(args: list[str]) -> None:
    _run([sys.executable, "-m", "pip", *args])


def _clear_xgbwwdata_modules() -> None:
    for mod_name in list(sys.modules):
        if mod_name == "xgbwwdata" or mod_name.startswith("xgbwwdata."):
            del sys.modules[mod_name]


def _remove_repo_shadowing(repo: Path) -> None:
    shadow_paths = {str(repo), str(repo.parent / "xgbwwdata")}
    sys.path[:] = [p for p in sys.path if p not in shadow_paths]


def bootstrap(repo: Path) -> None:
    repo = repo.resolve()
    req_file = repo / "requirements.txt"

    _pip_install(["install", "-U", "pip", "setuptools", "wheel"])
    if req_file.exists():
        _pip_install(["install", "-r", str(req_file)])
    _pip_install(["install", "-e", str(repo), "--no-build-isolation", "--no-deps"])

    importlib.invalidate_caches()
    if hasattr(site, "main"):
        site.main()

    _clear_xgbwwdata_modules()
    _remove_repo_shadowing(repo)

    import xgbwwdata  # noqa: PLC0415

    exports = [name for name in dir(xgbwwdata) if not name.startswith("_")]
    print("module:", xgbwwdata)
    print("__file__:", getattr(xgbwwdata, "__file__", None))
    print("__path__:", getattr(xgbwwdata, "__path__", None))
    print("exports:", exports)
    print("import OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the cloned xgbwwdata repository (default: current working directory).",
    )
    args = parser.parse_args()
    bootstrap(Path(args.repo))


if __name__ == "__main__":
    main()
