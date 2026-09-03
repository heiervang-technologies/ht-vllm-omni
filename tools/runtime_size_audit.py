#!/usr/bin/env python3
"""Report logical installed-package and accelerator-library sizes as JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import site
import sys
from pathlib import Path
from typing import Any


def _logical_size(path: Path) -> int:
    """Return logical bytes without following directory symlinks."""
    try:
        if path.is_symlink() or path.is_file():
            return path.lstat().st_size
        if not path.is_dir():
            return 0
    except OSError:
        return 0

    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        kept_dirs = []
        for name in dirs:
            candidate = root_path / name
            try:
                if candidate.is_symlink():
                    total += candidate.lstat().st_size
                else:
                    kept_dirs.append(name)
            except OSError:
                continue
        dirs[:] = kept_dirs
        for name in files:
            try:
                total += (root_path / name).lstat().st_size
            except OSError:
                continue
    return total


def _distribution_sizes() -> list[dict[str, Any]]:
    rows = []
    for distribution in importlib.metadata.distributions():
        files = distribution.files or ()
        seen: set[Path] = set()
        total = 0
        present = 0
        for relative in files:
            path = Path(distribution.locate_file(relative)).absolute()
            if path in seen:
                continue
            seen.add(path)
            try:
                total += path.lstat().st_size
                present += 1
            except OSError:
                continue
        rows.append(
            {
                "name": distribution.metadata.get("Name", "<unknown>"),
                "version": distribution.version,
                "logical_bytes": total,
                "files_present": present,
                "files_declared": len(files),
            }
        )
    rows.sort(key=lambda row: (-row["logical_bytes"], row["name"].lower()))
    return rows


def _component_paths() -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    for site_root in site.getsitepackages():
        root = Path(site_root)
        for name in ("nvidia", "torch", "triton", "onnxruntime"):
            candidate = root / name
            if candidate.exists():
                candidates.add(candidate)
    candidates.update(path for path in Path("/usr/local").glob("cuda*") if path.exists())
    return [
        {"path": str(path), "logical_bytes": _logical_size(path)}
        for path in sorted(candidates, key=lambda item: str(item))
    ]


def build_report(top: int) -> dict[str, Any]:
    distributions = _distribution_sizes()
    return {
        "python": {"executable": sys.executable, "version": sys.version},
        "site_packages": site.getsitepackages(),
        "distribution_count": len(distributions),
        "top_distributions": distributions[:top],
        "accelerator_components": _component_paths(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=50, help="number of largest distributions to report")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")
    json.dump(build_report(args.top), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
