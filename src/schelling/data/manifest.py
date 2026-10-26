"""Data manifests: upstream URL, revision, licence and sha256 of every file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class ManifestMismatchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    source: str,
    url: str,
    revision: str,
    license_: str,
    files: list[Path],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """No timestamps: building twice from the same inputs must give an identical manifest."""
    return {
        "source": source,
        "url": url,
        "revision": revision,
        "license": license_,
        "files": {
            p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(files, key=lambda p: p.name)
        },
        "stats": stats,
    }


def verify_manifest(manifest: dict[str, Any], base_dir: Path) -> None:
    for name, meta in manifest["files"].items():
        path = base_dir / name
        if not path.exists() or sha256_file(path) != meta["sha256"]:
            raise ManifestMismatchError(f"checksum mismatch for {name} ({manifest['source']})")
