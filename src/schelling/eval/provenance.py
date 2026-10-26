from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_sha() -> str:
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    status = _git("status", "--porcelain")
    return bool(status) if status is not None else True


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
