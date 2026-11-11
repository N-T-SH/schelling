"""Cached Fireworks chat client (OpenAI-compatible endpoint).

Rule 7: every response is cached to disk under a hash of the full request, so a
repeated call costs nothing and a run can be replayed offline. The API key is read
from the environment only (``FIREWORKS_API_KEY``, loaded from a git-ignored ``.env``)
and is never logged, cached or put in an exception message.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_CACHE = Path("cache/llm")

# transport(url, headers, body) -> parsed JSON; injectable so tests need no network.
Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def _urllib_transport(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            parsed: dict[str, Any] = json.load(resp)
            return parsed
    except urllib.error.HTTPError as err:
        # Status only: the response body can echo request details.
        raise RuntimeError(f"Fireworks HTTP {err.code}") from None


def request_key(model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class FireworksClient:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE,
        transport: Transport | None = None,
        api_key: str | None = None,
        retries: int = 3,
    ) -> None:
        self.cache_dir = cache_dir
        self._transport = transport or _urllib_transport
        self._api_key = api_key
        self._retries = retries
        self.network_calls = 0

    def _key(self) -> str:
        key = self._api_key or os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("FIREWORKS_API_KEY is not set (put it in the git-ignored .env)")
        return key

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **extra: Any,
    ) -> dict[str, Any]:
        """Return the cached or fresh response: {"text", "usage", "cached"}."""
        params = {"max_tokens": max_tokens, "temperature": temperature, **extra}
        digest = request_key(model, messages, params)
        path = self.cache_dir / digest[:2] / f"{digest}.json"
        if path.exists():
            hit: dict[str, Any] = json.loads(path.read_text())
            return {**hit, "cached": True}
        body = {"model": model, "messages": messages, **params}
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(self._retries):
            try:
                self.network_calls += 1
                raw = self._transport(f"{BASE_URL}/chat/completions", headers, body)
                break
            except RuntimeError as err:
                last = err
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Fireworks request failed after retries: {last}")
        out = {
            "text": raw["choices"][0]["message"]["content"],
            "usage": raw.get("usage", {}),
            "model": model,
            "request_key": digest,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        with (self.cache_dir / "usage.jsonl").open("a") as fh:
            fh.write(json.dumps({"request_key": digest, "model": model, **out["usage"]}) + "\n")
        return {**out, "cached": False}
