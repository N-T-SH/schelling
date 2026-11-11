from pathlib import Path
from typing import Any

import pytest

from schelling.llm.fireworks import FireworksClient, load_dotenv, request_key


def fake(calls: list[dict[str, Any]]) -> Any:
    def transport(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        calls.append({"url": url, "auth": headers["Authorization"], "body": body})
        return {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }

    return transport


MSG = [{"role": "user", "content": "hi"}]


def test_second_identical_call_is_served_from_disk(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    client = FireworksClient(tmp_path, fake(calls), api_key="k")
    first = client.chat("m", MSG)
    second = FireworksClient(tmp_path, fake(calls), api_key="k").chat("m", MSG)
    assert (first["cached"], second["cached"]) == (False, True)
    assert second["text"] == "hello" and len(calls) == 1


def test_different_params_are_different_requests(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    client = FireworksClient(tmp_path, fake(calls), api_key="k")
    client.chat("m", MSG, temperature=0.0)
    client.chat("m", MSG, temperature=0.7)
    assert len(calls) == 2
    assert request_key("m", MSG, {"a": 1}) != request_key("m", MSG, {"a": 2})


def test_key_is_sent_but_never_cached(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    FireworksClient(tmp_path, fake(calls), api_key="secret-xyz").chat("m", MSG)
    assert calls[0]["auth"] == "Bearer secret-xyz"
    assert all("secret-xyz" not in p.read_text() for p in tmp_path.rglob("*.json*"))


def test_missing_key_fails_before_any_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    calls: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        FireworksClient(tmp_path, fake(calls)).chat("m", MSG)
    assert calls == []


def test_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# c\nFW_TEST_A='one'\nexport FW_TEST_B=two\n")
    monkeypatch.setenv("FW_TEST_B", "kept")
    monkeypatch.delenv("FW_TEST_A", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["FW_TEST_A"] == "one" and os.environ["FW_TEST_B"] == "kept"
