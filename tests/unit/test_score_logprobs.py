"""Pure helpers of scripts/d01/score_logprobs.py (no model is loaded)."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PATH = Path(__file__).parents[2] / "scripts" / "d01" / "score_logprobs.py"
_spec = importlib.util.spec_from_file_location("score_logprobs", _PATH)
assert _spec is not None and _spec.loader is not None
sl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sl)


def test_fast_path_when_each_key_is_one_new_token() -> None:
    assert sl.fast_key_ids([1, 2, 3], [[1, 2, 3, 7], [1, 2, 3, 8]]) == [7, 8]


@pytest.mark.parametrize(
    "with_keys",
    [
        [[1, 2, 3, 7, 9], [1, 2, 3, 8]],  # a key takes two tokens
        [[1, 2, 5, 7], [1, 2, 3, 8]],  # the key merged with the prompt's last token
        [[1, 2, 3, 7], [1, 2, 3, 7]],  # two keys share a token
    ],
)
def test_fast_path_refused(with_keys: list[list[int]]) -> None:
    assert sl.fast_key_ids([1, 2, 3], with_keys) is None


def test_common_prefix_and_normalise() -> None:
    assert sl.common_prefix([1, 2, 3], [1, 2, 9, 9]) == 2
    p = sl.normalise([math.log(0.2), math.log(0.6)])
    assert p == pytest.approx([0.25, 0.75])
    assert sum(sl.normalise([-1000.0, -1001.0, -1002.0])) == pytest.approx(1.0)


def test_resume_reads_complete_lines_and_skips_cut_ones(tmp_path: Path) -> None:
    (tmp_path / "part-1.jsonl").write_text(json.dumps({"prompt_id": "a"}) + "\n")
    (tmp_path / "part-2.jsonl").write_text(json.dumps({"prompt_id": "b"}) + '\n{"prompt_')
    (tmp_path / "part-3.tmp").write_text(json.dumps({"prompt_id": "c"}) + "\n")
    assert sl.done_ids(tmp_path) == {"a", "b"}


def test_read_prompts_filters_arm(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"prompt_id": i, "arm": a})
            for i, a in [("1", "population"), ("2", "persona")]
        )
    )
    assert [r["prompt_id"] for r in sl.read_prompts(path, "persona")] == ["2"]
    assert len(sl.read_prompts(path, "all")) == 2
    assert [len(b) for b in sl.batches(list(range(5)), 2)] == [2, 2, 1]
