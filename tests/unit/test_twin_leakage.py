"""A04: the target item's answer must not appear anywhere in that person's context."""

import pytest

from schelling.data.adapters.twin2k import build_context
from schelling.schema import PersonContext

pytestmark = pytest.mark.unit

ANSWERS = {
    f"QID{i}": (f"question {i}?", [f"opt{i}a", f"opt{i}b", f"SENTINEL-{i}"], 2) for i in range(1, 8)
}


@pytest.mark.parametrize("target", sorted(ANSWERS))
def test_target_answer_absent_from_context(target: str) -> None:
    ctx = build_context(ANSWERS, target_item=target, attributes={"age": "30"}, k=50)
    assert isinstance(ctx, PersonContext)
    blob = ctx.model_dump_json()
    assert f"question {target[3:]}?" not in blob
    assert f"SENTINEL-{target[3:]}" not in blob
    # other items are still present: masking must not empty the context
    assert len(ctx.history) == len(ANSWERS) - 1


def test_k_limits_history() -> None:
    ctx = build_context(ANSWERS, target_item="QID1", attributes={}, k=3)
    assert len(ctx.history) == 3
