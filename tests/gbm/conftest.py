from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(scope="session")
def twin_frame():  # type: ignore[no-untyped-def]
    import pandas as pd

    from schelling.data.adapters import twin2k

    built = twin2k.build(FIXTURES / "twin2k")
    return pd.DataFrame([r.model_dump() for r in built.records]), built.persons
