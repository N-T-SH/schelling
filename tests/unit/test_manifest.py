from pathlib import Path

import pytest

from schelling.data.manifest import ManifestMismatchError, build_manifest, verify_manifest

pytestmark = pytest.mark.unit


def test_manifest_round_trip_and_checksum_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n")
    m = build_manifest("src", "https://example.org", "rev1", "cc-by-4.0", [f], {"records": 1})
    assert m == build_manifest(
        "src", "https://example.org", "rev1", "cc-by-4.0", [f], {"records": 1}
    )
    verify_manifest(m, tmp_path)
    f.write_text("x\n2\n")
    with pytest.raises(ManifestMismatchError):
        verify_manifest(m, tmp_path)
