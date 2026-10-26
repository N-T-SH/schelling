"""Regenerate schemas/*.json from the pydantic models."""

from pathlib import Path

from schelling.schema import export_json_schemas

if __name__ == "__main__":
    for path in export_json_schemas(Path(__file__).resolve().parents[1] / "schemas"):
        print(f"wrote {path}")
