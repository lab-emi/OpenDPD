"""frontend/mocks/*.json must equal the contract examples they are generated from."""

import json
from pathlib import Path

import pytest

from opendpd.schemas.__main__ import export_mocks

ROOT = Path(__file__).resolve().parents[2]
MOCKS = ROOT / "frontend" / "mocks"


@pytest.mark.skipif(not MOCKS.is_dir(), reason="frontend checkout not present")
def test_committed_mocks_match_examples(tmp_path):
    written = export_mocks(tmp_path)
    fresh = {p.name: json.loads(p.read_text()) for p in written}
    committed = {p.name: json.loads(p.read_text()) for p in MOCKS.glob("*.json")}
    assert set(fresh) == set(committed), "run: python -m opendpd.schemas export-mocks --out frontend/mocks"
    for name, data in fresh.items():
        assert committed[name] == data, f"{name} is stale; run: python -m opendpd.schemas export-mocks --out frontend/mocks"
        assert data["_mock"] is True
