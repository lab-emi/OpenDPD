"""The built frontend must not reference the network (plan S06: works offline)."""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "opendpd" / "studio" / "static"


@pytest.mark.skipif(not (STATIC / "index.html").exists(), reason="frontend not built")
def test_built_assets_reference_no_external_hosts():
    pattern = re.compile(r"https?://(?!www\.w3\.org|localhost|127\.0\.0\.1)[A-Za-z0-9.-]+")
    for path in [STATIC / "index.html", *STATIC.glob("assets/*.js"), *STATIC.glob("assets/*.css")]:
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = {m.group(0) for m in pattern.finditer(text)}
        # library metadata/homepage strings inside JS bundles are not loads; only
        # attributes that trigger a fetch matter.
        loads = {m.group(0) for m in re.finditer(r"(?:src|href|url)\s*[=:(]\s*[\"']?(https?://[^\"' )]+)", text)}
        assert not loads, f"{path.name} loads external resources: {sorted(loads)[:5]}"
        # Plotly embeds inert URL strings (topojson base for geo traces, chart-studio
        # links); none is fetched by the scatter charts Studio draws.
        assert not any(h for h in hits if "fonts.googleapis" in h or "unpkg.com" in h or "jsdelivr" in h), hits
    info = STATIC / "build-info.json"
    assert info.exists(), "build-info.json (version stamp for /readyz) missing"
