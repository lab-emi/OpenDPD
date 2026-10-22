"""Capture real NR test-model responses as fixtures for the Studio unit tests.

The vitest suites run without a backend, so the numbers they assert (N_RB, FFT size, sample counts, bin positions, preset ids, error texts)
must be the ones the real backend produces. This script asks the real local API (FastAPI test client, temporary workspace) for them:

    PYTHONPATH=. python scripts/capture_nr_tm_fixtures.py          # rewrite frontend/src/test/nrTmFixtureResponses.json
    PYTHONPATH=. python scripts/capture_nr_tm_fixtures.py --check  # exit 1 when the committed file differs from the backend

Input: `frontend/src/test/nrTmFixtureRecipes.json`
    {"resolve": [recipe, ...],                      POST /signal-generator/nr-test-models/resolve, answer or 422 envelope kept verbatim
     "generate": [{"name": str, "recipe": recipe}]} resolve + POST /signal-generator/batches + GET /signal-generator/signals/{id}

A recipe is the complete object the Studio sends (`NrTmRecipe`: no optional key left out), so tests look an answer up by the recipe they send.
The generated signals keep every scalar and every allocation / EVM-bin list verbatim; the long plot arrays (PSD, time, constellation, CCDF) are
decimated to `DECIMATE` points because no test reads them. Run this again after the backend changes; `fixtureResolve` throws for a recipe that is not
in the file and names the command.
"""
import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "frontend/src/test/nrTmFixtureRecipes.json"
RESPONSES = ROOT / "frontend/src/test/nrTmFixtureResponses.json"
API = "/api/v1"
TM = f"{API}/signal-generator/nr-test-models"
DECIMATE = 8
LONG = ("time_us", "time_i", "time_q", "time_envelope", "frequency_mhz", "psd_dbfs_hz", "constellation_i", "constellation_q",
        "reference_i", "reference_q", "ccdf_db", "ccdf_probability", "evm_per_symbol_percent")


def decimate(values):
    if len(values) <= DECIMATE:
        return values
    step = (len(values) - 1) / (DECIMATE - 1)
    return [values[round(i * step)] for i in range(DECIMATE)]


def capture():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

    from opendpd import __version__
    from opendpd.server.app import create_app
    from opendpd.server.security import CSRF_HEADER

    wanted = json.loads(RECIPES.read_text())
    with tempfile.TemporaryDirectory() as workspace:
        with TestClient(create_app(Path(workspace) / "ws", bootstrap_token="fixtures"), base_url="http://127.0.0.1:8877") as client:
            auth = client.post(f"{API}/session/bootstrap", json={"token": "fixtures"}).json()
            client.headers[CSRF_HEADER] = auth["csrf_token"]
            client.headers["origin"] = "http://127.0.0.1:8877"
            resolutions = []
            for recipe in wanted["resolve"]:
                response = client.post(f"{TM}/resolve", json=recipe)
                resolutions.append({"request": recipe, "status": response.status_code, "body": response.json()})
            signals = {}
            for item in wanted["generate"]:
                resolved = client.post(f"{TM}/resolve", json=item["recipe"])
                resolved.raise_for_status()
                batch = client.post(f"{API}/signal-generator/batches", json={"configs": [resolved.json()["config"]]})
                batch.raise_for_status()
                signal = client.get(f"{API}/signal-generator/signals/{batch.json()[0]['signal_id']}").json()
                analysis = signal["analysis"]
                for key in LONG:
                    if key in analysis:
                        analysis[key] = decimate(analysis[key])
                signals[item["name"]] = {"recipe": item["recipe"], "signal": signal}
    return {"_generated_by": "scripts/capture_nr_tm_fixtures.py", "_opendpd_version": __version__,
            "_note": "Real answers of the local API for the recipes in nrTmFixtureRecipes.json. Long plot arrays of generated signals are decimated to "
                     f"{DECIMATE} points; every other field is verbatim. Regenerate with the command in the script header; do not edit by hand.",
            "resolutions": resolutions, "signals": signals}


def render(data):
    lines = ["{"]
    for key in ("_generated_by", "_opendpd_version", "_note"):
        lines.append(f"{json.dumps(key)}: {json.dumps(data[key])},")
    lines.append('"resolutions": [')
    lines.append(",\n".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in data["resolutions"]))
    lines.append("],")
    lines.append('"signals": {')
    lines.append(",\n".join(f"{json.dumps(name)}: {json.dumps(entry, sort_keys=True, separators=(',', ':'))}" for name, entry in data["signals"].items()))
    lines.append("}}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed file instead of writing it")
    text = render(capture())
    json.loads(text)
    if parser.parse_args().check:
        if RESPONSES.read_text() != text:
            print(f"{RESPONSES.relative_to(ROOT)} differs from what the backend answers now; run this script without --check", file=sys.stderr)
            return 1
        print("fixtures match the backend")
        return 0
    RESPONSES.write_text(text)
    print(f"wrote {RESPONSES.relative_to(ROOT)}: {len(text) // 1024} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
