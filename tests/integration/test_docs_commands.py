"""S14: the commands the tutorials show are executed in CI and every documented command form parses.

The tutorials use placeholders (WS, run-…, <pa_run_id>); this test runs the concrete form of every
command family in a temporary workspace, checks that no documented family is left out (benchmark
commands run in the weekly workflow and in the protocol tests), and checks that every flag the
tutorials mention is an option of that sub-command, so the docs cannot drift from the CLI unnoticed.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from opendpd.commands import build_parser
from tests.fixtures.synthetic import Impairments, synthesize

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = [ROOT / "docs" / "tutorials" / "gui-quickstart.md", ROOT / "docs" / "tutorials" / "headless-cli.md",
             ROOT / "docs" / "tutorials" / "waveform-evaluation.md", ROOT / "docs" / "tutorials" / "measured-dpd.md",
             ROOT / "docs" / "tutorials" / "adaptation-benchmark.md", ROOT / "docs" / "tutorials" / "streaming.md",
             ROOT / "docs" / "tutorials" / "deployment-export.md"]
OTHER_CI_SOURCES = [ROOT / ".github" / "workflows" / "weekly.yml", ROOT / "tests" / "integration" / "test_benchmark_protocol.py"]
NESTED = {"datasets", "benchmark", "waveforms", "measurements", "instruments", "adaptation"}

Family = Tuple[str, ...]


def documented_commands() -> List[Tuple[Path, List[str]]]:
    out = []
    for path in TUTORIALS:
        for block in re.findall(r"```(?:bash|sh)[^\n]*\n(.*?)```", path.read_text(encoding="utf-8"), re.S):
            for line in block.splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                tokens = shlex.split(line)
                if tokens and tokens[0] == "opendpd":
                    out.append((path, tokens[1:]))
    return out


def family_of(tokens: List[str]) -> Family:
    return tuple(tokens[:2]) if tokens and tokens[0] in NESTED else tuple(tokens[:1])


def option_strings() -> Dict[Family, Set[str]]:
    """Every option each (nested) sub-command accepts, straight from the argparse tree."""
    parser = build_parser()
    out: Dict[Family, Set[str]] = {}

    def walk(p, prefix: Family):
        subs = [a for a in p._actions if hasattr(a, "choices") and isinstance(a.choices, dict) and a.dest in ("command", "datasets_command", "benchmark_command", "waveforms_command", "measurements_command", "instruments_command", "adaptation_command")]
        if not subs:
            out[prefix] = {s for a in p._actions for s in a.option_strings}
            return
        for name, child in subs[0].choices.items():
            walk(child, prefix + (name,))

    walk(parser, ())
    return out


def test_every_documented_flag_is_an_option_of_its_command():
    options = option_strings()
    problems = []
    for path, tokens in documented_commands():
        fam = family_of(tokens)
        if fam not in options:
            problems.append(f"{path.name}: unknown command {' '.join(fam)}")
            continue
        for tok in tokens:
            if tok.startswith("--") and tok.split("=", 1)[0] not in options[fam]:
                problems.append(f"{path.name}: `opendpd {' '.join(tokens)}` uses {tok}, which `opendpd {' '.join(fam)}` does not accept")
    assert problems == [], "\n".join(problems)


def _cli(*args: str, cwd: Path, expect: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
    proc = subprocess.run([sys.executable, "-m", "opendpd.commands", *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == expect, f"opendpd {' '.join(args)} exited {proc.returncode}\n{proc.stderr[-3000:]}"
    return proc


def test_documented_commands_run_end_to_end(tmp_path):
    ws, ws2 = tmp_path / "WS", tmp_path / "WS2"
    executed: Set[Family] = set()

    def run(*args: str, expect: int = 0):
        executed.add(family_of(list(args)))
        return _cli(*args, cwd=tmp_path, expect=expect)

    run("datasets", "import-builtin", "DPA_200MHz", "--workspace", str(ws))
    listing = json.loads(run("datasets", "list", "--workspace", str(ws), "--json").stdout)
    assert [d["dataset_id"] for d in listing] == ["dpa-200mhz"]
    assert any(r["recipe_id"] == "pa-gru-smoke-v1" for r in json.loads(run("recipes", "--json").stdout))
    assert any(m["key"] == "gru" for m in json.loads(run("models", "--json").stdout))
    assert any(p["profile_id"] == "legacy-opendpd-v1" for p in json.loads(run("profiles", "--json").stdout))

    pa = json.loads(run("run", "--recipe", "pa-gru-smoke-v1", "--dataset", "dpa-200mhz", "--workspace", str(ws), "--json").stdout)
    assert pa["run"]["status"] == "succeeded" and pa["result"]["metrics"]
    pa_id = pa["run"]["run_id"]
    dpd = json.loads(run("run", "--recipe", "dpd-gru-smoke-v1", "--dataset", "dpa-200mhz", "--pa-run", pa_id, "--workspace", str(ws), "--json").stdout)
    assert dpd["run"]["status"] == "succeeded"
    dpd_id = dpd["run"]["run_id"]

    # what the GUI's "Download configuration (JSON)" serves is the resolved configuration; minus its
    # resolution block it is a user configuration the CLI validates and runs unchanged
    exported = json.loads((ws / "runs" / pa_id / "config.resolved.json").read_text())
    exported.pop("resolution", None)
    cfg_path = tmp_path / "experiment.json"
    cfg_path.write_text(json.dumps(exported))
    run("validate", "--config", str(cfg_path))
    again = json.loads(run("run", "--config", str(cfg_path), "--workspace", str(ws), "--json").stdout)
    assert again["run"]["status"] == "succeeded"

    def config_hash(run_id: str) -> str:
        return json.loads((ws / "runs" / run_id / "config.resolved.json").read_text())["resolution"]["config_sha256"]

    assert config_hash(again["run"]["run_id"]) == config_hash(pa_id)

    x, y = synthesize(20000, 3, impairments=Impairments(delay_samples=6))
    capture = tmp_path / "capture.csv"
    pd.DataFrame({"I_in": x[:, 0], "Q_in": x[:, 1], "I_out": y[:, 0], "Q_out": y[:, 1]}).to_csv(capture, index=False)
    run("datasets", "import", str(capture), "--id", "mine", "--fs", "800e6", "--bandwidth", "200e6", "--n-sub-ch", "10",
        "--nperseg", "2560", "--units", "normalized", "--workspace", str(ws))
    report = json.loads(run("datasets", "doctor", "mine", "--workspace", str(ws), "--json").stdout)
    assert any(i["code"] == "time_misalignment" for i in report["items"])
    run("datasets", "preprocess", "mine", "--version", "aligned-v1", "--delay", "6", "--preview", "--workspace", str(ws))
    run("datasets", "preprocess", "mine", "--version", "aligned-v1", "--delay", "6", "--workspace", str(ws))
    assert (ws / "datasets" / "mine" / "versions" / "aligned-v1" / "version.json").exists()

    applied = json.loads(run("apply", dpd_id, "--workspace", str(ws), "--json").stdout)
    assert applied["run"]["status"] == "succeeded"
    apply_id = applied["run"]["run_id"]

    # docs/tutorials/measured-dpd.md: the mock adapter refuses unarmed, captures when armed, and the import scores the
    # pair as mock dpd_measured evidence; a manual import of operator files takes the same path
    captures = tmp_path / "captures"
    assert "mock" in run("instruments", "list").stdout
    proc = _cli("instruments", "dry-run", "--apply-run", apply_id, "--out", str(captures), "--workspace", str(ws), cwd=tmp_path, expect=2)
    executed.add(("instruments", "dry-run"))
    assert "RF output stays off" in proc.stderr and not captures.exists()
    run("instruments", "dry-run", "--apply-run", apply_id, "--out", str(captures), "--workspace", str(ws), "--arm", "Docs Operator")
    measured = json.loads(run("measurements", "import", "--apply-run", apply_id, "--with-dpd", str(captures / "with_dpd.npy"),
                              "--without-dpd", str(captures / "without_dpd.npy"), "--conditions", str(captures / "conditions.json"),
                              "--mock", "--workspace", str(ws), "--json").stdout)
    assert measured["run"]["status"] == "succeeded" and measured["result"]["is_mock"] is True
    assert measured["result"]["evidence_type"] == "dpd_measured" and measured["result"]["measurement"]["captures"][0]["delay_samples"] == 37
    measured_id = measured["run"]["run_id"]
    manual = json.loads(run("measurements", "import", "--apply-run", apply_id, "--with-dpd", str(captures / "with_dpd.npy"),
                            "--without-dpd", str(captures / "without_dpd.npy"), "--conditions", str(captures / "conditions.json"),
                            "--power-with", "30.0", "--power-without", "30.0", "--workspace", str(ws), "--json").stdout)
    assert manual["result"]["is_mock"] is False and manual["result"]["measurement"]["declared_power_difference_db"] == 0.0
    assert any("not independently verified" in lim for lim in manual["result"]["limitations"])
    scored = json.loads(run("evaluate", measured_id, "--workspace", str(ws), "--profile", "general-spectral-v1", "--json").stdout)
    assert scored["evidence_type"] == "dpd_measured" and scored["is_mock"] is True
    run("report", measured_id, "--workspace", str(ws), "--format", "md", "--out", str(tmp_path / "measured-report.md"))
    assert "Capture (with dpd)" in (tmp_path / "measured-report.md").read_text()

    # docs/tutorials/waveform-evaluation.md: generate a reference waveform, capture it through a (synthetic) PA,
    # import with the binding, train, and read the pending profile's numbers
    pkg = tmp_path / "waveforms" / "lte20-seed1"
    generated = json.loads(run("waveforms", "generate", "--seed", "1", "--subframes", "2", "--out", str(pkg), "--json").stdout)
    shown = json.loads(run("waveforms", "show", str(pkg / "waveform.json"), "--json").stdout)
    assert shown["matches"] and shown["sha256"] == generated["sha256"]
    from scipy.signal import resample_poly
    from opendpd.core.waveforms import generate, read_package
    wf = generate(read_package(pkg)[0])
    played = resample_poly(np.roll(wf.x, -500), 4, 1)                 # captured at 122.88 MS/s, mid-waveform start
    distorted = played - 0.04 * np.abs(played) ** 2 * played
    capture_w = tmp_path / "lte20-capture.csv"
    pd.DataFrame({"I_in": played.real, "Q_in": played.imag, "I_out": distorted.real, "Q_out": distorted.imag}).to_csv(capture_w, index=False)
    bound = json.loads(run("datasets", "import", str(capture_w), "--id", "pa-lte20", "--fs", "122.88e6", "--bandwidth", "18e6",
                           "--n-sub-ch", "1", "--nperseg", "4096", "--units", "normalized", "--waveform", str(pkg / "waveform.json"),
                           "--workspace", str(ws), "--json").stdout)
    assert bound["signal"]["waveform"]["input_offset_samples"] == 500 and bound["signal"]["waveform"]["correlation"] > 0.99
    proc = _cli("datasets", "import", str(capture), "--id", "not-the-waveform", "--fs", "800e6", "--waveform", str(pkg),
                "--workspace", str(ws), cwd=tmp_path, expect=2)
    assert "does not correlate" in proc.stderr
    lte = json.loads(run("run", "--recipe", "pa-gru-smoke-v1", "--dataset", "pa-lte20", "--workspace", str(ws), "--json").stdout)
    assert lte["run"]["status"] == "succeeded"
    stored = json.loads((ws / "runs" / lte["run"]["run_id"] / "results" / "ofdm-lte20-evm-v1.json").read_text())
    evm = next(m for m in stored["metrics"] if m["name"] == "EVM_RMS")
    assert evm["status"] == "ok" and 0.1 < evm["value"] < 30.0, evm
    assert any("pending cross-validation" in lim for lim in stored["limitations"])
    again = json.loads(run("evaluate", lte["run"]["run_id"], "--workspace", str(ws), "--profile", "ofdm-lte20-evm-v1", "--json").stdout)
    assert next(m for m in again["metrics"] if m["name"] == "EVM_RMS")["value"] == pytest.approx(evm["value"], rel=1e-9)

    # docs/tutorials/adaptation-benchmark.md: a three-condition card (synthetic captures, each its own acquisition),
    # a pre-registered plan, every cell run through the executor and a report that says it is a rehearsal
    assert "2 conditions" in run("adaptation", "card", "apa-200mhz-batches-v1").stdout
    conditions = []
    for i, (cid, gain) in enumerate([("p30", 1.0), ("p32", 0.9), ("p34", 0.8)]):
        xa, ya = synthesize(12000, 21 + i, impairments=Impairments(gain=gain))
        cap = tmp_path / f"{cid}.csv"
        pd.DataFrame({"I_in": xa[:, 0], "Q_in": xa[:, 1], "I_out": ya[:, 0], "Q_out": ya[:, 1]}).to_csv(cap, index=False)
        run("datasets", "import", str(cap), "--id", f"unit2-{cid}", "--fs", "800e6", "--bandwidth", "200e6", "--n-sub-ch", "10",
            "--nperseg", "2560", "--units", "normalized", "--workspace", str(ws))
        conditions.append({"condition_id": cid, "dataset_id": f"unit2-{cid}", "role": "source" if i == 0 else "target",
                           "capture_batch": f"2026-09-0{1 + i // 2}-{'ab'[i % 2]}", "values": {"output_power_dbm": 30 + 2 * i}})
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"set_id": "my-pa-drive-v1", "device": "synthetic PA (tutorial)", "dimension": "output_power_dbm",
                                "conditions": conditions}))
    sealed = tmp_path / "card.sealed.json"
    run("adaptation", "card", str(card), "--workspace", str(ws), "--out", str(sealed))
    plan = tmp_path / "plan.json"
    run("adaptation", "plan", str(sealed), "--workspace", str(ws), "--pa-recipe", "pa-gru-smoke-v1", "--dpd-recipe", "dpd-gru-smoke-v1",
        "--budgets", "2000", "--seeds", "0", "--target-metric", "NMSE", "--target-threshold", "-25", "--out", str(plan))
    ran = json.loads(run("adaptation", "run", str(plan), "--workspace", str(ws), "--json").stdout)
    assert not ran["failed"] and ran["missing"] == 0 and len(ran["runs"]) == 14
    adaptation_md = tmp_path / "adaptation-report.md"
    proc = run("adaptation", "report", str(plan), "--workspace", str(ws), "--markdown", str(adaptation_md))
    assert "evidence bar NOT met" in proc.stdout and "14 cells, 0 without a number" in proc.stdout
    assert "rehearsal of the protocol" in adaptation_md.read_text() and (ws / "adaptation").glob("*.report.json")

    # docs/tutorials/streaming.md: the PA run re-scored under its streaming variant, with the evidence on the result
    streamed = json.loads(run("stream", pa_id, "--workspace", str(ws), "--chunk", "512", "--json").stdout)
    assert streamed["run"]["status"] == "succeeded" and streamed["run"]["task"] == "evaluate_pa"
    execution = streamed["result"]["execution"]
    assert execution["semantics"] == "streaming_stateful" and execution["chunk_samples"] == 512
    assert execution["consistency"]["within_tolerance"] and execution["lookahead_samples"] == 0
    assert streamed["result"]["models"][0]["model"]["key"] == "gru_stream"

    # docs/tutorials/deployment-export.md: a fixed-point-v1 package of the PA run with its verified C99 reference
    deployed = json.loads(run("deploy", pa_id, "--workspace", str(ws), "--out", str(tmp_path / "deploy.zip"), "--json").stdout)
    assert deployed["manifest"]["spec"]["spec_id"] == "fixed-point-v1" and (tmp_path / "deploy.zip").exists()
    assert deployed["manifest"]["verification"]["status"] in ("bit_exact", "not_run")
    assert [g["case_id"] for g in deployed["manifest"]["golden"]][:2] == ["normal", "extreme"]

    package = json.loads(run("export", pa_id, "--workspace", str(ws), "--kind", "share", "--json").stdout)
    zip_path = Path(package["path"])
    assert zip_path.exists()
    run("import", str(zip_path), "--workspace", str(ws2), "--inspect")
    run("import", str(zip_path), "--workspace", str(ws2))
    assert (ws2 / "runs" / pa_id / "run.json").exists()
    scored = json.loads(run("evaluate", pa_id, "--workspace", str(ws), "--profile", "legacy-opendpd-v1", "--json").stdout)
    assert scored["metrics"]
    run("report", pa_id, "--workspace", str(ws), "--format", "md", "--out", str(tmp_path / "report.md"))
    assert (tmp_path / "report.md").read_text().startswith("#")

    # every family the tutorials show is executed here or in another CI source
    elsewhere = "\n".join(p.read_text(encoding="utf-8") for p in OTHER_CI_SOURCES)
    missing = []
    for path, tokens in documented_commands():
        fam = family_of(tokens)
        if fam in executed:
            continue
        needle = " ".join(("opendpd",) + fam) if fam[0] != "benchmark" else f"benchmark {fam[1]}"
        if needle not in elsewhere and f'"{fam[-1]}"' not in elsewhere:
            missing.append(f"{path.name}: opendpd {' '.join(fam)}")
    assert missing == [], "documented commands not executed by any CI source: " + ", ".join(missing)
