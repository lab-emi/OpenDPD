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

import pandas as pd
import pytest

from opendpd.commands import build_parser
from tests.fixtures.synthetic import Impairments, synthesize

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = [ROOT / "docs" / "tutorials" / "gui-quickstart.md", ROOT / "docs" / "tutorials" / "headless-cli.md"]
OTHER_CI_SOURCES = [ROOT / ".github" / "workflows" / "weekly.yml", ROOT / "tests" / "integration" / "test_benchmark_protocol.py"]
NESTED = {"datasets", "benchmark"}

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
        subs = [a for a in p._actions if hasattr(a, "choices") and isinstance(a.choices, dict) and a.dest in ("command", "datasets_command", "benchmark_command")]
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
