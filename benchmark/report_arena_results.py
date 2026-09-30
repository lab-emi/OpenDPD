"""Write the Markdown report of the bundled Arena reference results.

Every number is read from the content-sealed reference bundle after the loader
has recomputed its operations, scores and rankings from the raw cases. Nothing
is trained, judged or edited here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

from opendpd.core import arena

OFFLINE = "offline_overlap_200_100"


def name(row):
    return row.display_name


def fixed(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def ranked(rows, ranking_id, semantics=OFFLINE):
    """The service's order: inside one execution cohort, by score, then size and key."""
    group = [row for row in rows if row.status == "succeeded" and row.eligible and row.execution_semantics == semantics
             and row.rankings[ranking_id].score is not None]
    return sorted(group, key=lambda row: (-row.rankings[ranking_id].score, row.parameters or 0, row.backbone, row.entry_id))


def weighted_fom(row, mul_weight):
    """Sensitivity of the best observed configuration, retaining fixed reference costs."""
    scores=[]
    for point in row.budgets:
        if not point.qualified or point.quality_db is None or point.quality_db <= 0:
            continue
        p = point.parameters / 1000
        a = (mul_weight*point.mul + point.add) / (1000*(mul_weight+1))
        scores.append(point.quality_db - 5*math.log10(p) - 5*math.log10(a))
    return max(scores, default=None)


def spearman(first, second):
    def positions(values):
        order = sorted(range(len(values)), key=lambda index: -values[index])
        result = [0.0] * len(values)
        for position, index in enumerate(order):
            result[index] = float(position)
        return result
    if len(first) < 3:
        return None
    a, b = positions(first), positions(second)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    spread = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / spread if spread else None


def budget_cell(point):
    if not point.available:
        return "—"
    if point.quality_conservative_db is None:
        return "failed"
    return fixed(point.quality_conservative_db) if point.qualified else f"({fixed(point.quality_conservative_db)})"


def report(rows, protocol, payload_sha):
    """Report every evaluated point without treating a missing size as a zero."""
    out = ["# OpenDPD 2.3 Arena reference results", "",
        "Arena uses only APA_200MHz_b. One frozen TRes-GRU PA is shared by all DPD architectures. "
        "PA and DPD train, validation and test inputs are unchanged slices of original measured captures. "
        "ILC-based entries are excluded. MP/GMP least-squares models use training-only PA feedback, not ILC data. "
        "EVM is measured on complete held-out symbols and ACLR on the emitted PA output. "
        "These are fixed-PA simulations, not hardware measurements.", "",
        "Every neural configuration is retrained for 240 complete passes over all training windows, "
        "batch 64 and stride 1. "
        "These v6 results use the same full-window "
        "recipe throughout; a fixed budget still does not establish each architecture's optimum.", "",
        f"{len(rows)} backbone/board entries; {sum(r.completed_cases for r in rows)} completed condition/seed/configuration cases.", "",
        "```text", "q = ½ · (ΔEVM + ΔACLR)",
        "FoM = mean(q) − 5 log10(P/1000) − 5 log10(OPs/2000)", "```", "",
        "Every size uses the same cost reference. Three seeds report mean and standard deviation; the main score "
        "does not subtract the standard deviation. Require output power within ±0.5 dB for every case and positive "
        "mean quality to rank. Negative FoM is preserved; unavailable configurations are not zero scores. "
        "Costs include DPD only, with OPs = MUL + ADD at the published nonlinear reference price.", "",
        "## Frozen PA models", "",
        "| Condition | Parameters | Validation NMSE (dB) | Test NMSE (dB) |",
        "|---|---:|---:|---:|"]
    for condition, pa in protocol.pa_models.items():
        out.append(f"| {condition} | {pa.parameters} | {pa.validation_nmse_db:.4f} | {pa.test_nmse_db:.4f} |")
    out += ["", "The PA is selected by validation NMSE only. "
        "Its held-out test NMSE is reported after selection and is not a hard EVM or ACLR ceiling.", "",
        "## Training budget", "",
        "All valid 200-sample windows are shuffled without replacement each epoch; the last partial batch is included. "
        "Validation-only spectral quality selects checkpoints, with feasible output power first. All final metrics use test data.", "",
        "| Condition | Full epochs | Windows/epoch | Updates/seed |", "|---|---:|---:|---:|"]
    for condition, record in arena.active_calibration().items():
        budget = arena.training_budget(record["counts"]["train"])
        out.append(f"| {condition} | 240 | {budget['frames_per_epoch']:,} | {budget['optimizer_updates']:,} |")
    out += ["", "## Leaders", "", "Best observed configurations in the offline cohort; backbone summaries do not replace the full sweep below.", "",
        "| Ranking | " + " | ".join(b.title for b in protocol.boards) + " |", "|---|" + "---|"*len(protocol.boards)]
    for ranking in protocol.rankings:
        cells=[]
        for board in protocol.boards:
            order=ranked([r for r in rows if r.board_id==board.board_id],ranking.ranking_id)
            cells.append(f"{name(order[0])} · {fixed(order[0].rankings[ranking.ranking_id].score)}" if order else "—")
        out.append(f"| **{ranking.title}** | " + " | ".join(cells) + " |")
    out += ["", "## Overall configuration results", "",
        "Every row below is a trained configuration. EVM dB and ACLR average seeds within one measured-source "
        "dataset; percent is converted from mean EVM dB. Studio's four Pareto plots compare EVM and ACLR against parameters and operations "
        "and execution cohorts. Diamonds mark each two-dimensional front, with seed error bars. "
        "Budget subrankings include every configuration whose actual parameter count fits the cap.", ""]
    for board in protocol.boards:
        out += [f"## {board.title}", ""]
        entries=[r for r in rows if r.board_id==board.board_id]
        for mode in sorted({r.execution_semantics for r in entries}):
            out += [f"### {mode}", "", "| Backbone | P | OPs/sample | Quality ± seed SD (dB) | EVM (%) | EVM (dB) | ACLR (dBc) | FoM (dB) | Status |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
            points=[(r,p) for r in entries if r.execution_semantics==mode for p in r.budgets if p.available and p.metrics]
            for r,p in sorted(points,key=lambda item: -(item[1].score if item[1].score is not None else -1e10)):
                valid=r.status=='succeeded' and p.qualified and p.quality_db is not None and p.quality_db>0
                out.append(f"| {name(r)} | {p.parameters} | {p.ops} | {fixed(p.quality_db)} ± {fixed(p.quality_std_db)} | "
                    f"{fixed(p.metrics.evm_pct)} | {fixed(p.metrics.evm_db)} | {fixed(p.metrics.aclr_db)} | "
                    f"{fixed(p.score) if valid else '—'} | {'Ranked' if valid else 'Unranked'} |")
            for r in entries:
                if r.execution_semantics==mode and r.status!='succeeded':
                    out.append(f"| {name(r)} | — | — | — | — | — | — | — | {r.status} |")
            out.append("")
    out += ["## Cost sensitivity", "", "Best observed FoM when MUL is priced as 1, 4 or 16 ADDs, "
        "using fixed references of 1000 MUL and 1000 ADD. PA cost is always excluded.", "",
        "| Board | Backbone | MUL:ADD 1:1 | 4:1 | 16:1 |", "|---|---|---:|---:|---:|"]
    for board in protocol.boards:
        for row in ranked([r for r in rows if r.board_id==board.board_id],'overall'):
            out.append(f"| {board.title} | {name(row)} | " + " | ".join(fixed(weighted_fom(row,w)) for w in (1,4,16)) + " |")
    out += ["", "## Reproducibility", "", "[Protocol](../protocols/dpd-arena-v6.md) · "
        "[PA qualification](arena-pa-qualification.md) · [Independent audit](arena-reference-audit.json)", "",
        f"Protocol SHA-256: `{protocol.protocol_sha256}`.", "", f"Training SHA-256: `{protocol.training_sha256}`.", "",
        f"Reference bundle content seal: `{payload_sha}`.", "", "No production deployment or hardware RF measurement was performed."]
    return "\n".join(out)+"\n"


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/performance/arena-reference-results.md"))
    args = parser.parse_args()
    rows = arena.load_official_rows()
    if not rows:
        raise SystemExit("No bundled Arena results to report")
    seal = json.loads((arena.ASSETS / arena.RESULTS_FILE).read_text())["sha256"]
    args.out.write_text(report(rows, arena.protocol(), seal))
    print(f"Wrote {args.out} for {len(rows)} rows")


if __name__ == "__main__":
    main()
