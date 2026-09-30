"""Versioned, data-only DPD Arena rules and auditable result aggregation.

Training and inference live in arena_runner; importing this module never loads
torch or executes user code. All scores are derived from server-owned cases.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from opendpd.core import arena_ops
from opendpd.schemas.arena import ArenaBackbone, ArenaMetricSummary, ArenaProtocol, ArenaRow

ASSETS = Path(__file__).resolve().parents[1] / "arena_assets"
PROTOCOL_ID = "dpd-arena-v6-apa-b"
RESULTS_FILE = "reference-results-v6-apa-b.json"
CALIBRATION_FILE = "calibration-v6-apa-b.json"
SEEDS = [0, 1, 2]
# Octave parameter budgets. A model belongs to the smallest budget that holds it.
BUDGETS = [250, 500, 1000, 2000]
# Everything that determines trained weights and raw judge observations. Hashing
# source bytes keeps this module lightweight and works in source and wheel installs.
TRAINING_SOURCE_FILES = (
    "models.py",
    "datasets/demodulator.py",
    "opendpd/core/arena_engine.py", "opendpd/core/arena_metrics.py", "opendpd/core/arena_cuda.py",
    "opendpd/core/arena_training_backends.py",
    "opendpd/core/registry.py", "opendpd/core/backbone_builders.py",
    "opendpd/core/backbone_template.py", "opendpd/core/template_network.py",
    "opendpd/core/polynomial.py", "opendpd/core/ilc.py",
    "opendpd/core/streaming.py", "opendpd/services/streaming.py",
    "opendpd/core/virtual_pa.py", "opendpd/core/virtual_pa_kernel.py",
    "opendpd/core/metrics/__init__.py", "opendpd/core/metrics/registry.py",
    "opendpd/core/metrics/spectral_v2.py", "opendpd/core/metrics/general_v1.py",
    "modules/cuda_fast_training.py", "quant/modules/ops.py",
    "backbones/apnrru.py", "backbones/bojanet.py", "backbones/deltagru.py",
    "backbones/deltajanet.py", "backbones/dgru.py", "backbones/dvrjanet.py",
    "backbones/finite_iq.py", "backbones/gmp.py", "backbones/gru.py", "backbones/lstm.py",
    "backbones/mcldnn.py", "backbones/pgjanet.py", "backbones/qgru.py",
    "backbones/qgru_amp1.py", "backbones/rvtdcnn.py", "backbones/tcn.py",
    "backbones/tres_deltagru.py", "backbones/tres_gru.py", "backbones/vdlstm.py",
        "backbones/cuda_graph_frozen_dgru.py", "backbones/triton_deltagru.py",
)
# Rules, presets, the cost model and scoring. A change here needs new scores,
# not new weights: the training fingerprint below stays the same.
SCORING_SOURCE_FILES = ("opendpd/core/arena.py", "opendpd/core/arena_ops.py",
                        "opendpd/core/arena_runner.py", "opendpd/schemas/arena.py")
SOURCE_FILES = TRAINING_SOURCE_FILES + SCORING_SOURCE_FILES
BOARDS = [
    dict(board_id="apa-200mhz-b", title="APA_200MHz_b", dataset="APA_200MHz_b",
         description="Measured APA capture B, 200 MHz and 256-QAM; one frozen TRes-GRU cascade.",
         evidence_type="measured_data_simulation", evidence_label="Measured-data simulation",
         conditions=["apa-200mhz-b"]),
]

DETERMINISTIC = {"mp_ls", "gmp_ls", "ilc_dpd"}
# ILC remains available in Studio's DPD workflow, outside the Arena catalogue.
EXCLUDED_BACKBONES = {"ilc_dpd"}
# Polynomial families, fixed before any Arena v2 score was observed. MP keeps
# Q = 5K until the 50-sample inference halo caps the memory depth.
MP_PRESETS = {250: dict(K=5, Q=25), 500: dict(K=7, Q=35), 1000: dict(K=10, Q=50), 2000: dict(K=15, Q=50)}
GMP_PRESETS = {250: dict(Ka=5, La=9, Kb=2, Lb=8, Mb=3, Kc=2, Lc=8, Mc=2),
               500: dict(Ka=5, La=14, Kb=3, Lb=12, Mb=3, Kc=3, Lc=12, Mc=2),
               1000: dict(Ka=5, La=20, Kb=4, Lb=20, Mb=3, Kc=4, Lc=20, Mc=2),
               2000: dict(Ka=8, La=25, Kb=5, Lb=32, Mb=3, Kc=5, Lc=32, Mc=2)}
TRAINING = dict(epochs=240, frames_per_epoch="all training windows", frame_length=200,
                frame_stride=1, shuffle="seeded permutation without replacement each epoch", batch_size=64,
                optimizer="adamw", learning_rate=0.005, weight_decay=0.01,
                lr_end=0.0001, lr_patience_validation_checks=10, lr_factor=0.5,
                lr_threshold_db=0.01, lr_threshold_mode="abs",
                validation_every_epochs=1, updates_per_seed="240 * ceil((training_samples - 199) / 64)",
                frame_draw_hash="SHA256 of concatenated epoch permutations as little-endian int64",
                execution="FP32; zero-threshold Delta and BOJANET training use dense equivalents; PGJANET/DVRJANET/APNRRU may compile on CUDA; original inference",
                gradient_clip=200.0, validation_nperseg=4096, validation_context=200,
                max_parameters=4096, precision="FP32, TF32 disabled (polynomial fitting: complex128)",
                inference_window=200, inference_hop=100, inference_center_start=50,
                threads=4, peak_limit="maximum training-input magnitude",
                output_power_tolerance_db=0.5, quality_metric="evm-aclr-v3",
                stimuli="original measured capture inputs only; fixed 200-sample metric context at each end",
                checkpoint_selection="feasible validation output power first, then minimum 0.5 * (in-band error + worse-side ACLR)",
                pa="one validation-selected frozen TRes-GRU per condition, at most 5000 parameters",
                test_during_training=False)
SCORING = dict(scoring_version="configuration-ops-v3", budgets=BUDGETS,
               quality="evm-aclr-v3: 0.5 · (symbol EVM improvement + worse-side output ACLR improvement)",
               in_band_weight=0.5, adjacent_band_weight=0.5,
               operations="ops = mul + add per output IQ sample; nonlinear functions priced by the reference table",
               reference_parameters=1000, reference_operations=2000,
               parameter_weight=0.5, operation_weight=0.5,
               aggregation="per configuration; seed and condition mean; backbone summaries show best observed configuration")
RANKINGS = [
    dict(ranking_id="overall", title="Overall FoM", unit="dB",
         description="Per-configuration mean quality minus fixed parameter and operation costs. Backbone summaries show their best observed configuration; unavailable points are not zero scores."),
    dict(ranking_id="linearization", title="Best linearization", unit="dB",
         description="Mean equal-weight EVM and output ACLR improvement. Seed standard deviation is reported separately."),
    dict(ranking_id="parameter_efficiency", title="Parameter efficiency", unit="dB",
         description="Mean quality minus 10·log10(parameters / 1000), using the same reference for every configuration."),
    dict(ranking_id="arithmetic_efficiency", title="Arithmetic efficiency", unit="dB",
         description="Mean quality minus 10·log10(OPs / 2000): linearization per multiplication and addition."),
    dict(ranking_id="evm", title="Best EVM improvement", unit="dB",
         description="Largest mean improvement of the demodulated EVM over the linear baseline at any valid budget: in-band quality alone."),
    dict(ranking_id="aclr", title="Best ACLR improvement", unit="dB",
         description="Mean improvement of the worse adjacent channel, measured on the PA output itself."),
    *(dict(ranking_id=f"budget-{budget}", title=f"≤ {budget:,} parameters", unit="dB",
           description=f"FoM of every valid configuration with at most {budget:,} parameters, including smaller configurations.")
      for budget in BUDGETS),
]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    return {relative: file_hash(root / relative) for relative in SOURCE_FILES}


def calibration():
    path = ASSETS / CALIBRATION_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def active_calibration():
    """Only the calibration records used by the published Arena boards."""
    records = calibration()
    return {condition: records[condition] for board in BOARDS
            for condition in board["conditions"] if condition in records}


def training_budget(n_samples):
    """Every valid training window, including the final partial batch."""
    frames = (n_samples - TRAINING["frame_length"]) // TRAINING["frame_stride"] + 1
    if frames <= 0:
        raise ValueError("Arena training capture is shorter than one frame")
    return dict(frames_per_epoch=frames,
                optimizer_updates=TRAINING["epochs"] * math.ceil(frames / TRAINING["batch_size"]),
                frame_exposures=TRAINING["epochs"] * frames)


def oracle_sha256(simulation):
    """Bind the known PA equations as tightly as a learned judge's weights."""
    return canonical_hash({"virtual_pa": simulation["virtual_pa"],
        "parameters": simulation["parameters"],
        "implementation": file_hash(Path(__file__).with_name("virtual_pa_kernel.py"))})


def judge_hashes(condition):
    """Training and test use exactly the same frozen PA, on every board."""
    return {"pa": condition["teacher"]["sha256"]}


def budget_class(parameters):
    """The smallest budget that holds a model; None above the swept range."""
    return next((budget for budget in BUDGETS if parameters <= budget), None)


def _base(key):
    from opendpd.core.registry import get_model
    return get_model(key).weights_from or key


def scale_template(definition, budget):
    """Largest common width multiplier of a template's free hidden widths within a budget.

    Widths tied to the 2-feature input/output, to iq_features or to a concat that
    feeds a shape-constrained node stay fixed. A design already in this budget's
    class is evaluated exactly as submitted.
    """
    from opendpd.core.backbone_template import MAX_FEATURES, TemplateError, validate_definition
    native = validate_definition(definition)["parameters"]
    if budget_class(native) == budget:
        return definition
    nodes = definition["nodes"]
    parent = {name: name for name in ["input", *(node["id"] for node in nodes)]}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for node in nodes:
        if node["op"] in ("relu", "tanh", "gelu", "silu", "layer_norm", "dropout", "identity", "add"):
            for source in node["inputs"]:
                parent[find(source)] = find(node["id"])
    generators = {node["id"]: node for node in nodes if node["op"] in ("linear", "gru", "lstm", "conv1d")}
    fixed = {find("input"), find(definition["output"]),
             *(find(node["id"]) for node in nodes if node["op"] == "iq_features")}
    changed = True
    while changed:        # a concat whose own width is constrained pins every width it is made of
        changed = False
        for node in nodes:
            if node["op"] != "concat":
                continue
            group = find(node["id"])
            tied = group in fixed or any(find(name) == group for name in generators) or any(
                other["op"] == "concat" and other is not node and find(other["id"]) == group for other in nodes)
            for source in node["inputs"]:
                if tied and find(source) not in fixed:
                    fixed.add(find(source))
                    changed = True
    free = {name: node["features"] for name, node in generators.items() if find(name) not in fixed}
    best = None
    factors = sorted({width / base for base in set(free.values()) for width in range(1, MAX_FEATURES + 1)})
    for factor in factors or [1.0]:
        scaled = dict(definition, nodes=[
            dict(node, features=max(1, min(MAX_FEATURES, int(factor * free[node["id"]] + 1e-9))))
            if node["id"] in free else node for node in nodes])
        try:
            count = validate_definition(scaled)["parameters"]
        except TemplateError:
            continue
        if count <= budget and (best is None or count > best[0]):
            best = (count, scaled)
    return best[1] if best and budget_class(best[0]) == budget else None


def model_parameters(key, budget, supplied=None):
    """The one configuration a backbone enters at a budget, or None if it has none.

    Hidden-size families take the largest registered size within the budget. A
    configuration that also fits a smaller budget belongs to that class instead,
    so fixed-size and coarse-grained models are absent where they cannot scale.
    """
    from opendpd.core.backbone_template import canonical_definition, parse_definition
    from opendpd.core.registry import get_model
    if budget not in BUDGETS:
        raise ValueError(f"Arena budgets are {BUDGETS}")
    base = _base(key)
    if base in EXCLUDED_BACKBONES:
        raise ValueError("ILC is excluded from Arena rankings and submissions")
    params = get_model(base).defaults()
    if base == "user_template":
        scaled = scale_template(parse_definition((supplied or params)["definition"]), budget)
        return None if scaled is None else {"definition": canonical_definition(scaled)}
    if supplied:
        raise ValueError("Arena accepts only server-verified template definitions")
    if base == "mp_ls":
        params = {**MP_PRESETS[budget], "rcond": 1e-4}
    elif base == "gmp_ls":
        params = {**GMP_PRESETS[budget], "rcond": 1e-4}
    elif "hidden_size" in params:
        spec = get_model(base).param("hidden_size")
        fitting = [size for size in range(int(spec.minimum), int(spec.maximum) + 1)
                   if arena_ops.parameter_count(base, {**params, "hidden_size": size}) <= budget]
        if not fitting:
            return None
        params["hidden_size"] = fitting[-1]
    return params if budget_class(arena_ops.parameter_count(base, params)) == budget else None


def sweep(key, supplied=None):
    """Every protocol budget with its configuration (None = unavailable)."""
    return [dict(budget=budget, model_parameters=model_parameters(key, budget, supplied)) for budget in BUDGETS]


def bundled_backbones():
    from opendpd.core.registry import list_models
    labels = {"qgru": "QGRU (FP32)", "qgru_amp1": "QGRU amp1 (FP32)",
              "deltagru": "DeltaGRU (thresholds 0)", "deltajanet": "DeltaJANET (thresholds 0)",
              "tres_deltagru": "TRes-DeltaGRU (thresholds 0)",
              "user_template": "Template GRU (bundled example)"}
    return [ArenaBackbone(key=m.key, display_name=labels.get(m.key, m.display_name), family=m.family,
                          deterministic=m.key in DETERMINISTIC)
            for m in list_models() if "dpd" in m.roles and m.key not in EXCLUDED_BACKBONES]


def training_fingerprint():
    """What fixes the weights and raw observations; scoring text and code are outside it."""
    root = Path(__file__).resolve().parents[2]
    calibrated = calibration()
    return canonical_hash({"training": TRAINING, "seeds": SEEDS, "calibration": calibrated,
        "judge_hashes": {condition_id: judge_hashes(record) for condition_id, record in calibrated.items()},
        "sources": {relative: file_hash(root / relative) for relative in TRAINING_SOURCE_FILES}})


def cost_model():
    return dict(unit="per output IQ sample, steady-state streaming, batch 1, dense FP32",
                operations="ops = mul + add", nonlinear_reference=arena_ops.NONLINEAR_REFERENCE,
                nonlinear_cost={name: dict(mul=mul, add=add) for name, (mul, add) in arena_ops.NONLINEAR_COST.items()})


def protocol():
    rules = [{'title': 'APA_200MHz_b measured benchmark',
  'description': 'Every condition uses one frozen, validation-selected TRes-GRU for DPD training and test '
                 'evaluation. Arena uses the APA_200MHz_b capture: 200 MHz, 256-QAM. '
                 'PA identification uses only this measured capture. DPD '
                 'training, validation and test inputs are unchanged slices of the original measured captures. '},
 {'title': 'Parameter-budget sweep',
  'description': 'Evaluate each backbone at up to 250, 500, 1000 and 2000 real parameters. A distinct '
                 'configuration is trained once, recorded at its smallest fitting budget, and may enter every '
                 'larger budget ranking. Missing configurations are unavailable, never zero-valued measurements.'},
 {'title': 'Fixed training budget',
  'description': 'At each configuration: 240 full passes over every 200-sample training window at stride 1, '
                 'batch 64, including the final partial batch. Three seeds share their shuffled window orders '
                 'across backbones. Update counts depend on the dataset size. Validation runs each epoch; '
                 'AdamW starts at 0.005 and a plateau scheduler reduces it to at least 0.0001.'},
 {'title': 'Frozen PA and held-out scoring',
  'description': 'One TRes-GRU PA for APA_200MHz_b, selected using original-capture validation NMSE from '
                 'seeded candidates of at most 5000 parameters. '
                 'APA models are retrained after input-only symbol synchronization defines new disjoint splits. '
                 'PA test NMSE is reported after selection. DPD checkpoints minimize the equal-weight mean of '
                 'validation in-band error and worse-side ACLR, with feasible output power first. Validation '
                 'uses 4096-sample Welch segments and 200-sample end guards; it does not substitute proxy EVM '
                 'for final complete-symbol test EVM. Training workers load only train/validation arrays. Freeze all checkpoints '
                 'in a submission before loading test inputs. Final EVM, ACLR, power gates, FoM and ranks use '
                 'test observations only; validation metrics do not enter the score.'},
 {'title': 'Original measured inputs',
  'description': 'Use only original APA_200MHz_b measured-capture IQ samples for PA and DPD training, validation and test. '
                 'Reserve one complete input-synchronized '
                 'OFDM symbol per independently timed carrier for test; the preceding samples form disjoint '
                 'training and validation sets with 200-sample guards. No generated waveform, padding, '
                 'repetition or resampling supplies missing test samples.'},
 {'title': 'Common transmitter envelope',
  'description': 'Use the peak limit and target gain determined from the original PA training capture. One '
                 'linear-gain baseline is calibrated using only DPD training inputs. All configurations must keep '
                 'output power within 0.5 dB of the same target. EVM removes one common complex gain; output '
                 'power is checked separately.'},
 {'title': 'Hardware-agnostic cost',
  'description': 'Only DPD cost is counted: real parameters and algorithmic steady-state MUL plus ADD per output '
                 'IQ sample, batch 1, dense FP32. PA inference and training cost are excluded. Delays, indexing, '
                 'signs, exact powers of two and table reads are free. This ledger is not host timing or the '
                 'repeated work of the overlap-window evaluation harness.'},
 {'title': 'Nonlinear functions',
  'description': 'The published nonlinear reference implementation prices each scalar function using a '
                 'slope/intercept lookup table, at most 512 segments and at most 2^-12 reference-domain error. '
                 'Most functions cost 1 MUL + 1 ADD, Hardswish 2 + 2, atan2 3 + 4. Raw counts remain available '
                 'for other implementation assumptions.'},
 {'title': 'Linearization quality',
  'description': 'For each seed and condition, q = 0.5 times EVM improvement plus 0.5 times worse-side output '
                 'ACLR improvement, relative to the common baseline. Average conditions within a seed, then '
                 'average seeds. Standard deviation is reported separately and is not subtracted from the main '
                 'score. NMSE and AER are diagnostics only.'},
 {'title': 'Output-power gate',
  'description': 'Every prescribed condition and seed must keep output power within +/-0.5 dB. A configuration '
                 'with no positive mean quality improvement cannot earn a rank from a cost bonus. Retain negative '
                 'FoM values for otherwise valid configurations; there is no clipping to zero and no 3 dB '
                 'improvement threshold.'},
 {'title': 'Figure of merit',
  'description': 'For every configuration use the same reference costs: FoM = mean(q) - 5 log10(P/1000) - 5 '
                 'log10(OPs/2000). Halving both costs at the same EVM and ACLR adds 3.0103 dB. Parameter budgets '
                 'do not change the denominator. These weights express a declared tradeoff, not a physical '
                 'RF-efficiency law.'},
 {'title': 'Rankings',
  'description': 'Rank configurations individually. Optional backbone summaries show their best observed '
                 'configuration, not a sweep average or an unbiased estimate selected without test observations. '
                 'Budget rankings include every fitting configuration. Shipped/workspace and offline/stateful '
                 'results remain separate. Each EVM/ACLR-versus-cost plot has its own two-dimensional Pareto '
                 'front.'},
 {'title': 'Offline versus streaming',
  'description': 'Offline DPD uses 200-sample windows and retains the middle 100. Stateful variants carry state '
                 'and are evaluated separately. The PA processes the full record with continuous recurrent state. '
                 'Only original samples provide the fixed 200-sample metric context at both test-record ends. '},
 {'title': 'Classical training and ranking scope',
  'description': 'MP/GMP least-squares fits use only training-input feedback from the frozen PA: '
                 'PA(x_train)/G maps to x_train. They do not use ILC-generated data. The gradient-trained GMP '
                 'uses the same train/validation flow as neural DPD. ILC and ILC-to-MP are excluded from Arena '
                 'rankings, Pareto fronts and submissions. Deterministic fits use one seed; complex coefficients '
                 'count as two real parameters.'},
 {'title': 'EVM',
  'description': 'Demodulate the complete useful part of each OFDM symbol on its declared FFT grid; compare '
                 'occupied subcarriers of the cascade output with the original input reference after one common '
                 'complex gain. APA carriers are frequency-shifted and isolated separately at their frozen '
                 'input-only synchronized positions, using 1200 subcarriers per carrier. Report dB and percent. '
                 'Incomplete-symbol band error never substitutes for EVM.'},
 {'title': 'ACLR',
  'description': 'Compute adjacent-channel power from the cascade output itself, using the fixed spectral-v2 '
                 'Welch estimator. Normalize to its strongest in-band carrier, report negative dBc, and score the '
                 'worse (larger) of left and right. AER uses the error waveform and remains separately named '
                 'diagnostics; it never substitutes for ACLR.'},
 {'title': 'Reference scope',
  'description': 'Scores describe inversion of the stated frozen PA on held-out original measured inputs. Cascade outputs '
                 'are model predictions, not new predistorted hardware measurements. PA '
                 'NMSE is not a hard EVM or ACLR ceiling and does not validate new predistorted hardware inputs. '
                 'Seed variation measures training variability, not uncertainty across physical acquisitions.'}]
    payload = dict(protocol_id=PROTOCOL_ID, title="OpenDPD Arena · APA_200MHz_b",
        description="Compare DPD configurations on APA_200MHz_b using test EVM, output ACLR, parameters and arithmetic, with Pareto fronts.",
        boards=BOARDS, seeds=SEEDS, budgets=BUDGETS, rules=rules, training=TRAINING, scoring=SCORING,
        rankings=RANKINGS, cost_model=cost_model(),
        score_formula="q = 0.5 · (ΔEVM + ΔACLR), using symbol EVM and worse-side output ACLR in dB; Q = mean over conditions and seeds; FoM = Q − 5 log10(P/1000) − 5 log10(OPs/2000), OPs = MUL + ADD per output IQ sample. Every configuration uses the same frozen PA and fixed reference costs. Require output power within ±0.5 dB and Q > 0 to rank. Higher is better; seed standard deviation is reported separately.")
    calibrated = active_calibration()
    payload["pa_models"] = {identifier: dict(model="TRes-GRU", hidden_size=record["teacher"]["model"]["parameters"]["hidden_size"],
        parameters=record["teacher"]["parameters"], validation_nmse_db=record["teacher"]["pa_validation_nmse_db"],
        test_nmse_db=record["teacher"]["pa_test_nmse_db"], checkpoint_sha256=record["teacher"]["sha256"])
        for identifier, record in calibrated.items() if "pa_validation_nmse_db" in record["teacher"]}
    payload["training_sha256"] = training_fingerprint()
    manifest = source_manifest()
    payload["protocol_sha256"] = canonical_hash({**payload, "calibration": calibrated,
        "source_manifest": manifest,
        "judge_hashes": {condition_id: judge_hashes(record) for condition_id, record in calibrated.items()},
        "presets": _registered_presets(manifest)})
    return ArenaProtocol.model_validate(payload)


_PRESETS = {}


def _registered_presets(manifest):
    """Every backbone's sweep. It is a function of the bound sources and the declared
    preset tables, so the API does not rebuild it for each request."""
    key = canonical_hash([manifest, BUDGETS, MP_PRESETS, GMP_PRESETS])
    if key not in _PRESETS:
        _PRESETS.clear()
        _PRESETS[key] = json.dumps({m.key: sweep(m.key) for m in bundled_backbones()})
    return json.loads(_PRESETS[key])


def board(board_id):
    selected = next((b for b in protocol().boards if b.board_id == board_id), None)
    if selected is None:
        raise ValueError("Unknown Arena leaderboard")
    return selected


def _finite_number(value, field, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"Invalid Arena {field}: a finite{' positive' if positive else ''} number is required")
    return float(value)


def _verify_case(backbone, case, calibrated, reference_gains):
    condition_id = case["condition_id"]
    if case.get("evaluation_split") != "test":
        raise ValueError("Arena scoring requires held-out test observations")
    condition = calibrated.get(condition_id)
    if condition is None:
        raise ValueError(f"Missing Arena calibration for {condition_id}")
    attained, updates = case.get("attained_epochs"), case.get("optimizer_updates")
    selected_epoch = case.get("selected_epoch")
    if type(attained) is not int or type(updates) is not int:
        raise ValueError("Arena cases require exact attained epochs and optimizer updates")
    if backbone in DETERMINISTIC:
        if attained != 0 or updates != 0 or selected_epoch is not None:
            raise ValueError("Deterministic Arena fits cannot claim epochs, optimizer updates or a selected epoch")
    elif (attained != TRAINING["epochs"] or updates != training_budget(condition["counts"]["train"])["optimizer_updates"]
          or type(selected_epoch) is not int
          or not TRAINING["validation_every_epochs"] <= selected_epoch <= TRAINING["epochs"]
          or selected_epoch % TRAINING["validation_every_epochs"] != 0
          or not isinstance(case.get("frame_draw_sha256"), str)
          or re.fullmatch(r"[a-f0-9]{64}", case["frame_draw_sha256"]) is None):
        raise ValueError("Incomplete Arena training budget or invalid validation checkpoint/frame-draw hash")
    if case.get("data_sha256") != condition["data_sha256"]:
        raise ValueError(f"Arena dataset hash does not match {condition_id}")
    if case.get("teacher_sha256") != condition["teacher"]["sha256"]:
        raise ValueError(f"Arena teacher hash does not match {condition_id}")
    gain = _finite_number(case.get("reference_gain"), "reference gain", positive=True)
    if condition.get("reference_gain") is not None and gain != condition["reference_gain"]:
        raise ValueError("Arena reference gain differs from the frozen PA calibration")
    if reference_gains.setdefault(condition_id, gain) != gain:
        raise ValueError("Arena reference gain must be fixed across seeds and budgets of a condition")
    expected_judges = judge_hashes(condition)
    judges = case.get("judges")
    if not isinstance(judges, list) or not judges or any(not isinstance(j, dict) for j in judges):
        raise ValueError("Missing Arena judge observations")
    ids = [j.get("judge_id") for j in judges]
    if (len(ids) != len(expected_judges) or any(not isinstance(key, str) for key in ids)
            or set(ids) != set(expected_judges)):
        raise ValueError(f"Arena case requires exactly the frozen judges: {', '.join(expected_judges)}")
    for judge in judges:
        if judge.get("checkpoint_sha256") != expected_judges[judge["judge_id"]]:
            raise ValueError(f"Arena judge hash does not match {judge['judge_id']}")
        for key in ("nmse_db", "aclr_l_db", "aclr_r_db", "power_error_db", "ib_error_db", "evm_db",
                    "baseline_nmse_db", "baseline_aclr_l_db", "baseline_aclr_r_db", "baseline_ib_error_db",
                    "baseline_evm_db", "aer_l_db", "aer_r_db", "baseline_aer_l_db", "baseline_aer_r_db",
                    "reference_aclr_l_db", "reference_aclr_r_db"):
            _finite_number(judge.get(key), key)


def observation_quality(judge):
    """Test-only EVM/ACLR quality for one measured condition and frozen PA:
    in band and out of band weigh the same; out of band is the worse adjacent side."""
    evm = judge["baseline_evm_db"] - judge["evm_db"]
    adjacent = (max(judge["baseline_aclr_l_db"], judge["baseline_aclr_r_db"])
                - max(judge["aclr_l_db"], judge["aclr_r_db"]))
    return SCORING["in_band_weight"] * evm + SCORING["adjacent_band_weight"] * adjacent


def _budget_result(budget, params, count, cost, cases, seeds):
    """The power gate, conservative quality and cost-adjusted scores of one budget."""
    reasons, observations, qualities = [], [], []
    for seed in seeds:
        gains = []
        for case in (c for c in cases if c["seed"] == seed):
            for judge in case["judges"]:
                observations.append(judge)
                gains.append(observation_quality(judge))
                if abs(judge["power_error_db"]) > TRAINING["output_power_tolerance_db"]:
                    reasons.append("Output power falls outside the fixed-target ±0.5 dB envelope")
        qualities.append(statistics.fmean(gains))
    quality_std = statistics.stdev(qualities) if len(qualities) > 1 else 0.0
    conservative = statistics.fmean(qualities) - quality_std
    mean = lambda key: statistics.fmean(float(j[key]) for j in observations)
    worse = lambda prefix, name: statistics.fmean(max(j[f"{prefix}{name}_l_db"], j[f"{prefix}{name}_r_db"])
                                                  for j in observations)
    aclr, baseline_aclr, aer, baseline_aer = worse("", "aclr"), worse("baseline_", "aclr"), worse("", "aer"), worse("baseline_", "aer")
    percent = lambda value: 100 * 10 ** (value / 20)
    metrics = ArenaMetricSummary(nmse_db=mean("nmse_db"), aclr_db=aclr, baseline_nmse_db=mean("baseline_nmse_db"),
        baseline_aclr_db=baseline_aclr, nmse_improvement_db=mean("baseline_nmse_db") - mean("nmse_db"),
        aclr_improvement_db=baseline_aclr - aclr, ib_error_db=mean("ib_error_db"),
        evm_db=mean("evm_db"), evm_pct=percent(mean("evm_db")), baseline_evm_db=mean("baseline_evm_db"),
        baseline_evm_pct=percent(mean("baseline_evm_db")), evm_improvement_db=mean("baseline_evm_db") - mean("evm_db"),
        aer_db=aer, baseline_aer_db=baseline_aer, aer_improvement_db=baseline_aer - aer)
    quality = statistics.fmean(qualities)
    parameter_ratio = count / SCORING["reference_parameters"]
    operation_ratio = cost["ops"] / SCORING["reference_operations"]
    parameter_efficiency = quality - 10 * math.log10(parameter_ratio)
    arithmetic_efficiency = quality - 10 * math.log10(operation_ratio)
    return dict(budget=budget, available=True, model_parameters=params, parameters=count,
        mul=cost["mul"], add=cost["add"], ops=cost["ops"], nonlinear=cost["nonlinear"],
        nonlinear_mul=cost["nonlinear_mul"], nonlinear_add=cost["nonlinear_add"], operation_items=cost["items"],
        parameter_ratio=parameter_ratio, operation_ratio=operation_ratio,
        qualified=not reasons, reasons=sorted(set(reasons)),
        quality_db=statistics.fmean(qualities), quality_std_db=quality_std, quality_conservative_db=conservative,
        parameter_efficiency_db=parameter_efficiency, arithmetic_efficiency_db=arithmetic_efficiency,
        score=(parameter_efficiency + arithmetic_efficiency) / 2, metrics=metrics,
        expected_cases=len(cases), completed_cases=len(cases))


def summarize_cases(backbone, board_id, cases, points):
    """Recompute every gate, cost and ranking score; missing or duplicate cases fail closed.

    ``points`` lists each protocol budget with the evaluated ``model_parameters``
    (None where the backbone has no configuration). Parameters and operations
    are derived here from that configuration, never read from the result.
    """
    selected = board(board_id)
    base = _base(backbone)
    if base in EXCLUDED_BACKBONES:
        raise ValueError("ILC is excluded from Arena rankings and submissions")
    seeds = SEEDS[:1] if backbone in DETERMINISTIC else SEEDS
    if (not isinstance(points, list) or [p.get("budget") if isinstance(p, dict) else None for p in points] != BUDGETS):
        raise ValueError(f"Arena results must cover the budgets {BUDGETS} in order")
    configurations = {}
    for point in points:
        params = point.get("model_parameters")
        if params is None:
            continue
        if not isinstance(params, dict):
            raise ValueError("Arena model parameters must be an object")
        if base != "user_template" and params != model_parameters(backbone, point["budget"]):
            raise ValueError(f"{backbone} does not use its registered {point['budget']}-parameter preset")
        count = arena_ops.parameter_count(base, params)
        if budget_class(count) != point["budget"]:
            raise ValueError(f"A {count}-parameter model does not belong to the {point['budget']} budget")
        configurations[point["budget"]] = (params, count, arena_ops.count(base, params))
    if any(not isinstance(c, dict) or not isinstance(c.get("condition_id"), str)
           or type(c.get("seed")) is not int or type(c.get("budget")) is not int for c in cases):
        raise ValueError("Arena cases require a canonical budget, condition and integer seed")
    expected = {(budget, condition, seed) for budget in configurations
                for condition in selected.conditions for seed in seeds}
    identities = [(c["budget"], c["condition_id"], c["seed"]) for c in cases]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Arena result does not cover each required budget, condition and seed exactly once")
    calibrated, reference_gains = calibration(), {}
    for case in cases:
        _verify_case(backbone, case, calibrated, reference_gains)
        if case.get("parameters") != configurations[case["budget"]][1]:
            raise ValueError("Arena case parameter count differs from its registered configuration")
    results = []
    for budget in BUDGETS:
        if budget not in configurations:
            results.append(dict(budget=budget, available=False, qualified=False,
                                reasons=["No configuration of this backbone belongs to this parameter budget"]))
            continue
        params, count, cost = configurations[budget]
        results.append(_budget_result(budget, params, count, cost,
                                      [c for c in cases if c["budget"] == budget], seeds))
    qualified = [r for r in results if r["qualified"] and r["quality_db"] > 0]
    available = [r for r in results if r["available"]]
    peak = lambda value: max(map(value, qualified))
    scores = {ranking["ranking_id"]: None for ranking in RANKINGS}
    if qualified:
        scores.update(overall=peak(lambda r: r["score"]), parameter_efficiency=peak(lambda r: r["parameter_efficiency_db"]),
            arithmetic_efficiency=peak(lambda r: r["arithmetic_efficiency_db"]),
            linearization=peak(lambda r: r["quality_db"]),
            evm=peak(lambda r: r["metrics"].evm_improvement_db), aclr=peak(lambda r: r["metrics"].aclr_improvement_db),
            **{f"budget-{b}": max((r["score"] for r in qualified if r["parameters"] <= b), default=None) for b in BUDGETS})
    best = max(qualified or available, key=lambda r: (r["score"], -r["parameters"]), default=None)
    reasons = [] if qualified else sorted({reason for r in results for reason in r["reasons"]})
    if not qualified and any(r["qualified"] for r in results):
        reasons.append("No configuration has positive mean EVM/ACLR improvement")
    return dict(eligible=bool(qualified), eligibility_reasons=reasons, score=scores["overall"],
        rankings={key: dict(score=value) for key, value in scores.items()}, budgets=results,
        qualified_budgets=len(qualified), available_budgets=len(available),
        best_budget=best and best["budget"], parameters=best and best["parameters"],
        quality_db=best and best["quality_db"], quality_conservative_db=best and best["quality_conservative_db"],
        metrics=best and best["metrics"], seeds=seeds,
        ops_per_parameter=statistics.fmean(r["ops"] / r["parameters"] for r in available) if available else None,
        expected_cases=len(expected), completed_cases=len(cases))


def load_official_rows():
    path = ASSETS / RESULTS_FILE
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if canonical_hash(content) != payload.get("sha256"):
        raise ValueError("Bundled Arena results failed integrity verification")
    current = protocol()
    if payload.get("protocol_sha256") != current.protocol_sha256:
        raise ValueError("Bundled Arena results belong to a different protocol")
    rows = []
    seen = set()
    identities = {(selected.board_id, model.key) for selected in current.boards for model in bundled_backbones()}
    for raw in payload["rows"]:
        row = ArenaRow.model_validate(raw)
        identity = (row.board_id, row.backbone)
        if identity not in identities or identity in seen or row.origin != "official" or row.protocol_sha256 != current.protocol_sha256:
            raise ValueError("Invalid official Arena entry identity")
        seen.add(identity)
        if row.status == "succeeded":
            if [p.model_parameters for p in row.budgets] != [p["model_parameters"] for p in sweep(row.backbone)]:
                raise ValueError("An official Arena entry does not use the registered sweep")
            derived = summarize_cases(row.backbone, row.board_id, row.cases,
                                      [p.model_dump(mode="json") for p in row.budgets])
            row = ArenaRow.model_validate({**row.model_dump(), **derived})
        elif row.score is not None or row.eligible:
            raise ValueError("An incomplete official Arena entry cannot be ranked")
        rows.append(row)
    return rows
