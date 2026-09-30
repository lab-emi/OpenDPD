"""The Arena cost model: hand-derived counts, executable nonlinear references, instrumented forwards."""

import math
import subprocess
import sys

import numpy as np
import pytest

from opendpd.core import arena, arena_ops


def test_gru_count_matches_the_equations_term_by_term():
    hidden, inputs = 3, 2
    cost = arena_ops.count("gru", {"hidden_size": hidden, "num_layers": 1})
    matrices = 3 * hidden * (inputs + hidden)                      # W_i x and W_h h for r, z, n
    gates = 3 * hidden                                             # r*(W_hn h), (1-z)*n, z*h
    output = 2 * hidden
    functions = 3 * hidden                                         # 2H sigmoid + H tanh, one table each
    assert cost["explicit_mul"] == matrices + gates + output
    assert cost["nonlinear"] == {"sigmoid": 2 * hidden, "tanh": hidden}
    assert cost["nonlinear_mul"] == cost["nonlinear_add"] == functions
    # dot products: n-1 adds each; two bias vectors; five elementwise sums per unit; output bias
    adds = 3 * hidden * (inputs - 1) + 3 * hidden * (hidden - 1) + 6 * hidden + 5 * hidden + 2 * (hidden - 1 + 1)
    assert cost["explicit_add"] == adds
    assert cost["mul"] == cost["explicit_mul"] + functions and cost["add"] == adds + functions
    assert cost["ops"] == cost["mul"] + cost["add"]
    assert sum(item["mul"] for item in cost["items"]) == cost["explicit_mul"]


def test_memory_polynomial_reuses_delayed_basis_terms_and_pays_complex_coefficients():
    order, depth = 5, 25
    cost = arena_ops.count("mp_ls", {"K": order, "Q": depth, "rcond": 1e-4})
    powers = 2 + (order - 1 - 2)            # |x|^2, then |x|^3 and |x|^4; |x| is the square root
    basis = 2 * (order - 1)                 # x*|x|^k for k = 1..K-1, complex x real, once per sample
    assert cost["explicit_mul"] == powers + basis + 4 * order * depth
    assert cost["explicit_add"] == 1 + 2 * order * depth + 2 * (order * depth - 1)
    assert cost["nonlinear"] == {"sqrt": 1}
    assert arena_ops.parameter_count("mp_ls", {"K": order, "Q": depth}) == 2 * order * depth


def test_delta_networks_pay_their_encoder_and_get_no_sparsity_discount():
    params = {"hidden_size": 8, "num_layers": 1, "thx": 0., "thh": 0.}
    dense, delta = arena_ops.count("tres_gru", params), arena_ops.count("tres_deltagru", params)
    assert delta["explicit_mul"] == dense["explicit_mul"]
    assert delta["nonlinear"]["compare"] == delta["nonlinear"]["abs"] == 6 + 8
    assert delta["add"] > dense["add"] and delta["ops"] > dense["ops"]


def test_structural_zero_padding_is_not_arithmetic_but_is_reported_as_executed():
    cost = arena_ops.count("rvtdcnn", {"hidden_size": 5})
    convolution = next(item for item in cost["items"] if item["component"].startswith("conv2d"))
    # 4x5 window, 3x3 kernel, padding (1, 0): edge rows meet one row of zeros.
    assert convolution["mul"] == 3 * 3 * (6 + 9 + 9 + 6) == 270
    assert convolution["kernel_mul"] == 3 * 12 * 9 == 324
    assert arena_ops._window_taps(5, 3, 1) == [2, 3, 3, 3, 2] and arena_ops._window_taps(1, 3, 1) == [1]


def test_streaming_variants_cost_what_their_base_costs():
    for budget in arena.BUDGETS:
        params = arena.model_parameters("gru", budget)
        assert arena_ops.count("gru_stream", params) == arena_ops.count("gru", params)


def test_unreviewed_models_and_functions_are_refused():
    with pytest.raises(ValueError, match="multi-layer"):
        arena_ops.count("deltagru", {"hidden_size": 4, "num_layers": 2})
    with pytest.raises(KeyError):
        arena_ops.Ledger().put("softmax", softmax=1)
    with pytest.raises(ValueError):
        arena_ops.Ledger().put("negative", mul=-1)


def test_template_graph_is_counted_node_by_node():
    import json
    from opendpd.core.backbone_template import DEFAULT_DEFINITION
    definition = json.loads(DEFAULT_DEFINITION)
    cost = arena_ops.count("user_template", {"definition": DEFAULT_DEFINITION})
    hidden = definition["nodes"][0]["features"]
    gru = arena_ops.count("gru", {"hidden_size": hidden, "num_layers": 1})
    assert cost["mul"] == gru["mul"] and cost["add"] == gru["add"] + 2      # the I/Q residual sum
    definition["nodes"].insert(0, {"id": "features", "op": "iq_features", "inputs": ["input"]})
    definition["nodes"][1]["inputs"] = ["features"]
    definition["nodes"].insert(2, {"id": "norm", "op": "layer_norm", "inputs": ["memory"]})
    definition["nodes"][3]["inputs"] = ["norm"]
    richer = arena_ops.count("user_template", {"definition": json.dumps(definition)})
    assert richer["nonlinear"]["sqrt"] == 1 and richer["nonlinear"]["rsqrt"] == 1
    assert richer["explicit_mul"] == cost["explicit_mul"] + 3 * hidden * 4 + 4 + (3 * hidden + 2)


# --- every table price is backed by a reference implementation at the declared accuracy -------------

TABLES = {       # function, reference domain after free range reduction, uniform segments
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), 0., 16., 256),          # sigma(-x) = 1 - sigma(x)
    "tanh": (np.tanh, 0., 8., 256),                                     # odd
    "gelu": (lambda x: x * .5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2))), -8., 8., 512),
    "silu": (lambda x: x / (1 + np.exp(-x)), -12., 12., 512),
    "sqrt": (np.sqrt, 1., 4., 64),                                      # even-exponent normalisation
    "reciprocal": (lambda x: 1 / x, 1., 2., 64),                        # leading-zero normalisation
    "rsqrt": (lambda x: 1 / np.sqrt(x), 1., 4., 128),
    "sin": (np.sin, 0., math.pi / 2, 64),                               # quadrant folding
    "cos": (np.cos, 0., math.pi / 2, 64),
    "atan": (np.arctan, 0., 1., 64),                                    # octant folding, inside atan2
}
SATURATION = {"sigmoid": (16., 1.), "tanh": (8., 1.), "gelu": (-8., 0.), "silu": (-12., 0.)}


def first_order_table(function, low, high, segments, x):
    """slope * x + intercept from a uniform table: one multiplication and one addition."""
    edges = np.linspace(low, high, segments + 1)
    values = function(edges)
    slope = np.diff(values) / np.diff(edges)
    intercept = values[:-1] - slope * edges[:-1]
    index = np.clip(((x - low) / (high - low) * segments).astype(int), 0, segments - 1)
    return slope[index] * x + intercept[index]


@pytest.mark.parametrize("name", sorted(TABLES))
def test_reference_table_meets_the_declared_accuracy(name):
    function, low, high, segments = TABLES[name]
    assert segments <= 512
    x = np.linspace(low, high, 200001)[:-1]
    assert np.max(np.abs(first_order_table(function, low, high, segments, x) - function(x))) <= 2 ** -12
    if name in SATURATION:          # clamping outside the table costs no arithmetic and stays accurate
        edge, limit = SATURATION[name]
        outside = np.linspace(edge, 4 * edge, 1001)
        reference = function(outside)
        saturated = outside if name in ("gelu", "silu") and edge > 0 else np.full_like(outside, limit)
        assert np.max(np.abs(saturated - reference)) <= 2 ** -12
    priced = "atan2" if name == "atan" else name
    assert arena_ops.NONLINEAR_COST[priced] == ((3, 4) if name == "atan" else (1, 1))


def test_gelu_and_silu_are_linear_above_their_table():
    for name, edge in (("gelu", 8.), ("silu", 12.)):
        function = TABLES[name][0]
        x = np.linspace(edge, 64., 1001)
        assert np.max(np.abs(x - function(x))) <= 2 ** -12      # pass-through needs no arithmetic


def test_atan2_reference_costs_three_products_and_four_additions():
    """Octant fold (1 compare), reciprocal table, ratio product, arctangent table, octant offset."""
    rng = np.random.default_rng(0)
    i, q = rng.normal(size=20000), rng.normal(size=20000)
    big, small = np.maximum(abs(i), abs(q)), np.minimum(abs(i), abs(q))
    exponent = np.floor(np.log2(big))
    inverse = first_order_table(TABLES["reciprocal"][0], 1., 2., 64, big / 2 ** exponent) / 2 ** exponent
    angle = first_order_table(np.arctan, 0., 1., 64, np.clip(small * inverse, 0., 1.))
    angle = np.where(abs(q) > abs(i), math.pi / 2 - angle, angle)
    angle = np.where(i < 0, math.pi - angle, angle) * np.where(q < 0, -1., 1.)
    assert np.max(np.abs(angle - np.arctan2(q, i))) <= 2 ** -11      # two chained tables
    assert arena_ops.NONLINEAR_COST["atan2"] == (1 + 1 + 1, 1 + 1 + 1 + 1)


def test_hardswish_is_exact_without_a_table():
    x = np.linspace(-8, 8, 4001)
    exact = x * np.clip(x + 3, 0, 6) * (1 / 6)          # one add, one bound compare, two products
    assert arena_ops.NONLINEAR_COST["hardswish"] == (2, 2)
    import torch
    assert np.allclose(exact, torch.nn.functional.hardswish(torch.from_numpy(x)).numpy())


# --- the analytic count against the real modules ------------------------------------------------------

def test_every_sweep_configuration_matches_its_instantiated_model_and_instrumented_forward():
    from benchmark.verify_arena_operations import report
    result = report()
    assert result["problems"] == []
    assert result["configurations"] == sum(point["model_parameters"] is not None
        for model in arena.bundled_backbones() for point in arena.sweep(model.key))
    assert result["instrumented"] >= 70          # polynomial fits execute in NumPy and have no kernels
    dense = [row for row in result["rows"] if row["backbone"] in ("gru", "lstm", "qgru", "dgru")]
    assert all(.95 < row["ops"] / (2 * row["parameters"]) < 1.1 for row in dense)


def test_cost_model_is_torch_free():
    completed = subprocess.run([sys.executable, "-c",
        "import sys; from opendpd.core import arena, arena_ops; "
        "[arena_ops.count(m.key, p['model_parameters']) for m in arena.bundled_backbones() "
        "for p in arena.sweep(m.key) if p['model_parameters'] is not None]; assert 'torch' not in sys.modules"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
