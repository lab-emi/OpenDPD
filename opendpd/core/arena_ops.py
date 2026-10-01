"""Hardware-agnostic arithmetic cost of a DPD model: parameters, MUL, ADD, OPs.

Arena v2 does not time a host. The deployment target of a predistorter is
unknown (CPU, GPU, FPGA or ASIC), so cost is what the model's equations must
compute for ONE output IQ sample in steady-state streaming execution, batch 1:

* ``mul``  real multiplications (variable x variable or variable x stored weight)
* ``add``  real additions, subtractions, accumulations and nonzero-threshold compares
* ``ops``  = mul + add

Counting rules (identical for every model, all itemised in the result):

1. Dense, FP32, as published: no pruning, temporal sparsity, quantisation,
   weight sharing, constant folding or layer fusion is credited. Delta networks
   run with zero thresholds in Arena, so they pay their delta encoders and
   receive no sparsity discount.
2. Free: delays/shift registers, indexing, concatenation, reshaping, sign
   tests, negation/conjugation, multiplexing, exact powers of two, table reads
   and numerical guards (eps, zero masks).
3. A dot product of n terms is n MUL and n-1 ADD, plus one ADD for a bias.
4. Parameter-free per-sample input features (|x|, |x|^2, |x|^3, cos, sin ...)
   and polynomial basis products are evaluated once per new input sample by
   their cheapest exact arithmetic, then reused through free delay lines.
   Everything downstream of a learned layer is recomputed for every output
   sample exactly as the forward pass does; taps that only ever meet structural
   zero padding of a per-sample window are not arithmetic and are not counted.
5. A scalar nonlinear function is priced by ONE reference implementation at a
   common accuracy: a uniform-segment first-order table (slope x input +
   intercept), at most 512 segments, max abs error <= 2**-12 on its reference
   domain (verified by tests/unit/test_arena_ops.py). Range reduction by
   symmetry, leading-zero normalisation and power-of-two shifts is free.
   Table storage is memory, not arithmetic, and is excluded.

This module is torch-free so the API can recompute every count from a model
key and its parameters instead of trusting numbers stored in a result.
"""
from __future__ import annotations

# (MUL, ADD) equivalents of one evaluation of each nonlinear primitive.
NONLINEAR_COST = {
    # exact, table-free
    "relu": (0, 0),        # sign test + multiplexer
    "abs": (0, 0),         # sign test + conditional negation
    "compare": (0, 1),     # subtract a nonzero threshold, test the sign
    "hardswish": (2, 2),   # x * clamp(x + 3, 0, 6) * (1/6): one add, one bound compare, two products
    # one first-order table evaluation: slope * x + intercept
    "sigmoid": (1, 1),
    "tanh": (1, 1),
    "gelu": (1, 1),
    "silu": (1, 1),
    "sqrt": (1, 1),        # even-exponent normalisation to [1, 4), table, shift back
    "reciprocal": (1, 1),  # normalisation to [1, 2), table, shift back
    "rsqrt": (1, 1),
    "sin": (1, 1),         # quadrant folding of a binary angle, table on [0, pi/2]
    "cos": (1, 1),
    # octant fold (1 compare), min/max ratio = reciprocal table + 1 product,
    # arctangent table on [0, 1], octant offset (1 add)
    "atan2": (3, 4),
}
NONLINEAR_REFERENCE = {
    "accuracy": "max abs error <= 2**-12 on the reference domain",
    "implementation": "uniform-segment first-order (slope/intercept) table, <= 512 segments",
    "excluded": "table storage, range reduction by symmetry/normalisation/shifts",
}


class Ledger:
    """Itemised operation count; every entry is visible in the published result."""

    def __init__(self):
        self.items = []

    def put(self, component, mul=0, add=0, kernel_mul=0, **nonlinear):
        """``kernel_mul``: products the reference PyTorch forward runs inside matrix and
        convolution kernels for this component. It can exceed ``mul`` where that code
        multiplies structural zero padding or evaluates one projection twice; an
        instrumented forward pass checks it (benchmark/verify_arena_operations.py)."""
        unknown = set(nonlinear) - set(NONLINEAR_COST)
        if unknown:
            raise KeyError(f"Unpriced nonlinear primitive: {sorted(unknown)}")
        counts = {name: int(count) for name, count in nonlinear.items() if count}
        if min([mul, add, kernel_mul, *counts.values()], default=0) < 0:
            raise ValueError("Operation counts cannot be negative")
        self.items.append(dict(component=component, mul=int(mul), add=int(add),
                               kernel_mul=int(kernel_mul), nonlinear=counts))
        return self

    def dense(self, component, n_in, n_out, bias=True, executions=1, **nonlinear):
        """n_out dot products of n_in terms (+ bias)."""
        return self.put(component, mul=n_in * n_out, add=n_out * (n_in - 1 + int(bool(bias))),
                        kernel_mul=executions * n_in * n_out, **nonlinear)

    def total(self):
        nonlinear = {}
        for item in self.items:
            for name, count in item["nonlinear"].items():
                nonlinear[name] = nonlinear.get(name, 0) + count
        explicit_mul = sum(item["mul"] for item in self.items)
        explicit_add = sum(item["add"] for item in self.items)
        nonlinear_mul = sum(NONLINEAR_COST[name][0] * count for name, count in nonlinear.items())
        nonlinear_add = sum(NONLINEAR_COST[name][1] * count for name, count in nonlinear.items())
        mul, add = explicit_mul + nonlinear_mul, explicit_add + nonlinear_add
        return dict(mul=mul, add=add, ops=mul + add, explicit_mul=explicit_mul, explicit_add=explicit_add,
                    nonlinear_mul=nonlinear_mul, nonlinear_add=nonlinear_add,
                    kernel_mul=sum(item["kernel_mul"] for item in self.items),
                    nonlinear=dict(sorted(nonlinear.items())), items=self.items)


# --- shared building blocks --------------------------------------------------

def _envelope(ledger, *, amp=True, cube=False, fourth=False, direction=False, label="input features"):
    """|x|^2 = i^2 + q^2 once per sample; higher features reuse it."""
    mul, add, nonlinear = 2, 1, {}
    if amp:
        nonlinear["sqrt"] = 1
    if cube:
        mul += 1            # |x|^3 = |x|^2 * |x|
    if fourth:
        mul += 1            # |x|^4 = (|x|^2)^2
    if direction:
        nonlinear["reciprocal"] = 1
        mul += 2            # cos = i / |x|, sin = q / |x|
    return ledger.put(label, mul=mul, add=add, **nonlinear)


def _gru(ledger, n_in, hidden, bias=True, layers=1, label="GRU"):
    for layer in range(layers):
        width = n_in if layer == 0 else hidden
        name = label if layers == 1 else f"{label} layer {layer + 1}"
        # r, z, n: W_i x (+b_i) and W_h h (+b_h); r * (W_hn h); h' = (1 - z) * n + z * h
        ledger.put(name + " matrices", mul=3 * hidden * (width + hidden), kernel_mul=3 * hidden * (width + hidden),
                   add=3 * hidden * (width - 1) + 3 * hidden * (hidden - 1) + (6 * hidden if bias else 0))
        ledger.put(name + " gates", mul=3 * hidden, add=5 * hidden, sigmoid=2 * hidden, tanh=hidden)
    return ledger


def _lstm(ledger, n_in, hidden, bias=True, layers=1, label="LSTM"):
    for layer in range(layers):
        width = n_in if layer == 0 else hidden
        name = label if layers == 1 else f"{label} layer {layer + 1}"
        ledger.put(name + " matrices", mul=4 * hidden * (width + hidden), kernel_mul=4 * hidden * (width + hidden),
                   add=4 * hidden * (width - 1) + 4 * hidden * (hidden - 1) + (8 * hidden if bias else 0))
        # four input+hidden sums, c' = f * c + i * g, h' = o * tanh(c')
        ledger.put(name + " gates", mul=3 * hidden, add=5 * hidden, sigmoid=3 * hidden, tanh=2 * hidden)
    return ledger


def _delta_cell(ledger, n_in, hidden, gates, label):
    """Delta encoder and delta-memory accumulation with zero thresholds (dense)."""
    # x - x_prev and h - h_prev, then |delta| against the (zero) threshold
    ledger.put(label + " delta encoder", add=n_in + hidden, abs=n_in + hidden, compare=n_in + hidden)
    # every product is accumulated into a delta memory: n adds for an n-term row
    ledger.put(label + " matrices", mul=gates * hidden * (n_in + hidden), add=gates * hidden * (n_in + hidden),
               kernel_mul=gates * hidden * (n_in + hidden))
    return ledger


def _residual_tcn(ledger):
    """TRes skip path: Conv1d(2->3, k=3, dilation 16) - Hardswish - Conv1d(3->2, k=1) - Hardswish, bias-free."""
    ledger.put("residual conv 2->3 (k=3)", mul=18, add=15, kernel_mul=18, hardswish=3)
    return ledger.put("residual conv 3->2 (k=1)", mul=6, add=4, kernel_mul=6, hardswish=2)


def _window_taps(size, kernel, padding):
    """Valid (non-padding) taps of a stride-1 kernel at every output position of one axis."""
    outputs = size + 2 * padding - kernel + 1
    return [sum(0 <= position - padding + tap < size for tap in range(kernel)) for position in range(outputs)]


def _polynomial(ledger, orders, products, coefficients, complex_coefficients=True):
    """Envelope powers, distinct basis products, then one coefficient per basis term."""
    highest = max(orders, default=0)
    powers = 2 + max(0, highest - 2)     # |x|^2 (2 MUL), then one MUL per further power; |x| itself is the sqrt
    ledger.put("envelope powers", mul=powers if highest else 0, add=1 if highest else 0,
               sqrt=1 if highest else 0)
    ledger.put("basis products x*|x|^k (complex x real, one per distinct lag/order)", mul=2 * products)
    if complex_coefficients:
        ledger.put("complex coefficients x complex basis", mul=4 * coefficients, add=2 * coefficients)
    else:
        ledger.put("real coefficients x complex basis", mul=2 * coefficients)
    return ledger.put("complex accumulation", add=2 * (coefficients - 1))


def _layers(key, parameters):
    layers = int(parameters.get("num_layers", 1))
    if layers != 1 and key not in ("gru", "lstm", "qgru", "qgru_amp1", "dgru", "vdlstm", "tres_gru"):
        raise ValueError(f"No reviewed multi-layer operation count for {key}")
    return layers


# --- bundled models ----------------------------------------------------------

def _template(ledger, definition):
    from opendpd.core.backbone_template import validate_definition
    widths = validate_definition(definition)["features"]
    for node in definition["nodes"]:
        op, name = node["op"], node["id"]
        n_in, n_out = widths[node["inputs"][0]], widths[name]
        bias = node.get("bias", True)
        if op == "linear":
            ledger.dense(f"{name}: linear", n_in, n_out, bias)
        elif op == "gru":
            _gru(ledger, n_in, n_out, True, node.get("layers", 1), f"{name}: GRU")
        elif op == "lstm":
            _lstm(ledger, n_in, n_out, True, node.get("layers", 1), f"{name}: LSTM")
        elif op == "conv1d":
            ledger.dense(f"{name}: causal conv1d", n_in * node["kernel_size"], n_out, bias)
        elif op == "layer_norm":
            # mean (n-1 adds, x 1/n), centre (n), variance (n squares, n-1 adds, x 1/n), rsqrt, scale, affine
            ledger.put(f"{name}: layer norm", mul=3 * n_in + 2, add=4 * n_in - 2, rsqrt=1)
        elif op in ("relu", "tanh", "gelu", "silu"):
            ledger.put(f"{name}: {op}", **{op: n_in})
        elif op == "iq_features":
            _envelope(ledger, amp=True, cube=True, fourth=True, label=f"{name}: IQ features")
        elif op == "add":
            ledger.put(f"{name}: add", add=(len(node["inputs"]) - 1) * n_out)
        # concat, identity and dropout (inactive at inference) are free
    return ledger


def count(key, parameters):
    """Itemised per-sample cost of a registered Arena model. Streaming variants equal their base."""
    from opendpd.core.registry import get_model
    key = get_model(key).weights_from or key
    p = dict(parameters or {})
    h = int(p.get("hidden_size", 0))
    layers = _layers(key, p)
    ledger = Ledger()
    if key == "gru":
        _gru(ledger, 2, h, True, layers).dense("output layer", h, 2)
    elif key == "lstm":
        _lstm(ledger, 2, h, True, layers).dense("output layer", h, 2)
    elif key == "qgru":
        _envelope(ledger, amp=False, fourth=True)
        _gru(ledger, 4, h, True, layers).dense("output layer", h, 2)
    elif key == "qgru_amp1":
        _envelope(ledger, cube=True)
        _gru(ledger, 4, h, True, layers).dense("output layer", h, 2)
    elif key == "dgru":
        _envelope(ledger, cube=True, direction=True)
        _gru(ledger, 6, h, True, layers)
        ledger.dense("hidden layer", h, h, relu=h).dense("output layer on [hidden, features]", h + 6, 2)
    elif key == "vdlstm":
        _envelope(ledger, direction=True)
        _lstm(ledger, 4, h, True, layers)
        ledger.dense("lambda_1", h, 4).dense("lambda_2", h, 4)
        ledger.put("lambda * cos, lambda * sin", mul=8).dense("output layer", 8, 2)
    elif key == "tres_gru":
        _envelope(ledger, cube=True)
        _gru(ledger, 6, h, False, layers).dense("output layer", h, 2, bias=False)
        _residual_tcn(ledger).put("residual sum", add=2)
    elif key == "tres_deltagru":
        _envelope(ledger, cube=True)
        _delta_cell(ledger, 6, h, 3, "DeltaGRU")
        ledger.put("DeltaGRU gates", mul=3 * h, add=3 * h, sigmoid=2 * h, tanh=h)
        ledger.dense("output layer", h, 2, bias=False)
        _residual_tcn(ledger).put("residual sum", add=2)
    elif key == "deltagru":
        _envelope(ledger, cube=True, direction=True)
        _delta_cell(ledger, 6, h, 3, "DeltaGRU")
        ledger.put("DeltaGRU gates", mul=3 * h, add=3 * h, sigmoid=2 * h, tanh=h).dense("output layer", h, 2)
    elif key == "deltajanet":
        _envelope(ledger, cube=True, direction=True)
        _delta_cell(ledger, 6, h, 2, "DeltaJANET")
        ledger.put("DeltaJANET gates", mul=2 * h, add=2 * h, sigmoid=2 * h).dense("output layer", h, 2)
    elif key == "tcn":
        _envelope(ledger, cube=True, direction=True)
        ledger.dense("pointwise conv 6->C", 6, h, hardswish=h)
        for dilation in (1, 2, 4, 8):
            ledger.dense(f"depthwise conv k=5, dilation {dilation}", 5, h, bias=False, hardswish=h)
        ledger.dense("pointwise conv C->2", h, 2, bias=False).put("residual sum", add=2)
    elif key == "rvtdcnn":
        _envelope(ledger, cube=True)
        taps = [rows * 3 for rows in _window_taps(4, 3, 1) for _ in range(3)]      # 4x5 window, 3x3 kernel, pad (1, 0)
        ledger.put("conv2d 1->3 over the 4x5 window", mul=3 * sum(taps), add=3 * sum(taps),
                   kernel_mul=3 * 9 * len(taps), tanh=3 * len(taps))
        ledger.dense("hidden layer", 3 * len(taps), h, tanh=h).dense("output layer", h, 2)
    elif key == "mcldnn":
        _envelope(ledger, cube=True)
        plane = [a * b for a in _window_taps(5, 3, 1) for b in _window_taps(5, 3, 1)]   # 5 features x 5 samples
        ledger.put("conv2d 1->C over the 5x5 window", mul=h * sum(plane), add=h * sum(plane),
                   kernel_mul=h * 9 * len(plane))
        line = _window_taps(5, 3, 1)
        ledger.put("grouped conv1d 5->5C along the window", mul=5 * h * sum(line), add=5 * h * sum(line),
                   kernel_mul=5 * h * 3 * len(line))
        merged = [a * b for a in _window_taps(h, 3, 1) for b in _window_taps(5, 3, 1)]  # 10 channels, C x 5 plane
        ledger.put("conv2d 10->1 over the C x 5 plane", mul=10 * sum(merged), add=10 * sum(merged),
                   kernel_mul=10 * 9 * len(merged))
        _lstm(ledger, 5 * h, 8, True, 1)
        ledger.dense("dense 8->16", 8, 16).dense("output layer", 16, 2)
    elif key == "pgjanet":
        _envelope(ledger, direction=True)       # cos(atan2(q, i)) = i / |x| exactly
        for gate in ("a", "p1", "p2"):
            ledger.dense(f"input gate {gate} on [h, feature]", h + 1, h, tanh=h)
        ledger.put("u = a p1 p2 (1-a)(1-p1)(1-p2)", mul=5 * h, add=3 * h)
        ledger.dense("forget gate on [h, u]", 2 * h, h, sigmoid=h).dense("candidate on [h, u]", 2 * h, h, tanh=h)
        ledger.put("state update", mul=2 * h, add=2 * h).dense("output layer", h, 2)
    elif key == "dvrjanet":
        units = int(p.get("num_dvr_units", 3))
        _envelope(ledger).put("phase", atan2=1).put("h_I + h_Q", add=h)
        ledger.put("phase filter W_p theta + W_ph h", mul=h + h * h, add=h * (h - 1) + h, kernel_mul=h + h * h)
        ledger.put("magnitude filter W_ax |x| + W_ah h", mul=h + h * h, add=h * (h - 1) + h, kernel_mul=h + h * h)
        ledger.put("DVR units sum_k c_k |v - k/K|", mul=units * h, add=(2 * units - 1) * h, abs=units * h)
        ledger.put("cos, sin of the filtered phase", mul=2 * h, cos=h, sin=h)   # and a * cos, a * sin
        ledger.dense("forget gate", h, h, sigmoid=h)
        ledger.dense("cosine candidate on [h_I, a cos]", 2 * h, h, tanh=h)
        ledger.dense("sine candidate on [h_Q, a sin]", 2 * h, h, tanh=h)
        ledger.put("state updates", mul=4 * h, add=3 * h).dense("I output", h, 1).dense("Q output", h, 1)
    elif key == "bojanet":
        ledger.put("complex FIR bank 16 taps -> 6 (4 real filters)", mul=4 * 96, add=4 * 6 * 15 + 12, kernel_mul=4 * 96)
        ledger.put("vector demodulators |v|, cos, sin", mul=6 * 2 + 12, add=6, sqrt=6, reciprocal=6)
        ledger.put("recurrent gates W_i L + W_h h", mul=2 * (12 * h + h * h), add=2 * (12 * h + h * (h - 1) + h),
                   kernel_mul=2 * (12 * h + h * h), sigmoid=h, tanh=h)
        ledger.put("state update", mul=2 * h, add=2 * h).put("phase rotation h cos, h sin", mul=2 * h)
        ledger.dense("I projection", h, 1, executions=2).dense("Q projection", h, 1, executions=2)
        ledger.put("output sums", add=2)
    elif key == "apnrru":
        state = 2 * h + 3
        ledger.put("complex FIR bank 16 taps -> 3 (4 real filters)", mul=4 * 48, add=4 * 3 * 15 + 6, kernel_mul=4 * 48)
        _envelope(ledger, direction=True, label="phase reference r = conj(x) / |x|")
        ledger.put("phase normalisation of 4 complex inputs", mul=16, add=8)
        ledger.put("state rotation h r and h r*", mul=8 * h, add=4 * h)
        ledger.dense("RRU input layer", state + 8, 16, tanh=16).dense("RRU state layer", 16, state, tanh=state)
        ledger.put("RRU update sigmoid(C h) + Z v", mul=2 * state, add=state, sigmoid=state)
        ledger.dense("I projection", h, 1, bias=False, executions=2)
        ledger.dense("Q projection", h, 1, bias=False, executions=2).put("output sums", add=2)
    elif key == "gmp":
        memory, degree = 11, 5           # fixed in backbones/gmp.py
        terms = memory * (1 + (degree - 1) * memory)
        _polynomial(ledger, range(1, degree), (degree - 1) * memory, terms, complex_coefficients=False)
    elif key in ("mp_ls", "ilc_dpd"):
        order, depth = int(p["K"]), int(p["Q"])
        _polynomial(ledger, range(order), order - 1, order * depth)
    elif key == "gmp_ls":
        ka, la, kb, lb, mb, kc, lc, mc = (int(p[name]) for name in ("Ka", "La", "Kb", "Lb", "Mb", "Kc", "Lc", "Mc"))
        orders = [*range(ka), *range(1, kb + 1), *range(1, kc + 1)]
        _polynomial(ledger, orders, (ka - 1) + kb * mb + kc * mc, ka * la + kb * lb * mb + kc * lc * mc)
    elif key == "user_template":
        from opendpd.core.backbone_template import parse_definition
        _template(ledger, parse_definition(p["definition"]))
    else:
        raise KeyError(f"No reviewed operation count for {key}")
    return ledger.total()


def parameter_count(key, parameters):
    """Real trainable parameters, analytically; tests compare it with the instantiated model."""
    from opendpd.core.registry import get_model
    key = get_model(key).weights_from or key
    p = dict(parameters or {})
    h = int(p.get("hidden_size", 0))
    layers = _layers(key, p)
    gru = lambda n, bias=True: (3 * h * (n + h + (2 if bias else 0))
                                + (layers - 1) * 3 * h * (2 * h + (2 if bias else 0)))
    lstm = lambda n: 4 * h * (n + h + 2) + (layers - 1) * 4 * h * (2 * h + 2)
    if key == "gru":
        return gru(2) + 2 * h + 2
    if key == "lstm":
        return lstm(2) + 2 * h + 2
    if key in ("qgru", "qgru_amp1"):
        return gru(4) + 2 * h + 2
    if key == "dgru":
        return gru(6) + h * h + h + 2 * (h + 6) + 2
    if key == "vdlstm":
        return lstm(4) + 2 * (4 * h + 4) + 18
    if key in ("tres_gru", "tres_deltagru"):
        return gru(6, bias=False) + 2 * h + 24
    if key == "deltagru":
        return gru(6) + 2 * h + 2
    if key == "deltajanet":
        return 2 * h * (6 + h + 2) + 2 * h + 2
    if key == "tcn":
        return 29 * h
    if key == "rvtdcnn":
        return 30 + 37 * h + 2 * h + 2
    if key == "mcldnn":
        return 190 * h + 589
    if key == "pgjanet":
        return 7 * h * h + 10 * h + 2
    if key == "dvrjanet":
        return 7 * h * h + 7 * h + int(p.get("num_dvr_units", 3)) + 2
    if key == "bojanet":
        return 2 * h * h + 28 * h + 194
    if key == "apnrru":
        return 70 * h + 343
    if key == "gmp":
        return 495
    if key in ("mp_ls", "ilc_dpd"):
        return 2 * int(p["K"]) * int(p["Q"])
    if key == "gmp_ls":
        return 2 * (int(p["Ka"]) * int(p["La"]) + int(p["Kb"]) * int(p["Lb"]) * int(p["Mb"])
                    + int(p["Kc"]) * int(p["Lc"]) * int(p["Mc"]))
    if key == "user_template":
        from opendpd.core.backbone_template import parse_definition, validate_definition
        return validate_definition(parse_definition(p["definition"]))["parameters"]
    raise KeyError(f"No reviewed parameter count for {key}")
