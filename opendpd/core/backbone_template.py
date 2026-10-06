"""A bounded network description, never an executable Python plugin.

The .py template is deliberately a single literal assignment. Neither uploads
nor community files are imported, compiled, evaluated or added to sys.path.
Only the trusted layer implementations in template_network.py execute.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

MAX_SOURCE_BYTES = 32_768
MAX_NODES = 24
MAX_PARAMETERS = 250_000
MAX_FEATURES = 128
MAX_TOTAL_FEATURES = 1024
MODEL_KEY = "user_template"

TEMPLATE = '''"""OpenDPD backbone template v1 — edit BACKBONE below.

This is a network description, not executable Python. Do not add imports,
functions, classes, expressions or external dependencies. Studio never runs
this file. Input and output are real IQ tensors [batch, time, 2]. All layers
preserve time length; Conv1d uses causal left padding. State resets per frame.

Ops: linear(features, bias), gru/lstm(features, layers), conv1d(features,
kernel_size, dilation, bias), relu, tanh, gelu, silu, layer_norm, dropout(p),
identity, iq_features (2 -> 6), add, concat. Unspecified layers/bias/dilation
default to 1/True/1. inputs refer to earlier node IDs or "input". add/concat
take 2–4 inputs. All other ops take one. Final output must have 2 features.
Limits: 24 nodes, 128 features/layer, 250,000 parameters, no future samples.

For contribution, replace author with your public attribution. Checking the
contribution box publishes this whole file under Apache-2.0 in a review PR.
"""

BACKBONE = {
    "schema_version": 1,
    "name": "My Residual GRU",
    "description": "A causal GRU with an IQ residual connection.",
    "author": "",
    "license": "Apache-2.0",
    "nodes": [
        {"id": "memory", "op": "gru", "inputs": ["input"], "features": 24},
        {"id": "project", "op": "linear", "inputs": ["memory"], "features": 2},
        {"id": "result", "op": "add", "inputs": ["input", "project"]},
    ],
    "output": "result",
}
'''


class TemplateError(ValueError):
    pass


def _fail(message):
    raise TemplateError(message)


def _literal(node, depth=0):
    if depth > 16:
        _fail("Template nesting exceeds 16 levels.")
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool):
        return node.value
    if isinstance(node, ast.List):
        return [_literal(item, depth + 1) for item in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value in result:
                _fail("Dictionary keys must be unique literal strings; unpacking is not allowed.")
            result[key.value] = _literal(value, depth + 1)
        return result
    _fail("Only literal dictionaries, lists, strings, numbers and booleans are allowed. Python code is not executed.")


def scan_source(source: bytes, filename: str = "backbone.py") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.py", filename) or ".." in filename:
        _fail("Choose a .py file with a simple filename (letters, numbers, underscores or hyphens).")
    if not source or len(source) > MAX_SOURCE_BYTES:
        _fail("Backbone files must be non-empty and at most 32 KiB.")
    try:
        # Like a Python .py reader, permit a leading BOM for parsing only.
        # Storage, hashing and PR publication still use the original bytes.
        text = source.decode("utf-8-sig")
    except UnicodeDecodeError:
        _fail("The template must be UTF-8 text.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text) or any(char in text for char in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"):
        _fail("Control characters and bidirectional text controls are not allowed.")
    try:
        tree = ast.parse(text, mode="exec")
    except (SyntaxError, RecursionError, ValueError, MemoryError):
        _fail("Invalid or excessively nested Python template.")
    if sum(1 for _ in ast.walk(tree)) > 4096:
        _fail("Template syntax exceeds the complexity limit.")
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if (len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1
            or not isinstance(body[0].targets[0], ast.Name) or body[0].targets[0].id != "BACKBONE"):
        _fail("Use the downloaded template: one BACKBONE literal only. Imports, classes, functions and executable statements are refused.")
    definition = _literal(body[0].value)
    validate_definition(definition)
    return definition


def _int(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _text(value, name, maximum, empty=False):
    if (not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip())
            or any(ord(c) < 32 or c in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069" for c in value)):
        _fail(f"{name} must be plain text, at most {maximum} characters.")


def validate_definition(value: dict) -> dict:
    """Validate even persisted/API-supplied JSON; infer shapes without torch."""
    fields = {"schema_version", "name", "description", "author", "license", "nodes", "output"}
    if not isinstance(value, dict) or set(value) != fields:
        _fail("BACKBONE must contain exactly schema_version, name, description, author, license, nodes and output.")
    _int(value["schema_version"], "schema_version", 1, 1)
    for key, maximum in (("name", 64), ("description", 400), ("author", 100)):
        _text(value[key], key, maximum, empty=key == "author")
    if value["license"] != "Apache-2.0":
        _fail("Template v1 uses the OpenDPD Apache-2.0 license.")
    nodes = value["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
        _fail("A backbone must contain 1–24 nodes.")
    widths, parents = {"input": 2}, {}
    parameters = total_features = 0
    options = {
        "linear": {"features", "bias"}, "gru": {"features", "layers"}, "lstm": {"features", "layers"},
        "conv1d": {"features", "kernel_size", "dilation", "bias"}, "dropout": {"p"},
        **{key: set() for key in ("relu", "tanh", "gelu", "silu", "layer_norm", "identity", "iq_features", "add", "concat")},
    }
    for node in nodes:
        if not isinstance(node, dict) or not {"id", "op", "inputs"} <= set(node):
            _fail("Each node needs id, op and inputs.")
        name, op, inputs = node["id"], node["op"], node["inputs"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) or name in widths:
            _fail("Node IDs must be unique lowercase identifiers; 'input' is reserved.")
        if not isinstance(op, str) or op not in options or set(node) - {"id", "op", "inputs"} - options[op]:
            _fail(f"Unsupported operation or options at node {name}.")
        if not isinstance(inputs, list) or not all(isinstance(item, str) and item in widths for item in inputs):
            _fail(f"{name}: inputs must refer to input or earlier nodes; cycles and forward references are refused.")
        if not (2 <= len(inputs) <= 4 if op in ("add", "concat") else len(inputs) == 1):
            _fail(f"{name}: incorrect number of inputs.")
        inc = widths[inputs[0]]
        out = inc
        if "bias" in node and type(node["bias"]) is not bool:
            _fail(f"{name}: bias must be a boolean.")
        bias = node.get("bias", True)
        if op in ("linear", "conv1d", "gru", "lstm"):
            out = _int(node.get("features"), f"{name}.features", 1, MAX_FEATURES)
        if op == "linear":
            parameters += out * (inc + int(bias))
        elif op in ("gru", "lstm"):
            layers = _int(node.get("layers", 1), f"{name}.layers", 1, 2)
            gates = 3 if op == "gru" else 4
            parameters += gates * out * (inc + out + 2) + (layers - 1) * gates * out * (2 * out + 2)
        elif op == "conv1d":
            kernel = _int(node.get("kernel_size"), f"{name}.kernel_size", 1, 33)
            dilation = _int(node.get("dilation", 1), f"{name}.dilation", 1, 8)
            if (kernel - 1) * dilation > 128:
                _fail(f"{name}: convolution history exceeds 128 samples.")
            parameters += out * (inc * kernel + int(bias))
        elif op == "layer_norm":
            parameters += 2 * inc
        elif op == "iq_features":
            if inc != 2:
                _fail(f"{name}: iq_features requires 2 input features.")
            out = 6
        elif op == "dropout":
            p = node.get("p")
            if type(p) not in (int, float) or not 0 <= p <= .8:
                _fail(f"{name}: dropout p must be between 0 and 0.8.")
        elif op == "add" and any(widths[item] != inc for item in inputs):
            _fail(f"{name}: add inputs must have the same feature count.")
        elif op == "concat":
            out = sum(widths[item] for item in inputs)
        if out > MAX_FEATURES:
            _fail(f"{name}: at most 128 output features are allowed.")
        total_features += out
        widths[name], parents[name] = out, inputs
    output = value["output"]
    if not isinstance(output, str) or output not in parents or widths[output] != 2:
        _fail("output must name a node with exactly 2 IQ features.")
    used, pending = set(), [output]
    while pending:
        item = pending.pop()
        if item != "input" and item not in used:
            used.add(item)
            pending.extend(parents[item])
    if used != set(parents):
        _fail("Every node must contribute to output; remove unused nodes.")
    if not 1 <= parameters <= MAX_PARAMETERS or total_features > MAX_TOTAL_FEATURES:
        _fail("Network exceeds the budget (1–250,000 parameters, 1,024 total node features).")
    return {"parameters": parameters, "nodes": len(nodes), "features": widths, "lookahead_samples": 0}


def canonical_definition(definition):
    validate_definition(definition)
    return json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def parse_definition(text: str) -> dict:
    if not isinstance(text, str) or len(text) > MAX_SOURCE_BYTES:
        _fail("Network definition must be bounded JSON text.")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("JSON definition keys must be unique.")
            result[key] = value
        return result
    try:
        definition = json.loads(text, object_pairs_hook=unique)
    except (ValueError, RecursionError):
        _fail("Invalid network definition JSON.")
    validate_definition(definition)
    return definition


def source_sha256(source: bytes):
    return hashlib.sha256(source).hexdigest()


DEFAULT_DEFINITION = canonical_definition(scan_source(TEMPLATE.encode()))
