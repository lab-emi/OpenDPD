"""Model registry: the single list of models the CLI, API and GUI may use.

Each descriptor records what a model *is* (family, parameters and their
bounds), what it *can do* (roles, execution semantics, look-ahead, export),
and what has actually been *tested* (devices with evidence). Nothing here is
inferred at runtime from ``models.py``; a test keeps the two in sync.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

ParamValue = Any


class RegistryError(ValueError):
    """Structured error: ``field`` names the offending parameter."""

    def __init__(self, field: str, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.field = field
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str                       # "int" | "float" | "bool" | "str"
    default: ParamValue
    description: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Tuple[ParamValue, ...]] = None
    # Legacy argparse fields per role, e.g. {"pa": "PA_hidden_size", "dpd": "DPD_hidden_size"}
    legacy_arg: Mapping[str, str] = field(default_factory=dict)

    def coerce(self, value: ParamValue) -> ParamValue:
        py_type = {"int": int, "float": float, "bool": bool, "str": str}[self.type]
        if self.type == "bool":
            if not isinstance(value, bool):
                raise RegistryError(self.name, f"{self.name} must be a boolean")
            return value
        if self.type == "int" and isinstance(value, bool):
            raise RegistryError(self.name, f"{self.name} must be an integer")
        if self.type == "int" and isinstance(value, float) and not value.is_integer():
            raise RegistryError(self.name, f"{self.name} must be an integer")
        try:
            value = py_type(value)
        except (TypeError, ValueError):
            raise RegistryError(self.name, f"{self.name} must be of type {self.type}") from None
        if self.minimum is not None and value < self.minimum:
            raise RegistryError(self.name, f"{self.name} must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise RegistryError(self.name, f"{self.name} must be <= {self.maximum}")
        if self.choices is not None and value not in self.choices:
            raise RegistryError(self.name, f"{self.name} must be one of {list(self.choices)}")
        return value


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    display_name: str
    family: str                          # recurrent | convolutional | polynomial | hybrid
    legacy_backbone: str                 # value of --PA_backbone / --DPD_backbone
    training_method: str                 # gradient | least_squares | ila
    roles: Tuple[str, ...]               # subset of ("pa", "dpd")
    params: Tuple[ParamSpec, ...]
    status: str                          # supported | experimental
    devices_tested: Tuple[str, ...]      # devices with recorded evidence
    lookahead_samples: Optional[int]     # None = not characterised
    lookahead_note: str
    execution_semantics: str = "offline_segmented"
    # streaming variants (plan S18): the key whose trained weights this variant executes; such a variant is never
    # trained itself, its results are never compared with the offline key's, and it keeps its own evidence
    weights_from: Optional[str] = None
    export_formats: Tuple[str, ...] = ()
    constraints: Optional[str] = None
    reference: Optional[str] = None
    evidence: Optional[str] = None       # where devices_tested comes from

    def param(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def defaults(self) -> Dict[str, ParamValue]:
        return {p.name: p.default for p in self.params}

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["params"] = [dict(asdict(p), legacy_arg=dict(p.legacy_arg)) for p in self.params]
        return data


# --- parameter specs shared by several models -------------------------------

def _poly(name: str, default: int, maximum: int, description: str, minimum: int = 1) -> ParamSpec:
    return ParamSpec(name, "int", default, description, minimum=minimum, maximum=maximum)


def _rcond() -> ParamSpec:
    return ParamSpec("rcond", "float", 0.0, "singular-value cutoff relative to the largest (column-normalised "
                     "truncated SVD); 0 = machine precision", minimum=0.0, maximum=0.999)


def _hidden(default: int, maximum: int = 1024) -> ParamSpec:
    return ParamSpec("hidden_size", "int", default, "Hidden state size of the backbone", minimum=1,
                     maximum=maximum, legacy_arg={"pa": "PA_hidden_size", "dpd": "DPD_hidden_size"})


def _layers() -> ParamSpec:
    return ParamSpec("num_layers", "int", 1, "Number of stacked recurrent layers", minimum=1, maximum=8,
                     legacy_arg={"pa": "PA_num_layers", "dpd": "DPD_num_layers"})


def _thx() -> ParamSpec:
    return ParamSpec("thx", "float", 0.0, "Input delta threshold (0 disables temporal pruning)", minimum=0.0,
                     legacy_arg={"pa": "thx", "dpd": "thx"})


def _thh() -> ParamSpec:
    return ParamSpec("thh", "float", 0.0, "Hidden-state delta threshold (0 disables temporal pruning)",
                     minimum=0.0, legacy_arg={"pa": "thh", "dpd": "thh"})


CAUSAL = "recurrent, causal: output at t depends on inputs up to t"
_CPU_WEEKLY = "tests/test_e2e_pipeline.py::TestAllBackbonesTraining (weekly CPU)"

MODELS: Tuple[ModelDescriptor, ...] = (
    ModelDescriptor(
        key="gru", display_name="GRU", family="recurrent", legacy_backbone="gru", training_method="gradient",
        roles=("pa", "dpd"), params=(_hidden(23), _layers()), status="supported",
        devices_tested=("cpu", "cuda"), lookahead_samples=0, lookahead_note=CAUSAL,
        reference="OpenDPD (ISCAS 2024)", evidence="S00 baseline CPU+CUDA smoke; benchmark_report.md (CUDA)",
    ),
    ModelDescriptor(
        key="tres_gru", display_name="TRes-GRU", family="hybrid", legacy_backbone="tres_gru",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(27), _layers()), status="supported",
        devices_tested=("cuda",), lookahead_samples=16,
        lookahead_note="dilated residual conv (kernel 3, dilation 16, symmetric padding) reads 16 future "
                       "samples; recurrent features read 1 future sample via torch.roll",
        reference="OpenDPDv2 (arXiv 2507.06849)", evidence="benchmark_report.md (CUDA)",
    ),
    ModelDescriptor(
        key="tres_deltagru", display_name="TRes-DeltaGRU", family="hybrid", legacy_backbone="tres_deltagru",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(15), _layers(), _thx(), _thh()),
        status="supported", devices_tested=("cuda",), lookahead_samples=16,
        lookahead_note="same temporal context as TRes-GRU (16-sample dilated conv, 1-sample roll)",
        reference="OpenDPDv2 (arXiv 2507.06849), DeltaDPD (MWTL 2025)", evidence="benchmark_report.md (CUDA)",
    ),
    ModelDescriptor(
        key="gmp", display_name="GMP (gradient-trained)", family="polynomial", legacy_backbone="gmp",
        training_method="gradient", roles=("pa", "dpd"), params=(), status="supported",
        devices_tested=("cpu",), lookahead_samples=0,
        lookahead_note="zero-padded past window of 11 samples; causal",
        constraints="memory length 11 and degree 5 are fixed in backbones/gmp.py (the --K / --gmp_memory_length "
                    "flags are not wired)", reference="OpenDPD (ISCAS 2024)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="mp_ls", display_name="MP (least squares)", family="polynomial", legacy_backbone="mp",
        training_method="least_squares", roles=("pa", "dpd"),
        params=(_poly("K", 5, 15, "nonlinearity order: envelope powers |x|^k for k = 0..K-1"),
                _poly("Q", 50, 500, "memory depth in samples"), _rcond()),
        status="supported", devices_tested=("cpu",), lookahead_samples=0,
        lookahead_note="causal; the Q-1 sample history is zero-filled at every segment start",
        constraints="PA: direct least squares on the train split. DPD: indirect learning (ILA) on the measured train "
                    "split, evaluated through a gradient-trained PA surrogate; not usable as a surrogate itself. "
                    "Deterministic (no seed, no epochs); rank, condition number and cutoff are recorded.",
        reference="OpenDPD benchmark (benchmark/benchmark_volterra.py, benchmark_report.md)",
        evidence="tests/unit/test_polynomial.py (analytic), tests/integration/test_baselines.py",
    ),
    ModelDescriptor(
        key="gmp_ls", display_name="GMP (least squares)", family="polynomial", legacy_backbone="gmp",
        training_method="least_squares", roles=("pa", "dpd"),
        params=(_poly("Ka", 5, 15, "aligned terms: envelope orders"), _poly("La", 15, 200, "aligned terms: memory depth"),
                _poly("Kb", 4, 15, "lagging envelope terms: orders (0 = none)", minimum=0),
                _poly("Lb", 15, 200, "lagging terms: memory depth"), _poly("Mb", 2, 20, "lagging terms: envelope lags"),
                _poly("Kc", 4, 15, "leading envelope terms: orders (0 = none)", minimum=0),
                _poly("Lc", 15, 200, "leading terms: memory depth"), _poly("Mc", 1, 20, "leading terms: envelope leads"),
                _rcond()),
        status="supported", devices_tested=("cpu",), lookahead_samples=None,
        lookahead_note="not characterised as a constant: the leading envelope terms |x(n+m)| read Mc future samples "
                       "(0 when Kc = 0); every result records the value for its parameters",
        constraints="Same fitting and roles as mp_ls. The cross-term basis is often ill-conditioned: a cutoff "
                    "(rcond, e.g. 1e-4) keeps the fit stable and the retained rank is recorded.",
        reference="OpenDPD benchmark (benchmark/benchmark_volterra.py, benchmark_report.md)",
        evidence="tests/unit/test_polynomial.py (analytic), tests/integration/test_baselines.py",
    ),
    ModelDescriptor(
        key="lstm", display_name="LSTM", family="recurrent", legacy_backbone="lstm", training_method="gradient",
        roles=("pa", "dpd"), params=(_hidden(23), _layers()), status="supported", devices_tested=("cpu",),
        lookahead_samples=0, lookahead_note=CAUSAL, reference="OpenDPD (ISCAS 2024)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="vdlstm", display_name="VDLSTM", family="recurrent", legacy_backbone="vdlstm", training_method="gradient",
        roles=("pa", "dpd"), params=(_hidden(23), _layers()), status="supported", devices_tested=("cpu",),
        lookahead_samples=0, lookahead_note=CAUSAL, reference="OpenDPD (ISCAS 2024)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="dgru", display_name="DGRU", family="recurrent", legacy_backbone="dgru", training_method="gradient",
        roles=("pa", "dpd"), params=(_hidden(23), _layers()), status="supported", devices_tested=("cpu",),
        lookahead_samples=0, lookahead_note=CAUSAL, reference="OpenDPD (ISCAS 2024)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="tcn", display_name="TCN", family="convolutional", legacy_backbone="tcn", training_method="gradient",
        roles=("pa", "dpd"), params=(_hidden(23),), status="supported", devices_tested=("cpu",),
        lookahead_samples=30,
        lookahead_note="four depthwise conv layers (kernel 5, dilations 1/2/4/8, symmetric padding) read "
                       "2+4+8+16 = 30 future samples", reference="TCN-DPD (IMS 2025)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="rvtdcnn", display_name="RVTDCNN", family="convolutional", legacy_backbone="rvtdcnn",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23),), status="supported",
        devices_tested=("cpu",), lookahead_samples=None,
        lookahead_note="windowed 2-D convolution; temporal context not characterised yet",
        reference="OpenDPD (ISCAS 2024)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="pgjanet", display_name="PGJANET", family="recurrent", legacy_backbone="pgjanet",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23),), status="supported",
        devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
        reference="OpenDPDv2 (arXiv 2507.06849)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="dvrjanet", display_name="DVRJANET", family="recurrent", legacy_backbone="dvrjanet",
        training_method="gradient", roles=("pa", "dpd"),
        params=(_hidden(23), ParamSpec("num_dvr_units", "int", 3, "Number of DVR units", minimum=1, maximum=32,
                                       legacy_arg={"pa": "num_dvr_units", "dpd": "num_dvr_units"})),
        status="supported", devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
        reference="OpenDPDv2 (arXiv 2507.06849)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="deltagru", display_name="DeltaGRU", family="recurrent", legacy_backbone="deltagru",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23), _layers(), _thx(), _thh()),
        status="experimental", devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
        reference="DeltaDPD (MWTL 2025)", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="deltajanet", display_name="DeltaJANET", family="recurrent", legacy_backbone="deltajanet",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23), _layers(), _thx(), _thh()),
        status="experimental", devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
        evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="qgru", display_name="QGRU (quantisation-aware GRU)", family="recurrent", legacy_backbone="qgru",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23), _layers()), status="experimental",
        devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
        reference="MP-DPD (MWTL 2024)", evidence="tests/test_e2e_pipeline.py::TestQuantizedTrainDPD (CPU)",
    ),
    ModelDescriptor(
        key="qgru_amp1", display_name="QGRU (amp1 variant)", family="recurrent", legacy_backbone="qgru_amp1",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23), _layers()), status="experimental",
        devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL, evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="bojanet", display_name="BOJANET", family="recurrent", legacy_backbone="bojanet",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(16, maximum=18),), status="experimental",
        devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
        constraints="hidden_size must be <= 18", evidence=_CPU_WEEKLY,
    ),
    ModelDescriptor(
        key="apnrru", display_name="APNRRU", family="recurrent", legacy_backbone="apnrru",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23),), status="experimental",
        devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
        constraints="not selectable through the legacy CLI (its choices list spells it 'apnrnn')",
        evidence="tests/test_backbones.py forward pass (CPU)",
    ),
    ModelDescriptor(
        key="mcldnn", display_name="MCLDNN", family="convolutional", legacy_backbone="mcldnn",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(8),), status="experimental",
        devices_tested=("cpu",), lookahead_samples=None,
        lookahead_note="2-D/1-D convolutions with symmetric padding; temporal context not characterised yet",
        evidence=_CPU_WEEKLY,
    ),
)

STREAMING = "streaming_stateful"
_STREAM_CONSTRAINT = ("evaluation variant of '{base}': its weights come from a finished {base} run (evaluate_pa, run_dpd); "
                      "training it directly is refused; its scores are re-evaluated under streaming semantics and are "
                      "neither compared with nor inherited from {base}'s offline_segmented results")

MODELS += (
    ModelDescriptor(
        key="gru_stream", display_name="GRU (streaming, stateful)", family="recurrent", legacy_backbone="gru",
        training_method="gradient", roles=("pa", "dpd"), params=(_hidden(23), _layers()), status="experimental",
        devices_tested=("cpu",), lookahead_samples=0,
        lookahead_note=CAUSAL + "; the hidden state is carried across chunks instead of being reset at every segment",
        execution_semantics=STREAMING, weights_from="gru", constraints=_STREAM_CONSTRAINT.format(base="gru"),
        reference="OpenDPD (ISCAS 2024); streaming contract docs/architecture/streaming.md",
        evidence="tests/unit/test_streaming.py, tests/integration/test_streaming_eval.py (CPU)",
    ),
    ModelDescriptor(
        key="gmp_stream", display_name="GMP (streaming, windowed)", family="polynomial", legacy_backbone="gmp",
        training_method="gradient", roles=("pa", "dpd"), params=(), status="experimental", devices_tested=("cpu",),
        lookahead_samples=0,
        lookahead_note="causal; the 20-sample history (memory 11 with envelope windows lagging by another 10, measured) is "
                       "carried across chunks instead of being zero-filled at every segment start",
        execution_semantics=STREAMING, weights_from="gmp", constraints=_STREAM_CONSTRAINT.format(base="gmp"),
        reference="OpenDPD (ISCAS 2024); streaming contract docs/architecture/streaming.md",
        evidence="tests/unit/test_streaming.py, tests/integration/test_streaming_eval.py (CPU)",
    ),
)

_BY_KEY: Dict[str, ModelDescriptor] = {m.key: m for m in MODELS}


def streaming_variant_of(key: str) -> Optional[ModelDescriptor]:
    """The registered streaming variant that executes ``key``'s weights, if any."""
    return next((m for m in MODELS if m.weights_from == key and m.execution_semantics == STREAMING), None)


def list_models() -> List[ModelDescriptor]:
    return list(MODELS)


def get_model(key: str) -> ModelDescriptor:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise RegistryError("model.key", f"unknown model '{key}'",
                            hint=f"known models: {', '.join(sorted(_BY_KEY))}") from None


def validate_parameters(key: str, parameters: Mapping[str, ParamValue], role: str) -> Dict[str, ParamValue]:
    """Return the complete, validated parameter dict for ``key`` in ``role``."""
    model = get_model(key)
    if role not in model.roles:
        raise RegistryError("model.key", f"model '{key}' cannot be used as {role}",
                            hint=f"roles: {', '.join(model.roles)}")
    known = {p.name for p in model.params}
    unknown = sorted(set(parameters) - known)
    if unknown:
        raise RegistryError(f"model.parameters.{unknown[0]}",
                            f"model '{key}' has no parameter '{unknown[0]}'",
                            hint=f"parameters: {', '.join(sorted(known)) or 'none'}")
    resolved: Dict[str, ParamValue] = {}
    for spec in model.params:
        value = parameters.get(spec.name, spec.default)
        try:
            resolved[spec.name] = spec.coerce(value)
        except RegistryError as err:
            raise RegistryError(f"model.parameters.{spec.name}", err.message, err.hint) from None
    return resolved
