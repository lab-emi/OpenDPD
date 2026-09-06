"""Fixed-point specification ``fixed-point-v1`` and the deployment package it verifies (plan S19).

The specification names every operator and every stored quantity of the GRU streaming variant: formats, the
rounding rule, where saturation happens, the accumulator width and the table approximation of the two
non-linearities. A deployment package carries the quantised weights, golden vectors, the bit-exact C99
reference, the verification of that reference against the software reference, and a report whose numbers
are labelled by how they were obtained.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field

from .common import SCHEMA_VERSION, Sha256, Slug, SoftwareProvenance, StrictModel, utcnow

SPEC_ID = "fixed-point-v1"
ROUNDING = "round half up: add 2^(s-1) then arithmetic shift right by s; left shifts are exact"
SATURATION = "every stored quantity saturates to its word width (x, h, y, gate values, table index); the accumulator never wraps"
NONLINEARITY = "table lookup without interpolation: index = saturate(round(pre-activation to LUT_FRAC)) + offset"


class WordFormat(StrictModel):
    """A signed two's-complement fixed-point word."""

    bits: int = Field(ge=2, le=64)
    frac: int = Field(ge=0, le=62)

    @property
    def scale(self) -> int:
        return 1 << self.frac

    @property
    def max_int(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def min_int(self) -> int:
        return -(1 << (self.bits - 1))


class TableSpec(StrictModel):
    """A non-linearity as a table: entries at every 2^-frac over [-range, range), values in ``value`` format."""

    function: Literal["sigmoid", "tanh"]
    range: float = Field(gt=0)
    index_frac: int = Field(ge=0, le=16)
    value: WordFormat

    @property
    def entries(self) -> int:
        return int(2 * self.range * (1 << self.index_frac))


class FixedPointSpec(StrictModel):
    """Every format and rule of the GRU fixed-point reference. Changing any of them is a new spec id."""

    spec_id: Literal["fixed-point-v1"] = SPEC_ID
    model_key: Literal["gru_stream"] = "gru_stream"          # the semantics the reference executes (state carried)
    x: WordFormat = WordFormat(bits=16, frac=14)              # input I/Q, ±2 full scale
    h: WordFormat = WordFormat(bits=16, frac=15)              # hidden state and gate values, (-1, 1)
    y: WordFormat = WordFormat(bits=16, frac=14)              # output I/Q, ±2 full scale
    weight_bits: int = Field(default=16, ge=4, le=32)         # per-tensor fraction chosen at export from max |w|
    pre: WordFormat = WordFormat(bits=32, frac=20)            # pre-activation (matmul results rescaled, biases added)
    accumulator_bits: int = Field(default=48, ge=32, le=64)   # exact integer dot products; the bound is checked
    sigmoid: TableSpec = TableSpec(function="sigmoid", range=8.0, index_frac=8, value=WordFormat(bits=16, frac=15))
    tanh: TableSpec = TableSpec(function="tanh", range=4.0, index_frac=8, value=WordFormat(bits=16, frac=15))
    rounding: str = ROUNDING
    saturation: str = SATURATION
    nonlinearity: str = NONLINEARITY


class TensorFormat(StrictModel):
    """How one weight tensor was quantised: its fraction and the range it had to hold."""

    name: str
    shape: List[int]
    bits: int
    frac: int
    max_abs_float: float = Field(ge=0)
    saturated: int = Field(ge=0)        # values clipped by the word width (0 for a sound export)


class GoldenCase(StrictModel):
    """One golden vector: inputs, expected outputs and final state, all in the spec's integer formats."""

    case_id: Slug
    description: str
    n_samples: int = Field(ge=1)
    resets_at: List[int] = Field(default_factory=list)   # sample indices where the state is reset before the sample
    input_sha256: Sha256
    output_sha256: Sha256
    state_sha256: Sha256
    trace_sha256: Sha256                                 # the state after every sample (localisation of a mismatch)


class MetricDelta(StrictModel):
    name: str
    unit: str
    float_value: Optional[float] = None
    fixed_value: Optional[float] = None
    delta: Optional[float] = None       # fixed - float; None when either is not applicable


class ResourceEstimate(StrictModel):
    """Numbers derived from the specification and the shapes; nothing here was measured."""

    label: Literal["theoretical"] = "theoretical"
    mac_per_sample: int = Field(ge=0)
    table_lookups_per_sample: int = Field(ge=0)
    weight_bytes: int = Field(ge=0)
    bias_bytes: int = Field(ge=0)
    state_bytes: int = Field(ge=0)
    table_bytes: int = Field(ge=0)
    sparsity: str = "none exploited: every MAC is executed"


class MeasuredExecution(StrictModel):
    """The C reference's speed on the machine that built the package: a property of that build, not of a deployment."""

    label: Literal["measured_execution_time"] = "measured_execution_time"
    what: str
    samples_per_second: float = Field(ge=0)
    machine: Dict[str, str] = Field(default_factory=dict)


class Verification(StrictModel):
    """Bit-exact comparison of the target implementation with the software reference on every golden vector."""

    backend: str
    status: Literal["bit_exact", "mismatch", "not_run"]
    compiler: Optional[str] = None
    cases_checked: int = Field(ge=0)
    mismatch_case: Optional[str] = None
    mismatch_step: Optional[int] = None
    mismatch_signal: Optional[str] = None
    detail: Optional[str] = None


class FixedPointReport(StrictModel):
    """Every number labelled by how it was obtained; the labels never mix."""

    quality_loss: List[MetricDelta]                 # float streaming model vs fixed-point reference, same profile
    metric_profile_id: Slug
    resources: ResourceEstimate
    measured_execution: Optional[MeasuredExecution] = None
    synthesis_estimate: Optional[str] = None       # None = not available (nothing was synthesised)
    measured_power: Optional[str] = None           # None = not available (nothing was measured)
    execution_assumptions: List[str]


class DeploymentManifest(StrictModel):
    schema_version: int = SCHEMA_VERSION
    spec: FixedPointSpec
    run_id: Slug
    model_key: str
    weights_sha256: Optional[Sha256] = None
    hidden_size: int = Field(ge=1)
    tensors: List[TensorFormat]
    golden: List[GoldenCase]
    verification: Verification
    report: FixedPointReport
    files: Dict[str, Sha256]                        # every file of the package by relative path
    software: SoftwareProvenance
    created_at: datetime = Field(default_factory=utcnow)
