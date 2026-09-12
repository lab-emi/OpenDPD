"""Dataset catalog and guided CSV creation contracts."""

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field

from opendpd.core.splits import DEFAULT_GUARD_SAMPLES, DEFAULT_RATIOS
from .common import StrictModel
from .dataset import DatasetOrigin, SignalSpec


class BuiltinDatasetInfo(StrictModel):
    name: str
    description: str
    dataset_format: str
    n_samples: Optional[int] = None
    signal: SignalSpec
    problem: Optional[str] = None
    has_demodulator: bool
    origin: DatasetOrigin
    raw_sha256: str


class DatasetImportDefaults(StrictModel):
    ratios: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_RATIOS))
    guard_samples: int = Field(default=DEFAULT_GUARD_SAMPLES, ge=0, le=100_000)


class CsvOptions(StrictModel):
    format: Literal["auto", "complex_pair", "iq_columns"] = "auto"
    header: Literal["auto", "present", "absent"] = "auto"
    # Logical role -> zero-based source column. Indices also work without headers.
    mapping: Dict[str, int] = Field(default_factory=dict)


class CsvIssue(StrictModel):
    code: str
    line: Optional[int] = None
    column: Optional[str] = None
    message: str
    fix: str


class CsvInspection(StrictModel):
    valid: bool = False
    data_valid: bool = False
    sha256: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    options: CsvOptions = Field(default_factory=CsvOptions)
    preview: List[List[str]] = Field(default_factory=list)
    n_samples: int = 0
    issue_count: int = 0
    issues: List[CsvIssue] = Field(default_factory=list)
    split: DatasetImportDefaults = Field(default_factory=DatasetImportDefaults)
    boundaries: Dict[str, Tuple[int, int]] = Field(default_factory=dict)
    split_counts: Dict[str, int] = Field(default_factory=dict)
