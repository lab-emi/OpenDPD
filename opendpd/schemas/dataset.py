"""Dataset manifests: what was imported, from where, with which hashes."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import ClassVar, Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from .common import SCHEMA_VERSION, FileRef, Sha256, Slug, StrictModel, utcnow
from opendpd.schemas.waveform import WaveformBinding


class DatasetOrigin(str, Enum):
    measured = "measured"
    synthetic = "synthetic"
    unknown = "unknown"


class DatasetSourceKind(str, Enum):
    builtin = "builtin"                  # shipped with the package (datasets/<name>)
    csv_import = "csv_import"            # user CSV with I_in,Q_in,I_out,Q_out (or mapped)
    numpy_import = "numpy_import"        # .npy/.npz with declared dtype/shape
    legacy_dir_import = "legacy_dir_import"  # existing OpenDPD split-CSV directory


class DatasetSource(StrictModel):
    kind: DatasetSourceKind
    name: Optional[str] = None            # builtin dataset name
    original_path: Optional[str] = None   # display only; stripped from share packages
    imported_at: datetime = Field(default_factory=utcnow)


class SignalSpec(StrictModel):
    """Signal metadata needed by evaluation profiles. Unknown values stay
    ``None``; they are never guessed. ``missing_for_legacy_evaluation`` names
    what the legacy profile still needs."""

    sample_rate_hz: Optional[float] = Field(default=None, gt=0)
    bandwidth_hz: Optional[float] = Field(default=None, gt=0)        # bw_main_ch
    sub_channel_bandwidth_hz: Optional[float] = Field(default=None, gt=0)
    n_sub_ch: Optional[int] = Field(default=None, ge=1)
    nperseg: Optional[int] = Field(default=None, ge=2)
    modulation: Optional[str] = None
    standard: Optional[str] = None
    amplitude_units: Literal["normalized", "volts", "unknown"] = "unknown"
    # Which reference waveform the input column was captured from (plan S15); set by an explicit import
    # option after the input has been correlated with the regenerated waveform. None = no known symbols.
    waveform: Optional[WaveformBinding] = None

    LEGACY_REQUIRED: ClassVar[Tuple[str, ...]] = ("sample_rate_hz", "bandwidth_hz", "n_sub_ch", "nperseg")

    def missing_for_legacy_evaluation(self) -> List[str]:
        return [name for name in self.LEGACY_REQUIRED if getattr(self, name) is None]

    @model_validator(mode="after")
    def _bandwidth_within_sample_rate(self) -> "SignalSpec":
        if self.sample_rate_hz and self.bandwidth_hz and self.bandwidth_hz > self.sample_rate_hz:
            raise ValueError("bandwidth_hz cannot exceed sample_rate_hz")
        return self


class SplitSpec(StrictModel):
    """Contiguous train/val/test split performed *before* framing, with an
    optional guard band so overlapping frames never straddle a boundary."""

    version: str = "contiguous-v1"
    method: Literal["contiguous"] = "contiguous"
    ratios: Dict[Literal["train", "val", "test"], float]
    guard_samples: int = Field(default=0, ge=0)
    boundaries: Optional[Dict[Literal["train", "val", "test"], Tuple[int, int]]] = None

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SplitSpec":
        if set(self.ratios) != {"train", "val", "test"}:
            raise ValueError("ratios must define train, val and test")
        if any(r < 0 for r in self.ratios.values()):
            raise ValueError("ratios must be non-negative")
        if abs(sum(self.ratios.values()) - 1.0) > 1e-6:
            raise ValueError("ratios must sum to 1")
        if self.boundaries is not None:
            for name, (start, end) in self.boundaries.items():
                if start < 0 or end < start:
                    raise ValueError(f"invalid boundary for {name}")
        return self


class PreprocessingParams(StrictModel):
    """User-confirmed preprocessing; applied by ``opendpd.core.preprocess``."""

    delay_samples: float = 0.0          # output lags input by this many samples (may be fractional)
    gain_db: float = 0.0                # linear output gain to divide out
    phase_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    interpolate_non_finite: bool = False
    remove_outliers: bool = False
    normalize: Literal["none", "peak_input"] = "none"


class DatasetVersion(StrictModel):
    """One materialised data version (raw-v1 or a preprocessing result), in the
    directory layout the trainer reads (split CSV files)."""

    version: Slug
    base_version: Optional[Slug] = None
    created_at: datetime = Field(default_factory=utcnow)
    params: Optional[PreprocessingParams] = None
    code_version: Optional[str] = None
    fit_range: Optional[Tuple[int, int]] = None   # samples used to fit normalisation (train split)
    record: Dict[str, object] = Field(default_factory=dict)
    n_samples: int = Field(ge=0)
    split: SplitSpec
    files: List[FileRef] = Field(default_factory=list)
    sha256: Optional[Sha256] = None


class DatasetManifest(StrictModel):
    schema_version: int = SCHEMA_VERSION
    dataset_id: Slug
    display_name: str = Field(min_length=1)
    origin: DatasetOrigin
    source: DatasetSource
    signal: SignalSpec = SignalSpec()
    files: List[FileRef] = Field(default_factory=list)
    n_samples: Optional[int] = Field(default=None, ge=0)
    columns: Optional[Dict[str, str]] = None   # logical name -> column in the source file
    split: SplitSpec
    preprocessing_version: str = "raw-v1"
    raw_sha256: Optional[Sha256] = None
    versions: List[DatasetVersion] = Field(default_factory=list)   # empty for built-ins: raw/ is the split dir
    notes: Optional[str] = None

    def version(self, name: str) -> Optional[DatasetVersion]:
        return next((v for v in self.versions if v.version == name), None)

    def missing_metadata(self) -> List[str]:
        return self.signal.missing_for_legacy_evaluation()

    @model_validator(mode="after")
    def _synthetic_must_say_so(self) -> "DatasetManifest":
        if self.origin == DatasetOrigin.synthetic and "synthetic" not in self.display_name.lower():
            raise ValueError("synthetic datasets must carry 'synthetic' in their display name")
        return self
