"""Data descriptions may never supply trainer arguments or executable paths."""
from typing import Literal
from pydantic import Field, field_validator
from .common import StrictModel
from opendpd.safe_paths import filename


class LegacySpec(StrictModel):
    description: str | None = Field(default=None, max_length=8192)
    dataset_format: Literal["split_csv", "single_csv"] = "split_csv"
    csv_filename: str = "data.csv"
    split_ratios: dict[str, float] = Field(default_factory=lambda: {"train": .6, "val": .2, "test": .2})
    split_indices: dict[str, int] | None = None
    input_signal_fs: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    bw_main_ch: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    bw_sub_ch: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    n_sub_ch: int | None = Field(default=None, gt=0)
    nperseg: int | None = Field(default=None, gt=0)
    standard: str | None = Field(default=None, max_length=256)
    modulation: str | None = Field(default=None, max_length=256)
    test_model: str | None = Field(default=None, max_length=256)
    papr_db: float | None = Field(default=None, allow_inf_nan=False)
    scs: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    ofdm_nfft: int | None = Field(default=None, gt=0)
    n_active: int | None = Field(default=None, gt=0)
    cp_first: int | None = Field(default=None, ge=0)
    cp_other: int | None = Field(default=None, ge=0)
    origin: Literal["measured", "synthetic", "unknown"] | None = None
    tutorial: bool = False
    generator: str | None = Field(default=None, max_length=256)
    generator_seed: int | None = None

    @field_validator("csv_filename")
    @classmethod
    def _filename(cls, value):
        return filename(value)

    @field_validator("split_ratios")
    @classmethod
    def _ratios(cls, value):
        import math
        if (set(value) != {"train", "val", "test"}
                or any(not math.isfinite(v) or v < 0 or v > 1 for v in value.values())
                or not math.isclose(sum(value.values()), 1.0, abs_tol=1e-6)):
            raise ValueError("split ratios must be train/val/test fractions summing to one")
        return value


def validate_spec(value):
    return LegacySpec.model_validate(value).model_dump(exclude_unset=True)
