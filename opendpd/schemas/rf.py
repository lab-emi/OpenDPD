"""Versioned, explicitly declared RF context. Never inferred from normalized IQ."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from .common import StrictModel, utcnow


class RFConditions(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)

    version: Literal["rf-conditions-v1"] = "rf-conditions-v1"
    source: Literal["user_declared", "instrument_import"] = "user_declared"
    recorded_at: datetime = Field(default_factory=utcnow)
    note: str = Field(min_length=1, max_length=2000)
    dut: Optional[str] = Field(default=None, max_length=200)
    carrier_frequency_hz: Optional[float] = Field(default=None, gt=0)
    average_output_power_dbm: Optional[float] = Field(default=None, ge=-300, le=300)
    input_power_dbm: Optional[float] = Field(default=None, ge=-300, le=300)
    pa_dc_power_w: Optional[float] = Field(default=None, gt=0)
    dc_rails_w: Dict[str, float] = Field(default_factory=dict)
    included_rails: List[str] = Field(default_factory=list)
    power_reference_plane: Optional[str] = Field(default=None, max_length=500)
    backoff_reference: Optional[str] = Field(default=None, max_length=500)
    temperature_c: Optional[float] = None
    supply_v: Optional[float] = Field(default=None, gt=0)
    bias: Optional[str] = Field(default=None, max_length=200)
    mode: Optional[str] = Field(default=None, max_length=200)
    load: Optional[str] = Field(default=None, max_length=200)
    vswr: Optional[float] = Field(default=None, ge=1)
    reflection_phase_deg: Optional[float] = Field(default=None, ge=-180, le=180)
    calibration_id: Optional[str] = Field(default=None, max_length=200)
    fixture: Optional[str] = Field(default=None, max_length=500)
    deembedding: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _physical_context(self):
        if self.vswr is not None and self.reflection_phase_deg is None:
            raise ValueError("VSWR requires reflection_phase_deg to describe the complex load")
        if any(not name.strip() or value <= 0 for name, value in self.dc_rails_w.items()):
            raise ValueError("DC rails require names and positive powers in W")
        if len(set(self.included_rails)) != len(self.included_rails) or any(name not in self.dc_rails_w for name in self.included_rails):
            raise ValueError("included_rails must name distinct provided DC rails")
        return self
