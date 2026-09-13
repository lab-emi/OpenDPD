"""Experiment configuration resolution: one normaliser for every entry point."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from opendpd import __version__
from opendpd.core.metrics import PROFILES
from opendpd.core.registry import RegistryError, get_model, validate_parameters
from opendpd.schemas import SCHEMA_VERSION, ExperimentConfig, ResolvedExperimentConfig, ResolutionInfo, TaskType

RESOLVER_VERSION = "config-resolver-v1"

# Checkpoint selection is a protocol decision (plan §9.3), not a user knob:
# PA models are selected by validation NMSE, DPD models by validation ACLR.
SELECTION_METRIC = {
    TaskType.train_pa: "NMSE",
    TaskType.train_dpd: "ACLR_AVG",
    TaskType.run_dpd: "not_applicable",
    TaskType.evaluate_measured: "not_applicable",
    TaskType.evaluate_pa: "not_applicable",
}

SMOKE_EPOCH_LIMIT = 10


@dataclass
class ConfigIssue:
    field: str
    message: str
    hint: Optional[str] = None


@dataclass
class ValidationReport:
    errors: List[ConfigIssue] = field(default_factory=list)
    warnings: List[ConfigIssue] = field(default_factory=list)
    resolved: Optional[ResolvedExperimentConfig] = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [asdict(e) for e in self.errors],
            "warnings": [asdict(w) for w in self.warnings],
            "resolved": self.resolved.model_dump(mode="json") if self.resolved else None,
        }


class ConfigError(ValueError):
    def __init__(self, issues: List[ConfigIssue]):
        super().__init__("; ".join(f"{i.field}: {i.message}" for i in issues))
        self.issues = issues


def canonical_json(config: ExperimentConfig) -> str:
    """Stable serialisation used for hashing (the resolution block is excluded)."""
    data = config.model_dump(mode="json", exclude={"resolution"})
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_sha256(config: ExperimentConfig) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _pydantic_issues(err: ValidationError) -> List[ConfigIssue]:
    issues = []
    for e in err.errors():
        loc = ".".join(str(part) for part in e["loc"]) or "config"
        issues.append(ConfigIssue(field=loc, message=e["msg"]))
    return issues


def validate(config_data: Any, warnings: Optional[List[ConfigIssue]] = None) -> ValidationReport:
    """Validate raw config data (dict or ExperimentConfig) without running anything.

    ``warnings`` seeds workspace-level findings (e.g. split guard vs frame
    length) so they are part of the frozen resolution like the resolver's own.
    """
    report = ValidationReport()
    if warnings:
        report.warnings.extend(warnings)
    try:
        config = config_data if isinstance(config_data, ExperimentConfig) \
            else ExperimentConfig.model_validate(config_data)
    except ValidationError as err:
        report.errors.extend(_pydantic_issues(err))
        return report
    try:
        report.resolved = resolve(config, warnings=report.warnings)
    except ConfigError as err:
        report.errors.extend(err.issues)
    return report


def resolve(config: ExperimentConfig, warnings: Optional[List[ConfigIssue]] = None) -> ResolvedExperimentConfig:
    """Fill every result-affecting default and freeze the configuration.

    Raises ``ConfigError`` with field-level issues. Model parameters come from
    the registry; training defaults come from ``TrainingConfig`` (which mirror
    ``arguments.py``); the checkpoint selection metric comes from the protocol.
    """
    issues: List[ConfigIssue] = []
    warn = warnings if warnings is not None else []
    if isinstance(config, ResolvedExperimentConfig):
        config = ExperimentConfig.model_validate(config.model_dump(exclude={"resolution"}))

    role = "pa" if config.task in (TaskType.train_pa, TaskType.evaluate_pa) else "dpd"
    params: Dict[str, Any] = dict(config.model.parameters)
    try:
        params = validate_parameters(config.model.key, config.model.parameters, role)
    except RegistryError as err:
        issues.append(ConfigIssue(err.field, err.message, err.hint))

    pa_reference = config.pa_reference
    if config.task in (TaskType.train_dpd, TaskType.run_dpd, TaskType.evaluate_pa):
        if pa_reference is None or pa_reference.model is None or pa_reference.checkpoint_sha256 is None:
            issues.append(ConfigIssue("pa_reference", "the PA surrogate is not bound to a finished PA run",
                                      "submit through the experiments service, which binds run_id to a "
                                      "checkpoint and its model specification"))
        else:
            try:
                pa_params = validate_parameters(pa_reference.model.key, pa_reference.model.parameters, "pa")
                pa_reference = pa_reference.model_copy(
                    update={"model": pa_reference.model.model_copy(update={"parameters": pa_params})})
            except RegistryError as err:
                issues.append(ConfigIssue(f"pa_reference.{err.field}", err.message, err.hint))
            if config.model.key in ("tres_deltagru", "deltagru", "deltajanet") and pa_reference.model is not None:
                for name in ("thx", "thh"):
                    if name in params and name in pa_reference.model.parameters \
                            and params[name] != pa_reference.model.parameters[name]:
                        warn.append(ConfigIssue(f"model.parameters.{name}",
                                                f"the legacy trainer applies one {name} to both the DPD and the "
                                                f"frozen PA model; the PA was trained with "
                                                f"{pa_reference.model.parameters[name]}"))

    selection = SELECTION_METRIC[config.task]
    requested = config.evaluation.checkpoint_selection_metric
    if requested is not None and requested != selection:
        issues.append(ConfigIssue("evaluation.checkpoint_selection_metric",
                                  f"checkpoint selection for {config.task.value} is fixed to validation "
                                  f"{selection} by protocol", "remove the field or use the protocol value"))

    if config.evaluation.profile_id not in PROFILES:
        issues.append(ConfigIssue("evaluation.profile_id",
                                  f"metric profile '{config.evaluation.profile_id}' is not registered",
                                  "one of: " + ", ".join(sorted(PROFILES))))

    model = get_model(config.model.key) if not issues else None
    if model is not None and model.weights_from is not None and config.task in (TaskType.train_pa, TaskType.train_dpd):
        issues.append(ConfigIssue("model.key", f"model '{model.key}' is a streaming variant that executes the weights of "
                                  f"'{model.weights_from}'; it is not trained",
                                  f"train {model.weights_from}, then `opendpd stream <run>` (evaluate_pa / run_dpd with model {model.key})"))
    if config.evaluation.chunk_samples is not None and (model is None or model.weights_from is None):
        issues.append(ConfigIssue("evaluation.chunk_samples", "chunk_samples applies to streaming variants only",
                                  "remove the field, or evaluate with a model whose execution_semantics is streaming_stateful"))
    least_squares = model is not None and model.training_method == "least_squares"
    if config.training.epochs <= SMOKE_EPOCH_LIMIT and config.task in (TaskType.train_pa, TaskType.train_dpd) and not least_squares:
        warn.append(ConfigIssue("training.epochs",
                                f"{config.training.epochs} epochs is a smoke/demo run, not a benchmark result"))
    budget = config.training.train_samples
    if budget is not None and budget < config.training.frame_length and not least_squares:
        issues.append(ConfigIssue("training.train_samples", f"a budget of {budget} samples is shorter than one frame "
                                  f"({config.training.frame_length})", "raise the budget or shorten the frame"))
    if config.initialization is not None and least_squares:
        issues.append(ConfigIssue("initialization", f"model '{config.model.key}' is fitted by least squares and has no "
                                  "initial weights", "use training.train_samples alone for a budgeted refit"))
    if least_squares and config.quantization is not None and config.quantization.enabled:
        issues.append(ConfigIssue("quantization.enabled", f"model '{config.model.key}' is fitted by least squares; "
                                  "quantisation-aware training does not apply", "disable quantization"))
    if model is not None and config.execution.device not in model.devices_tested:
        warn.append(ConfigIssue("execution.device",
                                f"model '{config.model.key}' has no recorded test evidence on "
                                f"{config.execution.device} (tested: {', '.join(model.devices_tested)})"))

    if issues:
        raise ConfigError(issues)

    data = config.model_dump()
    data["model"]["parameters"] = params
    data["evaluation"]["checkpoint_selection_metric"] = selection
    data["pa_reference"] = pa_reference.model_dump() if pa_reference is not None else None
    base = ExperimentConfig.model_validate(data)
    digest = config_sha256(base)
    return ResolvedExperimentConfig(
        **base.model_dump(),
        resolution=ResolutionInfo(
            resolver_version=RESOLVER_VERSION,
            defaults_source=f"opendpd {__version__} / schema {SCHEMA_VERSION} / registry {config.model.key}",
            config_sha256=digest,
            warnings=[f"{w.field}: {w.message}" for w in warn],
        ),
    )
