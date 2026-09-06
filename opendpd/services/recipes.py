"""Reference recipes: named, reviewed starting points with a stated purpose.

``smoke`` recipes exist to prove the pipeline works in seconds; their numbers
must never be presented as benchmark results. ``research`` recipes carry the
full OpenDPDv2 optimisation budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from opendpd.schemas import (
    DatasetRef,
    DPDReference,
    EvaluationConfig,
    EvidenceType,
    ExecutionConfig,
    ExperimentConfig,
    ModelSpec,
    PAReference,
    TaskType,
    TrainingConfig,
)


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    title: str
    purpose: str            # "smoke" | "research" | "baseline"
    task: TaskType
    model: ModelSpec
    training: TrainingConfig
    description: str
    limits: str
    expected_duration: str

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["task"] = self.task.value
        data["model"] = self.model.model_dump()
        data["training"] = self.training.model_dump()
        return data


_SMOKE = dict(epochs=3, frame_length=50, frame_stride=16, batch_size_eval=256)
_RESEARCH = dict(epochs=300, frame_length=200, frame_stride=1)

RECIPES: List[Recipe] = [
    Recipe("pa-gru-smoke-v1", "PA model, GRU, smoke", "smoke", TaskType.train_pa,
           ModelSpec(key="gru", parameters={"hidden_size": 23, "num_layers": 1}), TrainingConfig(**_SMOKE),
           "Fast end-to-end check of PA modelling on CPU.",
           "3 epochs, short frames, stride 16: numbers are far from converged and must not be compared.",
           "about 5 s on a laptop CPU with DPA_200MHz"),
    Recipe("dpd-gru-smoke-v1", "DPD, GRU, smoke", "smoke", TaskType.train_dpd,
           ModelSpec(key="gru", parameters={"hidden_size": 15, "num_layers": 1}), TrainingConfig(**_SMOKE),
           "Fast DPD learning through a PA surrogate trained with pa-gru-smoke-v1.",
           "Requires a PA run with the same seed and frame_length (50). Not a benchmark.",
           "about 5 s on a laptop CPU with DPA_200MHz"),
    Recipe("dpd-tres-deltagru-smoke-v1", "DPD, TRes-DeltaGRU, smoke", "smoke", TaskType.train_dpd,
           ModelSpec(key="tres_deltagru", parameters={"hidden_size": 15, "num_layers": 1, "thx": 0.0, "thh": 0.0}),
           TrainingConfig(**_SMOKE),
           "Fast check of the OpenDPDv2 DPD architecture through a PA surrogate.",
           "Same constraints as dpd-gru-smoke-v1; TRes models read 16 future samples (see registry).",
           "about 10 s on a laptop CPU with DPA_200MHz"),
    Recipe("pa-gru-research-v1", "PA model, GRU, OpenDPDv2 recipe", "research", TaskType.train_pa,
           ModelSpec(key="gru", parameters={"hidden_size": 23, "num_layers": 1}), TrainingConfig(**_RESEARCH),
           "Full 300-epoch PA modelling recipe (AdamW, ReduceLROnPlateau) as in the OpenDPDv2 defaults.",
           "Single seed. Benchmark-grade comparisons need the S12 protocol (3 pre-registered seeds).",
           "hours on CPU; minutes to tens of minutes on a GPU"),
    Recipe("dpd-tres-deltagru-research-v1", "DPD, TRes-DeltaGRU, OpenDPDv2 recipe", "research", TaskType.train_dpd,
           ModelSpec(key="tres_deltagru", parameters={"hidden_size": 15, "num_layers": 1, "thx": 0.0, "thh": 0.0}),
           TrainingConfig(**_RESEARCH),
           "Full 300-epoch DPD learning through a research-grade PA surrogate.",
           "Requires a research PA run (frame_length 200, seed 0). Surrogate evidence only.",
           "hours on CPU; tens of minutes on a GPU"),
]

_LS = dict(epochs=1, frame_length=200, frame_stride=1)      # not used by a least-squares fit (recorded for the layout)
_GMP_PA = {"Ka": 5, "La": 30, "Kb": 4, "Lb": 30, "Mb": 5, "Kc": 4, "Lc": 30, "Mc": 5, "rcond": 1e-4}
_GMP_DPD = {"Ka": 5, "La": 20, "Kb": 4, "Lb": 20, "Mb": 3, "Kc": 4, "Lc": 20, "Mc": 2, "rcond": 0.0}
RECIPES += [
    Recipe("pa-mp-ls-v1", "PA model, MP, least squares (benchmark baseline)", "baseline", TaskType.train_pa,
           ModelSpec(key="mp_ls", parameters={"K": 9, "Q": 150, "rcond": 0.0}), TrainingConfig(**_LS),
           "Memory polynomial identified by direct least squares on the train split: the classical PA-modeling "
           "baseline of benchmark_report.md (1,350 complex = 2,700 real parameters).",
           "Deterministic, no seed. Rank, condition number and cutoff are recorded with the result. The basis "
           "(train samples x 1,350 complex columns) is held in memory.",
           "seconds to a minute on CPU"),
    Recipe("pa-gmp-ls-v1", "PA model, GMP, truncated SVD (benchmark baseline)", "baseline", TaskType.train_pa,
           ModelSpec(key="gmp_ls", parameters=dict(_GMP_PA)), TrainingConfig(**_LS),
           "Generalised memory polynomial (aligned, lagging and leading envelope terms) with a 1e-4 singular-value "
           "cutoff, as in benchmark_report.md (2,700 real parameters).",
           "The cutoff is part of the protocol: the retained rank is reported and must not be tuned per result. "
           "Leading terms read Mc = 5 future samples.",
           "seconds to a minute on CPU"),
    Recipe("dpd-mp-ila-v1", "DPD, MP, indirect learning (benchmark baseline)", "baseline", TaskType.train_dpd,
           ModelSpec(key="mp_ls", parameters={"K": 5, "Q": 100, "rcond": 0.0}), TrainingConfig(**_LS),
           "Memory-polynomial predistorter identified by ILA on the measured train split (1,000 real parameters) "
           "and scored through a gradient-trained PA surrogate.",
           "A different training path from gradient DPD (DLA through the surrogate): comparable under one "
           "protocol, but the result states the path. Requires a gradient-trained PA run on the dataset.",
           "seconds on CPU"),
    Recipe("dpd-gmp-ila-v1", "DPD, GMP, indirect learning (benchmark baseline)", "baseline", TaskType.train_dpd,
           ModelSpec(key="gmp_ls", parameters=dict(_GMP_DPD)), TrainingConfig(**_LS),
           "Generalised memory-polynomial predistorter identified by ILA (1,000 real parameters), scored through "
           "a gradient-trained PA surrogate.",
           "Same path statement as dpd-mp-ila-v1; leading terms read Mc = 2 future samples.",
           "seconds on CPU"),
]

_BY_ID = {r.recipe_id: r for r in RECIPES}


def list_recipes() -> List[Recipe]:
    return list(RECIPES)


def get_recipe(recipe_id: str) -> Recipe:
    try:
        return _BY_ID[recipe_id]
    except KeyError:
        raise KeyError(f"unknown recipe '{recipe_id}'; known: {', '.join(sorted(_BY_ID))}") from None


def instantiate(recipe_id: str, dataset_id: str, *, pa_run_id: Optional[str] = None,
                dpd_run_id: Optional[str] = None, device: str = "cpu", seed: Optional[int] = None,
                name: Optional[str] = None) -> ExperimentConfig:
    recipe = get_recipe(recipe_id)
    training = recipe.training if seed is None else recipe.training.model_copy(update={"seed": seed})
    evidence = EvidenceType.pa_modeling if recipe.task == TaskType.train_pa else EvidenceType.dpd_surrogate
    pa_ref = PAReference(run_id=pa_run_id) if pa_run_id else None
    dpd_ref = DPDReference(run_id=dpd_run_id) if dpd_run_id else None
    if recipe.task == TaskType.train_dpd and pa_ref is None:
        raise ValueError(f"recipe '{recipe_id}' needs pa_run_id (a succeeded PA run on '{dataset_id}')")
    return ExperimentConfig(
        task=recipe.task, recipe_id=recipe.recipe_id, name=name or recipe.title,
        dataset=DatasetRef(id=dataset_id), model=recipe.model, training=training,
        evaluation=EvaluationConfig(evidence_type=evidence),
        execution=ExecutionConfig(device=device), pa_reference=pa_ref, dpd_reference=dpd_ref,
        notes=f"{recipe.purpose}: {recipe.limits}",
    )


def run_dpd_config(dataset_id: str, dpd_run_id: str, *, pa_run_id: Optional[str] = None,
                   device: str = "cpu") -> ExperimentConfig:
    """A run_dpd task for an existing DPD run: export u = DPD(x) for the test split and score it through a PA
    surrogate. Without ``pa_run_id`` the DPD's training surrogate is used; another PA run of the same dataset
    gives a new, separately stored result (model and training are bound from the DPD run)."""
    return ExperimentConfig(
        task=TaskType.run_dpd, name=f"apply {dpd_run_id}" + (f" through {pa_run_id}" if pa_run_id else ""),
        dataset=DatasetRef(id=dataset_id), model=ModelSpec(key="gru"),
        evaluation=EvaluationConfig(evidence_type=EvidenceType.dpd_surrogate),
        execution=ExecutionConfig(device=device), dpd_reference=DPDReference(run_id=dpd_run_id),
        pa_reference=PAReference(run_id=pa_run_id) if pa_run_id else None,
    )
