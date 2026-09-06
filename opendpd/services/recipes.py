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
    purpose: str            # "smoke" | "research"
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
