"""conditions-v1 schema rules: what a card, an entry and a plan refuse before anything runs."""

import pytest
from pydantic import ValidationError

from opendpd.schemas import AdaptationEntry, AdaptationPlan, Condition, ConditionSet, ModelSpec, TargetRule, TaskType, TrainingConfig


def _cond(cid, role="target", dataset=None, batch=None):
    return Condition(condition_id=cid, dataset_id=dataset or cid, role=role, capture_batch=batch or f"b-{cid}")


def _card(*conditions):
    return ConditionSet(set_id="card", device="one PA", dimension="drive", conditions=list(conditions))


def _entry(entry_id="pa", task=TaskType.train_pa, **kw):
    return AdaptationEntry(entry_id=entry_id, task=task, model=ModelSpec(key="gru", parameters={"hidden_size": 8, "num_layers": 1}),
                           training=TrainingConfig(epochs=1), **kw)


def test_a_card_needs_exactly_one_source_and_one_dataset_per_condition():
    card = _card(_cond("a", "source"), _cond("b"))
    assert card.source.condition_id == "a" and [c.condition_id for c in card.targets] == ["b"] and card.independent_batches
    assert card.compute_sha256() == card.model_copy(update={"created_at": None}).compute_sha256()   # the seal ignores the time
    with pytest.raises(ValidationError, match="exactly one condition is the source"):
        _card(_cond("a"), _cond("b"))
    with pytest.raises(ValidationError, match="exactly one condition is the source"):
        _card(_cond("a", "source"), _cond("b", "source"))
    with pytest.raises(ValidationError, match="one capture cannot be two conditions"):
        _card(_cond("a", "source", dataset="same"), _cond("b", dataset="same"))
    with pytest.raises(ValidationError, match="condition ids must be unique"):
        _card(_cond("a", "source"), _cond("a"))
    assert not _card(_cond("a", "source", batch="one"), _cond("b", batch="one")).independent_batches


def test_entries_declare_their_surrogate_and_leave_the_budget_to_the_plan():
    with pytest.raises(ValidationError, match="needs pa_entry"):
        _entry("dpd", TaskType.train_dpd)
    with pytest.raises(ValidationError, match="a PA entry has no pa_entry"):
        _entry("pa", pa_entry="pa")
    with pytest.raises(ValidationError, match="leaves it unset"):
        AdaptationEntry(entry_id="pa", task=TaskType.train_pa, model=ModelSpec(key="gru", parameters={"hidden_size": 8, "num_layers": 1}),
                        training=TrainingConfig(epochs=1, train_samples=100))


def test_a_plan_needs_a_sealed_card_consistent_entries_and_distinct_axes():
    card = _card(_cond("a", "source"), _cond("b"))
    sealed = card.model_copy(update={"card_sha256": card.compute_sha256()})
    with pytest.raises(ValidationError, match="must be sealed"):
        AdaptationPlan(condition_set=card, entries=[_entry()])
    plan = AdaptationPlan(condition_set=sealed, entries=[_entry(), _entry("dpd", TaskType.train_dpd, pa_entry="pa")],
                          target=TargetRule(metric="NMSE", threshold=-30.0))
    assert plan.tasks == ["zero_update", "few_shot", "full_retrain"] and plan.budgets == [2000] and plan.seeds == [0]
    assert plan.target.reached(-31.0) is True and plan.target.reached(-29.0) is False and plan.target.reached(None) is None
    assert TargetRule(metric="ACLR_AVG", threshold=-40.0, better="higher").reached(-39.0) is True
    with pytest.raises(ValidationError, match="not in the plan"):
        AdaptationPlan(condition_set=sealed, entries=[_entry("dpd", TaskType.train_dpd, pa_entry="pa")])
    with pytest.raises(ValidationError, match="entry ids must be unique"):
        AdaptationPlan(condition_set=sealed, entries=[_entry(), _entry()])
    with pytest.raises(ValidationError, match="budgets must be distinct positive"):
        AdaptationPlan(condition_set=sealed, entries=[_entry()], budgets=[2000, 2000])
    with pytest.raises(ValidationError, match="seeds must be distinct"):
        AdaptationPlan(condition_set=sealed, entries=[_entry()], seeds=[0, 0])
    with pytest.raises(ValidationError, match="tasks must be distinct"):
        AdaptationPlan(condition_set=sealed, entries=[_entry()], tasks=["zero_update", "zero_update"])
