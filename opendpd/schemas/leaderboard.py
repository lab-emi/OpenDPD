"""Versioned leaderboards, submissions and reviews (``leaderboard-v1``, plan S20).

A board is one track (PA modelling, surrogate DPD, measured DPD; the standard-evaluation, robustness and
deployment tracks stay closed until their stage gates pass). Entries rank only inside one comparability key
(data, operating point, metric profile version, split, execution semantics, resource class); other keys are shown
apart and never ranked against each other. An entry keeps its history: submission, reviews, recomputations,
retractions and corrections are appended, never erased. A board calls itself a *reference benchmark* until enough
external submissions were accepted and independently recomputed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .benchmark import MetricStats, SeedScore, canonical_sha256
from .common import SCHEMA_VERSION, EvidenceType, Sha256, Slug, StrictModel, utcnow

PROTOCOL_ID = "leaderboard-v1"
MIN_EXTERNAL_ACCEPTED = 3            # complete external submissions accepted after review
MIN_INDEPENDENT_RECOMPUTATIONS = 2   # accepted entries recomputed by someone who is not their author
RECOMPUTE_TOLERANCE = 1e-6           # a stored result re-scored from its checkpoint reproduces to this (dB)
MIN_SEEDS = 3                        # below this the entry is shown, with its uncertainty marked as not established

Track = Literal["pa_modeling", "dpd_surrogate", "dpd_measured", "standard_evaluation", "robustness", "deployment"]
TRACK_EVIDENCE: Dict[str, EvidenceType] = {"pa_modeling": EvidenceType.pa_modeling, "dpd_surrogate": EvidenceType.dpd_surrogate,
                                           "dpd_measured": EvidenceType.dpd_measured}
TRACK_RANK_METRIC: Dict[str, str] = {"pa_modeling": "NMSE", "dpd_surrogate": "ACLR_AVG", "dpd_measured": "ACLR_AVG"}
# tracks that open only after the stage they depend on passed its human gate
TRACK_GATES: Dict[str, str] = {
    "standard_evaluation": "opens when the S15 cross-validation record of ofdm-lte20-evm-v1 is filled (docs/protocols/waveform-profiles.md §7)",
    "robustness": "opens when a measured condition set meets the S17 evidence bar (docs/protocols/conditions-v1.md §7)",
    "deployment": "opens when the S19 fixed-point rules are approved (docs/protocols/fixed-point-v1.md §8)",
}
SubmitterKind = Literal["external", "maintainer"]
EntryStatus = Literal["submitted", "under_review", "accepted", "rejected", "retracted", "corrected"]
EvidenceGrade = Literal["self_reported", "reviewed", "independently_recomputed"]
TRANSITIONS: Dict[str, List[str]] = {"submitted": ["under_review", "accepted", "rejected"], "under_review": ["accepted", "rejected"],
                                     "accepted": ["retracted", "corrected"], "corrected": ["retracted", "corrected"],
                                     "rejected": [], "retracted": []}


class Submitter(StrictModel):
    name: str = Field(min_length=1)
    kind: SubmitterKind
    affiliation: Optional[str] = None


class MethodCard(StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    reference: Optional[str] = None          # paper, preprint or report
    code_url: Optional[str] = None
    licence: str = Field(min_length=1)       # SPDX identifier, "proprietary" or "not_provided"


class ModelCard(StrictModel):
    model_key: str
    parameters: Dict[str, object] = Field(default_factory=dict)
    n_parameters: Optional[int] = None
    lookahead_samples: Optional[int] = None
    execution_semantics: str = "offline_segmented"
    training_path: Optional[str] = None


class DataCard(StrictModel):
    dataset_id: Slug
    raw_sha256: Optional[Sha256] = None
    availability: Literal["public", "on_request", "private"]
    statement: str = Field(min_length=1)     # how to obtain it, or why it is not public (never presented as public)
    operating_point: str = "as captured"     # measured tracks: PA | drive | power | chain | rate, as declared


class PackageRef(StrictModel):
    """One share package = one run (checkpoint, result, resolved configuration); one per seed."""

    path: str = Field(min_length=1)          # relative to the submission card
    sha256: Sha256
    run_id: Slug


class ResourceBudget(StrictModel):
    budget_class: str = "unbounded"          # the declared class entries compete in, e.g. "params<=1000"
    device: str
    mean_wall_clock_s: Optional[float] = None   # measured: training wall clock of the reported runs on that device; never power


class ResultSummary(StrictModel):
    metric_profile_id: Slug
    metric_profile_version: int
    split_version: str
    evidence_type: EvidenceType
    seeds: List[SeedScore] = Field(min_length=1)          # per seed: run, hashes, every metric
    metrics: Dict[str, MetricStats]                        # over the seeds
    resources: ResourceBudget

    @model_validator(mode="after")
    def _aggregate_matches_seeds(self) -> "ResultSummary":
        seeds = [s.seed for s in self.seeds]
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be distinct")
        for name, stats in self.metrics.items():
            if stats.n != len(self.seeds):
                raise ValueError(f"metric '{name}' aggregates {stats.n} values, the card lists {len(self.seeds)} seeds")
        return self


class LicenceCheck(StrictModel):
    code: str = Field(min_length=1)
    weights: str = Field(min_length=1)
    data: str = Field(min_length=1)
    redistribution_allowed: bool
    statement: str = Field(min_length=1)


class SubmissionCard(StrictModel):
    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["leaderboard-v1"] = PROTOCOL_ID
    submission_id: Slug
    track: Track
    submitter: Submitter
    method: MethodCard
    model: ModelCard
    data: DataCard
    result: ResultSummary
    packages: List[PackageRef] = Field(default_factory=list)
    benchmark_report_sha256: Optional[Sha256] = None       # maintainers' entries copied from a benchmark-v1 report
    benchmark_plan_sha256: Optional[Sha256] = None
    failure_conditions: List[str] = Field(default_factory=list)   # where the method fails or was not tried
    licence: LicenceCheck
    conflict_of_interest: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    isolated_validation: str = Field(min_length=1)   # where the package was produced; never a lab-connected machine
    submitted_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _well_formed(self) -> "SubmissionCard":
        expected = TRACK_EVIDENCE.get(self.track)
        if expected is None:
            raise ValueError(f"track '{self.track}' is not open: {TRACK_GATES[self.track]}")
        if self.result.evidence_type != expected:
            raise ValueError(f"track '{self.track}' takes {expected.value} evidence, the result is {self.result.evidence_type.value}")
        if not self.packages and self.benchmark_report_sha256 is None:
            raise ValueError("an entry is traceable to a share package per seed or to a benchmark-v1 report; neither is given")
        if not self.packages and self.submitter.kind != "maintainer":
            raise ValueError("an external submission is traceable to its share packages; a report hash alone is the maintainers' path")
        runs = {s.run_id for s in self.result.seeds}
        for p in self.packages:
            if p.run_id not in runs:
                raise ValueError(f"package '{p.path}' carries run '{p.run_id}', which is not one of the card's seeds")
        return self


class Recomputation(StrictModel):
    by: str = Field(min_length=1)
    kind: SubmitterKind
    date: datetime = Field(default_factory=utcnow)
    within_tolerance: bool
    max_abs_delta: Optional[float] = None
    tolerance: float = RECOMPUTE_TOLERANCE
    note: str = Field(min_length=1)


class Review(StrictModel):
    reviewer: str = Field(min_length=1)
    kind: SubmitterKind
    date: datetime = Field(default_factory=utcnow)
    decision: Literal["accepted", "rejected", "needs_changes"]
    notes: str = Field(min_length=1)
    recomputation: Optional[Recomputation] = None


class HistoryEvent(StrictModel):
    date: datetime = Field(default_factory=utcnow)
    action: Literal["submitted", "reviewed", "accepted", "rejected", "retracted", "corrected", "created"]
    by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    previous_metrics: Optional[Dict[str, MetricStats]] = None   # corrections keep what was shown before


class ComparabilityKey(StrictModel):
    """Entries rank only within one key; everything else is shown apart as non-comparable."""

    track: Track
    dataset_id: Slug
    raw_sha256: Optional[Sha256] = None
    operating_point: str
    metric_profile: str          # "<id> v<version>"
    split_version: str
    execution_semantics: str
    budget_class: str

    @classmethod
    def of(cls, card: "SubmissionCard") -> "ComparabilityKey":
        return cls(track=card.track, dataset_id=card.data.dataset_id, raw_sha256=card.data.raw_sha256,
                   operating_point=card.data.operating_point,
                   metric_profile=f"{card.result.metric_profile_id} v{card.result.metric_profile_version}",
                   split_version=card.result.split_version, execution_semantics=card.model.execution_semantics,
                   budget_class=card.result.resources.budget_class)

    def label(self) -> str:
        return (f"{self.dataset_id} (raw {(self.raw_sha256 or 'unknown')[:12]}), {self.operating_point}, {self.metric_profile}, "
                f"split {self.split_version}, {self.execution_semantics}, budget {self.budget_class}")


class BoardEntry(StrictModel):
    entry_id: Slug
    submission: SubmissionCard
    key: ComparabilityKey
    status: EntryStatus = "submitted"
    evidence_grade: EvidenceGrade = "self_reported"
    reviews: List[Review] = Field(default_factory=list)
    history: List[HistoryEvent] = Field(default_factory=list)

    @property
    def ranked(self) -> bool:
        return self.status in ("accepted", "corrected")


class Leaderboard(StrictModel):
    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["leaderboard-v1"] = PROTOCOL_ID
    board_id: Slug
    version: str = Field(min_length=1)           # a frozen version; a new version is a new file, never an overwrite
    track: Track
    supersedes: Optional[Sha256] = None          # board_sha256 of the previous version
    entries: List[BoardEntry] = Field(default_factory=list)
    history: List[HistoryEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    board_sha256: Optional[Sha256] = None

    @model_validator(mode="after")
    def _well_formed(self) -> "Leaderboard":
        if self.track not in TRACK_EVIDENCE:
            raise ValueError(f"track '{self.track}' is not open: {TRACK_GATES[self.track]}")
        ids = [e.entry_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("entry ids must be unique")
        for e in self.entries:
            if e.submission.track != self.track:
                raise ValueError(f"entry '{e.entry_id}' is a {e.submission.track} submission on a {self.track} board")
        return self

    def entry(self, entry_id: str) -> BoardEntry:
        for e in self.entries:
            if e.entry_id == entry_id:
                return e
        raise KeyError(entry_id)

    @property
    def external_accepted(self) -> int:
        return sum(e.ranked and e.submission.submitter.kind == "external" for e in self.entries)

    @property
    def independent_recomputations(self) -> int:
        return sum(e.ranked and e.submission.submitter.kind == "external" and e.evidence_grade == "independently_recomputed"
                   for e in self.entries)

    @property
    def community_bar_met(self) -> bool:
        return self.external_accepted >= MIN_EXTERNAL_ACCEPTED and self.independent_recomputations >= MIN_INDEPENDENT_RECOMPUTATIONS

    @property
    def label(self) -> str:
        return "community leaderboard" if self.community_bar_met else "reference benchmark"

    def content_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", exclude={"board_sha256"}))

    def sealed(self) -> "Leaderboard":
        return self.model_copy(update={"board_sha256": self.content_sha256()})

    @property
    def intact(self) -> bool:
        return self.board_sha256 is not None and self.board_sha256 == self.content_sha256()


class CheckItem(StrictModel):
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str


class SubmissionCheck(StrictModel):
    submission_id: Slug
    items: List[CheckItem]
    recomputation: Optional[Recomputation] = None

    @property
    def passed(self) -> bool:
        return all(i.status != "fail" for i in self.items)
