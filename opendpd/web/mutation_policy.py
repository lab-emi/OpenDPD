"""Shared validation, per-IP rates and storage admission for public numeric work."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re

from opendpd.schemas.signal_generator import GeneratorConfig
from opendpd.schemas.virtual_pa import VirtualPARequest
from opendpd.schemas.dataset_catalog import SyntheticSuiteRequest, DatasetPublicationDraft
from opendpd.services.workspace import WorkspaceError
from opendpd.web.policy import reject
from opendpd.web.runtime import directory_bytes


@dataclass(frozen=True)
class MutationRule:
    pattern: str
    model: type | None
    rate_key: str
    per_minute: int
    estimate: str


MUTATIONS = (
    MutationRule(r'/signal-analyzer/analyze', None, 'signal-analysis', 36, 'none'),
    MutationRule(r'/signal-generator/signals', GeneratorConfig, 'signal-generator', 24, 'generator'),
    MutationRule(r'/signal-generator/signals/sg-[a-f0-9]{64}/dataset', None, 'signal-dataset', 6, 'input_dataset'),
    MutationRule(r'/pa-library/simulations', VirtualPARequest, 'virtual-pa', 24, 'pa'),
    MutationRule(r'/pa-library/simulations/vpa-[a-f0-9]{64}/dataset', None, 'virtual-pa-dataset', 6, 'paired_dataset'),
    MutationRule(r'/datasets/synthetic', SyntheticSuiteRequest, 'synthetic', 3, 'synthetic'),
    MutationRule(r'/dataset-publications/prepare', DatasetPublicationDraft, 'publication-preview', 6, 'publication'),
)


def estimate_storage(rule, payload, path, ws):
    from opendpd.services.signal_generator import read_signal
    from opendpd.services.virtual_pa import read_simulation
    if rule.estimate == 'generator':
        return payload.sample_count * 100 + 2_000_000
    if rule.estimate == 'synthetic':
        return payload.samples_per_capture * payload.repeats * 3 * 160
    if rule.estimate == 'publication':
        return (ws.get_dataset(payload.dataset_id).n_samples or 0) * 100
    if rule.estimate == 'input_dataset':
        return read_signal(ws, path.split('/')[3]).analysis.sample_count * 160
    if rule.estimate == 'pa':
        return read_signal(ws, payload.input_signal_id).analysis.sample_count * 100 + 2_000_000
    if rule.estimate == 'paired_dataset':
        return read_simulation(ws, path.split('/')[3]).analysis.n_samples * 160
    return 0


async def enforce_mutation_budget(manager, tenant, path, body):
    rule = next((item for item in MUTATIONS if re.fullmatch(item.pattern, path)), None)
    if rule is None:
        return
    try:
        payload = rule.model.model_validate(body) if rule.model else body
    except ValueError:
        reject(422, 'invalid_request', 'Check the signal parameters, sample count and required dataset metadata.')
    manager.rate_limit(rule.rate_key + ':' + tenant.ip_key, rule.per_minute)
    if rule.estimate == 'none':
        return
    try:
        required = await asyncio.to_thread(estimate_storage, rule, payload, path, tenant.app.state.ws)
    except WorkspaceError:
        reject(404, 'source_not_found', 'The source is unavailable in this temporary workspace. Create or select it first.')
    used = await asyncio.to_thread(directory_bytes, tenant.root)
    if used + required > manager.config.max_workspace_bytes:
        reject(413, 'workspace_limit', 'This operation and its exports would exceed temporary storage. Use fewer samples.')
