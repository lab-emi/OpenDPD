"""Reject short spectral DPD runs before a worker is queued, using actual splits."""
import numpy as np
import pytest

from opendpd.schemas.dataset import SignalSpec
from opendpd.services.datasets import import_arrays
from opendpd.services.experiments import submission_issues
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration


def dataset(tmp_path, count, ratios=None):
    ws = Workspace.create(tmp_path / 'ws')
    x = np.random.default_rng(2).normal(size=(count, 2)).astype(np.float32)
    import_arrays(ws, x, x * .8, dataset_id='short-capture', guard_samples=0,
        ratios=ratios or {'train': 1/3, 'val': 1/3, 'test': 1/3},
        signal=SignalSpec(sample_rate_hz=40e6, bandwidth_hz=5e6, n_sub_ch=1, nperseg=4096))
    return ws


@pytest.mark.parametrize('count,blocked', [(12285, True), (12288, False)])
def test_actual_samples_must_fill_one_psd_segment(tmp_path, count, blocked):
    ws = dataset(tmp_path, count)
    config = instantiate('dpd-tres_gru-research-v1', 'short-capture', pa_run_id='placeholder-pa')
    config.execution.device = 'cpu'
    errors, _ = submission_issues(ws, config)
    assert bool(errors) is blocked
    if blocked:
        assert errors[0].field == 'dataset.id'
        assert '4,096 real samples' in errors[0].message
        assert 'validation: 4,095' in errors[0].message
        assert 'longer capture' in errors[0].hint
    pa = instantiate('pa-tres_gru-research-v1', 'short-capture')
    pa.execution.device = 'cpu'
    assert not submission_issues(ws, pa)[0]


def test_test_split_is_checked_only_when_trainer_evaluates_it(tmp_path):
    ws = dataset(tmp_path, 20000, {'train': .5, 'val': .4, 'test': .1})
    config = instantiate('dpd-tres_gru-research-v1', 'short-capture', pa_run_id='placeholder-pa')
    config.execution.device = 'cpu'
    assert 'test: 2,000' in submission_issues(ws, config)[0][0].message
    config.training.eval_test = False
    assert not submission_issues(ws, config)[0]
