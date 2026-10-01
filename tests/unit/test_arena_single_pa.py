"""Regression checks for the scientific changes in the single-PA protocol."""
import math

import numpy as np
import pytest
import torch

from opendpd.core import arena, arena_engine as engine
from opendpd.core.arena_metrics import evm_db, quality_db
from opendpd.schemas import SignalSpec


def test_quality_measures_emitted_leakage_not_error_waveform():
    baseline = dict(evm_db=-20., aclr_l_db=-30., aclr_r_db=-35., aer_l_db=-80., aer_r_db=-90.)
    candidate = dict(evm_db=-26., aclr_l_db=-32., aclr_r_db=-40., aer_l_db=-10., aer_r_db=-10.)
    assert quality_db(candidate, baseline) == 4.
    candidate.update(aer_l_db=-300., aer_r_db=-300.)
    assert quality_db(candidate, baseline) == 4.


def test_incomplete_symbol_metadata_is_not_relabelled_evm():
    x = np.ones((1024, 2))
    with pytest.raises(ValueError, match='complete OFDM'):
        evm_db(x, x, SignalSpec(sample_rate_hz=1024), dict(useful_samples=None))


def test_all_calibrations_use_one_shared_pa_and_complete_independent_symbols():
    wanted = {'apa-200mhz-b': 'APA_200MHz_b'}
    protocol = arena.protocol()
    assert {board.board_id: board.dataset for board in protocol.boards} == wanted
    assert all(board.conditions == [board.board_id] and board.evidence_type == 'measured_data_simulation'
               for board in protocol.boards)
    assert set(arena.active_calibration()) == set(wanted)
    for identifier, meta in arena.active_calibration().items():
        assert meta.get('origin', 'measured') == 'measured'
        assert meta.get('dataset', wanted[identifier]) == wanted[identifier]
        assert meta['simulation'] is None
        assert arena.judge_hashes(meta) == {'pa': meta['teacher']['sha256']}
        assert meta['teacher']['model']['key'] == 'tres_gru'
        assert meta['teacher']['parameters'] <= 5000
        assert meta['calibration_split'] == meta['split']
        assert meta['data_file'] == meta['calibration_data_file']
        assert meta['stimuli']['kind'] == 'unchanged original measured capture samples'
        with np.load(arena.ASSETS/meta['source_capture_file']) as source, np.load(arena.ASSETS/meta['data_file']) as data:
            for split, (first,last) in meta['split']['boundaries'].items():
                for kind in ('x','y'):
                    full = np.concatenate([source[f'{kind}_{part}'] for part in ('train','val','test')])
                    np.testing.assert_array_equal(data[f'{kind}_{split}'], full[first:last])
                window = meta['stimuli']['splits'][split]
                assert 'seed' not in window
                assert (window['metric_start'],window['metric_stop']) == (200,last-first-200)
        grid = meta['evm_grid']
        if identifier.startswith('apa-'):
            first,last = meta['split']['boundaries']['test']
            assert len(grid['carriers']) == 5 and grid['n_active_per_carrier'] == 1200
            for carrier in grid['carriers']:
                assert len(carrier['occupied_bins']) == 1200
                assert all(first+200 <= pos and pos+32768 <= last-200 for pos in carrier['fft_starts'])


def test_long_pa_preserves_state_and_lookahead_across_internal_chunks():
    torch.set_num_threads(4)
    torch.manual_seed(6)
    model = engine.build_model('tres_gru', dict(hidden_size=7, num_layers=1)).eval()
    x = np.random.default_rng(9).normal(0, .1, (33001, 2)).astype('float32')
    with torch.no_grad():
        expected = model(torch.from_numpy(x)[None])[0].numpy()
    np.testing.assert_allclose(engine.pa_output(model, x), expected, atol=2e-6, rtol=1e-5)


def test_cost_reference_is_independent_of_budget_and_rewards_smaller_models():
    obs = dict(evm_db=-30., baseline_evm_db=-20., aclr_l_db=-40., aclr_r_db=-42.,
        baseline_aclr_l_db=-30., baseline_aclr_r_db=-32., aer_l_db=-44., aer_r_db=-46.,
        baseline_aer_l_db=-34., baseline_aer_r_db=-36., nmse_db=-30., baseline_nmse_db=-20.,
        ib_error_db=-30., power_error_db=0.)
    cases = [dict(seed=0, judges=[obs])]
    def point(budget, p, a):
        cost = dict(ops=a, mul=a//2, add=a//2, nonlinear={}, nonlinear_mul=0, nonlinear_add=0, items=[])
        return arena._budget_result(budget, {}, p, cost, cases, [0])['score']
    assert point(250, 250, 500) == point(2000, 250, 500)
    assert point(250, 250, 500) - point(2000, 2000, 4000) == pytest.approx(10*math.log10(8))
