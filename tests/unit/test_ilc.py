import numpy as np
import pytest
from opendpd.core.ilc import learn


def test_linear_plant_converges_to_analytic_inverse_and_respects_peak():
    x = .2*np.exp(1j*np.arange(512))
    result = learn(lambda u: (2+.3j)*u, x, 1., 1/(2+.3j), learning_gain=.5, peak_limit=.3, target_nmse_db=-70)
    assert result.stop_reason == 'target_reached'
    np.testing.assert_allclose(result.input, x/(2+.3j), atol=4e-5)
    assert all(b['nmse_db'] < a['nmse_db'] for a,b in zip(result.history,result.history[1:]))
    assert max(abs(result.input)) <= .3


def test_saturated_unreachable_target_keeps_best_and_reports_failure_to_reach_target():
    x = np.ones(512,complex)*.5
    result = learn(lambda u: np.tanh(u.real)+1j*np.tanh(u.imag), x, 4., 1., peak_limit=.7)
    assert result.stop_reason != 'target_reached'
    assert max(abs(result.input)) <= .7
    assert result.history[-1]['nmse_db'] <= result.history[0]['nmse_db']


def test_backtracking_recovers_oversized_step():
    x=np.ones(256,complex)*.1
    result=learn(lambda u: 2*u, x, 1., 3., learning_gain=1., peak_limit=3.)
    assert result.history[1]['learning_gain'] < 1.
    assert result.history[-1]['nmse_db'] < -40


def test_bad_plant_and_cancellation_are_not_hidden():
    x=np.ones(256,complex)*.1
    with pytest.raises(ValueError, match='non-finite'):
        learn(lambda u: u*np.nan, x, 1., 1.)
    def cancelled():
        raise RuntimeError('cancel')
    with pytest.raises(RuntimeError,match='cancel'):
        learn(lambda u:u, x, 1., 1.,cancel=cancelled)
