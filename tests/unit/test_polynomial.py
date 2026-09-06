"""Least-squares MP/GMP baselines: analytic checks of the basis, the solve, its diagnostics and the torch module."""

import numpy as np
import pytest

from opendpd.core.polynomial import (
    PolynomialModel,
    coefficient_count,
    fit_least_squares,
    gmp_basis,
    lookahead_samples,
    mp_basis,
    segmented_basis,
)


def _signal(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.3


def test_mp_basis_columns_are_delayed_envelope_powers():
    x = _signal(64)
    phi = mp_basis(x, K=3, Q=4)
    assert phi.shape == (64, 12)
    d1 = np.concatenate([[0], x[:-1]])
    np.testing.assert_allclose(phi[:, 2 * 4 + 1], d1 * np.abs(d1) ** 2)      # k = 2, q = 1
    np.testing.assert_allclose(phi[:, 0], x)                                 # k = 0, q = 0: the linear term


def test_gmp_basis_has_aligned_lagging_and_leading_terms():
    x = _signal(64)
    cfg = dict(Ka=2, La=3, Kb=1, Lb=2, Mb=2, Kc=1, Lc=2, Mc=1)
    phi = gmp_basis(x, **cfg)
    assert phi.shape == (64, coefficient_count("gmp_ls", cfg)) == (64, 6 + 4 + 2)
    lag = np.concatenate([[0, 0], x[:-2]])                                   # |x(n-2)| envelope of the l=0, m=2 term
    np.testing.assert_allclose(phi[:, 6 + 1], x * np.abs(lag))
    lead = np.concatenate([x[1:], [0]])                                      # |x(n+1)| envelope of the l=0, m=1 term
    np.testing.assert_allclose(phi[:, 10], x * np.abs(lead))


def test_least_squares_recovers_known_coefficients_and_reports_the_fit():
    x = _signal()
    phi = mp_basis(x, 3, 4)
    rng = np.random.default_rng(1)
    w_true = rng.normal(size=12) + 1j * rng.normal(size=12)
    w, diag = fit_least_squares(phi, phi @ w_true)
    np.testing.assert_allclose(w, w_true, rtol=1e-9, atol=1e-11)
    assert diag.rank == 12 and diag.n_coefficients == 12 and diag.n_observations == len(x)
    assert diag.train_nmse_db < -200 and diag.condition_number >= 1 and diag.column_norm_ratio >= 1
    assert diag.rcond == 0.0 and diag.retained_condition_number == pytest.approx(diag.condition_number)


def test_indirect_learning_on_a_linear_pa_identifies_the_identity():
    x = _signal()
    gain = 2.5
    y = gain * x                                        # a linear PA: the postdistorter Phi(y/G) w ~= x is x itself
    w, diag = fit_least_squares(mp_basis(y / gain, 3, 4), x)
    expected = np.zeros(12, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(w, expected, atol=1e-9)
    assert diag.train_nmse_db < -200


def test_rank_deficiency_is_reported_and_the_cutoff_truncates():
    x = _signal()
    phi = np.concatenate([mp_basis(x, 2, 2), mp_basis(x, 1, 1)], axis=1)     # the last column repeats the first
    w, diag = fit_least_squares(phi, phi[:, 1] * 2.0)
    assert diag.n_coefficients == 5 and diag.rank == 4 and diag.condition_number > 1e12
    assert diag.retained_condition_number < 1e6
    w2, diag2 = fit_least_squares(mp_basis(x, 3, 4), x, rcond=0.5)
    assert diag2.rank < 12 and diag2.rcond == 0.5


def test_bad_systems_are_refused():
    x = _signal(8)
    with pytest.raises(ValueError, match="underdetermined"):
        fit_least_squares(mp_basis(x, 3, 4), x)
    with pytest.raises(ValueError, match="finite"):
        fit_least_squares(mp_basis(_signal(64), 2, 2), np.full(64, np.nan))
    with pytest.raises(ValueError, match="rcond"):
        fit_least_squares(mp_basis(_signal(64), 2, 2), _signal(64), rcond=1.0)


def test_torch_module_applies_the_segmented_basis_and_round_trips_its_state():
    import torch

    x = _signal(1500)
    w = np.linspace(0.1, 1.2, 12) * (1 + 0.5j)
    model = PolynomialModel("mp_ls", {"K": 3, "Q": 4}, w)
    xt = torch.from_numpy(np.stack([x.real, x.imag], -1).astype(np.float32)).reshape(3, 500, 2)
    out = model(xt)
    expected = segmented_basis("mp_ls", {"K": 3, "Q": 4}, x, 500) @ w
    np.testing.assert_allclose(out.numpy().reshape(-1, 2)[:, 0], expected.real, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out.numpy().reshape(-1, 2)[:, 1], expected.imag, rtol=1e-5, atol=1e-6)
    assert model.n_real_parameters == 24
    fresh = PolynomialModel("mp_ls", {"K": 3, "Q": 4})
    fresh.load_state_dict(model.state_dict())
    assert torch.equal(fresh.coefficients, model.coefficients)
    with pytest.raises(ValueError, match="coefficients expected"):
        PolynomialModel("mp_ls", {"K": 3, "Q": 4}, w[:5])


def test_lookahead_is_the_leading_envelope_lead_only():
    assert lookahead_samples("mp_ls", {"K": 5, "Q": 50}) == 0
    assert lookahead_samples("gmp_ls", {"Kc": 4, "Mc": 2}) == 2
    assert lookahead_samples("gmp_ls", {"Kc": 0, "Mc": 2}) == 0
