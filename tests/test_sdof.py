import numpy as np
from vibtool.sdof import (SDOF, damping_from_log_decrement, damping_from_half_power,
                          transmissibility, isolator_stiffness, unbalance_response)


def test_natural_frequency_and_zeta():
    s = SDOF.from_zeta(m=2.0, k=800.0, zeta=0.05)
    assert np.isclose(s.wn, 20.0)
    assert np.isclose(s.zeta, 0.05)
    assert np.isclose(s.wd, 20.0 * np.sqrt(1 - 0.05**2))


def test_free_response_matches_log_decrement():
    s = SDOF.from_zeta(1.0, 100.0, 0.03)
    Td = 2 * np.pi / s.wd
    t = np.array([0.0, 3 * Td])
    # with v0 = 0 the displacement samples at multiples of Td sit exactly on
    # the decaying envelope times cos(0), so the identification is exact
    x = s.free_response(t, x0=1.0, v0=0.0)
    zeta_id = damping_from_log_decrement(x[0], x[1], n=3)
    assert abs(zeta_id - 0.03) < 1e-10


def test_free_response_initial_conditions_all_regimes():
    t = np.linspace(0, 1, 5)
    for z in (0.2, 1.0, 2.0):
        s = SDOF.from_zeta(1.0, 50.0, z)
        x = s.free_response(t, x0=0.7, v0=-1.3)
        assert np.isclose(x[0], 0.7)
        h = 1e-6
        v0 = (s.free_response(h, 0.7, -1.3) - s.free_response(0.0, 0.7, -1.3)) / h
        assert abs(v0 + 1.3) < 1e-3


def test_magnification_at_resonance():
    s = SDOF.from_zeta(1.0, 1.0, 0.1)
    assert np.isclose(s.magnification(1.0), 1 / (2 * 0.1))
    assert np.isclose(abs(s.frf(s.wn)) * s.k, 5.0)
    assert np.isclose(s.phase(1.0), np.pi / 2)


def test_half_power_from_numerical_bandwidth():
    # locate the half power points on the magnification curve numerically, so
    # the check is independent of the identification formula itself
    from scipy.optimize import brentq

    zeta = 0.02
    s = SDOF.from_zeta(1.0, 100.0, zeta)
    peak = s.magnification(np.sqrt(1 - 2 * zeta**2))
    level = peak / np.sqrt(2.0)
    r1 = brentq(lambda r: s.magnification(r) - level, 0.5, 1.0 - zeta / 2)
    r2 = brentq(lambda r: s.magnification(r) - level, 1.0 + zeta / 2, 1.5)
    fn = s.fn
    zeta_id = damping_from_half_power(r1 * fn, r2 * fn, fn)
    # the formula is a small damping approximation, good to order zeta squared
    assert np.isclose(zeta_id, zeta, rtol=1e-2)


def test_steady_state_amplitude_and_phase():
    s = SDOF.from_zeta(2.0, 800.0, 0.1)
    amp, ph = s.steady_state(F0=10.0, w=s.wn)
    assert np.isclose(amp, 10.0 / 800.0 / (2 * 0.1))
    assert np.isclose(ph, np.pi / 2)
    amp0, ph0 = s.steady_state(F0=10.0, w=1e-6)
    assert np.isclose(amp0, 10.0 / 800.0, rtol=1e-6)
    assert abs(ph0) < 1e-6


def test_input_validation():
    import pytest

    with pytest.raises(ValueError):
        SDOF(0.0, 1.0)
    with pytest.raises(ValueError):
        SDOF(1.0, -1.0)
    with pytest.raises(ValueError):
        SDOF(1.0, 1.0, -0.1)
    with pytest.raises(ValueError):
        damping_from_log_decrement(1.0, 2.0)
    with pytest.raises(ValueError):
        isolator_stiffness(50.0, 1800.0, 0.1, zeta=-0.1)


def test_transmissibility_closed_form_points():
    assert np.isclose(transmissibility(np.sqrt(2), 0.3), 1.0)
    assert np.isclose(transmissibility(0.0, 0.3), 1.0)
    assert np.isclose(transmissibility(1.0, 0.1), np.sqrt(1 + 0.04) / 0.2)


def test_isolator_design_hits_target():
    m, rpm, T = 50.0, 1800.0, 0.1
    k = isolator_stiffness(m, rpm, T)
    w = rpm * 2 * np.pi / 60
    r = w / np.sqrt(k / m)
    assert np.isclose(transmissibility(r, 0.0), T)
    k2 = isolator_stiffness(m, rpm, T, zeta=0.1)
    r2 = w / np.sqrt(k2 / m)
    assert np.isclose(transmissibility(r2, 0.1), T, rtol=1e-6)


def test_unbalance_limits():
    assert np.isclose(unbalance_response(50.0, 0.05), 1.0, rtol=1e-3)
    assert np.isclose(unbalance_response(1.0, 0.1), 1 / 0.2)
