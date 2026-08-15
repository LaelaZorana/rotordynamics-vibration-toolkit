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
    x = s.free_response(t, x0=1.0, v0=s.zeta * s.wn * 1.0 * 0)  # generic start
    # peaks of the envelope-modulated cosine occur near multiples of Td
    zeta_id = damping_from_log_decrement(x[0], x[1], n=3)
    assert abs(zeta_id - 0.03) < 2e-3


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


def test_half_power():
    zeta = 0.02
    fn = 50.0
    f1, f2 = fn * (1 - zeta), fn * (1 + zeta)
    assert np.isclose(damping_from_half_power(f1, f2, fn), zeta)


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
