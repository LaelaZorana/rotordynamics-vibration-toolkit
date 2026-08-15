import numpy as np
import pytest
from vibtool.beam import beam_natural_frequencies, circular_section
from vibtool.rotor import ShaftFE

E, rho = 210e9, 7850.0


def _fe_frequencies(fixed_dofs_fn, n_el=40):
    shaft = ShaftFE(1.0, 0.02, E, rho, n_el=n_el)
    fixed = fixed_dofs_fn(shaft)
    w, _ = shaft.undamped_frequencies(n_modes=6, fixed_dofs=fixed)
    assert np.allclose(w[0::2], w[1::2], rtol=1e-9)
    return w[0::2]


def test_clamped_free_against_fe():
    # the FE shaft is an independent discretisation of the same beam, so it
    # cross checks the tabulated beta L roots without reusing them
    def clamp_root(shaft):
        return [shaft.dof(0, k) for k in ("x", "xs", "y", "ys")]

    A, I = circular_section(0.02)
    w_cf = beam_natural_frequencies(E, I, rho, A, 1.0, "clamped-free", 3)
    w_fe = _fe_frequencies(clamp_root)
    # FE Rayleigh beam sits slightly below Euler-Bernoulli
    assert np.allclose(w_fe[:3], w_cf, rtol=5e-3)


def test_clamped_clamped_against_fe():
    def clamp_both(shaft):
        last = shaft.n_nodes - 1
        return [shaft.dof(n, k) for n in (0, last) for k in ("x", "xs", "y", "ys")]

    A, I = circular_section(0.02)
    w_cc = beam_natural_frequencies(E, I, rho, A, 1.0, "clamped-clamped", 3)
    w_fe = _fe_frequencies(clamp_both)
    assert np.allclose(w_fe[:3], w_cc, rtol=5e-3)


def test_clamped_pinned_against_fe():
    def clamp_pin(shaft):
        last = shaft.n_nodes - 1
        return [shaft.dof(0, k) for k in ("x", "xs", "y", "ys")] + [
            shaft.dof(last, "x"),
            shaft.dof(last, "y"),
        ]

    A, I = circular_section(0.02)
    w_cp = beam_natural_frequencies(E, I, rho, A, 1.0, "clamped-pinned", 3)
    w_fe = _fe_frequencies(clamp_pin)
    assert np.allclose(w_fe[:3], w_cp, rtol=5e-3)


def test_free_free_equals_clamped_clamped_roots():
    A, I = circular_section(0.02)
    w_ff = beam_natural_frequencies(E, I, rho, A, 1.0, "free-free", 4)
    w_cc = beam_natural_frequencies(E, I, rho, A, 1.0, "clamped-clamped", 4)
    assert np.allclose(w_ff, w_cc)


def test_asymptotic_branch_continues_smoothly():
    A, I = circular_section(0.02)
    for bc in ("clamped-free", "clamped-clamped", "clamped-pinned"):
        w = beam_natural_frequencies(E, I, rho, A, 1.0, bc, 8)
        assert np.all(np.diff(w) > 0)
        # the n = 6 asymptotic root should sit close to the pattern set by
        # the tabulated n = 4 and 5 roots
        step = w[4] - w[3]
        assert abs((w[5] - w[4]) / step - 1.0) < 0.35


def test_unknown_bc_raises():
    with pytest.raises(ValueError):
        beam_natural_frequencies(1, 1, 1, 1, 1, "welded-taped", 2)
