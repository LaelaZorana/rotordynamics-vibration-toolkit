import numpy as np
from vibtool.rotor import Jeffcott, ShaftFE
from vibtool.beam import beam_natural_frequencies, circular_section

E, rho = 210e9, 7850.0


def test_jeffcott_critical_speed():
    j = Jeffcott(m=10.0, k=4e6, c=200.0, e=1e-4)
    assert np.isclose(j.omega_cr, np.sqrt(4e6 / 10.0))
    amp, phase = j.response(np.array([j.omega_cr, 100 * j.omega_cr]))
    assert np.isclose(amp[0], j.e / (2 * j.zeta))
    assert np.isclose(phase[0], np.pi / 2)
    assert np.isclose(amp[1], j.e, rtol=1e-3)  # heavy side flips in, whirl radius tends to e
    assert phase[1] > 0.99 * np.pi


def test_shaft_fe_first_bending_vs_euler_bernoulli():
    L, d = 1.0, 0.025
    A, I = circular_section(d)
    shaft = ShaftFE(L, d, E, rho, n_el=12)
    w_fe, _ = shaft.undamped_frequencies(n_modes=4, fixed_dofs=shaft.pinned_end_dofs())
    w_eb = beam_natural_frequencies(E, I, rho, A, L, "pinned-pinned", 3)
    # x and y planes give repeated pairs; take unique values
    w_fe_u = np.unique(np.round(w_fe, 6))
    assert abs(w_fe_u[0] - w_eb[0]) / w_eb[0] < 0.01
    assert abs(w_fe_u[1] - w_eb[1]) / w_eb[1] < 0.01


def test_shaft_fe_converged_modes_sit_just_below_euler_bernoulli():
    """Rayleigh rotary inertia lowers frequencies slightly relative to the
    Euler-Bernoulli closed form; the gap grows with mode number and stays small
    for a slender shaft (L/d = 40)."""
    L, d = 1.0, 0.025
    A, I = circular_section(d)
    w_eb = beam_natural_frequencies(E, I, rho, A, L, "pinned-pinned", 3)
    s = ShaftFE(L, d, E, rho, n_el=24)
    w, _ = s.undamped_frequencies(n_modes=6, fixed_dofs=s.pinned_end_dofs())
    wu = np.unique(np.round(w, 6))
    rel = (w_eb - wu[:3]) / w_eb
    assert np.all(rel > 0)
    assert rel[0] < rel[1] < rel[2]
    assert rel[2] < 0.005


def test_campbell_forward_backward_split_and_critical():
    shaft = ShaftFE(1.0, 0.03, E, rho, n_el=10)
    shaft.add_bearing(0, 1e7, 100.0)
    shaft.add_bearing(10, 1e7, 100.0)
    shaft.add_disk(5, m=5.0, Id=0.02, Ip=0.04)
    w0, d0 = shaft.whirl_frequencies(0.0, 4)
    assert np.isclose(w0[0], w0[1], rtol=1e-6)  # isotropic: repeated at rest
    w1, d1 = shaft.whirl_frequencies(500.0, 4)
    assert w1[0] < w0[0] < w1[1]  # backward drops, forward rises
    assert d1[0] == -1 and d1[1] == 1
    Om = np.linspace(1.0, 1500.0, 300)
    crits = shaft.critical_speeds(Om, n_modes=4)
    assert len(crits) >= 2
    assert crits[0][1] == -1 and crits[1][1] == 1
    assert crits[0][0] < w0[0] < crits[1][0]
