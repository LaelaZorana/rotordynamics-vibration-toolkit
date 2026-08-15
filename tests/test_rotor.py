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
    # x and y planes give repeated pairs: check the pairing, then take one
    assert np.allclose(w_fe[0::2], w_fe[1::2], rtol=1e-9)
    w_fe_u = w_fe[0::2]
    # tolerances match the accuracy advertised in the README validation table
    assert np.isclose(w_fe_u[0] / (2 * np.pi), 50.768, rtol=1e-4)
    assert np.isclose(w_fe_u[1] / (2 * np.pi), 202.97, rtol=1e-3)
    assert abs(w_fe_u[0] - w_eb[0]) / w_eb[0] < 2e-4
    assert abs(w_fe_u[1] - w_eb[1]) / w_eb[1] < 8e-4


def test_shaft_fe_converged_modes_sit_just_below_euler_bernoulli():
    """Rayleigh rotary inertia lowers frequencies slightly relative to the
    Euler-Bernoulli closed form; the gap grows with mode number and stays small
    for a slender shaft (L/d = 40)."""
    L, d = 1.0, 0.025
    A, I = circular_section(d)
    w_eb = beam_natural_frequencies(E, I, rho, A, L, "pinned-pinned", 3)
    s = ShaftFE(L, d, E, rho, n_el=24)
    w, _ = s.undamped_frequencies(n_modes=6, fixed_dofs=s.pinned_end_dofs())
    assert np.allclose(w[0::2], w[1::2], rtol=1e-9)
    wu = w[0::2]
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


def test_unsupported_shaft_whirl_raises():
    import pytest

    shaft = ShaftFE(1.0, 0.03, E, rho, n_el=8)
    with pytest.raises(ValueError):
        shaft.whirl_frequencies(0.0, 4)
    # a supported shaft returns no rigid body noise: all frequencies well
    # above zero and matching the undamped solve at rest
    shaft.add_bearing(0, 1e7)
    shaft.add_bearing(8, 1e7)
    w, _ = shaft.whirl_frequencies(0.0, 4)
    w_ref, _ = shaft.undamped_frequencies(n_modes=4)
    assert w.min() > 1.0
    assert np.allclose(np.sort(w), np.sort(w_ref), rtol=1e-6)


def test_whirl_with_fixed_dofs_matches_bearing_free_pinned_shaft():
    shaft = ShaftFE(1.0, 0.025, E, rho, n_el=12)
    fixed = shaft.pinned_end_dofs()
    w, _ = shaft.whirl_frequencies(0.0, 4, fixed_dofs=fixed)
    w_ref, _ = shaft.undamped_frequencies(n_modes=4, fixed_dofs=fixed)
    assert np.allclose(np.sort(w), np.sort(w_ref), rtol=1e-6)


def test_fe_unbalance_matches_jeffcott_limit():
    # a light shaft with a heavy rigid central disk on very stiff bearings is
    # a Jeffcott rotor with k = 48 E I / L^3
    L, d = 1.0, 0.01
    A, I = circular_section(d)
    rho_light = 10.0  # near massless shaft so the disk dominates
    shaft = ShaftFE(L, d, E, rho_light, n_el=12)
    shaft.add_bearing(0, 1e10)
    shaft.add_bearing(12, 1e10)
    m_disk, ecc = 10.0, 1e-4
    shaft.add_disk(6, m=m_disk, Id=1e-5, Ip=2e-5)
    k_eq = 48.0 * E * I / L**3
    j = Jeffcott(m=m_disk, k=k_eq, e=ecc)
    # sample away from the undamped resonance
    Om = np.array([0.5, 2.0, 4.0]) * j.omega_cr
    resp = shaft.unbalance_response(Om, node=6, me=m_disk * ecc)
    amp_j, _ = j.response(Om)
    assert np.allclose(resp[:, 6], amp_j, rtol=1e-2)


def test_gyroscopic_tilt_matches_rigid_disk_closed_form():
    # tilt mode of a rigid disk at midspan of a light pinned shaft: rotational
    # stiffness k_theta = 12 E I / L, forward and backward whirl at
    # (+-Ip Om + sqrt((Ip Om)^2 + 4 Id k_theta)) / (2 Id)
    L, d = 1.0, 0.01
    A, I = circular_section(d)
    rho_light = 10.0  # near massless shaft so the rigid disk formula applies
    shaft = ShaftFE(L, d, E, rho_light, n_el=16)
    Id_disk, Ip_disk = 0.05, 0.08
    shaft.add_disk(8, m=20.0, Id=Id_disk, Ip=Ip_disk)
    k_theta = 12.0 * E * I / L
    Om = 100.0
    w, _ = shaft.whirl_frequencies(Om, 4, fixed_dofs=shaft.pinned_end_dofs())
    disc = np.sqrt((Ip_disk * Om) ** 2 + 4.0 * Id_disk * k_theta)
    wb = (-Ip_disk * Om + disc) / (2.0 * Id_disk)
    wf = (Ip_disk * Om + disc) / (2.0 * Id_disk)
    assert np.isclose(w[2], wb, rtol=5e-3)
    assert np.isclose(w[3], wf, rtol=5e-3)


def test_construction_validation():
    import pytest

    with pytest.raises(ValueError):
        ShaftFE(1.0, 0.03, E, rho, n_el=0)
    with pytest.raises(ValueError):
        ShaftFE(-1.0, 0.03, E, rho)
    shaft = ShaftFE(1.0, 0.03, E, rho, n_el=4)
    with pytest.raises(ValueError):
        shaft.add_disk(9, 1.0, 0.01, 0.02)
    with pytest.raises(ValueError):
        shaft.add_bearing(2, -1e7)
    with pytest.raises(ValueError):
        Jeffcott(m=-1.0, k=1e6)
