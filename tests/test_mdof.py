import numpy as np
from vibtool.mdof import MDOF, rayleigh_fit, two_dof_analytic


def test_two_dof_eigen_vs_analytic():
    m1, m2, k1, k2 = 2.0, 1.0, 400.0, 100.0
    M = np.diag([m1, m2])
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    res = MDOF(M, K).eigen()
    assert np.allclose(res.omega, two_dof_analytic(m1, m2, k1, k2), rtol=1e-10)
    # mass orthonormality
    assert np.allclose(res.modes.T @ M @ res.modes, np.eye(2))


def test_rayleigh_fit_and_modal_damping():
    M = np.diag([2.0, 1.0])
    K = np.array([[500.0, -100.0], [-100.0, 100.0]])
    sys = MDOF(M, K)
    w = sys.eigen().omega
    a, b = rayleigh_fit(w[0], 0.02, w[1], 0.05)
    sys.set_rayleigh_damping(a, b)
    assert np.allclose(sys.modal_damping(), [0.02, 0.05])


def test_modal_superposition_matches_direct():
    M = np.diag([2.0, 1.0, 1.5])
    K = np.array([[300.0, -100.0, 0], [-100.0, 250.0, -150.0], [0, -150.0, 150.0]])
    sys = MDOF(M, K)
    sys.set_rayleigh_damping(0.5, 1e-3)
    w = np.linspace(0.5, 30, 60)
    f = np.array([0, 1.0, 0])
    assert np.allclose(sys.frf(w, f), sys.direct_frf(w, f), rtol=1e-8, atol=1e-12)
