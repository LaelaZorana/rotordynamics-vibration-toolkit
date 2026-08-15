import warnings

import numpy as np
import pytest
from vibtool.mdof import MDOF


def test_negative_eigenvalue_warns():
    with pytest.warns(UserWarning):
        MDOF(np.eye(2), np.diag([1.0, -1.0])).eigen()


def test_nonproportional_damping_warns():
    M = np.diag([2.0, 1.0])
    K = np.array([[500.0, -100.0], [-100.0, 100.0]])
    sys = MDOF(M, K, C=np.array([[5.0, 0.0], [0.0, 0.0]]))
    with pytest.warns(UserWarning):
        sys.modal_damping()


def test_n_modes_zero_means_zero():
    M = np.diag([2.0, 1.0])
    K = np.array([[500.0, -100.0], [-100.0, 100.0]])
    sys = MDOF(M, K)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X = sys.frf(np.array([1.0]), np.array([1.0, 0.0]), n_modes=0)
    assert np.allclose(X, 0.0)
