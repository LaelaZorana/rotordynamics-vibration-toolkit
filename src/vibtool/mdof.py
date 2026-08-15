"""Multi degree of freedom systems: eigenanalysis, mode shapes, proportional
damping and modal superposition harmonic response."""
from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import numpy as np
from scipy.linalg import eigh


@dataclass
class ModalResult:
    omega: np.ndarray  # natural frequencies, rad/s, ascending
    modes: np.ndarray  # mass normalised mode shapes, columns

    @property
    def freq_hz(self) -> np.ndarray:
        return self.omega / (2.0 * np.pi)


@dataclass
class MDOF:
    """M x'' + C x' + K x = f(t). C defaults to zero (undamped)."""

    M: np.ndarray
    K: np.ndarray
    C: np.ndarray | None = field(default=None)

    def __post_init__(self):
        self.M = np.asarray(self.M, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        if self.C is None:
            self.C = np.zeros_like(self.K)
        self.C = np.asarray(self.C, dtype=float)

    @property
    def ndof(self) -> int:
        return self.M.shape[0]

    def eigen(self) -> ModalResult:
        """Undamped eigenproblem K phi = w^2 M phi with mass normalised modes."""
        lam, phi = eigh(self.K, self.M)
        if lam.min() < -1e-8 * max(abs(lam.max()), 1.0):
            warnings.warn("significantly negative eigenvalue found, check K and M", stacklevel=2)
        lam = np.clip(lam, 0.0, None)
        return ModalResult(np.sqrt(lam), phi)

    def set_rayleigh_damping(self, alpha: float, beta: float) -> None:
        """C = alpha M + beta K."""
        self.C = alpha * self.M + beta * self.K

    def modal_damping(self) -> np.ndarray:
        """Modal damping ratios assuming C is diagonalised by the modes."""
        res = self.eigen()
        Cm = res.modes.T @ self.C @ res.modes
        off = Cm - np.diag(np.diag(Cm))
        if np.linalg.norm(off) > 1e-6 * max(np.linalg.norm(np.diag(Cm)), 1e-300):
            warnings.warn(
                "C is not diagonalised by the undamped modes, modal damping "
                "ratios are approximate, use direct_frf for the exact response",
                stacklevel=2,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.diag(Cm) / (2.0 * res.omega)
        return np.nan_to_num(z)

    def frf(self, w, f: np.ndarray, n_modes: int | None = None) -> np.ndarray:
        """Harmonic response amplitude vector X(w) by modal superposition.

        Returns array of shape (len(w), ndof) of complex displacements for a
        force vector f applied as f cos(w t). Assumes proportional damping.
        """
        w = np.atleast_1d(np.asarray(w, dtype=float))
        res = self.eigen()
        n = self.ndof if n_modes is None else n_modes
        phi = res.modes[:, :n]
        wn = res.omega[:n]
        zeta = self.modal_damping()[:n]
        q_force = phi.T @ np.asarray(f, dtype=float)
        X = np.zeros((w.size, self.ndof), dtype=complex)
        for i, wi in enumerate(w):
            H = 1.0 / (wn**2 - wi**2 + 2j * zeta * wn * wi)
            X[i] = phi @ (H * q_force)
        return X

    def direct_frf(self, w, f: np.ndarray) -> np.ndarray:
        """Direct inversion FRF, valid for any damping matrix. Same shape as frf."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        f = np.asarray(f, dtype=float)
        X = np.zeros((w.size, self.ndof), dtype=complex)
        for i, wi in enumerate(w):
            Z = self.K - wi**2 * self.M + 1j * wi * self.C
            X[i] = np.linalg.solve(Z, f)
        return X


def rayleigh_fit(w1: float, z1: float, w2: float, z2: float) -> tuple[float, float]:
    """Solve for alpha, beta such that zeta = alpha/(2w) + beta w/2 at two frequencies."""
    A = np.array([[1.0 / (2.0 * w1), w1 / 2.0], [1.0 / (2.0 * w2), w2 / 2.0]])
    alpha, beta = np.linalg.solve(A, [z1, z2])
    return float(alpha), float(beta)


def two_dof_analytic(m1, m2, k1, k2) -> np.ndarray:
    """Closed form natural frequencies of the chain m1-k1-ground, m2-k2-m1."""
    a = m1 * m2
    b = -(m1 * k2 + m2 * (k1 + k2))
    c = k1 * k2
    disc = np.sqrt(b**2 - 4 * a * c)
    lam = np.array([(-b - disc) / (2 * a), (-b + disc) / (2 * a)])
    return np.sqrt(lam)
