"""Euler-Bernoulli continuous beam natural frequencies, closed form."""
from __future__ import annotations

import numpy as np

# beta_n * L roots of the frequency equations for the first five modes,
# tabulated values as in Blevins, Formulas for Natural Frequency and Mode
# Shape, Table 8-1, and Inman, Engineering Vibration, Table 6.4
_BETA_L = {
    "pinned-pinned": lambda n: n * np.pi,
    "clamped-free": lambda n: [1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839][n - 1]
    if n <= 5
    else (2 * n - 1) * np.pi / 2,
    "clamped-clamped": lambda n: [4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966][n - 1]
    if n <= 5
    else (2 * n + 1) * np.pi / 2,
    "free-free": lambda n: [4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966][n - 1]
    if n <= 5
    else (2 * n + 1) * np.pi / 2,
    "clamped-pinned": lambda n: [3.92660231, 7.06858275, 10.21017612, 13.35176878, 16.49336143][n - 1]
    if n <= 5
    else (4 * n + 1) * np.pi / 4,
}


def beam_natural_frequencies(E, I, rho, A, L, bc="pinned-pinned", n_modes=3):
    """Natural frequencies (rad/s) of a uniform Euler-Bernoulli beam.

    E, I, rho, A, L in SI units. bc is one of the keys in _BETA_L.
    """
    if bc not in _BETA_L:
        raise ValueError(f"unknown boundary condition {bc}")
    out = []
    for n in range(1, n_modes + 1):
        bl = _BETA_L[bc](n)
        out.append((bl / L) ** 2 * np.sqrt(E * I / (rho * A)))
    return np.array(out)


def circular_section(d):
    """Area and second moment of area for a solid circular section of diameter d."""
    A = np.pi * d**2 / 4.0
    I = np.pi * d**4 / 64.0
    return A, I
