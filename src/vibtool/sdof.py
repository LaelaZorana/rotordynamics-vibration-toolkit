"""Single degree of freedom vibration: free and forced response, damping
identification, transmissibility and isolator design, rotating unbalance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SDOF:
    """Mass-spring-damper system m x'' + c x' + k x = f(t)."""

    m: float
    k: float
    c: float = 0.0

    @classmethod
    def from_zeta(cls, m: float, k: float, zeta: float) -> "SDOF":
        return cls(m, k, 2.0 * zeta * np.sqrt(k * m))

    @property
    def wn(self) -> float:
        """Undamped natural frequency in rad/s."""
        return float(np.sqrt(self.k / self.m))

    @property
    def fn(self) -> float:
        """Undamped natural frequency in Hz."""
        return self.wn / (2.0 * np.pi)

    @property
    def zeta(self) -> float:
        return self.c / (2.0 * np.sqrt(self.k * self.m))

    @property
    def wd(self) -> float:
        """Damped natural frequency in rad/s (zero if overdamped)."""
        z = self.zeta
        return self.wn * np.sqrt(1.0 - z**2) if z < 1.0 else 0.0

    def free_response(self, t, x0: float = 0.0, v0: float = 0.0):
        """Closed form free vibration for under, critically and over damped cases."""
        t = np.asarray(t, dtype=float)
        wn, z = self.wn, self.zeta
        if z < 1.0:
            wd = self.wd
            A = x0
            B = (v0 + z * wn * x0) / wd
            return np.exp(-z * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
        if np.isclose(z, 1.0):
            return (x0 + (v0 + wn * x0) * t) * np.exp(-wn * t)
        s = wn * np.sqrt(z**2 - 1.0)
        A = (v0 + (z * wn + s) * x0) / (2.0 * s)
        B = x0 - A
        return np.exp(-z * wn * t) * (A * np.exp(s * t) + B * np.exp(-s * t))

    def frf(self, w):
        """Complex receptance X/F for harmonic force at frequency w (rad/s)."""
        w = np.asarray(w, dtype=float)
        return 1.0 / (self.k - self.m * w**2 + 1j * self.c * w)

    def magnification(self, r):
        """Dynamic magnification factor |X| / (F/k) versus frequency ratio r."""
        r = np.asarray(r, dtype=float)
        z = self.zeta
        return 1.0 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * z * r) ** 2)

    def phase(self, r):
        """Response phase lag in radians versus frequency ratio r."""
        r = np.asarray(r, dtype=float)
        return np.arctan2(2.0 * self.zeta * r, 1.0 - r**2)

    def steady_state(self, F0: float, w: float):
        """Amplitude and phase lag of steady state response to F0 sin(w t)."""
        r = w / self.wn
        return F0 / self.k * float(self.magnification(r)), float(self.phase(r))


def damping_from_log_decrement(x_i: float, x_n: float, n: int = 1) -> float:
    """Damping ratio from peak amplitudes n cycles apart."""
    delta = np.log(x_i / x_n) / n
    return float(delta / np.sqrt(4.0 * np.pi**2 + delta**2))


def damping_from_half_power(f1: float, f2: float, fn: float) -> float:
    """Damping ratio from half power (minus 3 dB) bandwidth of an FRF peak."""
    return float((f2 - f1) / (2.0 * fn))


def transmissibility(r, zeta):
    """Force (or absolute motion) transmissibility versus frequency ratio."""
    r = np.asarray(r, dtype=float)
    num = 1.0 + (2.0 * zeta * r) ** 2
    den = (1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2
    return np.sqrt(num / den)


def isolator_stiffness(m: float, rpm: float, target_T: float, zeta: float = 0.0) -> float:
    """Isolator stiffness that gives transmissibility target_T at the given rpm.

    For undamped isolators the closed form is r^2 = 1 + 1/T. With damping the
    frequency ratio is found numerically on the isolation branch (r > sqrt 2).
    """
    if not 0.0 < target_T < 1.0:
        raise ValueError("target_T must be between 0 and 1 for isolation")
    w = rpm * 2.0 * np.pi / 60.0
    if zeta == 0.0:
        r = np.sqrt(1.0 + 1.0 / target_T)
    else:
        from scipy.optimize import brentq

        f = lambda r: transmissibility(r, zeta) - target_T
        hi = 10.0
        while f(hi) > 0:
            hi *= 2.0
        r = brentq(f, np.sqrt(2.0) * 1.0001, hi)
    wn = w / r
    return float(m * wn**2)


def unbalance_response(r, zeta):
    """Normalised rotating unbalance amplitude M X / (m e) versus speed ratio."""
    r = np.asarray(r, dtype=float)
    return r**2 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2)
