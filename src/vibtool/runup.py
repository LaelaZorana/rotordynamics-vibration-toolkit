"""Critical speed and damping identification from a 1x run up measurement.

Three estimates from the same data. Peak picking with parabolic interpolation
gives the critical speed. The half power bandwidth of the resonance gives a
damping ratio, valid for light damping. A Kasa least squares circle fit to the
runout compensated polar plot gives a second damping estimate from the angle
swept around the modal circle, which is less sensitive to amplitude noise."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunupFit:
    critical_speed_rpm: float
    zeta_half_power: float
    zeta_circle: float
    runout_um: complex  # low speed vector subtracted before fitting
    circle_center: complex
    circle_radius: float


def _parabolic_peak(x, y):
    """Vertex of the parabola through the peak sample and its neighbours."""
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return x[i], y[i]
    denom = y[i - 1] - 2.0 * y[i] + y[i + 1]
    if denom == 0.0:
        return x[i], y[i]
    delta = 0.5 * (y[i - 1] - y[i + 1]) / denom
    xv = x[i] + delta * (x[i + 1] - x[i])
    yv = y[i] - 0.25 * (y[i - 1] - y[i + 1]) * delta
    return float(xv), float(yv)


def _crossing(x, y, level, rising):
    """Interpolated x where y crosses level, searching from the peak outward."""
    i_pk = int(np.argmax(y))
    idx = range(i_pk, 0, -1) if rising else range(i_pk, len(y) - 1)
    for i in idx:
        j = i - 1 if rising else i + 1
        if (y[i] - level) * (y[j] - level) <= 0 and y[i] != y[j]:
            t = (level - y[i]) / (y[j] - y[i])
            return float(x[i] + t * (x[j] - x[i]))
    return None


def _kasa_circle(re, im):
    """Algebraic least squares circle fit, Kasa method."""
    A = np.column_stack([2.0 * re, 2.0 * im, np.ones_like(re)])
    b = re**2 + im**2
    (xc, yc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(c + xc**2 + yc**2)
    return complex(xc, yc), float(r)


def fit_runup(speed_rpm, amp_um, phase_deg, n_runout: int = 5, band: float = 0.25) -> RunupFit:
    """Identify the critical speed and damping from a 1x run up.

    speed_rpm, amp_um, phase_deg are equal length arrays for one probe, with
    phase as lag in degrees. The mean 1x vector of the first n_runout points is
    treated as slow roll runout and subtracted. band is the half width of the
    speed window around the peak, as a fraction of the peak speed, used for the
    circle fit."""
    speed = np.asarray(speed_rpm, dtype=float)
    amp = np.asarray(amp_um, dtype=float)
    phase = np.asarray(phase_deg, dtype=float)
    if not (speed.size == amp.size == phase.size and speed.size > 2 * n_runout):
        raise ValueError("need matching arrays with more than 2 n_runout samples")

    vec = amp * np.exp(-1j * np.radians(phase))
    runout = complex(np.mean(vec[:n_runout]))
    comp = vec - runout
    mag = np.abs(comp)

    # peak pick with parabolic refinement
    n_cr, peak = _parabolic_peak(speed, mag)

    # half power bandwidth on the compensated amplitude
    level = peak / np.sqrt(2.0)
    n1 = _crossing(speed, mag, level, rising=True)
    n2 = _crossing(speed, mag, level, rising=False)
    if n1 is None or n2 is None:
        raise ValueError("half power points not bracketed by the sweep")
    zeta_hp = (n2 - n1) / (2.0 * n_cr)

    # circle fit in a window around the peak
    sel = np.abs(speed - n_cr) < band * n_cr
    if np.count_nonzero(sel) < 5:
        raise ValueError("too few points near resonance for the circle fit")
    center, radius = _kasa_circle(comp[sel].real, comp[sel].imag)
    # angles of the half power points seen from the circle centre span 90
    # degrees each side of resonance for a single mode, so the sweep rate of
    # the angle gives zeta: tan(theta/2) = (1 - r^2) / (2 zeta r) leads to
    # zeta = (n2^2 - n1^2) / (2 n_cr (n1 tan(t1/2) + n2 tan(t2/2))) and with
    # theta = 90 degrees at both half power points the tangents are one
    zeta_circle = (n2**2 - n1**2) / (2.0 * n_cr * (n1 + n2))

    return RunupFit(
        critical_speed_rpm=n_cr,
        zeta_half_power=float(zeta_hp),
        zeta_circle=float(zeta_circle),
        runout_um=runout,
        circle_center=center,
        circle_radius=radius,
    )
