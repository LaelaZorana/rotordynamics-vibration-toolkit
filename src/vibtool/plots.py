"""Plot helpers. Each function returns the matplotlib figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .sdof import transmissibility


def plot_transmissibility(zetas=(0.05, 0.1, 0.2, 0.5), r=None):
    r = np.linspace(0.05, 4.0, 800) if r is None else r
    fig, ax = plt.subplots(figsize=(6, 4))
    for z in zetas:
        ax.plot(r, transmissibility(r, z), label=f"zeta = {z}")
    ax.axvline(np.sqrt(2), color="gray", ls="--", lw=0.8)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("frequency ratio r = w / wn")
    ax.set_ylabel("transmissibility T")
    ax.set_title("Vibration transmissibility")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mode_shapes(z, modes, omega, n=3, title="Shaft bending mode shapes"):
    """modes: array (ndof, n_modes) with x translation at every 4th dof."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(n):
        shape = modes[0::4, i]
        peak = np.max(np.abs(shape))
        if peak == 0.0:
            raise ValueError(f"mode {i} has zero x translation, pass in plane mode vectors")
        shape = shape / peak
        ax.plot(z, shape, marker="o", ms=3, label=f"mode {i+1}: {omega[i]/(2*np.pi):.1f} Hz")
    ax.set_xlabel("axial position z (m)")
    ax.set_ylabel("normalised deflection")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_campbell(Omega, W, D, crits=None):
    rpm = Omega * 60 / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in range(W.shape[1]):
        fw = D[:, k] > 0
        ax.plot(rpm[fw], W[fw, k] / (2 * np.pi), "b.", ms=2)
        ax.plot(rpm[~fw], W[~fw, k] / (2 * np.pi), "r.", ms=2)
    ax.plot(rpm, Omega / (2 * np.pi), "k--", lw=1, label="1X synchronous")
    if crits:
        for Om, d in crits:
            ax.plot(Om * 60 / (2 * np.pi), Om / (2 * np.pi), "ko", mfc="yellow", ms=7)
    ax.plot([], [], "b.", label="forward whirl")
    ax.plot([], [], "r.", label="backward whirl")
    ax.set_xlabel("spin speed (rpm)")
    ax.set_ylabel("whirl frequency (Hz)")
    ax.set_title("Campbell diagram")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_unbalance_response(Omega, amp, phase=None, label="disk"):
    rpm = Omega * 60 / (2 * np.pi)
    if phase is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(rpm, amp * 1e6, label=label)
    axes[0].set_ylabel("whirl radius (um)")
    axes[0].set_title("Unbalance response")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    if phase is not None:
        axes[1].plot(rpm, np.degrees(phase))
        axes[1].set_ylabel("phase lag (deg)")
        axes[1].grid(alpha=0.3)
    axes[-1].set_xlabel("spin speed (rpm)")
    fig.tight_layout()
    return fig
