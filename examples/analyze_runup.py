"""Identify the first critical speed and damping from the synthetic run up in
data/runup_turbine_synthetic.csv, and plot the Bode and polar views with the
fit results. Run from repo root:
    python examples/analyze_runup.py
"""
import os, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vibtool import load_runup, fit_runup

ROOT = os.path.join(os.path.dirname(__file__), "..")
run = load_runup(os.path.join(ROOT, "data", "runup_turbine_synthetic.csv"))

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for probe in run.probes:
    fit = fit_runup(run.speed_rpm, run.amp_um[probe], run.phase_deg[probe])
    print(
        f"{probe}: critical {fit.critical_speed_rpm:.0f} rpm, "
        f"zeta {fit.zeta_half_power:.4f} half power, {fit.zeta_circle:.4f} circle fit, "
        f"runout {abs(fit.runout_um):.1f} um"
    )
    comp = run.complex_vector(probe) - fit.runout_um
    axes[0, 0].plot(run.speed_rpm, np.abs(comp), label=probe)
    axes[1, 0].plot(run.speed_rpm, -np.degrees(np.angle(comp)) % 360.0, ".", ms=2, label=probe)
    axes[0, 1].plot(comp.real, comp.imag, ".", ms=3, label=probe)
    th = np.linspace(0, 2 * np.pi, 200)
    axes[0, 1].plot(
        fit.circle_center.real + fit.circle_radius * np.cos(th),
        fit.circle_center.imag + fit.circle_radius * np.sin(th),
        lw=0.8,
    )
    axes[0, 0].axvline(fit.critical_speed_rpm, color="gray", ls="--", lw=0.8)

axes[0, 0].set_ylabel("compensated 1x amplitude (um)")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[1, 0].set_ylabel("phase lag (deg)")
axes[1, 0].set_xlabel("speed (rpm)")
axes[1, 0].grid(alpha=0.3)
axes[0, 1].set_xlabel("real (um)")
axes[0, 1].set_ylabel("imag (um)")
axes[0, 1].set_title("polar plot with Kasa circle fit")
axes[0, 1].axis("equal")
axes[0, 1].grid(alpha=0.3)
axes[1, 1].axis("off")
axes[1, 1].text(
    0.05,
    0.6,
    "synthetic data, truth:\ncritical 4200 rpm\nzeta 0.045",
    fontsize=11,
    family="monospace",
)
fig.suptitle("Run up identification, synthetic turbine data")
fig.tight_layout()
out = os.path.join(ROOT, "figures", "runup_identification.png")
fig.savefig(out, dpi=150)
print("figure written to", os.path.abspath(out))
