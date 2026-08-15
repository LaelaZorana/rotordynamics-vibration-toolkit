"""Generate the synthetic run up measurement in data/runup_turbine_synthetic.csv.

The 1x vectors come from the toolkit's own Jeffcott model of the small steam
turbine rotor, with a slow roll runout vector added at each probe and gaussian
noise on amplitude and phase. Probe B sits 90 degrees from probe A around the
shaft, so with isotropic bearings it sees the same amplitude with the phase
shifted by 90 degrees. The random seed is fixed so the file is reproducible.

Run from the repo root:
    python examples/generate_runup_data.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vibtool import Jeffcott

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "runup_turbine_synthetic.csv")

rng = np.random.default_rng(20260815)

# modal mass, stiffness and damping chosen to put the first critical near
# 4200 rpm with zeta 0.045, and an effective eccentricity of 12 micrometres
m = 470.0
wn = 4200.0 * 2 * np.pi / 60.0
k = m * wn**2
zeta_true = 0.045
c = 2 * zeta_true * np.sqrt(k * m)
e = 12e-6
rotor = Jeffcott(m=m, k=k, c=c, e=e)

# linear run up from 300 to 7000 rpm over 180 seconds, one sample per second
t = np.arange(0.0, 181.0, 1.0)
rpm = 300.0 + (7000.0 - 300.0) * t / t[-1]
Om = rpm * 2 * np.pi / 60.0
amp, phase = rotor.response(Om)
amp_um = amp * 1e6

# slow roll runout vectors, micrometres, and measurement noise
runout_A = 3.1 * np.exp(-1j * np.radians(35.0))
runout_B = 2.7 * np.exp(-1j * np.radians(128.0))
vec = amp_um * np.exp(-1j * phase)
vec_A = vec + runout_A + rng.normal(0, 0.25, t.size) + 1j * rng.normal(0, 0.25, t.size)
vec_B = vec * np.exp(-1j * np.pi / 2) + runout_B + rng.normal(0, 0.25, t.size) + 1j * rng.normal(0, 0.25, t.size)

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["time_s", "speed_rpm", "probe_A_amp_um", "probe_A_phase_deg", "probe_B_amp_um", "probe_B_phase_deg"])
    for i in range(t.size):
        w.writerow(
            [
                f"{t[i]:.1f}",
                f"{rpm[i]:.1f}",
                f"{abs(vec_A[i]):.3f}",
                f"{-np.degrees(np.angle(vec_A[i])) % 360.0:.2f}",
                f"{abs(vec_B[i]):.3f}",
                f"{-np.degrees(np.angle(vec_B[i])) % 360.0:.2f}",
            ]
        )

print("wrote", os.path.abspath(OUT))
print(f"true critical {wn * 60 / 2 / np.pi:.1f} rpm, true zeta {zeta_true}")
