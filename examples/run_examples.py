"""Generate all figures into figures/. Run from repo root:
    python examples/run_examples.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vibtool import ShaftFE, Jeffcott, isolator_stiffness
from vibtool.plots import plot_transmissibility, plot_mode_shapes, plot_campbell, plot_unbalance_response

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# 1. transmissibility and an isolator design
plot_transmissibility().savefig(os.path.join(OUT, "transmissibility.png"), dpi=150)
k = isolator_stiffness(m=120.0, rpm=1500.0, target_T=0.1, zeta=0.05)
print(f"isolator: total k = {k:.0f} N/m for T = 0.1 at 1500 rpm (120 kg, zeta 0.05)")

# 2. shaft FE model with two bearings and a central disk
E, rho = 210e9, 7850.0
shaft = ShaftFE(L=1.0, d=0.03, E=E, rho=rho, n_el=12)
shaft.add_bearing(0, k=1e7, c=200.0)
shaft.add_bearing(12, k=1e7, c=200.0)
shaft.add_disk(6, m=5.0, Id=0.02, Ip=0.04)
w, modes = shaft.undamped_frequencies(n_modes=6)
plot_mode_shapes(shaft.z, modes[:, ::2], w[::2], n=3).savefig(os.path.join(OUT, "mode_shapes.png"), dpi=150)

Om = np.linspace(1.0, 2000.0, 400)
W, D = shaft.campbell(Om, n_modes=6)
crits = shaft.critical_speeds(Om, n_modes=6)
plot_campbell(Om, W, D, crits).savefig(os.path.join(OUT, "campbell.png"), dpi=150)
for c, d in crits:
    print(f"critical speed {c*60/2/np.pi:8.1f} rpm  ({'forward' if d > 0 else 'backward'})")

# 3. unbalance response of the FE rotor at the disk and Jeffcott comparison
resp = shaft.unbalance_response(Om, node=6, me=5.0 * 50e-6)
plot_unbalance_response(Om, resp[:, 6], label="FE rotor, disk node").savefig(
    os.path.join(OUT, "unbalance_fe.png"), dpi=150)
j = Jeffcott(m=5.0, k=2e6, c=300.0, e=50e-6)
amp, ph = j.response(Om)
plot_unbalance_response(Om, amp, ph, label="Jeffcott").savefig(os.path.join(OUT, "unbalance_jeffcott.png"), dpi=150)
print(f"Jeffcott critical speed {j.rpm_cr:.1f} rpm")
print("figures written to", os.path.abspath(OUT))
