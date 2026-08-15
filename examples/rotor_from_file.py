"""Load the two rotor definitions from data/, build the FE models, and report
undamped criticals, Campbell criticals, the ISO 21940-11 unbalance budget and
the predicted response to the G2.5 unbalance case. Run from repo root:
    python examples/rotor_from_file.py
"""
import os, sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vibtool import load_rotor, load_balance_grades, load_vibration_zones
from vibtool.io import classify_vibration, permissible_unbalance
from vibtool.plots import plot_unbalance_response

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")

grades = load_balance_grades(os.path.join(DATA, "standards", "iso21940-11_balance_grades.csv"))
g25 = next(g for g in grades if g["grade"] == "G2.5")
zones = load_vibration_zones(os.path.join(DATA, "standards", "iso10816-3_zone_limits.csv"))

for fname in ("rotors/steam_turbine_rotor.yaml", "rotors/ev_motor_rotor.json"):
    spec = load_rotor(os.path.join(DATA, fname))
    shaft = spec.build_shaft()
    disk_mass = sum(float(d["mass_kg"]) for d in spec.disks)
    total_mass = shaft.rho * shaft.A * shaft.L + disk_mass
    print(f"\n{spec.name}: length {spec.length:.3f} m, model mass {total_mass:.0f} kg, "
          f"service speed {spec.service_speed_rpm:.0f} rpm")

    w, _ = shaft.undamped_frequencies(n_modes=4)
    print(f"  first undamped criticals: {w[0]/2/np.pi:.1f} and {w[2]/2/np.pi:.1f} Hz")

    Om_max = 1.6 * spec.service_speed_rpm * 2 * np.pi / 60.0
    Om = np.linspace(1.0, Om_max, 400)
    for c, d in shaft.critical_speeds(Om, n_modes=6)[:4]:
        sense = "forward" if d > 0 else "backward" if d < 0 else "undetermined"
        print(f"  Campbell critical {c*60/2/np.pi:8.1f} rpm  {sense}")

    U_per = permissible_unbalance(g25["eper_omega_mm_s"], total_mass, spec.service_speed_rpm)
    print(f"  ISO 21940-11 {g25['grade']} budget: U_per = {U_per*1e6:.0f} g mm total")

    case = spec.unbalance_cases[0]
    node = spec.node_of(shaft, case["station"])
    resp = shaft.unbalance_response(Om, node=node, me=float(case["unbalance_kg_m"]))
    worst = resp.max(axis=1)
    i_srv = int(np.argmin(np.abs(Om - spec.service_speed_rpm * 2 * np.pi / 60.0)))
    # rough velocity reading for the zone check: v_rms = omega x / sqrt(2)
    v_rms = Om[i_srv] * worst[i_srv] / np.sqrt(2.0) * 1e3
    group = "group1_large_300kW_to_50MW" if total_mass > 300 else "group2_medium_15kW_to_300kW"
    zone = classify_vibration(v_rms, zones, group, "flexible")
    print(f"  case {case['name']}: peak amplitude {worst.max()*1e6:.1f} um, "
          f"{v_rms:.2f} mm/s rms at service speed, ISO 10816-3 zone {zone}")

    fig = plot_unbalance_response(Om, worst, label=f"{spec.name}, worst node, {case['name']}")
    fig.savefig(os.path.join(OUT, f"unbalance_{spec.name}.png"), dpi=150)

print("\nfigures written to", os.path.abspath(OUT))
