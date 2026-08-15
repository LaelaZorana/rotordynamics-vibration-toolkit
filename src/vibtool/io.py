"""Loaders for rotor definition files, run up measurements and standards tables.

Rotor definitions are YAML or JSON files with a station table, disks, bearings
and unbalance cases. Run up measurements are CSV files of 1x vibration vectors
versus speed. The standards tables are small CSVs stored in data/ with their
sources listed in data/README.md."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RotorSpec:
    """A rotor definition loaded from file. Raw dictionaries keep every field
    from the file, and build_shaft turns the model section into a ShaftFE."""

    name: str
    description: str
    material: dict
    model: dict
    stations: list = field(default_factory=list)
    disks: list = field(default_factory=list)
    bearings: list = field(default_factory=list)
    unbalance_cases: list = field(default_factory=list)
    service_speed_rpm: float = 0.0

    @property
    def length(self) -> float:
        return float(sum(s["length_m"] for s in self.stations))

    def station_z(self, name: str) -> float:
        """Axial position of the start of a named station."""
        z = 0.0
        for s in self.stations:
            if s["name"] == name:
                return z
            z += s["length_m"]
        raise KeyError(f"no station named {name}")

    def build_shaft(self):
        """Build a ShaftFE from the model section.

        The FE model in this package is a uniform shaft, so the file supplies
        an equivalent uniform diameter chosen to match the bending stiffness of
        the real stepped shaft. Disks and bearings are placed at the nearest
        node to their axial position."""
        from .rotor import ShaftFE

        n_el = int(self.model["n_el"])
        shaft = ShaftFE(
            L=self.length,
            d=float(self.model["equivalent_diameter_m"]),
            E=float(self.material["E_Pa"]),
            rho=float(self.material["rho_kg_m3"]),
            n_el=n_el,
        )
        for brg in self.bearings:
            node = self._nearest_node(shaft, self.station_z(brg["station"]))
            shaft.add_bearing(node, k=float(brg["k_N_per_m"]), c=float(brg["c_Ns_per_m"]))
        for dsk in self.disks:
            node = self._nearest_node(shaft, self.station_z(dsk["station"]))
            shaft.add_disk(
                node,
                m=float(dsk["mass_kg"]),
                Id=float(dsk["Id_kg_m2"]),
                Ip=float(dsk["Ip_kg_m2"]),
            )
        return shaft

    @staticmethod
    def _nearest_node(shaft, z: float) -> int:
        return int(np.argmin(np.abs(shaft.z - z)))

    def node_of(self, shaft, station: str) -> int:
        """FE node nearest to the start of a named station."""
        return self._nearest_node(shaft, self.station_z(station))


def load_rotor(path) -> RotorSpec:
    """Load a rotor definition from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    required = ("name", "material", "model", "stations")
    for key in required:
        if key not in raw:
            raise ValueError(f"rotor file {path} is missing the {key} section")
    return RotorSpec(
        name=raw["name"],
        description=raw.get("description", ""),
        material=raw["material"],
        model=raw["model"],
        stations=raw["stations"],
        disks=raw.get("disks", []),
        bearings=raw.get("bearings", []),
        unbalance_cases=raw.get("unbalance_cases", []),
        service_speed_rpm=float(raw.get("service_speed_rpm", 0.0)),
    )


@dataclass
class RunUp:
    """A run up measurement: 1x vibration vectors at two probes versus speed."""

    time_s: np.ndarray
    speed_rpm: np.ndarray
    amp_um: dict  # probe name to amplitude array, micrometres peak
    phase_deg: dict  # probe name to phase lag array, degrees

    @property
    def probes(self) -> list:
        return list(self.amp_um)

    def complex_vector(self, probe: str) -> np.ndarray:
        """1x vector in the complex plane, micrometres, phase as lag."""
        return self.amp_um[probe] * np.exp(-1j * np.radians(self.phase_deg[probe]))


def load_runup(path) -> RunUp:
    """Load a run up CSV with columns time_s, speed_rpm and per probe pairs
    named like probe_A_amp_um, probe_A_phase_deg."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    amp = {}
    phase = {}
    for name in cols:
        if name.endswith("_amp_um"):
            probe = name[: -len("_amp_um")]
            amp[probe] = cols[name]
            phase[probe] = cols[f"{probe}_phase_deg"]
    if not amp:
        raise ValueError(f"{path} has no *_amp_um columns")
    return RunUp(time_s=cols["time_s"], speed_rpm=cols["speed_rpm"], amp_um=amp, phase_deg=phase)


def load_balance_grades(path) -> list:
    """Load the ISO 21940-11 balance grade table. Returns a list of dicts with
    grade, eper_omega_mm_s and example machinery."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["eper_omega_mm_s"] = float(r["eper_omega_mm_s"])
    return rows


def permissible_unbalance(grade_value_mm_s: float, rotor_mass_kg: float, service_speed_rpm: float) -> float:
    """Permissible residual unbalance in kg m from an ISO 21940-11 grade value.

    U_per = G * m / omega with G in m/s, m in kg and omega in rad/s."""
    omega = service_speed_rpm * 2.0 * np.pi / 60.0
    if omega <= 0:
        raise ValueError("service speed must be positive")
    return grade_value_mm_s * 1e-3 * rotor_mass_kg / omega


def load_vibration_zones(path) -> list:
    """Load the ISO 10816-3 zone boundary table. Returns a list of dicts with
    machine group, support type, boundary and velocity in mm/s rms."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["velocity_mm_s_rms"] = float(r["velocity_mm_s_rms"])
    return rows


def classify_vibration(velocity_mm_s_rms: float, zones: list, group: str, support: str) -> str:
    """Return the ISO 10816-3 zone letter for a broadband velocity reading."""
    bounds = sorted(
        (r for r in zones if r["machine_group"] == group and r["support"] == support),
        key=lambda r: r["velocity_mm_s_rms"],
    )
    if not bounds:
        raise ValueError(f"no zone boundaries for group {group}, support {support}")
    letters = "ABCD"
    for i, r in enumerate(bounds):
        if velocity_mm_s_rms <= r["velocity_mm_s_rms"]:
            return letters[i]
    return "D"
