from pathlib import Path

import numpy as np
import pytest
from vibtool.io import (
    classify_vibration,
    load_balance_grades,
    load_rotor,
    load_runup,
    load_vibration_zones,
    permissible_unbalance,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_load_steam_turbine_yaml_and_build():
    spec = load_rotor(DATA / "rotors" / "steam_turbine_rotor.yaml")
    assert spec.name == "small-steam-turbine"
    assert np.isclose(spec.length, 1.70)
    assert spec.station_z("coupling_stub") == 0.0
    assert np.isclose(spec.station_z("de_journal"), 0.15)
    shaft = spec.build_shaft()
    assert len(shaft.bearings) == 2
    assert len(shaft.disks) == 4
    w, _ = shaft.undamped_frequencies(n_modes=2)
    # first bending of a 1.7 m turbine rotor should land in tens of Hz
    assert 10.0 < w[0] / (2 * np.pi) < 200.0


def test_load_ev_motor_json_and_build():
    spec = load_rotor(DATA / "rotors" / "ev_motor_rotor.json")
    assert spec.service_speed_rpm == 15000
    shaft = spec.build_shaft()
    assert len(shaft.bearings) == 2
    w, _ = shaft.undamped_frequencies(n_modes=2)
    assert w[0] > 0.0


def test_unknown_station_raises():
    spec = load_rotor(DATA / "rotors" / "ev_motor_rotor.json")
    with pytest.raises(KeyError):
        spec.station_z("no_such_station")


def test_missing_section_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "x", "material": {}, "model": {}}')
    with pytest.raises(ValueError):
        load_rotor(bad)


def test_balance_grades_and_permissible_unbalance():
    grades = load_balance_grades(DATA / "standards" / "iso21940-11_balance_grades.csv")
    g25 = next(g for g in grades if g["grade"] == "G2.5")
    assert g25["eper_omega_mm_s"] == 2.5
    # G2.5, 470 kg, 7500 rpm: U_per = 2.5e-3 * 470 / 785.4 = 1.496e-3 kg m
    U = permissible_unbalance(2.5, 470.0, 7500.0)
    assert np.isclose(U, 1.496e-3, rtol=1e-3)
    with pytest.raises(ValueError):
        permissible_unbalance(2.5, 470.0, 0.0)


def test_vibration_zone_classification():
    zones = load_vibration_zones(DATA / "standards" / "iso10816-3_zone_limits.csv")
    assert len(zones) == 12
    g1 = "group1_large_300kW_to_50MW"
    assert classify_vibration(1.0, zones, g1, "rigid") == "A"
    assert classify_vibration(3.0, zones, g1, "rigid") == "B"
    assert classify_vibration(5.0, zones, g1, "rigid") == "C"
    assert classify_vibration(10.0, zones, g1, "rigid") == "D"
    with pytest.raises(ValueError):
        classify_vibration(1.0, zones, "no_group", "rigid")


def test_load_runup_columns_and_vectors():
    run = load_runup(DATA / "runup_turbine_synthetic.csv")
    assert set(run.probes) == {"probe_A", "probe_B"}
    assert run.speed_rpm[0] < run.speed_rpm[-1]
    v = run.complex_vector("probe_A")
    assert v.shape == run.speed_rpm.shape
    assert np.allclose(np.abs(v), run.amp_um["probe_A"])
