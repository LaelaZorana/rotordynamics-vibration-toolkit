from pathlib import Path

import numpy as np
import pytest
from vibtool import Jeffcott, load_runup
from vibtool.runup import fit_runup

DATA = Path(__file__).resolve().parents[1] / "data"


def _clean_runup(zeta, fn_rpm=4200.0, runout=2.0 + 1.0j):
    m = 470.0
    wn = fn_rpm * 2 * np.pi / 60.0
    k = m * wn**2
    c = 2 * zeta * np.sqrt(k * m)
    rotor = Jeffcott(m=m, k=k, c=c, e=12e-6)
    rpm = np.linspace(300.0, 7000.0, 300)
    amp, phase = rotor.response(rpm * 2 * np.pi / 60.0)
    vec = amp * 1e6 * np.exp(-1j * phase) + runout
    return rpm, np.abs(vec), -np.degrees(np.angle(vec)) % 360.0


def test_fit_recovers_noise_free_jeffcott():
    zeta = 0.045
    rpm, amp, phase = _clean_runup(zeta)
    fit = fit_runup(rpm, amp, phase)
    # the amplitude peak of the unbalance response sits slightly above the
    # undamped critical, at wn / sqrt(1 - 2 zeta^2)
    expected_peak = 4200.0 / np.sqrt(1.0 - 2.0 * zeta**2)
    assert np.isclose(fit.critical_speed_rpm, expected_peak, rtol=2e-3)
    assert np.isclose(fit.zeta_half_power, zeta, rtol=0.15)
    assert np.isclose(fit.zeta_circle, zeta, rtol=0.15)
    assert np.isclose(fit.runout_um, 2.0 + 1.0j, atol=0.1)
    # the compensated resonance sweeps a circle of diameter close to the
    # resonant amplitude e / (2 zeta)
    assert np.isclose(2.0 * fit.circle_radius, 12.0 / (2.0 * zeta), rtol=0.1)


def test_fit_on_committed_synthetic_file():
    run = load_runup(DATA / "runup_turbine_synthetic.csv")
    for probe in run.probes:
        fit = fit_runup(run.speed_rpm, run.amp_um[probe], run.phase_deg[probe])
        assert np.isclose(fit.critical_speed_rpm, 4200.0, rtol=0.02)
        assert np.isclose(fit.zeta_half_power, 0.045, rtol=0.25)
        assert np.isclose(fit.zeta_circle, 0.045, rtol=0.25)


def test_fit_input_validation():
    with pytest.raises(ValueError):
        fit_runup([1, 2, 3], [1, 2, 3], [0, 0, 0])
    rpm = np.linspace(300.0, 1000.0, 50)
    amp = np.ones_like(rpm)
    with pytest.raises(ValueError):
        # flat response has no bracketed half power points
        fit_runup(rpm, amp, np.zeros_like(rpm))
