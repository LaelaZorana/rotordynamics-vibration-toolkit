import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from vibtool.plots import (
    plot_campbell,
    plot_mode_shapes,
    plot_transmissibility,
    plot_unbalance_response,
)
from vibtool.rotor import ShaftFE

E, rho = 210e9, 7850.0


def teardown_function():
    plt.close("all")


def test_plot_transmissibility_returns_figure():
    fig = plot_transmissibility()
    assert fig.axes


def test_plot_mode_shapes_first_mode_is_symmetric():
    # the C1 regression check: with the y plane fixed, the first bending mode
    # of a symmetric rotor must be symmetric about midspan in the figure data
    shaft = ShaftFE(1.0, 0.03, E, rho, n_el=12)
    shaft.add_bearing(0, 1e7, 200.0)
    shaft.add_bearing(12, 1e7, 200.0)
    shaft.add_disk(6, m=5.0, Id=0.02, Ip=0.04)
    y_dofs = [shaft.dof(n, k) for n in range(shaft.n_nodes) for k in ("y", "ys")]
    w, modes = shaft.undamped_frequencies(n_modes=3, fixed_dofs=y_dofs)
    fig = plot_mode_shapes(shaft.z, modes, w, n=3)
    line = fig.axes[0].lines[0]
    y = line.get_ydata()
    assert np.allclose(y, y[::-1], atol=1e-6)
    assert np.isclose(np.max(np.abs(y)), 1.0)


def test_plot_mode_shapes_rejects_zero_plane():
    shaft = ShaftFE(1.0, 0.03, E, rho, n_el=4)
    modes = np.zeros((shaft.ndof, 1))
    modes[2, 0] = 1.0  # pure y motion, zero x translation everywhere
    with pytest.raises(ValueError):
        plot_mode_shapes(shaft.z, modes, np.array([10.0]), n=1)


def test_plot_campbell_and_unbalance_smoke():
    shaft = ShaftFE(1.0, 0.03, E, rho, n_el=8)
    shaft.add_bearing(0, 1e7, 100.0)
    shaft.add_bearing(8, 1e7, 100.0)
    Om = np.linspace(1.0, 1000.0, 40)
    W, D = shaft.campbell(Om, n_modes=4)
    fig = plot_campbell(Om, W, D, crits=[(500.0, 1.0)])
    assert fig.axes
    resp = shaft.unbalance_response(Om, node=4, me=1e-4)
    fig2 = plot_unbalance_response(Om, resp[:, 4], phase=np.zeros_like(Om))
    assert len(fig2.axes) == 2
