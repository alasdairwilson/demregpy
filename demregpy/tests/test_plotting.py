import matplotlib
import numpy as np
import pytest

from demregpy.plotting import plot_dem, plot_loci_curves

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def test_plot_dem_sets_labels_and_scale():
    logt = np.array([5.8, 5.9, 6.0])
    dem = np.array([1e20, 2e20, 1.5e20])
    elogt = np.array([0.05, 0.05, 0.05])
    edem = np.array([1e19, 2e19, 1e19])

    ax, _container = plot_dem(
        logt,
        dem,
        elogt=elogt,
        edem=edem,
        label="Example",
        color="tab:red",
    )

    assert ax.get_xlabel() == r"$\log_{10} T$"
    assert ax.get_ylabel() == r"DEM [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]"
    assert ax.get_yscale() == "log"


def test_plot_dem_rejects_non_1d_input():
    logt = np.array([[5.8, 5.9]])
    dem = np.array([1e20, 2e20])

    with pytest.raises(ValueError, match="one-dimensional"):
        plot_dem(logt, dem)


def test_plot_loci_curves_sets_labels_and_scale():
    logt = np.array([5.8, 5.9, 6.0, 6.1])
    dn_in = np.array([100.0, 60.0])
    tresp = np.array(
        [
            [1e-25, 3e-26],
            [3e-25, 5e-26],
            [2e-25, 4e-25],
            [5e-26, 2e-25],
        ]
    )

    ax, lines = plot_loci_curves(
        logt,
        dn_in,
        tresp,
        channels=["A", "B"],
    )

    assert ax.get_xlabel() == r"$\log_{10} T$"
    assert ax.get_ylabel() == r"Loci Curve [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]"
    assert ax.get_yscale() == "log"
    assert len(lines) == 3
    assert lines[-1].get_label() == "Minimum loci"


def test_plot_loci_curves_accepts_fig_input():
    fig = plt.figure()
    logt = np.array([5.8, 5.9, 6.0, 6.1])
    dn_in = np.array([100.0, 60.0])
    tresp = np.array(
        [
            [1e-25, 3e-26],
            [3e-25, 5e-26],
            [2e-25, 4e-25],
            [5e-26, 2e-25],
        ]
    )

    ax, _lines = plot_loci_curves(logt, dn_in, tresp, fig=fig)

    assert ax.figure is fig


def test_plot_loci_curves_rejects_bad_shapes():
    logt = np.array([5.8, 5.9, 6.0, 6.1])
    dn_in = np.array([100.0, 60.0])
    tresp = np.array([1e-25, 3e-26, 4e-25, 2e-25])

    with pytest.raises(ValueError, match=r"tresp must have shape \(nt, nf\)"):
        plot_loci_curves(logt, dn_in, tresp)


def test_plot_loci_curves_rejects_mismatched_channel_labels():
    logt = np.array([5.8, 5.9, 6.0, 6.1])
    dn_in = np.array([100.0, 60.0])
    tresp = np.array(
        [
            [1e-25, 3e-26],
            [3e-25, 5e-26],
            [2e-25, 4e-25],
            [5e-26, 2e-25],
        ]
    )

    with pytest.raises(ValueError, match="channels must have the same length as dn_in"):
        plot_loci_curves(logt, dn_in, tresp, channels=["A"])
