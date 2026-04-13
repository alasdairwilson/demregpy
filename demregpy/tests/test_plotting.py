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


@pytest.mark.parametrize(
    ("logt", "dem", "kwargs", "message"),
    [
        (
            np.array([5.8, 5.9]),
            np.array([1e20, 2e20, 3e20]),
            {},
            "logt and dem must have the same shape",
        ),
        (
            np.array([5.8, 5.9]),
            np.array([1e20, 2e20]),
            {"elogt": np.array([0.1, 0.1, 0.1])},
            "elogt must have the same shape as logt",
        ),
        (
            np.array([5.8, 5.9]),
            np.array([1e20, 2e20]),
            {"edem": np.array([1e19, 1e19, 1e19])},
            "edem must have the same shape as dem",
        ),
    ],
)
def test_plot_dem_rejects_mismatched_shapes(logt, dem, kwargs, message):
    with pytest.raises(ValueError, match=message):
        plot_dem(logt, dem, **kwargs)


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


def test_plot_loci_curves_raw_em_mode_respects_fig_and_ylim():
    fig, ax = plt.subplots()
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

    returned_ax, lines = plot_loci_curves(
        logt,
        dn_in,
        tresp,
        fig=fig,
        dem_space=False,
        show_minimum=False,
        ylim=(1e26, 1e28),
    )

    assert returned_ax is ax
    assert returned_ax.get_ylabel() == r"EM Loci Curve [$\mathrm{cm}^{-5}$]"
    assert len(lines) == 2
    assert returned_ax.get_ylim() == (1e26, 1e28)


def test_plot_loci_curves_rejects_multi_axes_figure():
    fig, _axes = plt.subplots(1, 2)
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

    with pytest.raises(ValueError, match="fig must have exactly one axes unless ax is provided"):
        plot_loci_curves(logt, dn_in, tresp, fig=fig)
