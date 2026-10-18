from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pytest

from demregpy.plotting import plot_dem, plot_demogram, plot_loci_curves

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _clean_figures():
    yield
    plt.close("all")


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


def test_plot_demogram_sets_labels_and_defaults():
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
        [1.5e20, 2.5e20, 2e20],
    ])

    ax, mesh = plot_demogram(logt, dem)

    assert ax.get_xlabel() == "Time Step"
    assert ax.get_ylabel() == r"$\log_{10} T$"
    assert isinstance(mesh, matplotlib.collections.QuadMesh)
    assert isinstance(mesh.norm, matplotlib.colors.LogNorm)
    assert mesh.norm.vmin == 1e20
    assert mesh.norm.vmax == 3e20


def test_plot_demogram_accepts_times_keyword_and_positional():
    logt = np.array([5.8, 6.0, 6.2])
    times = np.array([0.0, 5.0, 10.0])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
        [1.5e20, 2.5e20, 2e20],
    ])

    # Via keyword argument
    ax1, _mesh1 = plot_demogram(logt, dem, times=times)
    assert ax1.get_xlabel() == "Time"

    # Via 3 positional arguments: plot_demogram(times, logt, dem)
    ax2, _mesh2 = plot_demogram(times, logt, dem)
    assert ax2.get_xlabel() == "Time"


def test_plot_demogram_accepts_logt_bin_edges():
    # Non-uniform logt bin edges with length nt + 1 = 5 (for nt = 4)
    logt_edges = np.array([5.0, 5.2, 5.7, 6.3, 7.0])
    dem = np.array([
        [1e20, 2e20, 3e20, 4e20],
        [2e20, 3e20, 1e20, 2e20],
    ])

    _ax, mesh = plot_demogram(logt_edges, dem)
    assert isinstance(mesh, matplotlib.collections.QuadMesh)
    coords = mesh.get_coordinates()
    # The mesh Y-coordinates should match the exact non-uniform bin edges
    np.testing.assert_allclose(coords[:, 0, 1], logt_edges)


def test_plot_demogram_accepts_times_bin_edges():
    # times has length n_times + 1 = 4 (for n_times = 3)
    times_edges = np.array([0.0, 10.0, 25.0, 40.0])
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
        [1.5e20, 2.5e20, 2e20],
    ])

    ax, mesh = plot_demogram(logt, dem, times=times_edges)
    assert ax.get_xlabel() == "Time"
    coords = mesh.get_coordinates()
    np.testing.assert_allclose(coords[0, :, 0], times_edges)

    # Also test as 3 positional arguments with datetime edges
    t0 = datetime(2021, 1, 1, 0, 0)
    dt_edges = [t0 + timedelta(minutes=10 * i) for i in range(4)]
    ax2, mesh2 = plot_demogram(dt_edges, logt, dem)
    assert ax2.get_xlabel() == "Time"
    assert isinstance(mesh2, matplotlib.collections.QuadMesh)


def test_plot_demogram_constant_and_infinite_values():
    logt = np.array([5.8, 6.0, 6.2])
    # Constant positive DEM
    dem_const = np.full((3, 3), 1e20)
    _ax, mesh = plot_demogram(logt, dem_const)
    assert mesh.norm.vmin < mesh.norm.vmax
    assert mesh.norm.vmin > 0

    # DEM containing +inf and NaN
    dem_inf = np.array([
        [1e20, np.inf, 3e20],
        [np.nan, 2e20, 1e20],
    ])
    _ax2, mesh2 = plot_demogram(logt, dem_inf)
    assert np.isfinite(mesh2.norm.vmin)
    assert np.isfinite(mesh2.norm.vmax)
    assert mesh2.norm.vmax == 3e20


def test_plot_demogram_flat_shading():
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
    ])

    # With centers and explicit shading='flat'
    _ax, mesh = plot_demogram(logt, dem, shading="flat")
    assert isinstance(mesh, matplotlib.collections.QuadMesh)


def test_plot_demogram_custom_cax():
    _fig, (ax, cax) = plt.subplots(1, 2)
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
    ])

    _ax, mesh = plot_demogram(logt, dem, ax=ax, colorbar=True, cbar_kwargs={"cax": cax})
    assert mesh.colorbar is not None
    assert mesh.colorbar.ax is cax



def test_plot_demogram_linear_scale_and_negative_values():
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [-0.5, 0.0, 0.5],
        [-0.2, 0.3, 0.8],
    ])

    _ax, mesh = plot_demogram(
        logt,
        dem,
        log=False,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    assert isinstance(mesh.norm, matplotlib.colors.Normalize)
    assert mesh.norm.vmin == -1.0
    assert mesh.norm.vmax == 1.0


def test_plot_demogram_with_colorbar():
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
    ])

    _ax, mesh = plot_demogram(logt, dem, colorbar=True, clabel="Custom DEM")

    assert mesh.colorbar is not None
    assert mesh.colorbar.ax.get_ylabel() == "Custom DEM"


def test_plot_demogram_datetime_times():
    t0 = datetime(2011, 2, 15, 1, 44, 0)
    times = [t0 + timedelta(minutes=2 * i) for i in range(3)]
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
        [1.5e20, 2.5e20, 2e20],
    ])

    # As datetime objects
    ax, _mesh = plot_demogram(logt, dem, times=times)
    assert ax.get_xlabel() == "Time"

    # As ISO format strings
    iso_times = [t.isoformat() for t in times]
    ax2, _mesh2 = plot_demogram(logt, dem, times=iso_times)
    assert ax2.get_xlabel() == "Time"


def test_plot_demogram_categorical_strings():
    logt = np.array([5.8, 6.0, 6.2])
    times = ["Phase A", "Phase B", "Phase C"]
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
        [1.5e20, 2.5e20, 2e20],
    ])

    ax, _mesh = plot_demogram(logt, dem, times=times)
    tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert tick_labels == times


def test_plot_demogram_masks_non_positive_in_log_mode():
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 0.0, -1e19],
        [2e20, 3e20, 1e20],
    ])

    _ax, mesh = plot_demogram(logt, dem, log=True)
    # The non-positive elements should be masked in the plotted mesh
    array = mesh.get_array()
    assert np.ma.is_masked(array)
    assert array.mask.any()


def test_plot_demogram_accepts_fig_and_ax():
    fig = plt.figure()
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
    ])

    ax, _mesh = plot_demogram(logt, dem, fig=fig)
    assert ax.figure is fig

    _fig2, ax2 = plt.subplots()
    returned_ax, _mesh2 = plot_demogram(logt, dem, ax=ax2)
    assert returned_ax is ax2


def test_plot_demogram_rejects_multi_axes_figure():
    fig, _axes = plt.subplots(1, 2)
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
    ])

    with pytest.raises(ValueError, match="fig must have exactly one axes unless ax is provided"):
        plot_demogram(logt, dem, fig=fig)


def test_plot_demogram_rejects_bad_shapes_and_arguments():
    logt = np.array([5.8, 6.0, 6.2])
    dem = np.array([
        [1e20, 2e20, 3e20],
        [2e20, 3e20, 1e20],
    ])

    # 1D dem
    with pytest.raises(ValueError, match="two-dimensional"):
        plot_demogram(logt, np.array([1e20, 2e20, 3e20]))

    # < 2 time steps
    with pytest.raises(ValueError, match="at least two time steps"):
        plot_demogram(logt, np.array([[1e20, 2e20, 3e20]]))

    # < 2 temperature bins
    with pytest.raises(ValueError, match="at least two temperature bins"):
        plot_demogram(np.array([5.8]), np.array([[1e20], [2e20]]))

    # 2D logt
    with pytest.raises(ValueError, match="one-dimensional"):
        plot_demogram(np.array([[5.8, 6.0]]), dem)

    # logt length mismatch
    with pytest.raises(ValueError, match="logt length"):
        plot_demogram(np.array([5.8, 6.0]), dem)

    # non-strictly-increasing logt
    with pytest.raises(ValueError, match="strictly increasing"):
        plot_demogram(np.array([6.0, 5.8, 6.2]), dem)

    # times length mismatch (matches neither n_times=2 nor n_times+1=3)
    with pytest.raises(ValueError, match="times length"):
        plot_demogram(logt, dem, times=np.array([0, 10, 20, 30, 40]))

    # times passed positionally and by keyword
    with pytest.raises(ValueError, match="cannot be passed both positionally and as a keyword argument"):
        plot_demogram(np.array([0, 1]), logt, dem, times=np.array([0, 1]))

    # too many positional args
    with pytest.raises(TypeError, match="takes at most 3 positional arguments"):
        plot_demogram(1, 2, 3, 4)

    # vmin <= 0 when log=True
    with pytest.raises(ValueError, match="vmin must be positive when log=True"):
        plot_demogram(logt, dem, log=True, vmin=-1.0)

    # vmax <= vmin
    with pytest.raises(ValueError, match="vmax must be greater than vmin"):
        plot_demogram(logt, dem, vmin=10.0, vmax=5.0)

