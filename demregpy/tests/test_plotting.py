import numpy as np

from demregpy.plotting import plot_dem


def test_plot_dem_sets_labels_and_scale():
    import matplotlib

    matplotlib.use("Agg")

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
    assert ax.get_ylabel() == "DEM"
    assert ax.get_yscale() == "log"


def test_plot_dem_rejects_non_1d_input():
    logt = np.array([[5.8, 5.9]])
    dem = np.array([1e20, 2e20])

    try:
        plot_dem(logt, dem)
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:
        raise AssertionError("plot_dem should reject non-1d input")
