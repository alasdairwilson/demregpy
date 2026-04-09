"""
Plotting helpers for demregpy.
"""

import importlib

import numpy as np

__all__ = [
    "plot_dem",
]


def plot_dem(
    logt,
    dem,
    *,
    elogt=None,
    edem=None,
    ax=None,
    label=None,
    color=None,
    ecolor=None,
    fmt="o",
    capsize=0,
    elinewidth=2,
    xlabel=r"$\log_{10} T$",
    ylabel="DEM",
    yscale="log",
    **kwargs,
):
    """
    Plot a one-dimensional DEM with optional horizontal and vertical error bars.

    Parameters
    ----------
    logt : array_like
        Temperature-bin centres in log10(T).
    dem : array_like
        DEM values for each temperature bin.
    elogt : array_like, optional
        Horizontal uncertainty in log10(T).
    edem : array_like, optional
        Vertical uncertainty on the DEM.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on. If not given, a new figure and axes are created.
    label : str, optional
        Label for the plotted series.
    color : str, optional
        Matplotlib colour for markers and line.
    ecolor : str, optional
        Matplotlib colour for the error bars. Defaults to ``color`` if given.
    fmt : str, optional
        Errorbar marker and line format.
    capsize : float, optional
        Errorbar cap size.
    elinewidth : float, optional
        Errorbar line width.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    yscale : str or None, optional
        Y-axis scale. Defaults to ``"log"``.
    **kwargs
        Additional keyword arguments passed to ``Axes.errorbar``.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        Axes used for the plot.
    container : `matplotlib.container.ErrorbarContainer`
        The Matplotlib errorbar container.
    """
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "plot_dem requires matplotlib. Install demregpy with the 'plot' "
            "extra to use plotting helpers."
        ) from exc

    logt = np.asarray(logt)
    dem = np.asarray(dem)
    if logt.ndim != 1 or dem.ndim != 1:
        raise ValueError("plot_dem expects one-dimensional logt and dem arrays")
    if logt.shape != dem.shape:
        raise ValueError("logt and dem must have the same shape")

    if elogt is not None:
        elogt = np.asarray(elogt)
        if elogt.shape != logt.shape:
            raise ValueError("elogt must have the same shape as logt")
    if edem is not None:
        edem = np.asarray(edem)
        if edem.shape != dem.shape:
            raise ValueError("edem must have the same shape as dem")

    if ax is None:
        _, ax = plt.subplots()

    if ecolor is None and color is not None:
        ecolor = color

    container = ax.errorbar(
        logt,
        dem,
        xerr=elogt,
        yerr=edem,
        fmt=fmt,
        label=label,
        color=color,
        ecolor=ecolor,
        capsize=capsize,
        elinewidth=elinewidth,
        **kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)
    return ax, container
