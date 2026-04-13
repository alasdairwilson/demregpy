"""
Plotting helpers for demregpy.
"""

import importlib

import numpy as np

__all__ = [
    "plot_dem",
    "plot_loci_curves",
]


def _resolve_axes(plt, *, ax=None, fig=None):
    if ax is not None:
        return ax
    if fig is None:
        _, ax = plt.subplots()
        return ax
    if len(fig.axes) == 0:
        return fig.add_subplot(111)
    if len(fig.axes) == 1:
        return fig.axes[0]
    raise ValueError("fig must have exactly one axes unless ax is provided")


def _logt_bin_widths(logt):
    logt = np.asarray(logt, dtype=float)
    if logt.ndim != 1:
        raise ValueError("logt must be one-dimensional")
    if logt.size < 2:
        raise ValueError("logt must contain at least two temperature points")
    if not np.all(np.diff(logt) > 0):
        raise ValueError("logt must be strictly increasing")

    edges = np.empty(logt.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (logt[:-1] + logt[1:])
    edges[0] = logt[0] - 0.5 * (logt[1] - logt[0])
    edges[-1] = logt[-1] + 0.5 * (logt[-1] - logt[-2])
    return np.diff(edges)


def _default_loci_ylabel(dem_space):
    if dem_space:
        return r"Loci Curve [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]"
    return r"EM Loci Curve [$\mathrm{cm}^{-5}$]"


def _axis_positive_ydata(ax):
    values = []
    for line in ax.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        mask = np.isfinite(ydata) & (ydata > 0)
        if np.any(mask):
            values.append(ydata[mask])
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def _loci_envelope(loci):
    valid = np.isfinite(loci) & (loci > 0)
    envelope = np.full(loci.shape[0], np.nan, dtype=float)
    if np.any(valid):
        envelope[valid.any(axis=1)] = np.min(np.where(valid, loci, np.inf), axis=1)[valid.any(axis=1)]
    return envelope


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
    ylabel=r"DEM [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]",
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
        Matplotlib container returned by ``Axes.errorbar``.
    """
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError("plot_dem requires matplotlib to be installed.") from exc

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

    ax = _resolve_axes(plt, ax=ax)

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


def plot_loci_curves(
    logt,
    dn_in,
    tresp,
    *,
    channels=None,
    ax=None,
    fig=None,
    dem_space=True,
    show_minimum=True,
    minimum_kwargs=None,
    xlabel=r"$\log_{10} T$",
    ylabel=None,
    yscale="log",
    ylim=None,
    **kwargs,
):
    """
    Plot filter loci curves for a single observation vector.

    By default the response matrix is scaled in the same bin-aware way used by
    ``dn2dem``, so the loci curves can be overplotted directly on a DEM axis.
    Set ``dem_space=False`` to plot the raw ``DN / R(T)`` EM loci curves instead.

    Parameters
    ----------
    logt : array_like
        Temperature-bin centres in log10(T).
    dn_in : array_like
        Input counts for one spectrum, with shape ``(nf,)``.
    tresp : array_like
        Temperature response matrix with shape ``(nt, nf)``.
    channels : sequence of str, optional
        Channel labels for the individual loci curves.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on.
    fig : `matplotlib.figure.Figure`, optional
        Figure to draw on if ``ax`` is not given. If the figure already has an
        axes, it must have exactly one.
    dem_space : bool, optional
        Apply the same bin-width scaling used by ``dn2dem`` so the curves are in
        DEM-like units and can be overplotted directly on a DEM axis.
    show_minimum : bool, optional
        Plot the pointwise minimum loci envelope.
    minimum_kwargs : dict, optional
        Keyword arguments passed to the minimum-envelope line.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label. If omitted, a default is chosen from ``dem_space``.
    yscale : str or None, optional
        Y-axis scale. Defaults to ``"log"``.
    ylim : tuple[float, float], optional
        Explicit y limits. If omitted, robust limits are inferred from the loci
        envelope and any existing lines already on the axes.
    **kwargs
        Additional keyword arguments passed to ``Axes.plot`` for each loci curve.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        Axes used for the plot.
    lines : list[`matplotlib.lines.Line2D`]
        Line objects for the individual loci curves, followed by the minimum
        envelope if ``show_minimum=True``.
    """
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError("plot_loci_curves requires matplotlib to be installed.") from exc

    logt = np.asarray(logt, dtype=float)
    dn_in = np.asarray(dn_in, dtype=float)
    tresp = np.asarray(tresp, dtype=float)

    if logt.ndim != 1 or dn_in.ndim != 1:
        raise ValueError("plot_loci_curves expects one-dimensional logt and dn_in arrays")
    if tresp.ndim != 2:
        raise ValueError("tresp must have shape (nt, nf)")
    if tresp.shape[0] != logt.size:
        raise ValueError("The first axis of tresp must match the length of logt")
    if tresp.shape[1] != dn_in.size:
        raise ValueError("The second axis of tresp must match the length of dn_in")
    if channels is not None and len(channels) != dn_in.size:
        raise ValueError("channels must have the same length as dn_in")

    ax = _resolve_axes(plt, ax=ax, fig=fig)
    existing_positive = _axis_positive_ydata(ax)

    response = tresp
    if dem_space:
        dlogt = _logt_bin_widths(logt)
        response = tresp * (10.0 ** logt * np.log(10.0 ** dlogt))[:, np.newaxis]

    with np.errstate(divide="ignore", invalid="ignore"):
        loci = dn_in[np.newaxis, :] / response
    loci[~np.isfinite(loci)] = np.nan
    loci[loci <= 0] = np.nan

    labels = channels if channels is not None else [None] * dn_in.size
    lines = []
    for idx, label in enumerate(labels):
        (line,) = ax.plot(logt, loci[:, idx], label=label, **kwargs)
        lines.append(line)

    envelope = _loci_envelope(loci)
    if show_minimum:
        minimum_style = {
            "color": "k",
            "linestyle": "--",
            "linewidth": 2,
            "label": "Minimum loci",
        }
        if minimum_kwargs is not None:
            minimum_style.update(minimum_kwargs)
        (line,) = ax.plot(logt, envelope, **minimum_style)
        lines.append(line)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_default_loci_ylabel(dem_space) if ylabel is None else ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        envelope_positive = envelope[np.isfinite(envelope) & (envelope > 0)]
        if envelope_positive.size or existing_positive.size:
            y_reference = np.concatenate([arr for arr in (existing_positive, envelope_positive) if arr.size > 0])
        else:
            y_reference = loci[np.isfinite(loci) & (loci > 0)]
        if y_reference.size:
            ymin = 0.5 * np.min(y_reference)
            ymax = 5.0 * np.max(y_reference)
            if np.isfinite(ymin) and np.isfinite(ymax) and 0 < ymin < ymax:
                ax.set_ylim(ymin, ymax)
    return ax, lines
