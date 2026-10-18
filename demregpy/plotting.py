"""
Plotting helpers for demregpy.
"""

import importlib
from datetime import datetime

import numpy as np

__all__ = [
    "plot_dem",
    "plot_demogram",
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


def _centers_to_edges(c):
    c = np.asarray(c, dtype=float)
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


def _datetime_centers_to_edges(c):
    if isinstance(c, np.ndarray) and c.dtype.kind == "M":
        edges = np.empty(c.size + 1, dtype=c.dtype)
        half_dt = (c[1:] - c[:-1]) // 2
        edges[1:-1] = c[:-1] + half_dt
        edges[0] = c[0] - half_dt[0]
        edges[-1] = c[-1] + half_dt[-1]
        return edges
    edges = [None] * (len(c) + 1)
    for i in range(1, len(c)):
        dt = (c[i] - c[i - 1]) / 2
        edges[i] = c[i - 1] + dt
    edges[0] = c[0] - (c[1] - c[0]) / 2
    edges[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return edges


def plot_demogram(
    logt,
    dem,
    *args,
    times=None,
    ax=None,
    fig=None,
    cmap="magma",
    norm=None,
    vmin=None,
    vmax=None,
    log=True,
    colorbar=False,
    cbar_kwargs=None,
    xlabel=None,
    ylabel=r"$\log_{10} T$",
    clabel=None,
    aspect="auto",
    shading="auto",
    **kwargs,
):
    r"""
    Plot a Differential Emission Measure diagram (DEMogram) across time.

    A DEMogram visualises the temporal evolution of the DEM over temperature,
    typically from a time series of multi-channel observations.

    Parameters
    ----------
    logt : array_like
        Temperature-bin centres in log10(T) with shape ``(nt,)``, or bin edges
        with shape ``(nt + 1,)``. If 3 positional arguments are supplied, the
        arguments are interpreted as ``(times, logt, dem)``.
    dem : array_like
        Recovered DEM values with shape ``(n_times, nt)``.
    *args : tuple
        Optional third positional argument allowing ``plot_demogram(times, logt, dem)``.
    times : array_like, optional
        Time values corresponding to each time step (centers with shape ``(n_times,)``
        or bin edges with shape ``(n_times + 1,)``), such as seconds, minutes,
        or timestamps (``datetime`` objects, ISO format strings, or `astropy.time.Time`).
        If omitted, defaults to integer time steps ``0, 1, ..., n_times - 1``.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on.
    fig : `matplotlib.figure.Figure`, optional
        Figure to draw on if ``ax`` is not given. If the figure already has an
        axes, it must have exactly one.
    cmap : str or `matplotlib.colors.Colormap`, optional
        Matplotlib colormap. Defaults to ``"magma"``.
    norm : `matplotlib.colors.Normalize`, optional
        Colormap normalization. If provided, overrides ``log``, ``vmin``, and ``vmax``.
    vmin : float, optional
        Minimum data value for color scaling. If ``log=True`` and ``vmin`` is
        omitted, it defaults to the minimum positive finite value in ``dem``.
    vmax : float, optional
        Maximum data value for color scaling. If ``log=True`` and ``vmax`` is
        omitted, it defaults to the maximum positive finite value in ``dem``.
    log : bool, optional
        Whether to use a logarithmic color scale (`matplotlib.colors.LogNorm`).
        Defaults to ``True``. Any values less than or equal to zero are masked
        as NaN. Set to ``False`` for linear color scaling or relative change maps.
    colorbar : bool, optional
        Whether to draw a colorbar next to the plot. Defaults to ``False``.
    cbar_kwargs : dict, optional
        Additional keyword arguments passed to `matplotlib.figure.Figure.colorbar`.
    xlabel : str, optional
        X-axis label. Defaults to ``"Time"`` if ``times`` is provided, or
        ``"Time Step"`` otherwise.
    ylabel : str, optional
        Y-axis label. Defaults to ``r"$\log_{10} T$"``.
    clabel : str, optional
        Label for the colorbar when ``colorbar=True``. Defaults to
        ``r"DEM [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]"``.
    aspect : str or float, optional
        Aspect ratio of the axes. Defaults to ``"auto"``.
    shading : str, optional
        Shading style for `matplotlib.axes.Axes.pcolormesh`. Defaults to ``"auto"``.
    **kwargs
        Additional keyword arguments passed to `matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        Axes used for the plot.
    mesh : `matplotlib.collections.QuadMesh`
        Matplotlib QuadMesh collection returned by ``pcolormesh``.
    """
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        colors = importlib.import_module("matplotlib.colors")
    except ImportError as exc:
        raise ImportError("plot_demogram requires matplotlib to be installed.") from exc

    if len(args) == 1:
        if times is not None:
            raise ValueError("times cannot be passed both positionally and as a keyword argument")
        times_val = logt
        logt_val = dem
        dem_val = args[0]
    elif len(args) == 0:
        times_val = times
        logt_val = logt
        dem_val = dem
    else:
        raise TypeError(f"plot_demogram takes at most 3 positional arguments, but {2 + len(args)} were given")

    dem_val = np.asarray(dem_val, dtype=float)
    if dem_val.ndim != 2:
        raise ValueError("plot_demogram expects a two-dimensional dem array with shape (n_times, nt)")

    n_times, nt = dem_val.shape
    if n_times < 2:
        raise ValueError("dem must contain at least two time steps; use plot_dem for a single observation")
    if nt < 2:
        raise ValueError("dem must contain at least two temperature bins")

    logt_val = np.asarray(logt_val, dtype=float)
    if logt_val.ndim != 1:
        raise ValueError("logt must be a one-dimensional array")
    if logt_val.size not in (nt, nt + 1):
        raise ValueError(
            f"logt length ({logt_val.size}) does not match DEM temperature axis ({nt}) or edges ({nt + 1})"
        )

    if not np.all(np.diff(logt_val) > 0):
        raise ValueError("logt must be strictly increasing")

    logt_is_edges = logt_val.size == nt + 1

    categorical_labels = None
    is_datetime = False
    if times_val is None:
        times_plot = np.arange(n_times)
        default_xlabel = "Time Step"
    else:
        default_xlabel = "Time"
        if hasattr(times_val, "to_datetime"):
            times_plot = times_val.to_datetime()
            is_datetime = True
        elif hasattr(times_val, "datetime"):
            times_plot = times_val.datetime
            is_datetime = True
        elif isinstance(times_val, (list, tuple)) and len(times_val) > 0 and isinstance(times_val[0], datetime):
            times_plot = list(times_val)
            is_datetime = True
        else:
            times_arr = np.asarray(times_val)
            if times_arr.ndim != 1:
                raise ValueError("times must be a one-dimensional array or sequence")
            if times_arr.size not in (n_times, n_times + 1):
                raise ValueError(
                    f"times length ({times_arr.size}) does not match DEM time axis ({n_times}) or edges ({n_times + 1})"
                )
            if times_arr.dtype.kind == "M":
                times_plot = times_arr
                is_datetime = True
            elif times_arr.dtype.kind in ("U", "S", "O"):
                try:
                    times_plot = [datetime.fromisoformat(str(t)) for t in times_arr]
                    is_datetime = True
                except (ValueError, TypeError):
                    times_plot = np.arange(times_arr.size)
                    categorical_labels = [str(t) for t in times_arr]
            else:
                times_plot = np.asarray(times_arr, dtype=float)

    times_len = len(times_plot)
    if times_len not in (n_times, n_times + 1):
        raise ValueError(
            f"times length ({times_len}) does not match DEM time axis ({n_times}) or edges ({n_times + 1})"
        )
    times_is_edges = times_len == n_times + 1

    ax = _resolve_axes(plt, ax=ax, fig=fig)

    plot_data = dem_val.T
    if norm is None:
        if log:
            positive = np.isfinite(dem_val) & (dem_val > 0)
            if vmin is None and vmax is None:
                if np.any(positive):
                    vmin = float(np.min(dem_val[positive]))
                    vmax = float(np.max(dem_val[positive]))
                    if vmin == vmax:
                        vmin = vmin * 0.1
                        vmax = vmax * 10.0
                else:
                    vmin = 1.0
                    vmax = 10.0
            elif vmin is None:
                vmin = float(np.min(dem_val[positive])) if np.any(positive) else 1.0
                if vmin >= vmax:
                    vmin = vmax * 0.1
            elif vmax is None:
                vmax = float(np.max(dem_val[positive])) if np.any(positive) else 10.0
                if vmax <= vmin:
                    vmax = vmin * 10.0

            if vmin <= 0:
                raise ValueError("vmin must be positive when log=True")
            if vmax <= vmin:
                raise ValueError("vmax must be greater than vmin")
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            plot_data = np.where(plot_data > 0, plot_data, np.nan)
        else:
            if vmin is not None and vmax is not None and vmax <= vmin:
                raise ValueError("vmax must be greater than vmin")
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if logt_is_edges or times_is_edges or shading == "flat":
        logt_mesh = logt_val if logt_is_edges else _centers_to_edges(logt_val)
        if times_is_edges:
            times_mesh = times_plot
        else:
            if categorical_labels is not None:
                times_mesh = np.arange(n_times + 1) - 0.5
            elif is_datetime:
                times_mesh = _datetime_centers_to_edges(times_plot)
            else:
                times_mesh = _centers_to_edges(times_plot)
        plot_shading = "flat" if shading == "auto" else shading
    else:
        logt_mesh = logt_val
        times_mesh = times_plot
        plot_shading = shading

    mesh = ax.pcolormesh(
        times_mesh,
        logt_mesh,
        plot_data,
        cmap=cmap,
        norm=norm,
        shading=plot_shading,
        **kwargs,
    )

    if aspect is not None:
        ax.set_aspect(aspect)
    if categorical_labels is not None:
        ax.set_xticks(np.arange(n_times))
        ax.set_xticklabels(categorical_labels)
    ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        cb_kwargs = dict(cbar_kwargs) if cbar_kwargs is not None else {}
        if "cax" not in cb_kwargs and "ax" not in cb_kwargs:
            cb_kwargs["ax"] = ax
        if "label" not in cb_kwargs:
            cb_kwargs["label"] = clabel if clabel is not None else r"DEM [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]"
        ax.figure.colorbar(mesh, **cb_kwargs)

    return ax, mesh
