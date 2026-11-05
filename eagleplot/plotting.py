"""Plotting module for creating scientific visualizations.

This module provides a high-level interface for creating publication-quality
plots using matplotlib. It includes functions for common plot types, figure
management, and styling utilities.

Examples
--------
>>> from eagleplot.plotting import line_plot, histogram, save_figure
>>> fig, ax = line_plot(x, y, xlabel='Time (s)', ylabel='Temperature (K)')
>>> save_figure(fig, 'output.png', dpi=300)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.projections.polar import PolarAxes

# Okabe-Ito colorblind-safe palette (8 colors)
COLORBLIND_PALETTE: List[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Optional imports for advanced features
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ============================================================================
# Figure Management
# ============================================================================


def create_figure(
    figsize: Tuple[float, float] = (8, 6), dpi: int = 100, **kwargs
) -> Tuple[Figure, Axes]:
    """Create a new figure with axes.

    Parameters
    ----------
    figsize : tuple of (width, height), default (8, 6)
        Figure size in inches
    dpi : int, default 100
        Dots per inch (resolution)
    kwargs : dict
        Additional arguments passed to plt.subplots()

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    ax : Axes
        Matplotlib axes object

    Examples
    --------
    >>> fig, ax = create_figure(figsize=(10, 6), dpi=150)
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **kwargs)
    return fig, ax


def configure_axes(
    ax: Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = True,
    legend: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Axes:
    """Configure axes properties.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to configure
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    grid : bool, default True
        Whether to show grid
    legend : bool, default False
        Whether to show legend
    xlim : tuple of (min, max), optional
        X-axis limits
    ylim : tuple of (min, max), optional
        Y-axis limits
    kwargs : dict
        Additional axes properties

    Returns
    -------
    ax : Axes
        Configured axes object

    Examples
    --------
    >>> configure_axes(ax, xlabel='Time', ylabel='Value', title='Results')
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Apply any additional properties
    for key, value in kwargs.items():
        if hasattr(ax, f"set_{key}"):
            getattr(ax, f"set_{key}")(value)

    return ax


def save_figure(
    fig: Figure,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs,
) -> None:
    """Save figure to file.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    filepath : str or Path
        Output file path
    format : str, optional
        File format (png, pdf, svg, etc.). If None, inferred from filepath
    dpi : int, default 300
        Resolution for raster formats
    bbox_inches : str, default 'tight'
        Bounding box setting
    kwargs : dict
        Additional arguments passed to fig.savefig()

    Examples
    --------
    >>> save_figure(fig, 'plot.png', dpi=300)
    >>> save_figure(fig, 'plot.pdf', format='pdf')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filepath, format=format, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


# ============================================================================
# Core Plotting Functions
# ============================================================================


def histogram(
    data: np.ndarray,
    bins: Union[int, str, np.ndarray] = "auto",
    xlabel: Optional[str] = None,
    ylabel: str = "Frequency",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color: str = "steelblue",
    alpha: float = 0.7,
    density: bool = False,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Create a histogram.

    Parameters
    ----------
    data : np.ndarray
        Data to plot
    bins : int, str, or array, default 'auto'
        Number of bins or bin edges
    xlabel : str, optional
        X-axis label
    ylabel : str, default 'Frequency'
        Y-axis label
    title : str, optional
        Plot title
    figsize : tuple, default (8, 6)
        Figure size in inches
    color : str, default 'steelblue'
        Bar color
    alpha : float, default 0.7
        Transparency (0-1)
    density : bool, default False
        If True, plot probability density instead of counts
    kwargs : dict
        Additional arguments passed to ax.hist()

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    info : dict
        Dictionary containing:
        - 'counts': Histogram counts
        - 'bins': Bin edges
        - 'mean': Data mean
        - 'std': Data standard deviation

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> fig, ax, info = histogram(data, bins=30, xlabel='Value')
    >>> print(f"Mean: {info['mean']:.2f}")
    """
    data = np.asarray(data).flatten()

    fig, ax = create_figure(figsize=figsize)

    # matplotlib types `bins` as int | Sequence[float] | str | None; an ndarray of
    # explicit edges is valid at runtime but not per the stub, so convert it.
    _bins: Union[int, str, List[float], None] = (
        [float(edge) for edge in bins] if isinstance(bins, np.ndarray) else bins
    )
    counts, bin_edges, patches = ax.hist(
        data,
        bins=_bins,
        color=color,
        alpha=alpha,
        density=density,
        **kwargs,  # type: ignore[arg-type]
    )

    # Configure axes
    if ylabel == "Frequency" and density:
        ylabel = "Density"
    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    # Calculate statistics
    info = {
        "counts": counts,
        "bins": bin_edges,
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }

    return fig, ax, info


def line_plot(
    x: np.ndarray,
    y: Union[np.ndarray, List[np.ndarray]],
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    legend: bool = True,
    yerr: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create a line plot.

    Parameters
    ----------
    x : np.ndarray
        X-axis data
    y : np.ndarray or list of np.ndarray
        Y-axis data. Can be single array or list of arrays for multiple lines
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    figsize : tuple, default (8, 6)
        Figure size in inches
    labels : list of str, optional
        Line labels for legend
    colors : list of str, optional
        Line colors
    linestyles : list of str, optional
        Line styles ('-', '--', '-.', ':')
    markers : list of str, optional
        Marker styles ('o', 's', '^', etc.)
    legend : bool, default True
        Whether to show legend (only if labels provided)
    yerr : np.ndarray or list of np.ndarray, optional
        Y error values.  A single array is applied to the first series;
        a list of arrays matches each series in order.
    kwargs : dict
        Additional arguments passed to ax.plot() / ax.errorbar()

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> fig, ax = line_plot(x, y, xlabel='x', ylabel='sin(x)')

    >>> # With error bars
    >>> err = 0.05 * np.ones_like(y)
    >>> fig, ax = line_plot(x, y, yerr=err)
    """
    x = np.asarray(x)

    # Handle single vs multiple y arrays
    if isinstance(y, np.ndarray) and y.ndim == 1:
        y = [y]
    elif not isinstance(y, list):
        y = [np.asarray(y)]
    else:
        y = [np.asarray(yi) for yi in y]

    n_lines = len(y)

    # Normalise yerr to a list aligned with y series
    if yerr is not None:
        if isinstance(yerr, np.ndarray) and yerr.ndim == 1:
            yerr_list: List[Optional[np.ndarray]] = [yerr] + [None] * (n_lines - 1)
        elif isinstance(yerr, list):
            yerr_list = list(yerr) + [None] * max(0, n_lines - len(yerr))
        else:
            yerr_list = [np.asarray(yerr)] + [None] * (n_lines - 1)
    else:
        yerr_list = [None] * n_lines

    fig, ax = create_figure(figsize=figsize)

    # Set default colors, linestyles, markers
    if colors is None:
        colors = [f"C{i}" for i in range(n_lines)]
    if linestyles is None:
        linestyles = ["-"] * n_lines
    if markers is None:
        markers = [None] * n_lines
    if labels is None:
        labels = [None] * n_lines

    # Plot each line (with or without error bars)
    for i, yi in enumerate(y):
        c = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]
        lbl = labels[i]
        yi_err = yerr_list[i]
        if yi_err is not None:
            ax.errorbar(
                x,
                yi,
                yerr=yi_err,
                color=c,
                linestyle=ls,
                marker=mk,
                label=lbl,
                capsize=3,
                **kwargs,
            )
        else:
            ax.plot(
                x,
                yi,
                color=c,
                linestyle=ls,
                marker=mk,
                label=lbl,
                **kwargs,
            )

    # Configure axes
    show_legend = legend and any(label is not None for label in labels)
    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, legend=show_legend)

    return fig, ax


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color: Union[str, np.ndarray] = "steelblue",
    size: Union[float, np.ndarray] = 50,
    alpha: float = 0.7,
    colorbar: bool = False,
    yerr: Optional[np.ndarray] = None,
    xerr: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create a scatter plot.

    Parameters
    ----------
    x : np.ndarray
        X-axis data
    y : np.ndarray
        Y-axis data
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    figsize : tuple, default (8, 6)
        Figure size in inches
    color : str or np.ndarray, default 'steelblue'
        Point color(s). Can be single color or array for color mapping
    size : float or np.ndarray, default 50
        Point size(s). Can be single value or array
    alpha : float, default 0.7
        Transparency (0-1)
    colorbar : bool, default False
        Whether to add colorbar (only if color is array)
    yerr : np.ndarray, optional
        Symmetric Y error bar values
    xerr : np.ndarray, optional
        Symmetric X error bar values
    kwargs : dict
        Additional arguments passed to ax.scatter()

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> fig, ax = scatter_plot(x, y, xlabel='X', ylabel='Y')

    >>> # With error bars
    >>> fig, ax = scatter_plot(x, y, yerr=0.1*np.ones_like(y))
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = create_figure(figsize=figsize)

    scatter = ax.scatter(x, y, c=color, s=size, alpha=alpha, **kwargs)

    # Add colorbar if color is array
    if colorbar and isinstance(color, np.ndarray):
        plt.colorbar(scatter, ax=ax)

    # Overlay error bars if provided
    if yerr is not None or xerr is not None:
        ecolor = color if isinstance(color, str) else "gray"
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            xerr=xerr,
            fmt="none",
            color=ecolor,
            alpha=alpha,
            capsize=3,
        )

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    return fig, ax


def heatmap(
    data: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "viridis",
    colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_n_colors: Optional[int] = None,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create a heatmap.

    Parameters
    ----------
    data : np.ndarray
        2D array to visualize
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    figsize : tuple, default (8, 6)
        Figure size in inches
    cmap : str, default 'viridis'
        Colormap name
    colorbar : bool, default True
        Whether to show colorbar
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    cmap_n_colors : int, optional
        If set, discretize the colormap into this many distinct colors (2–256)
    xticklabels : list of str, optional
        X-axis tick labels
    yticklabels : list of str, optional
        Y-axis tick labels
    kwargs : dict
        Additional arguments passed to ax.imshow()

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> data = np.random.rand(10, 10)
    >>> fig, ax = heatmap(data, title='Random Data', cmap='hot')
    >>> fig, ax = heatmap(data, cmap='viridis', cmap_n_colors=8, vmin=0.2, vmax=0.8)
    """
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    fig, ax = create_figure(figsize=figsize)

    actual_cmap = plt.get_cmap(cmap, cmap_n_colors) if cmap_n_colors is not None else cmap
    im = ax.imshow(data, cmap=actual_cmap, vmin=vmin, vmax=vmax, aspect="auto", **kwargs)

    if colorbar:
        plt.colorbar(im, ax=ax)

    # Set tick labels if provided
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)

    return fig, ax


def contour_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    levels: Union[int, np.ndarray] = 10,
    cmap: str = "viridis",
    filled: bool = True,
    colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_n_colors: Optional[int] = None,
    linewidths: Optional[float] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create a contour plot.

    Parameters
    ----------
    X : np.ndarray
        2D array of X coordinates (from meshgrid)
    Y : np.ndarray
        2D array of Y coordinates (from meshgrid)
    Z : np.ndarray
        2D array of Z values
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    figsize : tuple, default (8, 6)
        Figure size in inches
    levels : int or array, default 10
        Number of contour levels or specific level values
    cmap : str, default 'viridis'
        Colormap name
    filled : bool, default True
        Whether to create filled contours (contourf) or line contours (contour)
    colorbar : bool, default True
        Whether to show colorbar
    vmin : float, optional
        Minimum value for color/level normalization
    vmax : float, optional
        Maximum value for color/level normalization
    cmap_n_colors : int, optional
        If set, discretize the colormap into this many distinct colors (2–256)
    linewidths : float, optional
        Line width for unfilled contour lines (ignored when filled=True)
    kwargs : dict
        Additional arguments passed to ax.contour() or ax.contourf()

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> x = np.linspace(-3, 3, 100)
    >>> y = np.linspace(-3, 3, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = np.sqrt(X**2 + Y**2)
    >>> fig, ax = contour_plot(X, Y, Z, xlabel='X', ylabel='Y')
    >>> fig, ax = contour_plot(X, Y, Z, filled=False, cmap_n_colors=10, vmin=0, vmax=3)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            f"X, Y, Z must have same shape. Got X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}"
        )

    fig, ax = create_figure(figsize=figsize)

    actual_cmap = plt.get_cmap(cmap, cmap_n_colors) if cmap_n_colors is not None else cmap

    plot_kwargs = dict(kwargs)
    if (vmin is not None or vmax is not None) and "norm" not in plot_kwargs:
        plot_kwargs["norm"] = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if filled:
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=actual_cmap, **plot_kwargs)
    else:
        if linewidths is not None and "linewidths" not in plot_kwargs:
            plot_kwargs["linewidths"] = linewidths
        contour = ax.contour(X, Y, Z, levels=levels, cmap=actual_cmap, **plot_kwargs)

    if colorbar:
        plt.colorbar(contour, ax=ax)

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)

    return fig, ax


# ============================================================================
# Extended Plot Types (M14)
# ============================================================================


def bar_chart(
    categories: Union[List, np.ndarray],
    values: Union[np.ndarray, List],
    yerr: Optional[Union[np.ndarray, List]] = None,
    xerr: Optional[Union[np.ndarray, List]] = None,
    hue: Optional[List[str]] = None,
    kind: str = "simple",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color: Union[str, List[str]] = "steelblue",
    alpha: float = 0.8,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Create a bar chart (simple, grouped, or stacked).

    Parameters
    ----------
    categories : list or np.ndarray
        Category labels for the X axis
    values : np.ndarray
        Bar heights.  Shape ``(n_categories,)`` for simple/stacked;
        shape ``(n_groups, n_categories)`` for grouped bars.
    yerr : array-like, optional
        Y error bar values.  Shape matches *values*.
    xerr : array-like, optional
        X error bar values (simple bars only).
    hue : list of str, optional
        Group labels for grouped/stacked charts.
    kind : str, default 'simple'
        ``'simple'``, ``'grouped'``, or ``'stacked'``.
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    figsize : tuple, default (8, 6)
        Figure size in inches
    color : str or list of str, default 'steelblue'
        Bar color(s).
    alpha : float, default 0.8
        Transparency (0–1)
    kwargs : dict
        Extra keyword arguments forwarded to ``ax.bar()``.

    Returns
    -------
    fig : Figure
    ax : Axes
    info : dict
        ``kind``, ``n_categories``, ``n_groups``

    Examples
    --------
    >>> fig, ax, info = bar_chart(['A', 'B', 'C'], [3, 5, 2])
    >>> # Grouped
    >>> fig, ax, info = bar_chart(
    ...     ['A', 'B'], [[2, 4], [3, 1]],
    ...     hue=['G1', 'G2'], kind='grouped',
    ... )
    """
    cats = list(categories)
    vals = np.asarray(values)
    fig, ax = create_figure(figsize=figsize)

    if kind == "simple":
        bar_color = color if isinstance(color, str) else color[0]
        ax.bar(
            cats,
            vals,
            yerr=yerr,
            xerr=xerr,
            color=bar_color,
            alpha=alpha,
            capsize=4,
            **kwargs,
        )
        n_groups = 1

    elif kind == "grouped":
        vals = vals if vals.ndim == 2 else vals.reshape(1, -1)
        n_groups = vals.shape[0]
        n_cats = len(cats)
        colors_list = color if isinstance(color, list) else [f"C{i}" for i in range(n_groups)]
        x = np.arange(n_cats)
        width = 0.8 / n_groups
        offsets = np.linspace(
            -(n_groups - 1) * width / 2,
            (n_groups - 1) * width / 2,
            n_groups,
        )
        for i in range(n_groups):
            g_err = (
                yerr[i]
                if (
                    yerr is not None
                    and hasattr(yerr, "__len__")
                    and len(yerr) == n_groups
                    and hasattr(yerr[0], "__len__")
                )
                else yerr
            )
            lbl = hue[i] if hue and i < len(hue) else f"Group {i + 1}"
            ax.bar(
                x + offsets[i],
                vals[i],
                width=width,
                yerr=g_err,
                label=lbl,
                color=colors_list[i % len(colors_list)],
                alpha=alpha,
                capsize=3,
                **kwargs,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.legend()

    elif kind == "stacked":
        vals = vals if vals.ndim == 2 else vals.reshape(1, -1)
        n_groups = vals.shape[0]
        colors_list = color if isinstance(color, list) else [f"C{i}" for i in range(n_groups)]
        bottom = np.zeros(len(cats))
        for i in range(n_groups):
            lbl = hue[i] if hue and i < len(hue) else f"Group {i + 1}"
            ax.bar(
                cats,
                vals[i],
                bottom=bottom,
                label=lbl,
                color=colors_list[i % len(colors_list)],
                alpha=alpha,
                **kwargs,
            )
            bottom = bottom + vals[i]
        ax.legend()

    else:
        raise ValueError(
            f"Unknown bar chart kind: '{kind}'. Use 'simple', 'grouped', or 'stacked'."
        )

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    info: Dict[str, Any] = {
        "kind": kind,
        "n_categories": len(cats),
        "n_groups": n_groups if kind != "simple" else 1,
    }
    return fig, ax, info


def waterfall_plot(
    x: np.ndarray,
    y_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    offset: Union[float, str] = "auto",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7),
    cmap: str = "viridis",
    alpha: float = 0.9,
    linewidth: float = 1.5,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Create a waterfall (stacked-offset line) plot.

    Each row of *y_matrix* is plotted as a separate line shifted upward
    by a constant *offset*, producing the characteristic waterfall
    appearance used in time-resolved spectroscopy.

    Parameters
    ----------
    x : np.ndarray
        Shared X axis (wavenumber, wavelength, time, …)
    y_matrix : np.ndarray
        2-D array of shape ``(n_traces, n_points)``.  Each row is one
        spectrum / signal trace.
    labels : list of str, optional
        Legend labels for each trace.
    offset : float or 'auto', default 'auto'
        Vertical spacing between traces.  ``'auto'`` uses 60 % of the
        largest peak-to-peak amplitude across all traces.
    xlabel : str, optional
    ylabel : str, optional
    title : str, optional
    figsize : tuple, default (10, 7)
    cmap : str, default 'viridis'
        Colormap used to color successive traces.
    alpha : float, default 0.9
    linewidth : float, default 1.5
    kwargs : dict
        Extra keyword arguments forwarded to ``ax.plot()``.

    Returns
    -------
    fig : Figure
    ax : Axes
    info : dict
        ``n_traces``, ``offset``

    Examples
    --------
    >>> x = np.linspace(400, 4000, 1000)
    >>> spectra = np.random.rand(5, 1000)
    >>> fig, ax, info = waterfall_plot(x, spectra)
    """
    x = np.asarray(x)
    y_mat = np.asarray(y_matrix)
    if y_mat.ndim == 1:
        y_mat = y_mat.reshape(1, -1)

    n_traces = y_mat.shape[0]

    if offset == "auto":
        ptp = np.ptp(y_mat, axis=1)
        offset_val = float(np.max(ptp) * 0.6) if np.max(ptp) > 0 else 1.0
    else:
        offset_val = float(offset)

    fig, ax = create_figure(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap, n_traces)

    for i in range(n_traces):
        y_shifted = y_mat[i] + i * offset_val
        clr = cmap_obj(i / max(n_traces - 1, 1))
        lbl = labels[i] if (labels and i < len(labels)) else None
        ax.plot(
            x,
            y_shifted,
            color=clr,
            alpha=alpha,
            linewidth=linewidth,
            label=lbl,
            **kwargs,
        )

    configure_axes(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend=labels is not None,
    )

    info = {"n_traces": n_traces, "offset": offset_val}
    return fig, ax, info


def dual_axis_plot(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel1: Optional[str] = None,
    ylabel2: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color1: str = "steelblue",
    color2: str = "tomato",
    linestyle1: str = "-",
    linestyle2: str = "--",
    label1: Optional[str] = None,
    label2: Optional[str] = None,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Create a dual Y-axis line plot.

    Plots *y1* on the left axis and *y2* on an independent right axis,
    which is useful when comparing signals that have different units or
    very different magnitudes.

    Parameters
    ----------
    x : np.ndarray
        Shared X axis
    y1 : np.ndarray
        Data for the left Y axis
    y2 : np.ndarray
        Data for the right Y axis
    xlabel : str, optional
    ylabel1 : str, optional
        Left Y-axis label
    ylabel2 : str, optional
        Right Y-axis label
    title : str, optional
    figsize : tuple, default (8, 6)
    color1 : str, default 'steelblue'
        Color for left-axis series
    color2 : str, default 'tomato'
        Color for right-axis series
    linestyle1 : str, default '-'
    linestyle2 : str, default '--'
    label1 : str, optional
        Legend label for left series
    label2 : str, optional
        Legend label for right series
    kwargs : dict
        Extra keyword arguments forwarded to both ``ax.plot()`` calls.

    Returns
    -------
    fig : Figure
    ax : Axes
        Primary (left) axes
    info : dict
        ``ax2`` — the secondary (right) Axes object

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> fig, ax, info = dual_axis_plot(
    ...     x, np.sin(x), np.exp(0.3 * x),
    ...     ylabel1='sin(x)', ylabel2='exp(0.3x)',
    ... )
    """
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    fig, ax1 = create_figure(figsize=figsize)
    ax2 = ax1.twinx()

    lbl1 = label1 or ylabel1 or "Y1"
    lbl2 = label2 or ylabel2 or "Y2"

    ax1.plot(x, y1, color=color1, linestyle=linestyle1, label=lbl1, **kwargs)
    ax2.plot(x, y2, color=color2, linestyle=linestyle2, label=lbl2, **kwargs)

    if title:
        ax1.set_title(title)
    if xlabel:
        ax1.set_xlabel(xlabel)
    if ylabel1:
        ax1.set_ylabel(ylabel1, color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
    if ylabel2:
        ax2.set_ylabel(ylabel2, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

    # Merged legend from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    info: Dict[str, Any] = {"ax2": ax2}
    return fig, ax1, info


# ============================================================================
# Extended Plot Types (M21)
# ============================================================================


def z_colored_scatter(
    x,
    y,
    z,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(8, 6),
    cmap="viridis",
    colorbar=True,
    colorbar_label=None,
    s=20,
    alpha=0.8,
    vmin=None,
    vmax=None,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Scatter plot with a third variable encoded as point color.

    Parameters
    ----------
    x, y : array-like
        Point coordinates.
    z : array-like
        Values to map to color.
    cmap : str, default 'viridis'
        Matplotlib colormap name.
    colorbar : bool, default True
        Whether to add a colorbar.
    colorbar_label : str, optional
        Label for the colorbar.
    s : float, default 20
        Marker size.
    alpha : float, default 0.8
        Marker opacity.
    vmin, vmax : float, optional
        Color scale limits.

    Returns
    -------
    fig, ax, info : tuple
        info keys: 'n_points', 'z_min', 'z_max'.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    fig, ax = create_figure(figsize=figsize)
    sc = ax.scatter(x, y, c=z, cmap=cmap, s=s, alpha=alpha, vmin=vmin, vmax=vmax, **kwargs)
    if colorbar:
        cb = plt.colorbar(sc, ax=ax)
        if colorbar_label:
            cb.set_label(colorbar_label)
    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    info: Dict[str, Any] = {
        "n_points": len(x),
        "z_min": float(np.nanmin(z)),
        "z_max": float(np.nanmax(z)),
    }
    return fig, ax, info


def bubble_chart(
    x,
    y,
    sizes,
    z=None,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(8, 6),
    cmap="viridis",
    colorbar=True,
    colorbar_label=None,
    color="steelblue",
    alpha=0.6,
    size_scale=1.0,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Scatter plot with size-encoded bubbles and optional color encoding.

    Parameters
    ----------
    x, y : array-like
        Point coordinates.
    sizes : array-like
        Values that control bubble area.
    z : array-like, optional
        Values mapped to color.
    size_scale : float, default 1.0
        Multiplicative scale applied to *sizes*.

    Returns
    -------
    fig, ax, info : tuple
        info keys: 'n_points', 'size_min', 'size_max'.
    """
    x, y, s = np.asarray(x), np.asarray(y), np.asarray(sizes, dtype=float)
    s_scaled = s * size_scale
    fig, ax = create_figure(figsize=figsize)
    if z is not None:
        z_arr = np.asarray(z)
        sc = ax.scatter(x, y, s=s_scaled, c=z_arr, cmap=cmap, alpha=alpha, **kwargs)
        if colorbar:
            cb = plt.colorbar(sc, ax=ax)
            if colorbar_label:
                cb.set_label(colorbar_label)
    else:
        ax.scatter(x, y, s=s_scaled, color=color, alpha=alpha, **kwargs)
    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    info: Dict[str, Any] = {
        "n_points": len(x),
        "size_min": float(np.nanmin(s)),
        "size_max": float(np.nanmax(s)),
    }
    return fig, ax, info


def polar_plot(
    theta,
    r,
    title=None,
    figsize=(7, 7),
    color="steelblue",
    linestyle="-",
    linewidth=1.5,
    fill=False,
    fill_alpha=0.15,
    theta_direction=-1,
    theta_zero_location="N",
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Create a polar line/area plot.

    Parameters
    ----------
    theta : array-like
        Angles in radians.
    r : array-like
        Radial values.
    fill : bool, default False
        Whether to fill the area under the curve.
    theta_direction : int, default -1
        -1 for clockwise, 1 for counter-clockwise.
    theta_zero_location : str, default 'N'
        Position of 0°: 'N', 'E', 'S', or 'W'.

    Returns
    -------
    fig, ax, info : tuple
        info keys: 'n_points', 'r_max'.
    """
    theta_arr = np.asarray(theta, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    # subplot_kw guarantees a PolarAxes at runtime; plt.subplots is typed as returning
    # the base Axes, which has no set_theta_* methods.
    polar_ax = cast("PolarAxes", ax)
    polar_ax.set_theta_zero_location(theta_zero_location)
    polar_ax.set_theta_direction(theta_direction)
    ax.plot(theta_arr, r_arr, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
    if fill:
        ax.fill(theta_arr, r_arr, alpha=fill_alpha, color=color)
    if title:
        ax.set_title(title, pad=15)
    info: Dict[str, Any] = {"n_points": len(theta_arr), "r_max": float(np.nanmax(r_arr))}
    return fig, ax, info


def pair_plot(
    data,
    hue=None,
    vars=None,
    kind="scatter",
    diag_kind="hist",
    title=None,
    figsize=None,
    palette=None,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Create a pair plot (scatter matrix) using seaborn.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    hue : str, optional
        Column name to use for color encoding.
    vars : list of str, optional
        Columns to include; defaults to all numeric columns.
    kind : str, default 'scatter'
        Off-diagonal plot type: 'scatter', 'kde', 'hist', 'reg'.
    diag_kind : str, default 'hist'
        Diagonal plot type: 'hist', 'kde', 'auto'.

    Returns
    -------
    fig, ax, info : tuple
        ax is grid.axes[0][0]; info keys: 'n_vars', 'hue'.

    Raises
    ------
    ImportError
        If seaborn is not installed.
    """
    if not HAS_SEABORN:
        raise ImportError("seaborn is required for pair_plot. Install it with: pip install seaborn")
    import seaborn as sns

    kw: Dict[str, Any] = dict(kind=kind, diag_kind=diag_kind)
    if hue is not None:
        kw["hue"] = hue
    if vars is not None:
        kw["vars"] = vars
    if palette is not None:
        kw["palette"] = palette
    if figsize is not None:
        n_vars = len(vars) if vars else len(data.select_dtypes("number").columns)
        kw["height"] = figsize[1] / max(n_vars, 1)
    grid = sns.pairplot(data, **kw, **kwargs)
    if title:
        grid.figure.suptitle(title, y=1.02)
    n_vars = len(vars) if vars else len(data.select_dtypes(include="number").columns)
    info: Dict[str, Any] = {"n_vars": n_vars, "hue": hue}
    ax = grid.axes[0][0]
    return grid.figure, ax, info


def histogram_2d(
    x,
    y,
    bins=30,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(8, 6),
    cmap="viridis",
    colorbar=True,
    mode="hist2d",
    gridsize=30,
    vmin=None,
    vmax=None,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Create a 2D histogram or hexbin density plot.

    Parameters
    ----------
    x, y : array-like
        Input data arrays.
    bins : int, default 30
        Number of bins (used when mode='hist2d').
    mode : {'hist2d', 'hexbin'}, default 'hist2d'
        Plot mode.
    gridsize : int, default 30
        Hexbin grid size (used when mode='hexbin').

    Returns
    -------
    fig, ax, info : tuple
        info keys: 'n_points', 'mode'.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fig, ax = create_figure(figsize=figsize)
    if mode == "hexbin":
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if colorbar:
            plt.colorbar(hb, ax=ax, label="count")
    else:
        h, xedges, yedges, img = ax.hist2d(
            x, y, bins=bins, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs
        )
        if colorbar:
            plt.colorbar(img, ax=ax, label="count")
    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)
    info: Dict[str, Any] = {"n_points": len(x), "mode": mode}
    return fig, ax, info


def interactive_3d_scatter(
    x,
    y,
    z,
    color=None,
    size=None,
    labels=None,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title=None,
    colorscale="Viridis",
    opacity=0.8,
    marker_size=5,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """Create an interactive 3D scatter plot using Plotly.

    Parameters
    ----------
    x, y, z : array-like
        Point coordinates.
    color : array-like, optional
        Values mapped to marker color.
    size : array-like, optional
        Values mapped to marker size.
    labels : array-like, optional
        Hover text labels.
    colorscale : str, default 'Viridis'
        Plotly colorscale name.
    opacity : float, default 0.8
        Marker opacity.
    marker_size : int, default 5
        Default marker size (overridden by *size* if given).

    Returns
    -------
    fig, info : tuple
        info keys: 'n_points'.

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive_3d_scatter. Install it with: pip install plotly"
        )
    x_arr, y_arr, z_arr = np.asarray(x), np.asarray(y), np.asarray(z)

    marker_kw: Dict[str, Any] = {"size": marker_size, "opacity": opacity}
    if color is not None:
        c_arr = np.asarray(color)
        marker_kw["color"] = c_arr
        marker_kw["colorscale"] = colorscale
        marker_kw["showscale"] = True
    if size is not None:
        marker_kw["size"] = np.asarray(size)

    scatter3d = go.Scatter3d(
        x=x_arr,
        y=y_arr,
        z=z_arr,
        mode="markers",
        marker=marker_kw,
        text=labels,
        **kwargs,
    )
    fig = go.Figure(data=[scatter3d])
    fig.update_layout(
        title=title or "",
        scene=dict(
            xaxis_title=xlabel or "X",
            yaxis_title=ylabel or "Y",
            zaxis_title=zlabel or "Z",
        ),
    )
    info: Dict[str, Any] = {"n_points": len(x_arr)}
    return fig, info


def scatter_with_regression(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    color: str = "steelblue",
    line_color: str = "tomato",
    alpha: float = 0.7,
    show_ci: bool = True,
    ci_alpha: float = 0.15,
    show_equation: bool = True,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Scatter plot with a linear regression line and optional confidence band.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    xlabel, ylabel, title : str, optional
    figsize : tuple, default (8, 6)
    color : str, default 'steelblue'
        Scatter point color.
    line_color : str, default 'tomato'
        Regression line color.
    alpha : float, default 0.7
        Scatter point opacity.
    show_ci : bool, default True
        Shade a 95% confidence band around the regression line.
    ci_alpha : float, default 0.15
        Opacity of the confidence band.
    show_equation : bool, default True
        Annotate the plot with the regression equation and R².
    kwargs
        Extra keyword arguments forwarded to ``ax.scatter()``.

    Returns
    -------
    fig : Figure
    ax : Axes
    info : dict
        ``slope``, ``intercept``, ``r_squared``, ``p_value``, ``stderr``
    """
    from scipy import stats as _stats

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    result = _stats.linregress(x_arr, y_arr)
    slope, intercept, r_value, p_value, stderr = (
        result.slope,
        result.intercept,
        result.rvalue,
        result.pvalue,
        result.stderr,
    )

    fig, ax = create_figure(figsize=figsize)
    ax.scatter(x_arr, y_arr, color=color, alpha=alpha, zorder=3, **kwargs)

    x_fit = np.linspace(x_arr.min(), x_arr.max(), 300)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color=line_color, linewidth=1.8, label="Regression line")

    if show_ci:
        n = len(x_arr)
        t_crit = _stats.t.ppf(0.975, df=n - 2)
        se_fit = stderr * np.sqrt(
            1.0 / n + (x_fit - x_arr.mean()) ** 2 / np.sum((x_arr - x_arr.mean()) ** 2)
        )
        ax.fill_between(
            x_fit,
            y_fit - t_crit * se_fit,
            y_fit + t_crit * se_fit,
            color=line_color,
            alpha=ci_alpha,
            label="95% CI",
        )

    if show_equation:
        eq = (
            f"y = {slope:.4g}x {'+' if intercept >= 0 else '−'} {abs(intercept):.4g}\n"
            f"R² = {r_value**2:.4f}  p = {p_value:.3g}"
        )
        ax.annotate(
            eq,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, legend=show_ci)

    info: Dict[str, Any] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "stderr": float(stderr),
    }
    return fig, ax, info


def residual_plot(
    x: np.ndarray,
    y_actual: np.ndarray,
    y_fitted: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "Residual",
    title: Optional[str] = "Residual Plot",
    figsize: Tuple[float, float] = (8, 5),
    color_pos: str = "steelblue",
    color_neg: str = "tomato",
    alpha: float = 0.75,
    show_zero_line: bool = True,
    vs_fitted: bool = False,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Plot residuals from a curve fit.

    Parameters
    ----------
    x : array-like
        Original x-axis values.
    y_actual : array-like
        Observed y values.
    y_fitted : array-like
        Model-predicted y values (from any fit).
    xlabel : str, optional
        X-axis label; defaults to 'x' or 'Fitted value' when ``vs_fitted=True``.
    ylabel : str, default 'Residual'
    title : str, default 'Residual Plot'
    figsize : tuple, default (8, 5)
    color_pos : str, default 'steelblue'
        Color for positive residuals.
    color_neg : str, default 'tomato'
        Color for negative residuals.
    alpha : float, default 0.75
    show_zero_line : bool, default True
        Draw a horizontal reference line at y = 0.
    vs_fitted : bool, default False
        If True, plot residuals vs. fitted values instead of vs. x.
    kwargs
        Extra keyword arguments forwarded to ``ax.scatter()``.

    Returns
    -------
    fig : Figure
    ax : Axes
    info : dict
        ``n_points``, ``rmse``, ``max_residual``, ``mean_residual``
    """
    x_arr = np.asarray(x, dtype=float)
    ya = np.asarray(y_actual, dtype=float)
    yf = np.asarray(y_fitted, dtype=float)
    residuals = ya - yf

    x_plot = yf if vs_fitted else x_arr
    default_xlabel = "Fitted value" if vs_fitted else (xlabel or "x")

    colors = np.where(residuals >= 0, color_pos, color_neg)

    fig, ax = create_figure(figsize=figsize)
    ax.scatter(x_plot, residuals, c=colors, alpha=alpha, zorder=3, **kwargs)

    if show_zero_line:
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", zorder=2)

    configure_axes(ax, xlabel=default_xlabel, ylabel=ylabel, title=title)

    rmse = float(np.sqrt(np.mean(residuals**2)))
    info: Dict[str, Any] = {
        "n_points": len(residuals),
        "rmse": rmse,
        "max_residual": float(np.max(np.abs(residuals))),
        "mean_residual": float(np.mean(residuals)),
    }
    return fig, ax, info


def interactive_ternary(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    color: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    a_label: str = "A",
    b_label: str = "B",
    c_label: str = "C",
    title: Optional[str] = None,
    colorscale: str = "Viridis",
    marker_size: int = 8,
    opacity: float = 0.85,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """Create an interactive ternary scatter plot using Plotly.

    Each point represents a 3-component composition (a + b + c should ≈ 1 or
    a constant sum). Useful for phase diagrams, solvent composition studies,
    and materials science.

    Parameters
    ----------
    a, b, c : array-like
        Component fractions for each axis.
    color : array-like, optional
        Numeric values mapped to a colorscale.
    labels : list of str, optional
        Hover text labels for each point.
    a_label, b_label, c_label : str
        Axis labels for the three components.
    title : str, optional
    colorscale : str, default 'Viridis'
    marker_size : int, default 8
    opacity : float, default 0.85
    kwargs
        Extra keyword arguments forwarded to ``go.Scatterternary``.

    Returns
    -------
    fig : go.Figure
    info : dict
        ``n_points``

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive_ternary. Install it with: pip install plotly"
        )
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    c_arr = np.asarray(c, dtype=float)

    marker_kw: Dict[str, Any] = {"size": marker_size, "opacity": opacity}
    if color is not None:
        marker_kw["color"] = np.asarray(color, dtype=float)
        marker_kw["colorscale"] = colorscale
        marker_kw["showscale"] = True

    trace = go.Scatterternary(
        a=a_arr,
        b=b_arr,
        c=c_arr,
        mode="markers",
        marker=marker_kw,
        text=labels,
        hovertemplate=(
            f"{a_label}: %{{a:.3f}}<br>"
            f"{b_label}: %{{b:.3f}}<br>"
            f"{c_label}: %{{c:.3f}}<extra></extra>"
        ),
        **kwargs,
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=title or "",
        ternary={
            "aaxis": {"title": a_label},
            "baxis": {"title": b_label},
            "caxis": {"title": c_label},
        },
    )
    return fig, {"n_points": len(a_arr)}


def broken_axis_plot(
    x: np.ndarray,
    y: np.ndarray,
    breaks: List[Tuple[float, float]],
    axis: str = "x",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 5),
    color: str = "steelblue",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    d: float = 0.012,
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Line plot with one or more broken axis segments.

    Creates adjacent subplots for each unbroken segment and hides the inner
    spines to give the visual appearance of a continuous broken axis.

    Parameters
    ----------
    x, y : array-like
        Data arrays. x must be sorted ascending.
    breaks : list of (lo, hi) tuples
        Each tuple defines a region to *exclude* from the plot.
        E.g. ``breaks=[(800, 1200)]`` skips x values between 800 and 1200.
    axis : {'x', 'y'}, default 'x'
        Which axis to break.
    xlabel, ylabel, title : str, optional
    figsize : tuple, default (9, 5)
    color : str, default 'steelblue'
    linewidth : float, default 1.5
    alpha : float, default 1.0
    d : float, default 0.012
        Size of the diagonal break markers (in axes-fraction units).
    kwargs
        Extra keyword arguments forwarded to ``ax.plot()``.

    Returns
    -------
    fig : Figure
    ax : Axes
        The first (leftmost / bottom) axes object.
    info : dict
        ``n_segments`` — number of unbroken segments shown.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Build segment limits by excluding break regions
    all_breaks = sorted(breaks, key=lambda b: b[0])

    limits = []
    prev = x_arr.min()
    for lo, hi in all_breaks:
        limits.append((prev, lo))
        prev = hi
    limits.append((prev, x_arr.max()))
    # Remove empty segments
    limits = [(a, b) for a, b in limits if b > a]
    n_seg = len(limits)

    if axis == "x":
        widths = [b - a for a, b in limits]
        total = sum(widths)
        width_ratios = [w / total for w in widths]
        fig, axes = plt.subplots(
            1,
            n_seg,
            sharey=True,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.08},
        )
        if n_seg == 1:
            axes = [axes]
        for i, (ax_i, (lo, hi)) in enumerate(zip(axes, limits)):
            mask = (x_arr >= lo) & (x_arr <= hi)
            ax_i.plot(
                x_arr[mask], y_arr[mask], color=color, linewidth=linewidth, alpha=alpha, **kwargs
            )
            ax_i.set_xlim(lo, hi)
            # Hide inner spines
            if i > 0:
                ax_i.spines["left"].set_visible(False)
                ax_i.tick_params(left=False)
            if i < n_seg - 1:
                ax_i.spines["right"].set_visible(False)
            # Draw diagonal break markers
            kw_mark = dict(transform=ax_i.transAxes, color="k", clip_on=False, linewidth=1.2)
            if i > 0:
                ax_i.plot((-d, +d), (-d, +d), **kw_mark)
                ax_i.plot((-d, +d), (1 - d, 1 + d), **kw_mark)
            if i < n_seg - 1:
                ax_i.plot((1 - d, 1 + d), (-d, +d), **kw_mark)
                ax_i.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw_mark)
        # Shared labels
        if xlabel:
            fig.text(0.5, 0.01, xlabel, ha="center")
        if ylabel:
            axes[0].set_ylabel(ylabel)
        if title:
            fig.suptitle(title)
        primary_ax = axes[0]

    else:  # axis == "y"
        heights = [b - a for a, b in limits]
        total = sum(heights)
        height_ratios = [h / total for h in reversed(heights)]
        fig, axes = plt.subplots(
            n_seg,
            1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        )
        if n_seg == 1:
            axes = [axes]
        axes_ordered = list(reversed(axes))  # bottom to top
        for i, (ax_i, (lo, hi)) in enumerate(zip(axes_ordered, limits)):
            mask = (y_arr >= lo) & (y_arr <= hi)
            ax_i.plot(
                x_arr[mask], y_arr[mask], color=color, linewidth=linewidth, alpha=alpha, **kwargs
            )
            ax_i.set_ylim(lo, hi)
            if i > 0:
                ax_i.spines["bottom"].set_visible(False)
                ax_i.tick_params(bottom=False)
            if i < n_seg - 1:
                ax_i.spines["top"].set_visible(False)
            kw_mark = dict(transform=ax_i.transAxes, color="k", clip_on=False, linewidth=1.2)
            if i > 0:
                ax_i.plot((-d, +d), (-d, +d), **kw_mark)
                ax_i.plot((1 - d, 1 + d), (-d, +d), **kw_mark)
            if i < n_seg - 1:
                ax_i.plot((-d, +d), (1 - d, 1 + d), **kw_mark)
                ax_i.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw_mark)
        if ylabel:
            fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical")
        if xlabel:
            axes_ordered[0].set_xlabel(xlabel)
        if title:
            fig.suptitle(title)
        primary_ax = axes_ordered[0]

    info: Dict[str, Any] = {"n_segments": n_seg}
    return fig, primary_ax, info


def inset_plot(
    x,
    y,
    x_inset,
    y_inset,
    inset_bounds=None,
    indicate_region=None,
    title=None,
    xlabel=None,
    ylabel=None,
    inset_xlabel=None,
    inset_ylabel=None,
    color="steelblue",
    inset_color=None,
    linewidth=1.5,
    figsize=(8, 6),
    **kwargs,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Plot data with an inset subplot showing a zoomed or alternate view.

    Parameters
    ----------
    x : array-like
        X data for main plot.
    y : array-like
        Y data for main plot.
    x_inset : array-like
        X data for inset (can be same as x for zoom, or different data).
    y_inset : array-like
        Y data for inset.
    inset_bounds : list of float, optional
        [x0, y0, width, height] in axes fraction coordinates (0–1).
        Defaults to [0.55, 0.55, 0.4, 0.35].
    indicate_region : tuple of float, optional
        (x_start, x_end) to draw a grey shaded region on the main plot
        matching the inset x range.
    title : str, optional
    xlabel : str, optional
    ylabel : str, optional
    inset_xlabel : str, optional
        Axis label for the inset X axis.
    inset_ylabel : str, optional
        Axis label for the inset Y axis.
    color : str, optional
        Color for main plot line.  Default ``'steelblue'``.
    inset_color : str, optional
        Color for inset line.  Defaults to *color*.
    linewidth : float, optional
        Line width for both plots.  Default 1.5.
    figsize : tuple, default (8, 6)
        Figure size in inches.
    kwargs : dict
        Extra keyword arguments forwarded to ``ax.plot()`` for the main line.

    Returns
    -------
    fig : Figure
    ax : Axes
        Main axes.
    info : dict
        ``n_points``, ``n_inset_points``, ``inset_bounds``.

    Examples
    --------
    >>> x = np.linspace(0, 10, 200)
    >>> y = np.sin(x)
    >>> fig, ax, info = inset_plot(x, y, x[40:80], y[40:80],
    ...                            indicate_region=(2.0, 4.0))
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_inset = np.asarray(x_inset)
    y_inset = np.asarray(y_inset)

    if inset_bounds is None:
        inset_bounds = [0.55, 0.55, 0.4, 0.35]
    if inset_color is None:
        inset_color = color

    fig, ax = create_figure(figsize=figsize)
    ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Shade indicated region on main plot
    if indicate_region is not None:
        ax.axvspan(indicate_region[0], indicate_region[1], alpha=0.15, color="grey")

    # Create inset axes
    ax_inset = ax.inset_axes(inset_bounds)
    ax_inset.plot(x_inset, y_inset, color=inset_color, linewidth=linewidth)
    if inset_xlabel:
        ax_inset.set_xlabel(inset_xlabel, fontsize=8)
    if inset_ylabel:
        ax_inset.set_ylabel(inset_ylabel, fontsize=8)
    ax_inset.tick_params(labelsize=7)

    plt.tight_layout()

    info: Dict[str, Any] = {
        "n_points": len(x),
        "n_inset_points": len(x_inset),
        "inset_bounds": inset_bounds,
    }
    return fig, ax, info


# ============================================================================
# Styling Utilities
# ============================================================================


def set_style(style: str = "default") -> None:
    """Set matplotlib style.

    Parameters
    ----------
    style : str, default 'default'
        Style name. Options: 'default', 'seaborn', 'ggplot', 'bmh', 'dark_background'

    Examples
    --------
    >>> set_style('seaborn')
    >>> fig, ax = line_plot(x, y)
    """
    if style == "default":
        plt.style.use("default")
    elif style in plt.style.available:
        plt.style.use(style)
    else:
        available = ", ".join(plt.style.available[:10])
        raise ValueError(f"Unknown style: {style}. Available styles include: {available}, ...")


def get_color_palette(name: str = "tab10", n_colors: int = 10) -> List[str]:
    """Get a color palette.

    Parameters
    ----------
    name : str, default 'tab10'
        Palette name. Options: 'tab10', 'tab20', 'Set1', 'Set2', 'Set3', etc.
    n_colors : int, default 10
        Number of colors to return

    Returns
    -------
    colors : list of str
        List of color hex codes

    Examples
    --------
    >>> colors = get_color_palette('Set1', n_colors=5)
    >>> fig, ax = line_plot(x, [y1, y2], colors=colors[:2])
    """
    if name == "colorblind":
        palette = COLORBLIND_PALETTE[:n_colors]
        return [str(c) for c in palette]

    cmap = plt.get_cmap(name)

    if hasattr(cmap, "colors"):
        # For qualitative colormaps
        colors = [mcolors.rgb2hex(c) for c in cmap.colors[:n_colors]]
    else:
        # For continuous colormaps
        colors = [mcolors.rgb2hex(cmap(i / n_colors)) for i in range(n_colors)]

    return colors


def apply_publication_style(
    fig: Figure,
    ax: Union[Axes, List[Axes]],
    fontsize: int = 12,
    labelsize: int = 10,
    linewidth: float = 1.5,
) -> None:
    """Apply publication-quality styling to figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax : Axes or list of Axes
        Axes to style
    fontsize : int, default 12
        Font size for labels
    labelsize : int, default 10
        Font size for tick labels
    linewidth : float, default 1.5
        Line width for plot elements

    Examples
    --------
    >>> fig, ax = line_plot(x, y)
    >>> apply_publication_style(fig, ax)
    """
    # Handle single or multiple axes
    if not isinstance(ax, list):
        ax = [ax]

    for a in ax:
        # Set font sizes
        a.title.set_fontsize(fontsize + 2)
        a.xaxis.label.set_fontsize(fontsize)
        a.yaxis.label.set_fontsize(fontsize)
        a.tick_params(labelsize=labelsize)

        # Set line widths
        for line in a.lines:
            line.set_linewidth(linewidth)

        # Spine styling
        for spine in a.spines.values():
            spine.set_linewidth(1.0)

    # Tight layout
    fig.tight_layout()


# ============================================================================
# Seaborn Statistical Plots
# ============================================================================


def distribution_plot(
    data: Union[np.ndarray, List], kind: str = "hist", kde: bool = True, **kwargs
) -> Tuple[Figure, Axes]:
    """Create distribution plots using Seaborn.

    Parameters
    ----------
    data : array-like
        Data to plot
    kind : str, default 'hist'
        Plot type: 'hist', 'kde', 'ecdf'
    kde : bool, default True
        Whether to overlay kernel density estimate
    kwargs : dict
        Additional arguments passed to seaborn plotting function

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    ax : Axes
        Matplotlib axes object

    Raises
    ------
    ImportError
        If seaborn is not installed

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> fig, ax = distribution_plot(data, kind='hist', kde=True)
    """
    if not HAS_SEABORN:
        raise ImportError(
            "Seaborn is required for distribution_plot. Install it with: pip install seaborn"
        )

    figsize = kwargs.pop("figsize", (8, 6))
    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    fig, ax = create_figure(figsize=figsize)

    if kind == "hist":
        sns.histplot(data=data, kde=kde, ax=ax, **kwargs)
    elif kind == "kde":
        sns.kdeplot(data=data, ax=ax, **kwargs)
    elif kind == "ecdf":
        sns.ecdfplot(data=data, ax=ax, **kwargs)
    else:
        raise ValueError(f"Unknown kind: {kind}. Supported: 'hist', 'kde', 'ecdf'")

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    return fig, ax


def box_plot(
    data: Union[np.ndarray, "pd.DataFrame"],
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    kind: str = "box",
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create box plots or violin plots using Seaborn.

    Parameters
    ----------
    data : array-like or DataFrame
        Data to plot
    x : str, optional
        Column name for x-axis (for DataFrame input)
    y : str, optional
        Column name for y-axis (for DataFrame input)
    hue : str, optional
        Column name for color grouping
    kind : str, default 'box'
        Plot type: 'box', 'violin', 'boxen'
    kwargs : dict
        Additional arguments passed to seaborn function

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    ax : Axes
        Matplotlib axes object

    Raises
    ------
    ImportError
        If seaborn is not installed

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'group': ['A']*50 + ['B']*50,
    ...                    'value': np.random.randn(100)})
    >>> fig, ax = box_plot(df, x='group', y='value')
    """
    if not HAS_SEABORN:
        raise ImportError("Seaborn is required for box_plot. Install it with: pip install seaborn")

    figsize = kwargs.pop("figsize", (8, 6))
    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    fig, ax = create_figure(figsize=figsize)

    if kind == "box":
        sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    elif kind == "violin":
        sns.violinplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    elif kind == "boxen":
        sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    else:
        raise ValueError(f"Unknown kind: {kind}. Supported: 'box', 'violin', 'boxen'")

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    return fig, ax


def regression_plot(
    x: Union[np.ndarray, str],
    y: Union[np.ndarray, str],
    data: Optional["pd.DataFrame"] = None,
    order: int = 1,
    ci: int = 95,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create regression plot with confidence interval.

    Parameters
    ----------
    x : array-like or str
        X data or column name
    y : array-like or str
        Y data or column name
    data : DataFrame, optional
        DataFrame containing x and y columns
    order : int, default 1
        Polynomial order (1 = linear, 2 = quadratic, etc.)
    ci : int, default 95
        Confidence interval percentage
    kwargs : dict
        Additional arguments passed to seaborn.regplot

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    ax : Axes
        Matplotlib axes object

    Raises
    ------
    ImportError
        If seaborn is not installed

    Examples
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2*x + 1 + np.random.randn(50)
    >>> fig, ax = regression_plot(x, y, order=1)
    """
    if not HAS_SEABORN:
        raise ImportError(
            "Seaborn is required for regression_plot. Install it with: pip install seaborn"
        )

    figsize = kwargs.pop("figsize", (8, 6))
    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    fig, ax = create_figure(figsize=figsize)

    sns.regplot(x=x, y=y, data=data, order=order, ci=ci, ax=ax, **kwargs)

    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    return fig, ax


# ============================================================================
# Plotly Interactive Plots
# ============================================================================


def interactive_histogram(
    data: np.ndarray,
    bins: int = 30,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> "go.Figure":
    """Create interactive histogram using Plotly.

    Parameters
    ----------
    data : array-like
        Data to plot
    bins : int, default 30
        Number of bins
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    kwargs : dict
        Additional arguments passed to go.Histogram

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object (use fig.show() to display)

    Raises
    ------
    ImportError
        If plotly is not installed

    Examples
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> fig = interactive_histogram(data, bins=50, title='Distribution')
    >>> fig.write_html('histogram.html')
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for interactive_histogram. Install it with: pip install plotly"
        )

    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=bins, **kwargs)])

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or "Value",
        yaxis_title=ylabel or "Count",
        hovermode="x",
    )

    return fig


def interactive_scatter(
    x: np.ndarray,
    y: np.ndarray,
    color: Optional[np.ndarray] = None,
    size: Optional[np.ndarray] = None,
    hover_data: Optional[Dict] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> "go.Figure":
    """Create interactive scatter plot using Plotly.

    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    color : array-like, optional
        Values for color mapping
    size : array-like, optional
        Values for marker size
    hover_data : dict, optional
        Additional data to show on hover
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    kwargs : dict
        Additional arguments passed to go.Scatter

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object

    Raises
    ------
    ImportError
        If plotly is not installed

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> fig = interactive_scatter(x, y, title='Interactive Scatter')
    >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for interactive_scatter. Install it with: pip install plotly"
        )

    scatter_kwargs = {"x": x, "y": y, "mode": "markers", **kwargs}

    if color is not None:
        scatter_kwargs["marker"] = {
            "color": color,
            "colorscale": "Viridis",
            "showscale": True,
        }

    if size is not None:
        if "marker" not in scatter_kwargs:
            scatter_kwargs["marker"] = {}
        scatter_kwargs["marker"]["size"] = size

    if hover_data is not None:
        scatter_kwargs["customdata"] = hover_data

    fig = go.Figure(data=[go.Scatter(**scatter_kwargs)])

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or "X",
        yaxis_title=ylabel or "Y",
        hovermode="closest",
    )

    return fig


def interactive_line(
    x: Union[np.ndarray, List[np.ndarray]],
    y: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> "go.Figure":
    """Create interactive line plot using Plotly.

    Parameters
    ----------
    x : array-like or list of arrays
        X coordinates (single or multiple series)
    y : array-like or list of arrays
        Y coordinates (single or multiple series)
    labels : list of str, optional
        Line labels for legend
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    kwargs : dict
        Additional arguments passed to go.Scatter

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object

    Raises
    ------
    ImportError
        If plotly is not installed

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y1 = np.sin(x)
    >>> y2 = np.cos(x)
    >>> fig = interactive_line(x, [y1, y2], labels=['sin', 'cos'])
    >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for interactive_line. Install it with: pip install plotly"
        )

    fig = go.Figure()

    # Handle single or multiple lines
    if not isinstance(y, list):
        y = [y]
        x_data = [x]
    else:
        x_data = [x] * len(y) if not isinstance(x, list) else x  # type: ignore[assignment]

    if labels is None:
        labels = [f"Line {i + 1}" for i in range(len(y))]

    for i, (x_i, y_i) in enumerate(zip(x_data, y)):
        fig.add_trace(go.Scatter(x=x_i, y=y_i, mode="lines", name=labels[i], **kwargs))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or "X",
        yaxis_title=ylabel or "Y",
        hovermode="x unified",
        height=450,
        autosize=True,
    )

    return fig


def interactive_heatmap(
    data: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    colorscale: str = "Viridis",
    **kwargs,
) -> "go.Figure":
    """Create interactive heatmap using Plotly.

    Parameters
    ----------
    data : 2D array
        Data to visualize as heatmap
    x_labels : list of str, optional
        Labels for x-axis
    y_labels : list of str, optional
        Labels for y-axis
    title : str, optional
        Plot title
    colorscale : str, default 'Viridis'
        Colorscale name
    kwargs : dict
        Additional arguments passed to go.Heatmap

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object

    Raises
    ------
    ImportError
        If plotly is not installed

    Examples
    --------
    >>> data = np.random.rand(10, 10)
    >>> fig = interactive_heatmap(data, title='Interactive Heatmap')
    >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for interactive_heatmap. Install it with: pip install plotly"
        )

    fig = go.Figure(
        data=go.Heatmap(z=data, x=x_labels, y=y_labels, colorscale=colorscale, **kwargs)
    )

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")

    return fig


def interactive_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: Optional[str] = None,
    colorscale: str = "Viridis",
    **kwargs,
) -> "go.Figure":
    """Create interactive 3D surface plot using Plotly.

    Parameters
    ----------
    x : 2D array
        X coordinates (meshgrid format)
    y : 2D array
        Y coordinates (meshgrid format)
    z : 2D array
        Z values
    title : str, optional
        Plot title
    colorscale : str, default 'Viridis'
        Colorscale name
    kwargs : dict
        Additional arguments passed to go.Surface

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object

    Raises
    ------
    ImportError
        If plotly is not installed

    Examples
    --------
    >>> x = np.linspace(-5, 5, 50)
    >>> y = np.linspace(-5, 5, 50)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = np.sin(np.sqrt(X**2 + Y**2))
    >>> fig = interactive_3d_surface(X, Y, Z, title='3D Surface')
    >>> fig.show()
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for interactive_3d_surface. Install it with: pip install plotly"
        )

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale=colorscale, **kwargs)])

    fig.update_layout(title=title, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    return fig


# ============================================================================
# Export Utilities
# ============================================================================


def export_interactive(
    fig: "go.Figure",
    filepath: Union[str, Path],
    include_plotlyjs: str = "cdn",
    **kwargs,
) -> None:
    """Export Plotly figure to HTML file.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to export
    filepath : str or Path
        Output file path (.html)
    include_plotlyjs : str, default 'cdn'
        How to include plotly.js: 'cdn', 'directory', or True (embed)
    kwargs : dict
        Additional arguments passed to fig.write_html()

    Examples
    --------
    >>> fig = interactive_scatter(x, y)
    >>> export_interactive(fig, 'scatter.html')
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for export_interactive. Install it with: pip install plotly"
        )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(filepath), include_plotlyjs=include_plotlyjs, **kwargs)
