"""Quick Plot Page.

Select a loaded dataset, choose a plot type, configure parameters,
and generate a publication-ready or interactive visualization.
All 13 plot types from plottle.plotting are accessible here.
"""

from __future__ import annotations

import traceback
from pathlib import Path
import sys

import streamlit as st
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plottle.plotting import (
    histogram,
    line_plot,
    scatter_plot,
    bar_chart,
    heatmap,
    contour_plot,
    waterfall_plot,
    dual_axis_plot,
    broken_axis_plot,
    distribution_plot,
    box_plot,
    regression_plot,
    interactive_histogram,
    interactive_scatter,
    interactive_line,
    interactive_heatmap,
    interactive_3d_surface,
    z_colored_scatter,
    bubble_chart,
    polar_plot,
    histogram_2d,
    pair_plot,
    interactive_3d_scatter,
    scatter_with_regression,
    residual_plot,
    interactive_ternary,
    inset_plot,
)
from plottle.utils import (
    initialize_session_state,
    get_dataset,
    add_plot_to_history,
    get_session_summary,
    get_defaults,
    list_presets,
    save_preset,
    load_preset,
    delete_preset,
)
import matplotlib as mpl

from plottle.utils.plot_config import (
    PLOT_TYPES,
    COLOR_PALETTES,
    get_plot_config_widgets,
    get_plot_kwargs,
    selected_points,
    format_marker_label,
)

initialize_session_state()

# Map config.json default keys → shared widget keys.
# config.json uses plain names (e.g. "fontfamily"); widgets now use
# "_shared_*" keys so they persist across plot-type switches.
_DEFAULTS_KEY_MAP = {
    "fontfamily": "_shared_fontfamily",
    "fontcolor": "_shared_fontcolor",
    "linewidth": "_shared_lw",
    "grid": "_shared_grid",
    "grid_linestyle": "_shared_grid_ls",
    "grid_which": "_shared_grid_which",
    "legend_frameon": "_shared_leg_frame",
    "legend_framealpha": "_shared_leg_bg",
    "legend_position": "_shared_leg_pos",
    "color_palette": "_shared_palette",
    "fontsize": "_shared_ply_fontsize",
    # Per-element sizes: saved with new keys directly if user updated them
    "fontsize_label": "_shared_fs_label",
    "fontsize_tick": "_shared_fs_tick",
    "fontsize_title": "_shared_fs_title",
    "fontsize_legend": "_shared_fs_legend",
}

# Seed session state with persisted user defaults (first run only).
if "user_defaults_loaded" not in st.session_state:
    _ud = get_defaults()
    for _k, _v in _ud.items():
        _mapped = _DEFAULTS_KEY_MAP.get(_k, _k)
        if _mapped not in st.session_state:
            st.session_state[_mapped] = _v
    st.session_state["user_defaults_loaded"] = True

# Persistent plot storage for click-to-annotate.
if "qp_fig_dict" not in st.session_state:
    st.session_state.qp_fig_dict = None  # Plotly figure as dict
if "qp_mpl_fig" not in st.session_state:
    st.session_state.qp_mpl_fig = None  # Matplotlib figure object
if "qp_is_plotly" not in st.session_state:
    st.session_state.qp_is_plotly = False
if "qp_config" not in st.session_state:
    st.session_state.qp_config = {}
if "qp_annotations" not in st.session_state:
    st.session_state.qp_annotations = []  # [{x, y, label}]
if "qp_ann_overlays" not in st.session_state:
    st.session_state.qp_ann_overlays = []  # M16 annotation overlays
if "qp_current_plot_type" not in st.session_state:
    st.session_state.qp_current_plot_type = ""  # M16: used by _regen_with_overlays
if "qp_show_plotly_version" not in st.session_state:
    st.session_state.qp_show_plotly_version = False  # zoom/pan Plotly conversion
if "qp_mpl_png" not in st.session_state:
    st.session_state.qp_mpl_png = None  # PNG bytes — avoids stale MediaFileStorage hashes
if "qp_zoom_xlim" not in st.session_state:
    st.session_state.qp_zoom_xlim = None  # (xmin, xmax) or None
if "qp_zoom_ylim" not in st.session_state:
    st.session_state.qp_zoom_ylim = None  # (ymin, ymax) or None
if "qp_last_smart_default_for" not in st.session_state:
    st.session_state.qp_last_smart_default_for = None  # dataset name last auto-defaulted


def _suggest_plot_type(data) -> tuple:
    """Return (category, plot_label) suited to the data's shape and type."""
    if isinstance(data, np.ndarray):
        return ("Matplotlib", "Heatmap") if data.ndim >= 2 else ("Matplotlib", "Histogram")
    if isinstance(data, pd.DataFrame):
        n_numeric = len(data.select_dtypes(include=[np.number]).columns)
        n_rows = len(data)
        if n_numeric >= 2 and n_rows > 100:
            return "Matplotlib", "Line Plot"
        if n_numeric >= 2:
            return "Matplotlib", "Scatter Plot"
    return "Matplotlib", "Histogram"


# ── Style post-processing ────────────────────────────────────────────────────

_MPL_LS_TO_PLOTLY = {
    "--": "dash",
    "-": "solid",
    ":": "dot",
    "-.": "dashdot",
}


def _set_mpl_color_cycle(config: dict) -> None:
    """Apply the selected color palette to the matplotlib color cycle."""
    palette_name = config.get("color_palette", "Default")
    if palette_name == "Custom":
        colors = config.get("_custom_colors") or COLOR_PALETTES["Default"]
    else:
        colors = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Default"])
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)


def _apply_plotly_palette(fig, config: dict) -> None:
    """Re-color Plotly line/scatter traces using the selected palette."""
    palette_name = config.get("color_palette", "Default")
    if palette_name == "Custom":
        colors = config.get("_custom_colors") or COLOR_PALETTES["Default"]
    else:
        colors = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Default"])
    for i, trace in enumerate(fig.data):
        c = colors[i % len(colors)]
        try:
            trace.update(line=dict(color=c))
        except Exception:
            pass


def _apply_transform(arr: np.ndarray, transform: str, scale_value: float) -> np.ndarray:
    """Apply a named data transform to *arr*."""
    if transform == "normalize_max":
        mx = np.max(np.abs(arr))
        return arr / mx if mx != 0 else arr
    if transform == "normalize_01":
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx != mn else arr
    if transform == "scale_by":
        return arr * scale_value
    return arr


def _apply_y_transform(y: np.ndarray, config: dict) -> np.ndarray:
    return _apply_transform(
        y, config.get("y_transform", "none"), config.get("y_transform_value", 1.0)
    )


def _apply_x_transform(x: np.ndarray, config: dict) -> np.ndarray:
    return _apply_transform(
        x, config.get("x_transform", "none"), config.get("x_transform_value", 1.0)
    )


def _apply_z_transform(z: np.ndarray, config: dict) -> np.ndarray:
    return _apply_transform(
        z, config.get("z_transform", "none"), config.get("z_transform_value", 1.0)
    )


def _apply_mpl_style(fig, config: dict) -> None:
    """Apply GUI appearance settings to a Matplotlib figure."""
    import matplotlib.ticker as ticker

    if not fig.axes:
        return
    ax = fig.axes[0]

    # Axis limits
    xmin, xmax = config.get("xlim_min"), config.get("xlim_max")
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    ymin, ymax = config.get("ylim_min"), config.get("ylim_max")
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    # Axis scale
    if config.get("x_scale") == "log":
        ax.set_xscale("log")
    if config.get("y_scale") == "log":
        ax.set_yscale("log")

    # Axis notation formatters
    x_notation = config.get("x_notation", "default")
    if x_notation == "scientific":
        ax.xaxis.set_major_formatter(ticker.ScientificFormatter())
    elif x_notation == "engineering":
        ax.xaxis.set_major_formatter(ticker.EngFormatter())

    y_notation = config.get("y_notation", "default")
    if y_notation == "scientific":
        ax.yaxis.set_major_formatter(ticker.ScientificFormatter())
    elif y_notation == "engineering":
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

    # Colorbar (Z) notation — applied to any axes that is not the main axes
    z_notation = config.get("z_notation", "default")
    if z_notation != "default":
        _z_fmt = (
            ticker.ScientificFormatter() if z_notation == "scientific" else ticker.EngFormatter()
        )
        for _cax in fig.axes:
            if _cax is not ax:
                _cax.yaxis.set_major_formatter(_z_fmt)
                break

    # Grid
    show_grid = config.get("grid", True)
    if show_grid:
        grid_ls = config.get("grid_linestyle", "--")
        grid_which = config.get("grid_which", "major")
        ax.grid(True, linestyle=grid_ls, which=grid_which, alpha=0.45)
    else:
        ax.grid(False)

    # Font — per-element sizes with fallback to legacy "fontsize" key
    fontfamily = config.get("fontfamily", "sans-serif")
    fontcolor = config.get("fontcolor", "#262730")
    _fs = config.get("fontsize", 11)  # legacy single-size fallback
    fs_label = config.get("fontsize_label", _fs)
    fs_tick = config.get("fontsize_tick", max(_fs - 1, 6))
    fs_title = config.get("fontsize_title", _fs + 2)
    fs_legend = config.get("fontsize_legend", max(_fs - 1, 6))

    lp = {"family": fontfamily, "color": fontcolor}
    if ax.get_xlabel():
        ax.xaxis.label.set(size=fs_label, **lp)
    if ax.get_ylabel():
        ax.yaxis.label.set(size=fs_label, **lp)
    if ax.get_title():
        ax.title.set(size=fs_title, **lp)
    ax.tick_params(labelsize=fs_tick, labelcolor=fontcolor)

    # Line width (post-hoc on existing line objects)
    lw = config.get("linewidth")
    if lw is not None:
        for line in ax.lines:
            line.set_linewidth(lw)

    # Legend
    leg = ax.get_legend()
    if leg:
        frameon = config.get("legend_frameon", True)
        leg.set_frame_on(frameon)
        if frameon:
            leg.get_frame().set_alpha(
                config.get("legend_framealpha", 0.8),
            )
        pos = config.get("legend_position", "best")
        if pos == "outside right":
            leg = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        else:
            leg.set_loc(pos)
        for text in leg.get_texts():
            text.set_fontsize(fs_legend)
            text.set_color(fontcolor)

    # Figure caption (below plot as figure text)
    caption = config.get("figure_caption")
    if caption:
        fig.text(
            0.5,
            -0.02,
            caption,
            ha="center",
            va="top",
            fontsize=max(fs_label - 2, 8),
            color=fontcolor,
            style="italic",
            wrap=True,
        )

    fig.tight_layout()


def _apply_plotly_style(fig, config: dict) -> None:
    """Apply GUI appearance settings to a Plotly figure."""
    layout: dict = {}

    # Font
    font_dict: dict = {}
    if config.get("fontsize"):
        font_dict["size"] = config["fontsize"]
    if config.get("fontfamily"):
        font_dict["family"] = config["fontfamily"]
    if config.get("fontcolor"):
        font_dict["color"] = config["fontcolor"]
    if font_dict:
        layout["font"] = font_dict

    # Chart height
    if config.get("plotly_height"):
        layout["height"] = config["plotly_height"]

    # Grid
    show_grid = config.get("grid", True)
    grid_ls = _MPL_LS_TO_PLOTLY.get(
        config.get("grid_linestyle", "--"),
        "dash",
    )
    axis_grid = dict(
        showgrid=show_grid,
        griddash=grid_ls if show_grid else "dash",
    )
    layout["xaxis"] = axis_grid
    layout["yaxis"] = axis_grid.copy()

    # Axis limits
    xmin, xmax = config.get("xlim_min"), config.get("xlim_max")
    if xmin is not None and xmax is not None:
        layout.setdefault("xaxis", {})["range"] = [xmin, xmax]
    ymin, ymax = config.get("ylim_min"), config.get("ylim_max")
    if ymin is not None and ymax is not None:
        layout.setdefault("yaxis", {})["range"] = [ymin, ymax]

    # Legend
    legend_upd: dict = {}
    frameon = config.get("legend_frameon")
    if frameon is not None:
        legend_upd["borderwidth"] = 1 if frameon else 0
    alpha = config.get("legend_framealpha")
    if alpha is not None:
        if alpha < 0.05:
            legend_upd["bgcolor"] = "rgba(0,0,0,0)"
        else:
            legend_upd["bgcolor"] = f"rgba(255,255,255,{alpha:.2f})"
    if legend_upd:
        layout["legend"] = legend_upd

    # Y-axis scale
    y_scale = config.get("y_scale", "linear")
    if y_scale == "log":
        layout.setdefault("yaxis", {})["type"] = "log"

    # Y-axis notation
    y_notation = config.get("y_notation", "default")
    if y_notation == "scientific":
        layout.setdefault("yaxis", {})["exponentformat"] = "e"
    elif y_notation == "engineering":
        layout.setdefault("yaxis", {})["exponentformat"] = "SI"

    if layout:
        fig.update_layout(**layout)

    # Line width (trace-level — must come after update_layout)
    lw = config.get("linewidth")
    if lw is not None:
        fig.update_traces(
            line=dict(width=lw),
            selector=dict(mode="lines"),
        )
        fig.update_traces(
            line=dict(width=lw),
            selector=dict(mode="lines+markers"),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_col(data: any, col_key) -> np.ndarray:
    """Extract a single column/series from *data* as a 1-D numpy array.

    Parameters
    ----------
    data : DataFrame or ndarray
        Source dataset.
    col_key : str or None
        Column name (DataFrame), string column index (2-D ndarray), or
        ``None`` to return the full array flattened.
    """
    if isinstance(data, pd.DataFrame):
        return data[col_key].values
    arr = np.asarray(data)
    if col_key is None:
        return arr.flatten()
    return arr[:, int(col_key)]


def _numeric_cols(data: any):
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).columns.tolist()
    return []


def _grid_from_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_n: int = 100,
    method: str = "cubic",
) -> tuple:
    """Return (X, Y, Z) meshgrids suitable for contour_plot.

    If *x* and *y* already describe a structured (regular) grid the data is
    pivoted directly without interpolation.  Otherwise
    ``scipy.interpolate.griddata`` is used to interpolate onto an N×N grid.
    """
    from scipy.interpolate import griddata

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    n_expected = len(x_unique) * len(y_unique)

    # --- attempt structured-grid reshape ---------------------------------
    if n_expected == len(x):
        try:
            xi_sorted = np.sort(x_unique)
            yi_sorted = np.sort(y_unique)
            Xi, Yi = np.meshgrid(xi_sorted, yi_sorted)
            Zi = np.full(Xi.shape, np.nan)
            ix_map = {v: i for i, v in enumerate(xi_sorted)}
            iy_map = {v: i for i, v in enumerate(yi_sorted)}
            for xv, yv, zv in zip(x, y, z):
                Zi[iy_map[yv], ix_map[xv]] = zv
            if not np.any(np.isnan(Zi)):
                return Xi, Yi, Zi
        except Exception:
            pass  # fall through to griddata

    # --- scattered data: interpolate onto a regular grid -----------------
    xi = np.linspace(x.min(), x.max(), grid_n)
    yi = np.linspace(y.min(), y.max(), grid_n)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method=method)
    return Xi, Yi, Zi


def _generate_plot(plot_type: str, data: any, config: dict):
    """Dispatch to the correct plotting function.

    Returns
    -------
    fig : Figure or plotly Figure
    is_plotly : bool
    """
    kwargs = get_plot_kwargs(config)

    # ── Matplotlib ──────────────────────────────────────────────────────────
    if plot_type == "histogram":
        arr = _extract_col(data, config.get("_data_col"))
        arr = _apply_y_transform(arr, config)
        fig, ax, _ = histogram(arr, **kwargs)
        return fig, False

    if plot_type == "line_plot":
        x_col = config.get("_x_col")
        y_cols = config.get("_y_cols")

        if x_col is None and y_cols is None:
            # 1-D array
            x = np.arange(len(np.asarray(data).flatten()))
            y = [_apply_y_transform(np.asarray(data).flatten(), config)]
            labels = None
        elif isinstance(data, pd.DataFrame):
            x = data[x_col].values
            y = [_apply_y_transform(data[c].values, config) for c in y_cols]
            labels = list(y_cols)
        else:
            x = data[:, int(x_col)]
            y = [_apply_y_transform(data[:, int(c)], config) for c in y_cols]
            labels = [f"Col {c}" for c in y_cols]
        x = _apply_x_transform(x, config)

        # Custom legend labels override column names
        custom_labels = config.get("_custom_labels")
        if custom_labels and labels:
            labels = custom_labels[: len(labels)]

        if labels:
            kwargs["labels"] = labels
        _set_mpl_color_cycle(config)
        fig, ax = line_plot(x, y, **kwargs)

        # Secondary Y axis
        sec_y_cols = config.get("_secondary_y_cols")
        if sec_y_cols and isinstance(data, pd.DataFrame):
            palette = COLOR_PALETTES.get(
                config.get("color_palette", "Default"),
                COLOR_PALETTES["Default"],
            )
            n_primary = len(y)
            ax2 = ax.twinx()
            lw = config.get("linewidth", 1.5)
            for i, col in enumerate(sec_y_cols):
                y_sec = _apply_y_transform(data[col].values, config)
                color = palette[(n_primary + i) % len(palette)]
                sec_lbl = col
                ax2.plot(
                    x,
                    y_sec,
                    color=color,
                    linestyle="--",
                    linewidth=lw,
                    label=str(sec_lbl),
                )
            sec_label = config.get("_secondary_y_label", "")
            if sec_label:
                ax2.set_ylabel(sec_label)
            if config.get("_secondary_y_scale") == "log":
                ax2.set_yscale("log")
            # Merge legends from both axes
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best")

        return fig, False

    if plot_type == "scatter_plot":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        fig, ax = scatter_plot(x, y, **kwargs)
        return fig, False

    if plot_type == "heatmap":
        if config.get("_use_numeric_df"):
            mat = data[_numeric_cols(data)].values
        else:
            mat = np.asarray(data)
        mat = _apply_z_transform(mat, config)
        if config.get("cmap_reversed"):
            cmap_val = kwargs.get("cmap", "viridis")
            if not str(cmap_val).endswith("_r"):
                kwargs["cmap"] = str(cmap_val) + "_r"
        if config.get("z_scale") == "log":
            from matplotlib.colors import LogNorm

            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            kwargs["norm"] = LogNorm(
                vmin=vmin if (vmin and vmin > 0) else None,
                vmax=vmax if (vmax and vmax > 0) else None,
            )
        fig, ax = heatmap(mat, **kwargs)
        return fig, False

    if plot_type == "contour_plot":
        if config.get("_mode") == "dataframe":
            x_arr = _apply_x_transform(data[config["_x_col"]].values.astype(float), config)
            y_arr = data[config["_y_col"]].values.astype(float)
            z_arr = data[config["_z_col"]].values.astype(float)
            grid_n = int(config.get("_grid_n", 100))
            interp = config.get("_interp_method", "cubic")
            X, Y, Z = _grid_from_xyz(x_arr, y_arr, z_arr, grid_n, interp)
        else:
            Z = np.asarray(data)
            nrows, ncols = Z.shape
            X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        Z = _apply_z_transform(Z, config)
        if config.get("cmap_reversed"):
            cmap_val = kwargs.get("cmap", "viridis")
            if not str(cmap_val).endswith("_r"):
                kwargs["cmap"] = str(cmap_val) + "_r"
        if config.get("z_scale") == "log":
            from matplotlib.colors import LogNorm

            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            kwargs["norm"] = LogNorm(
                vmin=vmin if (vmin and vmin > 0) else None,
                vmax=vmax if (vmax and vmax > 0) else None,
            )
        fig, ax = contour_plot(X, Y, Z, **kwargs)
        return fig, False

    # ── Seaborn ─────────────────────────────────────────────────────────────
    if plot_type == "distribution_plot":
        arr = _extract_col(data, config.get("_data_col"))
        fig, ax = distribution_plot(arr, **kwargs)
        return fig, False

    if plot_type == "box_plot":
        if isinstance(data, pd.DataFrame):
            fig, ax = box_plot(data, **kwargs)
        else:
            fig, ax = box_plot(np.asarray(data).flatten(), **kwargs)
        return fig, False

    if plot_type == "regression_plot":
        x_col = config.get("_x_col")
        y_col = config.get("_y_col")
        if config.get("_use_df", False):
            fig, ax = regression_plot(x=x_col, y=y_col, data=data, **kwargs)
        else:
            x = _extract_col(data, x_col)
            y = _extract_col(data, y_col)
            fig, ax = regression_plot(x=x, y=y, **kwargs)
        return fig, False

    # ── Plotly ───────────────────────────────────────────────────────────────
    if plot_type == "interactive_histogram":
        arr = _apply_y_transform(_extract_col(data, config.get("_data_col")), config)
        fig = interactive_histogram(arr, **kwargs)
        return fig, True

    if plot_type == "interactive_scatter":
        x = _extract_col(data, config.get("_x_col"))
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        fig = interactive_scatter(x, y, **kwargs)
        return fig, True

    if plot_type == "interactive_line":
        x_col = config.get("_x_col")
        y_cols = config.get("_y_cols")

        if x_col is None and y_cols is None:
            arr = np.asarray(data).flatten()
            x = np.arange(len(arr))
            y = [_apply_y_transform(arr, config)]
            labels = ["Data"]
        elif isinstance(data, pd.DataFrame):
            x = data[x_col].values
            y = [_apply_y_transform(data[c].values, config) for c in y_cols]
            labels = list(y_cols)
        else:
            x = data[:, int(x_col)]
            y = [_apply_y_transform(data[:, int(c)], config) for c in y_cols]
            labels = [f"Col {c}" for c in y_cols]

        fig = interactive_line(x, y, labels=labels, **kwargs)
        _apply_plotly_palette(fig, config)
        return fig, True

    if plot_type == "interactive_heatmap":
        if config.get("_use_numeric_df"):
            mat = data[_numeric_cols(data)].values
        else:
            mat = np.asarray(data)
        fig = interactive_heatmap(mat, **kwargs)
        return fig, True

    if plot_type == "interactive_3d_surface":
        Z = np.asarray(data)
        nrows, ncols = Z.shape
        X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        fig = interactive_3d_surface(X, Y, Z, **kwargs)
        return fig, True

    # ── M14 Extended Plot Types ──────────────────────────────────────────────
    if plot_type == "bar_chart":
        cat_col = config.get("_cat_col")
        val_col = config.get("_val_col")
        yerr_col = config.get("_yerr_col")
        hue_cols = config.get("_hue_cols")

        if isinstance(data, pd.DataFrame):
            if cat_col is not None:
                cats = data[cat_col].astype(str).tolist()
            else:
                cats = [str(i) for i in range(len(data))]

            if hue_cols:
                # Grouped / stacked: each hue column is one group
                vals = np.array([_apply_y_transform(data[c].values, config) for c in hue_cols])
                kwargs["hue"] = hue_cols
            else:
                vals = _apply_y_transform(
                    data[val_col].values if val_col else data.iloc[:, 0].values,
                    config,
                )
            yerr = data[yerr_col].values if (yerr_col and yerr_col in data.columns) else None
        elif isinstance(data, np.ndarray):
            arr = np.asarray(data)
            if arr.ndim == 2 and cat_col is not None:
                cats = [str(v) for v in arr[:, int(cat_col)]]
                raw = arr[:, int(val_col)] if val_col else arr[:, 1]
                vals = _apply_y_transform(raw, config)
            else:
                arr_flat = arr.flatten()
                cats = [str(i) for i in range(len(arr_flat))]
                vals = _apply_y_transform(arr_flat, config)
            yerr = None
        else:
            cats = [str(i) for i in range(len(data))]
            vals = np.asarray(data).flatten()
            yerr = None

        _set_mpl_color_cycle(config)
        fig, ax, _ = bar_chart(cats, vals, yerr=yerr, **kwargs)
        return fig, False

    if plot_type == "waterfall_plot":
        x_col = config.get("_x_col")
        y_cols = config.get("_y_cols")

        if isinstance(data, pd.DataFrame) and x_col and y_cols:
            x = _apply_x_transform(data[x_col].values, config)
            y_mat = np.array([_apply_y_transform(data[c].values, config) for c in y_cols])
            labels_wf = list(y_cols)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            if x_col is not None and y_cols:
                x = _apply_x_transform(data[:, int(x_col)], config)
                y_mat = np.array([_apply_y_transform(data[:, int(c)], config) for c in y_cols])
                labels_wf = [f"Col {c}" for c in y_cols]
            else:
                x = np.arange(data.shape[1])
                y_mat = _apply_y_transform(data, config)
                labels_wf = None
        else:
            arr = np.asarray(data)
            x = np.arange(arr.shape[-1] if arr.ndim > 1 else len(arr))
            y_mat = arr.reshape(-1, x.shape[0]) if arr.ndim > 1 else arr.reshape(1, -1)
            labels_wf = None

        fig, ax, _ = waterfall_plot(x, y_mat, labels=labels_wf, **kwargs)
        return fig, False

    if plot_type == "dual_axis_plot":
        x_col = config.get("_x_col")
        y1_col = config.get("_y1_col")
        y2_col = config.get("_y2_col")

        if isinstance(data, pd.DataFrame):
            x = _apply_x_transform(data[x_col].values, config)
            y1 = _apply_y_transform(data[y1_col].values, config)
            y2 = _apply_y_transform(data[y2_col].values, config)
            kwargs.setdefault("label1", y1_col)
            kwargs.setdefault("label2", y2_col)
            kwargs.setdefault("ylabel1", y1_col)
            kwargs.setdefault("ylabel2", y2_col)
        elif isinstance(data, np.ndarray) and data.ndim == 2 and x_col is not None:
            x = _apply_x_transform(data[:, int(x_col)], config)
            y1 = _apply_y_transform(data[:, int(y1_col)], config)
            y2 = _apply_y_transform(data[:, int(y2_col)], config)
        else:
            raise ValueError("Dual Y-axis plot requires a DataFrame or 2-D array.")

        fig, ax, _ = dual_axis_plot(x, y1, y2, **kwargs)
        return fig, False

    if plot_type == "z_colored_scatter":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        z = _extract_col(data, config.get("_z_col"))
        fig, ax, _ = z_colored_scatter(x, y, z, **kwargs)
        return fig, False

    if plot_type == "bubble_chart":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        sizes = _extract_col(data, config.get("_size_col"))
        z_col = config.get("_z_col")
        z = _extract_col(data, z_col) if z_col else None
        if z is not None:
            kwargs["z"] = z
        fig, ax, _ = bubble_chart(x, y, sizes, **kwargs)
        return fig, False

    if plot_type == "polar_plot":
        theta_col = config.get("_theta_col")
        r_col = config.get("_r_col")
        if theta_col is None and r_col is None:
            arr = np.asarray(data).flatten()
            theta = np.linspace(0, 2 * np.pi, len(arr))
            r = arr
        else:
            theta = _extract_col(data, theta_col)
            r = _extract_col(data, r_col)
        fig, ax, _ = polar_plot(theta, r, **kwargs)
        return fig, False

    if plot_type == "histogram_2d":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        fig, ax, _ = histogram_2d(x, y, **kwargs)
        return fig, False

    if plot_type == "pair_plot":
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Pair plot requires a DataFrame.")
        vars_sel = config.get("vars")
        hue_sel = config.get("hue")
        pk = {k: v for k, v in kwargs.items() if k not in ("xlabel", "ylabel", "figsize")}
        if vars_sel:
            pk["vars"] = vars_sel
        if hue_sel:
            pk["hue"] = hue_sel
        fig, ax, _ = pair_plot(data, **pk)
        return fig, False

    if plot_type == "interactive_3d_scatter":
        x = _extract_col(data, config.get("_x_col"))
        y = _extract_col(data, config.get("_y_col"))
        z = _extract_col(data, config.get("_z_col"))
        color_col = config.get("_color_col")
        size_col = config.get("_size_col")
        if color_col is not None:
            kwargs["color"] = _extract_col(data, color_col)
        if size_col is not None:
            kwargs["size"] = _extract_col(data, size_col)
        fig, _ = interactive_3d_scatter(x, y, z, **kwargs)
        return fig, True

    if plot_type == "scatter_with_regression":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        fig, ax, _ = scatter_with_regression(x, y, **kwargs)
        return fig, False

    if plot_type == "residual_plot":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y_actual = _extract_col(data, config.get("_y_actual_col"))
        y_fitted = _extract_col(data, config.get("_y_fitted_col"))
        fig, ax, _ = residual_plot(x, y_actual, y_fitted, **kwargs)
        return fig, False

    if plot_type == "interactive_ternary":
        a = _extract_col(data, config.get("_a_col"))
        b = _extract_col(data, config.get("_b_col"))
        c = _extract_col(data, config.get("_c_col"))
        color_col = config.get("_color_col")
        if color_col is not None:
            kwargs["color"] = _extract_col(data, color_col)
        fig, _ = interactive_ternary(a, b, c, **kwargs)
        return fig, True

    if plot_type == "broken_axis_plot":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        breaks = config.get("_breaks", [(0.0, 0.0)])
        axis = config.get("axis", "x")
        lw = config.get("linewidth", 1.5)
        color = kwargs.get("color", "steelblue")
        title = kwargs.get("title")
        xlabel = kwargs.get("xlabel")
        ylabel = kwargs.get("ylabel")
        fig, ax, _ = broken_axis_plot(
            x,
            y,
            breaks=breaks,
            axis=axis,
            linewidth=lw,
            color=color,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        return fig, False

    if plot_type == "inset_plot":
        x = _apply_x_transform(_extract_col(data, config.get("_x_col")), config)
        y = _apply_y_transform(_extract_col(data, config.get("_y_col")), config)
        x_inset = _apply_x_transform(_extract_col(data, config.get("_xi_col")), config)
        y_inset = _apply_y_transform(_extract_col(data, config.get("_yi_col")), config)
        ib = [
            float(config.get("_ib_x0", 0.55)),
            float(config.get("_ib_y0", 0.55)),
            float(config.get("_ib_w", 0.4)),
            float(config.get("_ib_h", 0.35)),
        ]
        region_start = config.get("_region_start")
        region_end = config.get("_region_end")
        indicate_region = None
        if region_start is not None and region_end is not None:
            indicate_region = (float(region_start), float(region_end))
        fig, ax, _ = inset_plot(
            x,
            y,
            x_inset,
            y_inset,
            inset_bounds=ib,
            indicate_region=indicate_region,
            title=kwargs.get("title"),
            xlabel=kwargs.get("xlabel"),
            ylabel=kwargs.get("ylabel"),
            inset_xlabel=config.get("inset_xlabel"),
            inset_ylabel=config.get("inset_ylabel"),
            linewidth=float(config.get("linewidth", 1.5)),
            figsize=config.get("figsize", (8, 6)),
        )
        return fig, False

    raise ValueError(f"Unhandled plot type: {plot_type}")


# ── M16: Annotation helpers ───────────────────────────────────────────────────


def _fig_to_png_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes.

    Storing bytes (not the figure object) in session state avoids the
    ``MediaFileHandler: Missing file`` error that occurs when Streamlit's
    in-memory media store is cleared between reruns.
    """
    import io as _io

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def _regen_with_overlays() -> None:
    """Re-generate the current matplotlib figure and re-apply all overlays.

    Called after any change to ``qp_ann_overlays`` so the display stays in sync.
    Silently no-ops if the required state keys are missing or generation fails.
    """
    pt = st.session_state.get("qp_current_plot_type", "")
    ds = st.session_state.get("qp_dataset", "")
    cfg = st.session_state.get("qp_config", {})
    if not (pt and ds and cfg):
        return
    d = get_dataset(ds)
    if d is None:
        return
    try:
        fig, is_plotly = _generate_plot(pt, d, cfg)
        if not is_plotly:
            _apply_mpl_style(fig, cfg)
            overlays = st.session_state.get("qp_ann_overlays", [])
            if overlays and fig.axes:
                from plottle.annotations import apply_annotations

                apply_annotations(fig.axes[0], overlays)
            st.session_state.qp_mpl_fig = fig
            st.session_state.qp_mpl_png = _fig_to_png_bytes(fig)
    except Exception:  # noqa: BLE001
        pass  # preserve the last good figure; user will see the error on next generate


def _annotation_input_widgets(ann_type: str) -> dict | None:
    """Render input widgets for *ann_type* and return the overlay dict (or None).

    All widget keys are prefixed with ``ann_new_`` to avoid collision with other
    widgets on the page.
    """
    from plottle.annotations import ANNOTATION_COLORS, ANNOTATION_LINESTYLES

    color = st.selectbox(
        "Color",
        ANNOTATION_COLORS,
        key="ann_new_color",
    )
    label = st.text_input("Legend label (optional)", key="ann_new_label")

    if ann_type == "Horizontal line":
        y_val = st.number_input("y value", value=0.0, key="ann_new_y")
        ls = st.selectbox("Line style", ANNOTATION_LINESTYLES, key="ann_new_ls")
        lw = st.slider("Line width", 0.5, 4.0, 1.5, 0.25, key="ann_new_lw")
        return {
            "type": "hline",
            "y": y_val,
            "color": color,
            "linestyle": ls,
            "lw": lw,
            "label": label or "",
        }

    if ann_type == "Vertical line":
        x_val = st.number_input("x value", value=0.0, key="ann_new_x")
        ls = st.selectbox("Line style", ANNOTATION_LINESTYLES, key="ann_new_ls")
        lw = st.slider("Line width", 0.5, 4.0, 1.5, 0.25, key="ann_new_lw")
        return {
            "type": "vline",
            "x": x_val,
            "color": color,
            "linestyle": ls,
            "lw": lw,
            "label": label or "",
        }

    if ann_type == "Horizontal span":
        c1, c2 = st.columns(2)
        y1 = c1.number_input("y1", value=0.0, key="ann_new_y1")
        y2 = c2.number_input("y2", value=1.0, key="ann_new_y2")
        alpha = st.slider("Opacity", 0.05, 1.0, 0.25, 0.05, key="ann_new_alpha")
        return {
            "type": "hspan",
            "y1": y1,
            "y2": y2,
            "color": color,
            "alpha": alpha,
            "label": label or "",
        }

    if ann_type == "Vertical span":
        c1, c2 = st.columns(2)
        x1 = c1.number_input("x1", value=0.0, key="ann_new_x1")
        x2 = c2.number_input("x2", value=1.0, key="ann_new_x2")
        alpha = st.slider("Opacity", 0.05, 1.0, 0.25, 0.05, key="ann_new_alpha")
        return {
            "type": "vspan",
            "x1": x1,
            "x2": x2,
            "color": color,
            "alpha": alpha,
            "label": label or "",
        }

    if ann_type == "Text":
        text_val = st.text_input("Text", value="annotation", key="ann_new_text")
        c1, c2 = st.columns(2)
        x_pos = c1.number_input("x", value=0.0, key="ann_new_x")
        y_pos = c2.number_input("y", value=0.0, key="ann_new_y")
        fontsize = st.slider("Font size", 6, 24, 10, 1, key="ann_new_fs")
        use_arrow = st.checkbox("Add arrow", value=False, key="ann_new_arrow")
        tx, ty = x_pos, y_pos
        if use_arrow:
            ca, cb = st.columns(2)
            tx = ca.number_input("Arrow tail x", value=float(x_pos) + 0.5, key="ann_new_tx")
            ty = cb.number_input("Arrow tail y", value=float(y_pos) + 0.5, key="ann_new_ty")
        return {
            "type": "text",
            "x": x_pos,
            "y": y_pos,
            "text": text_val,
            "color": color,
            "fontsize": fontsize,
            "arrow": use_arrow,
            "tx": tx,
            "ty": ty,
            "label": label or "",
        }

    if ann_type == "Rectangle":
        c1, c2 = st.columns(2)
        x1 = c1.number_input("x1", value=0.0, key="ann_new_x1")
        y1 = c2.number_input("y1", value=0.0, key="ann_new_y1")
        c3, c4 = st.columns(2)
        x2 = c3.number_input("x2", value=1.0, key="ann_new_x2")
        y2 = c4.number_input("y2", value=1.0, key="ann_new_y2")
        alpha = st.slider("Opacity", 0.05, 1.0, 0.25, 0.05, key="ann_new_alpha")
        face = st.selectbox("Fill color", ["none"] + ANNOTATION_COLORS, key="ann_new_face")
        return {
            "type": "rectangle",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "color": color,
            "facecolor": face,
            "alpha": alpha,
            "label": label or "",
        }

    if ann_type == "Ellipse":
        c1, c2 = st.columns(2)
        cx = c1.number_input("Center x", value=0.0, key="ann_new_cx")
        cy = c2.number_input("Center y", value=0.0, key="ann_new_cy")
        c3, c4 = st.columns(2)
        w = c3.number_input("Width", value=1.0, min_value=0.001, key="ann_new_w")
        h = c4.number_input("Height", value=1.0, min_value=0.001, key="ann_new_h")
        alpha = st.slider("Opacity", 0.05, 1.0, 0.25, 0.05, key="ann_new_alpha")
        face = st.selectbox("Fill color", ["none"] + ANNOTATION_COLORS, key="ann_new_face")
        return {
            "type": "ellipse",
            "cx": cx,
            "cy": cy,
            "width": w,
            "height": h,
            "color": color,
            "facecolor": face,
            "alpha": alpha,
            "label": label or "",
        }

    return None


def _render_annotation_panel() -> None:
    """Render the full annotations expander panel for the current matplotlib figure."""
    from plottle.annotations import describe_overlay

    overlays: list = st.session_state.qp_ann_overlays

    # ── Existing annotations ──────────────────────────────────────────────────
    if overlays:
        st.markdown("**Current annotations**")
        for i, ov in enumerate(overlays):
            c_desc, c_del = st.columns([5, 1])
            c_desc.caption(describe_overlay(ov))
            if c_del.button("✕", key=f"ann_del_{i}", help="Remove"):
                st.session_state.qp_ann_overlays.pop(i)
                _regen_with_overlays()
                st.rerun(scope="fragment")

        if st.button("Clear all annotations", key="ann_clear_all"):
            st.session_state.qp_ann_overlays = []
            _regen_with_overlays()
            st.rerun(scope="fragment")

        st.markdown("---")

    # ── Add new annotation ────────────────────────────────────────────────────
    st.markdown("**Add annotation**")
    _ANN_TYPES = [
        "Horizontal line",
        "Vertical line",
        "Horizontal span",
        "Vertical span",
        "Text",
        "Rectangle",
        "Ellipse",
    ]
    ann_type = st.selectbox("Annotation type", _ANN_TYPES, key="ann_type_sel")
    new_overlay = _annotation_input_widgets(ann_type)

    if new_overlay is not None:
        if st.button("Add annotation", type="primary", key="ann_add_btn"):
            st.session_state.qp_ann_overlays.append(new_overlay)
            _regen_with_overlays()
            st.rerun(scope="fragment")


# ── Page layout ──────────────────────────────────────────────────────────────

st.title("Quick Plot")
st.caption(
    "Choose a dataset, select a plot type, configure parameters, and generate your visualization."
)

# ── Style Presets panel ───────────────────────────────────────────────────────
# Helpers to move values between _shared_* widget keys and config.json presets.


def _collect_preset_settings() -> dict:
    """Read all current shared widget values into a plain config dict."""
    settings: dict = {}
    for cfg_key, widget_key in _DEFAULTS_KEY_MAP.items():
        val = st.session_state.get(widget_key)
        if val is not None:
            settings[cfg_key] = val
    # Custom palette colours are not in _DEFAULTS_KEY_MAP
    custom = st.session_state.get("_custom_colors")
    if custom:
        settings["_custom_colors"] = custom
    # Save current plot type so it can be restored on load
    if st.session_state.get("qp_category"):
        settings["_preset_category"] = st.session_state["qp_category"]
    if st.session_state.get("qp_plottype"):
        settings["_preset_plot_label"] = st.session_state["qp_plottype"]
    return settings


def _apply_preset_to_widgets(settings: dict) -> None:
    """Write a preset dict back into the shared widget session-state keys."""
    for cfg_key, widget_key in _DEFAULTS_KEY_MAP.items():
        if cfg_key in settings:
            st.session_state[widget_key] = settings[cfg_key]
    if "_custom_colors" in settings:
        st.session_state["_custom_colors"] = settings["_custom_colors"]

    # Restore saved plot type (category + label)
    if "_preset_category" in settings:
        st.session_state["qp_category"] = settings["_preset_category"]
    if "_preset_plot_label" in settings:
        st.session_state["qp_plottype"] = settings["_preset_plot_label"]
        # Derive the plot-type key so _regen_with_overlays uses the right type
        _cat = settings.get("_preset_category", st.session_state.get("qp_category", ""))
        _label = settings["_preset_plot_label"]
        _key = next(
            (k for k, v in PLOT_TYPES.items() if v["label"] == _label and v["category"] == _cat),
            None,
        )
        if _key:
            st.session_state["qp_current_plot_type"] = _key
        # Prevent smart-default from immediately overriding the restored type
        st.session_state["qp_last_smart_default_for"] = None

    # Merge style values into qp_config so _regen_with_overlays picks them up
    _style_keys = {k for k in settings if not k.startswith("_")}
    if _style_keys and st.session_state.get("qp_config"):
        merged = dict(st.session_state["qp_config"])
        merged.update({k: settings[k] for k in _style_keys})
        st.session_state["qp_config"] = merged


with st.expander("Style Presets", expanded=False):
    _tab_save, _tab_load, _tab_delete = st.tabs(["Save", "Load", "Delete"])

    with _tab_save:
        st.caption(
            "Snapshot the current appearance settings under a name.  "
            "Existing presets with the same name are overwritten."
        )
        _save_name = st.text_input(
            "Preset name",
            placeholder="e.g. Publication, Poster, Dark BG",
            key="qp_preset_save_name",
        )
        if st.button("Save preset", type="primary", key="qp_preset_save_btn"):
            _name = _save_name.strip()
            if not _name:
                st.warning("Enter a preset name first.")
            else:
                save_preset(_name, _collect_preset_settings())
                st.success(f"Preset **{_name}** saved.")

    with _tab_load:
        _presets = list_presets()
        if not _presets:
            st.info("No presets saved yet — use the **Save** tab to create one.")
        else:
            _sel_preset = st.selectbox(
                "Choose preset",
                _presets,
                key="qp_preset_sel",
            )
            if st.button("Apply preset", type="primary", key="qp_preset_apply"):
                _apply_preset_to_widgets(load_preset(_sel_preset))
                _regen_with_overlays()  # redraw existing figure with new style
                st.success(f"Preset **{_sel_preset}** applied.")
                st.rerun()
            with st.expander("Preview values"):
                st.json(load_preset(_sel_preset))

    with _tab_delete:
        _presets_del = list_presets()
        if not _presets_del:
            st.info("No presets saved yet.")
        else:
            _del_preset = st.selectbox(
                "Choose preset to delete",
                _presets_del,
                key="qp_preset_del_sel",
            )
            if st.button("Delete preset", type="primary", key="qp_preset_del_btn"):
                if delete_preset(_del_preset):
                    st.success(f"Preset **{_del_preset}** deleted.")
                    st.rerun()
                else:
                    st.error("Preset not found.")

# Guard: need at least one dataset
summary = get_session_summary()
if summary["num_datasets"] == 0:
    st.warning("No datasets loaded. Go to **Data Upload** first.")
    st.stop()

# ── Dataset selector ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 1. Dataset")

dataset_names = list(st.session_state.datasets.keys())
default_idx = (
    dataset_names.index(summary["current_dataset"])
    if summary["current_dataset"] in dataset_names
    else 0
)

col1, col2 = st.columns([3, 1])
with col1:
    dataset_name = st.selectbox(
        "Select dataset",
        dataset_names,
        index=default_idx,
        key="qp_dataset",
    )
with col2:
    st.write("")
    st.write("")
    st.metric("Loaded", summary["num_datasets"])

if dataset_name:
    st.session_state.current_dataset = dataset_name

data = get_dataset(dataset_name)

# Data type summary
if isinstance(data, pd.DataFrame):
    st.caption(f"DataFrame — {data.shape[0]} rows × {data.shape[1]} cols")
elif isinstance(data, np.ndarray):
    st.caption(f"NumPy array — shape {data.shape}, dtype {data.dtype}")

# ── Plot type selector ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Plot Type")

# Group by category for a two-step selector
categories: dict = {}
for key, info in PLOT_TYPES.items():
    cat = info["category"]
    categories.setdefault(cat, {})[info["label"]] = key

# Auto-suggest a sensible default when the selected dataset changes
if dataset_name and dataset_name != st.session_state.qp_last_smart_default_for:
    _sug_cat, _sug_label = _suggest_plot_type(data)
    st.session_state.qp_category = _sug_cat
    st.session_state.qp_plottype = _sug_label
    st.session_state.qp_last_smart_default_for = dataset_name

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Library", list(categories.keys()), key="qp_category")
with col2:
    label_to_key = categories[category]
    # Guard: if stored label doesn't exist in this category, fall back to first
    if st.session_state.get("qp_plottype") not in label_to_key:
        st.session_state.qp_plottype = next(iter(label_to_key))
    plot_label = st.selectbox("Plot type", list(label_to_key.keys()), key="qp_plottype")

plot_type = label_to_key[plot_label]

# Brief description
_DESCRIPTIONS = {
    "histogram": "Distribution of a single numeric variable.",
    "line_plot": "One or more lines over a shared X axis.",
    "scatter_plot": "X vs Y scatter with optional color/size mapping.",
    "heatmap": "2-D matrix as a color-encoded grid.",
    "contour_plot": "Iso-value contours over a 2-D scalar field.",
    "distribution_plot": "Seaborn histogram, KDE, or ECDF.",
    "box_plot": "Seaborn box, violin, or boxen plot.",
    "regression_plot": "Seaborn scatter with regression line and CI.",
    "interactive_histogram": "Plotly interactive histogram (zoom / hover).",
    "interactive_scatter": "Plotly interactive scatter (zoom / hover).",
    "interactive_line": "Plotly interactive line (multiple traces).",
    "interactive_heatmap": "Plotly interactive heatmap (hover values).",
    "interactive_3d_surface": "Plotly 3-D surface (rotate / zoom).",
}
st.caption(_DESCRIPTIONS.get(plot_type, ""))

# ── Plot workspace ───────────────────────────────────────────────────────────
# PROTOTYPE (speed comparison). Everything from "3. Configuration" down runs as one
# st.fragment: changing a setting reruns only this part of the page, not the preset,
# dataset and plot-type sections above it. Exports encode on demand instead of on
# every rerun, and the plot redraws live as settings change.

_EXPORT_FORMATS = {
    "PNG (150 dpi)": ("png", 150, "image/png", "plot.png"),
    "PNG (300 dpi)": ("png", 300, "image/png", "plot_300dpi.png"),
    "PNG (600 dpi)": ("png", 600, "image/png", "plot_600dpi.png"),
    "SVG (vector)": ("svg", None, "image/svg+xml", "plot.svg"),
    "PDF (vector)": ("pdf", None, "application/pdf", "plot.pdf"),
}


def _streamlit_supports_lazy_download() -> bool:
    # st.download_button(data=callable) arrived in Streamlit 1.52.
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (1, 52)


_LAZY_DOWNLOAD = _streamlit_supports_lazy_download()


def _savefig_bytes(fig, fmt: str, dpi) -> bytes:
    import io as _io

    buf = _io.BytesIO()
    if dpi is None:
        fig.savefig(buf, format=fmt, bbox_inches="tight")
    else:
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def _cached_for_figure(key: str, make):
    """Compute *make()* once per displayed figure (keyed by qp_fig_version)."""
    cache = st.session_state.setdefault("qp_export_cache", {})
    version = st.session_state.get("qp_fig_version", 0)
    if cache.get("_version") != version:
        cache.clear()
        cache["_version"] = version
    if key not in cache:
        cache[key] = make()
    return cache[key]


def _dataset_csv(data, dataset_name: str) -> str:
    cache = st.session_state.setdefault("qp_csv_cache", {})
    key = (dataset_name, id(data))
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        cache["csv"] = data.to_csv(index=False)
    return cache["csv"]


def _close_if_unreferenced(fig) -> None:
    """Close a replaced matplotlib figure unless plot history still holds it."""
    if fig is None:
        return
    if any(h.get("figure") is fig for h in st.session_state.get("plot_history", [])):
        return
    import matplotlib.pyplot as plt

    plt.close(fig)


def _store_figure(fig, is_plotly: bool, config: dict, plot_type: str, *, new_plot: bool):
    """Make *fig* the displayed figure.

    ``new_plot`` (a different dataset or plot type, or an explicit Generate) resets the
    click markers, zoom and Plotly conversion; a live restyle keeps them.
    """
    old = st.session_state.get("qp_mpl_fig")
    if is_plotly:
        _apply_plotly_style(fig, config)
        st.session_state.qp_fig_dict = fig.to_dict()
        st.session_state.qp_mpl_fig = None
    else:
        _apply_mpl_style(fig, config)
        overlays = st.session_state.get("qp_ann_overlays", [])
        if overlays and fig.axes:
            from plottle.annotations import apply_annotations

            apply_annotations(fig.axes[0], overlays)
        if not new_plot and fig.axes and st.session_state.get("qp_zoom_xlim"):
            fig.axes[0].set_xlim(*st.session_state.qp_zoom_xlim)
            fig.axes[0].set_ylim(*st.session_state.qp_zoom_ylim)
        st.session_state.qp_fig_dict = None
        st.session_state.qp_mpl_fig = fig
        st.session_state.qp_mpl_png = _fig_to_png_bytes(fig)
    if old is not None and old is not fig:
        _close_if_unreferenced(old)
    st.session_state.qp_is_plotly = is_plotly
    st.session_state.qp_config = config
    st.session_state.qp_current_plot_type = plot_type  # M16
    st.session_state.qp_fig_version = st.session_state.get("qp_fig_version", 0) + 1
    if new_plot:
        st.session_state.qp_annotations = []
        st.session_state.qp_show_plotly_version = False
        st.session_state.qp_zoom_xlim = None
        st.session_state.qp_zoom_ylim = None


def _same_config(a, b) -> bool:
    try:
        return bool(a == b)
    except Exception:  # e.g. array-valued entries
        return False


def _export_panel_plotly(fig, data, dataset_name: str) -> None:
    with st.expander("Export plot & data"):
        _ec1, _ec2 = st.columns(2)
        with _ec1:
            st.markdown("**Interactive plot (HTML)**")
            try:
                html = (
                    (lambda: fig.to_html(include_plotlyjs="cdn").encode("utf-8"))
                    if _LAZY_DOWNLOAD
                    else _cached_for_figure(
                        "html", lambda: fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                    )
                )
                st.download_button(
                    "Download HTML",
                    data=html,
                    file_name="plot.html",
                    mime="text/html",
                    key="qp_dl_html",
                )
            except Exception as _e:
                st.warning(f"HTML export failed: {_e}")
        with _ec2:
            if isinstance(data, pd.DataFrame):
                st.markdown("**Dataset (CSV)**")
                st.download_button(
                    "Download CSV",
                    data=_dataset_csv(data, dataset_name),
                    file_name=f"{dataset_name}.csv",
                    mime="text/csv",
                    key="qp_dl_csv_plotly",
                )


def _export_panel_mpl(data, dataset_name: str) -> None:
    with st.expander("Export plot & data"):
        fig = st.session_state.qp_mpl_fig
        _c1, _c2 = st.columns([3, 1])
        with _c1:
            label = st.selectbox(
                "Format", list(_EXPORT_FORMATS), key="qp_export_fmt", label_visibility="collapsed"
            )
        fmt, dpi, mime, fname = _EXPORT_FORMATS[label]
        if fmt == "png" and dpi == 150:
            payload = st.session_state.qp_mpl_png  # already rendered for display
        elif _LAZY_DOWNLOAD:
            payload = lambda: _savefig_bytes(fig, fmt, dpi)  # noqa: E731 -- encoded on click
        else:
            payload = _cached_for_figure(label, lambda: _savefig_bytes(fig, fmt, dpi))
        with _c2:
            st.download_button(
                "Download",
                data=payload,
                file_name=fname,
                mime=mime,
                type="primary",
                key="qp_dl_plot",
                width="stretch",
            )
        if isinstance(data, pd.DataFrame):
            st.download_button(
                "Download dataset as CSV",
                data=_dataset_csv(data, dataset_name),
                file_name=f"{dataset_name}.csv",
                mime="text/csv",
                key="qp_dl_csv_mpl",
            )


# Keep the plot column in view while the (long) settings column scrolls.
st.markdown(
    """
    <style>
    div[data-testid="stColumn"]:has(> div > div > .st-key-qp_plot_panel),
    div[data-testid="stColumn"]:has(.st-key-qp_plot_panel) {
        position: sticky;
        top: 4rem;
        align-self: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.fragment
def _plot_workspace(plot_type: str, data, dataset_name: str) -> None:
    st.markdown("---")
    # Settings on the left, plot on the right: every change is visible without
    # scrolling. The plot column stays in view while the settings scroll.
    cfg_col, plot_col = st.columns([2, 3], gap="large")
    with cfg_col:
        st.markdown("## 3. Configuration")

        config = get_plot_config_widgets(plot_type, data)

        st.markdown("---")

        if not config:
            st.info("Adjust the configuration above to enable plot generation.")
        else:
            live = st.toggle(
                "Live preview",
                value=True,
                key="qp_live",
                help="Redraw the plot as you change settings. "
                "Generate Plot also saves it to the plot history.",
            )
            _btn_col, _clear_col = st.columns([3, 2])
            with _btn_col:
                _do_generate = st.button(
                    "Generate Plot", type="primary", key="qp_generate", width="stretch"
                )
            with _clear_col:
                if st.session_state.qp_is_plotly and st.session_state.qp_annotations:
                    if st.button("Clear markers", key="qp_clear_ann", width="stretch"):
                        st.session_state.qp_annotations = []
                        st.rerun(scope="fragment")

            _target = (dataset_name, plot_type)
            _prev = st.session_state.get("qp_preview_for")
            _same_plot = _prev is not None and _prev[0] == _target
            _needs_preview = live and not (_same_plot and _same_config(_prev[1], config))

            if _do_generate:
                try:
                    with st.spinner("Generating…"):
                        fig, is_plotly = _generate_plot(plot_type, data, config)
                        _store_figure(fig, is_plotly, config, plot_type, new_plot=True)
                        st.session_state.qp_preview_for = (_target, config)
                    add_plot_to_history(
                        {
                            "type": plot_type,
                            "dataset": dataset_name,
                            "config": {k: str(v) for k, v in config.items()},
                            "figure": fig if not is_plotly else None,
                        }
                    )
                    st.success(
                        f"Saved to plot history (total: {len(st.session_state.plot_history)})"
                    )
                except Exception as exc:
                    st.error(f"Error generating plot: {exc}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
            elif _needs_preview:
                try:
                    fig, is_plotly = _generate_plot(plot_type, data, config)
                    _store_figure(fig, is_plotly, config, plot_type, new_plot=not _same_plot)
                except Exception as exc:
                    st.warning(f"Preview not updated: {exc}")
                # Record the attempt either way, so a failing config isn't retried every rerun.
                st.session_state.qp_preview_for = (_target, config)

    with plot_col, st.container(key="qp_plot_panel"):
        # ── Persistent plot display with click-to-annotate ────────────────────────────
        if st.session_state.qp_is_plotly and st.session_state.qp_fig_dict:
            import plotly.graph_objects as go

            _fig = go.Figure(st.session_state.qp_fig_dict)

            # Overlay annotation markers from session state
            if st.session_state.qp_annotations:
                _ax = [a["x"] for a in st.session_state.qp_annotations]
                _ay = [a["y"] for a in st.session_state.qp_annotations]
                _al = [format_marker_label(a["x"], a["y"]) for a in st.session_state.qp_annotations]
                _fig.add_trace(
                    go.Scatter(
                        x=_ax,
                        y=_ay,
                        mode="markers+text",
                        text=_al,
                        textposition="top center",
                        marker=dict(symbol="x-open", size=14, color="red", line=dict(width=2)),
                        name="Markers",
                        showlegend=True,
                    )
                )

            st.markdown("### Plot")
            st.caption(
                "Click any data point to mark it.  Use **Clear markers** above to remove all marks."
                if not st.session_state.qp_annotations
                else f"{len(st.session_state.qp_annotations)} point(s) marked."
            )

            try:
                _event = st.plotly_chart(
                    _fig,
                    width="stretch",
                    on_select="rerun",
                    selection_mode="points",
                    key="qp_chart",
                )
                # Handle newly selected points (Streamlit returns each point as a dict)
                _added = False
                for _ann in selected_points(_event):
                    if _ann not in st.session_state.qp_annotations:
                        st.session_state.qp_annotations.append(_ann)
                        _added = True
                if _added:
                    # The chart above was drawn before this click was read; redraw the
                    # plot area so the new marker shows now, not on the next interaction.
                    st.rerun(scope="fragment")
            except TypeError:
                # on_select not supported in this Streamlit version
                st.plotly_chart(_fig, width="stretch")
                st.info(
                    "Upgrade Streamlit (≥ 1.33) to enable click-to-annotate.",
                )

            # Marked-points table
            if st.session_state.qp_annotations:
                with st.expander(f"Marked points ({len(st.session_state.qp_annotations)})"):
                    _ann_df = pd.DataFrame(st.session_state.qp_annotations)
                    st.dataframe(_ann_df, width="stretch")

            _export_panel_plotly(_fig, data, dataset_name)

        elif st.session_state.qp_mpl_fig is not None:
            st.markdown("### Plot")

            # ── Convert to Plotly (zoom/pan) toggle ───────────────────────────────
            _ctp_col1, _ctp_col2 = st.columns([3, 1])
            with _ctp_col1:
                if not st.session_state.qp_show_plotly_version:
                    if st.button(
                        "Convert to Plotly (zoom/pan)",
                        key="qp_convert_to_plotly",
                        help=(
                            "Convert the current matplotlib figure to an interactive "
                            "Plotly figure for zooming and panning."
                        ),
                    ):
                        st.session_state.qp_show_plotly_version = True
                        st.rerun(scope="fragment")
                else:
                    if st.button(
                        "Back to static",
                        key="qp_back_to_static",
                    ):
                        st.session_state.qp_show_plotly_version = False
                        st.rerun(scope="fragment")

            if st.session_state.qp_show_plotly_version:
                try:
                    import plotly.tools as _ply_tools

                    _mpl_fig_for_conv = st.session_state.qp_mpl_fig
                    _ply_fig = _ply_tools.mpl_to_plotly(_mpl_fig_for_conv)
                    _apply_plotly_style(_ply_fig, st.session_state.qp_config)
                    st.info(
                        "Interactive Plotly version — use mouse to zoom and pan. "
                        "Note: mpl_to_plotly is a best-effort conversion; "
                        "some styling may differ."
                    )
                    st.plotly_chart(_ply_fig, width="stretch", key="qp_plotly_converted")
                except Exception as _conv_err:
                    st.warning(
                        f"Could not convert figure to Plotly: {_conv_err}. "
                        "Showing static version instead."
                    )
                    st.image(st.session_state.qp_mpl_png, width="stretch")
            else:
                st.image(st.session_state.qp_mpl_png, width="stretch")

            # ── Zoom / Axis Range ─────────────────────────────────────────────────
            if not st.session_state.qp_show_plotly_version:
                _zoom_fig = st.session_state.qp_mpl_fig
                if _zoom_fig is not None and _zoom_fig.axes:
                    _zoom_ax = _zoom_fig.axes[0]
                    _zxlim = _zoom_ax.get_xlim()
                    _zylim = _zoom_ax.get_ylim()
                    with st.expander("Zoom / Axis Range"):
                        _zc1, _zc2, _zc3, _zc4 = st.columns(4)
                        with _zc1:
                            _zxmin = st.number_input(
                                "X min", value=float(_zxlim[0]), key="qp_zoom_xmin", format="%g"
                            )
                        with _zc2:
                            _zxmax = st.number_input(
                                "X max", value=float(_zxlim[1]), key="qp_zoom_xmax", format="%g"
                            )
                        with _zc3:
                            _zymin = st.number_input(
                                "Y min", value=float(_zylim[0]), key="qp_zoom_ymin", format="%g"
                            )
                        with _zc4:
                            _zymax = st.number_input(
                                "Y max", value=float(_zylim[1]), key="qp_zoom_ymax", format="%g"
                            )
                        _zbtn1, _zbtn2 = st.columns(2)
                        with _zbtn1:
                            if st.button("Apply zoom", key="qp_apply_zoom"):
                                if _zxmin < _zxmax and _zymin < _zymax:
                                    _zoom_ax.set_xlim(_zxmin, _zxmax)
                                    _zoom_ax.set_ylim(_zymin, _zymax)
                                    st.session_state.qp_zoom_xlim = (_zxmin, _zxmax)
                                    st.session_state.qp_zoom_ylim = (_zymin, _zymax)
                                    st.session_state.qp_mpl_png = _fig_to_png_bytes(_zoom_fig)
                                    st.rerun(scope="fragment")
                                else:
                                    st.error("Min must be less than max for both axes.")
                        with _zbtn2:
                            if st.button("Reset zoom", key="qp_reset_zoom"):
                                st.session_state.qp_zoom_xlim = None
                                st.session_state.qp_zoom_ylim = None
                                _regen_with_overlays()
                                st.rerun(scope="fragment")

            # ── M16: Annotations panel ────────────────────────────────────────────
            _ann_label = (
                f"Annotations ({len(st.session_state.qp_ann_overlays)} active)"
                if st.session_state.qp_ann_overlays
                else "Annotations"
            )
            with st.expander(_ann_label, expanded=bool(st.session_state.qp_ann_overlays)):
                _render_annotation_panel()

            _export_panel_mpl(data, dataset_name)
        else:
            st.info("The plot appears here as soon as the configuration is valid.")


_plot_workspace(plot_type, data, dataset_name)
