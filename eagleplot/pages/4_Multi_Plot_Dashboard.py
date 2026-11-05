"""Multi-Plot Dashboard Page.

Configure a grid of independent plots (up to 3 × 2 = 6 cells) for
side-by-side visual comparison across datasets or plot types.
Each cell has its own dataset, plot type, and column selection.
"""

import traceback
from pathlib import Path
import sys

import streamlit as st
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.plotting import (
    histogram,
    line_plot,
    scatter_plot,
    heatmap,
    contour_plot,
    distribution_plot,
    box_plot,
    regression_plot,
    interactive_histogram,
    interactive_scatter,
    interactive_line,
    interactive_heatmap,
    interactive_3d_surface,
)
from eagleplot.utils import (
    initialize_session_state,
    get_dataset,
    get_session_summary,
)
from eagleplot.utils.plot_config import PLOT_TYPES

initialize_session_state()


# ── Constants ─────────────────────────────────────────────────────────────────

LAYOUTS: dict = {
    "1 × 1": (1, 1),
    "1 × 2": (1, 2),
    "2 × 1": (2, 1),
    "2 × 2": (2, 2),
    "2 × 3": (2, 3),
    "3 × 2": (3, 2),
    "3 × 3": (3, 3),
    "3 × 4": (3, 4),
    "4 × 3": (4, 3),
    "4 × 4": (4, 4),
}

# Categorise plot types by what data they need
_NEEDS_1COL = frozenset({"histogram", "distribution_plot", "interactive_histogram"})
_NEEDS_XY = frozenset(
    {
        "scatter_plot",
        "line_plot",
        "regression_plot",
        "interactive_scatter",
        "interactive_line",
    }
)
_BOX_TYPES = frozenset({"box_plot"})
_MATRIX_TYPES = frozenset(
    {"heatmap", "contour_plot", "interactive_heatmap", "interactive_3d_surface"}
)

# Matplotlib types that accept the figsize kwarg
_MPL_TYPES = frozenset({"histogram", "line_plot", "scatter_plot", "heatmap", "contour_plot"})

_CELL_FIGSIZE = (5, 4)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _numeric_cols(data) -> list:
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).columns.tolist()
    return []


def _col_options(data):
    """Return (options_list, format_func) for a column selector."""
    if isinstance(data, pd.DataFrame):
        return _numeric_cols(data), str
    arr = np.asarray(data)
    if arr.ndim < 2:
        return [], str
    return list(range(arr.shape[1])), lambda i: f"Column {i}"


def _to_1d(data, col_ref) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        return data[col_ref].values.astype(float)
    arr = np.asarray(data, dtype=float)
    if col_ref is None:
        return arr.flatten()
    return arr[:, int(col_ref)]


def _auto_1d(data) -> np.ndarray:
    """Return first numeric column (DataFrame) or flatten the array."""
    if isinstance(data, pd.DataFrame):
        cols = _numeric_cols(data)
        return data[cols[0]].values.astype(float) if cols else np.array([])
    return np.asarray(data, dtype=float).flatten()


def _cell_col_widgets(cell_idx: int, plot_type: str, data):
    """Render column pickers appropriate for *plot_type*. Returns (x_col, y_col)."""
    x_col = y_col = None
    options, fmt = _col_options(data)

    if plot_type in _NEEDS_1COL:
        if options:
            x_col = st.selectbox(
                "Data column",
                options,
                format_func=fmt,
                key=f"cell_{cell_idx}_xcol",
            )
        else:
            st.caption("1-D array — all values used.")

    elif plot_type in _NEEDS_XY:
        if len(options) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox(
                    "X column",
                    options,
                    format_func=fmt,
                    key=f"cell_{cell_idx}_xcol",
                )
            with c2:
                y_col = st.selectbox(
                    "Y column",
                    options,
                    index=min(1, len(options) - 1),
                    format_func=fmt,
                    key=f"cell_{cell_idx}_ycol",
                )
        elif options:
            x_col = y_col = options[0]
            st.caption("Only one column available — used for both axes.")
        else:
            st.caption("1-D array — auto-indexed on x axis.")

    elif plot_type in _BOX_TYPES:
        if isinstance(data, pd.DataFrame):
            cols = _numeric_cols(data)
            st.caption(f"Box plot will show all {len(cols)} numeric column(s).")
        # no column picker — uses whole dataset

    else:  # _MATRIX_TYPES
        st.caption("Uses the full dataset matrix — no column selection needed.")

    return x_col, y_col


def _render_cell_plot(cfg: dict) -> None:
    """Dispatch to the correct plotting function and display inside the current column.

    Parameters
    ----------
    cfg : dict
        Keys: 'dataset', 'plot_type', 'x_col', 'y_col', 'cell_title'
    """
    data = get_dataset(cfg["dataset"])
    if data is None:
        st.error(f"Dataset '{cfg['dataset']}' not found.")
        return

    pt = cfg["plot_type"]
    x_col = cfg.get("x_col")
    y_col = cfg.get("y_col")
    title = cfg.get("cell_title") or f"{PLOT_TYPES[pt]['label']} — {cfg['dataset']}"

    try:
        # ── Histogram ────────────────────────────────────────────────────────
        if pt == "histogram":
            arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            fig, _, __ = histogram(arr, title=title, figsize=_CELL_FIGSIZE)
            st.pyplot(fig)

        # ── Line Plot ────────────────────────────────────────────────────────
        elif pt == "line_plot":
            if x_col is not None and y_col is not None:
                x = _to_1d(data, x_col)
                y = [_to_1d(data, y_col)]
                lbl = [str(y_col)]
                xlab = str(x_col)
            else:
                arr = _auto_1d(data)
                x = np.arange(len(arr))
                y = [arr]
                lbl = ["Data"]
                xlab = "Index"
            fig, _ = line_plot(x, y, labels=lbl, xlabel=xlab, title=title, figsize=_CELL_FIGSIZE)
            st.pyplot(fig)

        # ── Scatter Plot ─────────────────────────────────────────────────────
        elif pt == "scatter_plot":
            x = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            y = _to_1d(data, y_col) if y_col is not None else _auto_1d(data)
            fig, _ = scatter_plot(
                x,
                y,
                xlabel=str(x_col or "x"),
                ylabel=str(y_col or "y"),
                title=title,
                figsize=_CELL_FIGSIZE,
            )
            st.pyplot(fig)

        # ── Heatmap ──────────────────────────────────────────────────────────
        elif pt == "heatmap":
            if isinstance(data, pd.DataFrame):
                cols = _numeric_cols(data)
                mat = data[cols].values.astype(float)
            else:
                mat = np.asarray(data, dtype=float)
            if mat.ndim != 2:
                st.error("Heatmap requires a 2-D array.")
                return
            fig, _ = heatmap(mat, title=title, figsize=_CELL_FIGSIZE)
            st.pyplot(fig)

        # ── Contour Plot ─────────────────────────────────────────────────────
        elif pt == "contour_plot":
            Z = np.asarray(data, dtype=float)
            if Z.ndim != 2:
                st.error("Contour plot requires a 2-D array.")
                return
            nrows_z, ncols_z = Z.shape
            X, Y = np.meshgrid(np.arange(ncols_z), np.arange(nrows_z))
            fig, _ = contour_plot(X, Y, Z, title=title, figsize=_CELL_FIGSIZE)
            st.pyplot(fig)

        # ── Distribution Plot (Seaborn) ───────────────────────────────────────
        elif pt == "distribution_plot":
            arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            fig, _ = distribution_plot(arr, title=title)
            st.pyplot(fig)

        # ── Box Plot (Seaborn) ────────────────────────────────────────────────
        elif pt == "box_plot":
            if isinstance(data, pd.DataFrame):
                fig, _ = box_plot(data, title=title)
            else:
                fig, _ = box_plot(np.asarray(data, dtype=float).flatten(), title=title)
            st.pyplot(fig)

        # ── Regression Plot (Seaborn) ─────────────────────────────────────────
        elif pt == "regression_plot":
            if isinstance(data, pd.DataFrame) and isinstance(x_col, str):
                fig, _ = regression_plot(x=x_col, y=y_col, data=data, title=title)
            else:
                x = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
                y = _to_1d(data, y_col) if y_col is not None else _auto_1d(data)
                fig, _ = regression_plot(x=x, y=y, title=title)
            st.pyplot(fig)

        # ── Interactive Histogram (Plotly) ────────────────────────────────────
        elif pt == "interactive_histogram":
            arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            fig = interactive_histogram(arr, title=title)
            st.plotly_chart(fig, width="stretch")

        # ── Interactive Scatter (Plotly) ──────────────────────────────────────
        elif pt == "interactive_scatter":
            x = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            y = _to_1d(data, y_col) if y_col is not None else _auto_1d(data)
            fig = interactive_scatter(
                x, y, xlabel=str(x_col or "x"), ylabel=str(y_col or "y"), title=title
            )
            st.plotly_chart(fig, width="stretch")

        # ── Interactive Line (Plotly) ─────────────────────────────────────────
        elif pt == "interactive_line":
            if x_col is not None and y_col is not None:
                x = _to_1d(data, x_col)
                y = [_to_1d(data, y_col)]
                lbl = [str(y_col)]
                xlab = str(x_col)
            else:
                arr = _auto_1d(data)
                x = np.arange(len(arr))
                y = [arr]
                lbl = ["Data"]
                xlab = "Index"
            fig = interactive_line(x, y, labels=lbl, xlabel=xlab, title=title)
            st.plotly_chart(fig, width="stretch")

        # ── Interactive Heatmap (Plotly) ──────────────────────────────────────
        elif pt == "interactive_heatmap":
            if isinstance(data, pd.DataFrame):
                cols = _numeric_cols(data)
                mat = data[cols].values.astype(float)
            else:
                mat = np.asarray(data, dtype=float)
            if mat.ndim != 2:
                st.error("Interactive heatmap requires a 2-D array.")
                return
            fig = interactive_heatmap(mat, title=title)
            st.plotly_chart(fig, width="stretch")

        # ── 3D Surface (Plotly) ───────────────────────────────────────────────
        elif pt == "interactive_3d_surface":
            Z = np.asarray(data, dtype=float)
            if Z.ndim != 2:
                st.error("3D Surface requires a 2-D array.")
                return
            nrows_z, ncols_z = Z.shape
            X, Y = np.meshgrid(np.arange(ncols_z), np.arange(nrows_z))
            fig = interactive_3d_surface(X, Y, Z, title=title)
            st.plotly_chart(fig, width="stretch")

        else:
            st.error(f"Unhandled plot type: {pt}")

    except Exception as exc:
        st.error(f"Error: {exc}")
        with st.expander("Details"):
            st.code(traceback.format_exc())


# ── M16 helpers ───────────────────────────────────────────────────────────────


def _generate_cell_fig(cfg: dict):
    """Return ``(fig, is_plotly)`` for a cell config without displaying.

    Used by the combined export and the axis-sharing pass.
    Returns ``(None, False)`` on error or unsupported type.
    """
    data = get_dataset(cfg["dataset"])
    if data is None:
        return None, False

    pt = cfg["plot_type"]
    x_col = cfg.get("x_col")
    y_col = cfg.get("y_col")
    title = cfg.get("cell_title") or f"{PLOT_TYPES[pt]['label']} — {cfg['dataset']}"

    try:
        if pt == "histogram":
            arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            fig, _, __ = histogram(arr, title=title, figsize=_CELL_FIGSIZE)
            return fig, False

        elif pt == "line_plot":
            if x_col is not None and y_col is not None:
                x = _to_1d(data, x_col)
                y = [_to_1d(data, y_col)]
                lbl = [str(y_col)]
                xlab = str(x_col)
            else:
                arr = _auto_1d(data)
                x = np.arange(len(arr))
                y = [arr]
                lbl = ["Data"]
                xlab = "Index"
            fig, _ = line_plot(
                x,
                y,
                labels=lbl,
                xlabel=xlab,
                title=title,
                figsize=_CELL_FIGSIZE,
            )
            return fig, False

        elif pt == "scatter_plot":
            x = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            y = _to_1d(data, y_col) if y_col is not None else _auto_1d(data)
            fig, _ = scatter_plot(
                x,
                y,
                xlabel=str(x_col or "x"),
                ylabel=str(y_col or "y"),
                title=title,
                figsize=_CELL_FIGSIZE,
            )
            return fig, False

        elif pt == "heatmap":
            if isinstance(data, pd.DataFrame):
                cols = _numeric_cols(data)
                mat = data[cols].values.astype(float)
            else:
                mat = np.asarray(data, dtype=float)
            if mat.ndim != 2:
                return None, False
            fig, _ = heatmap(mat, title=title, figsize=_CELL_FIGSIZE)
            return fig, False

        elif pt == "contour_plot":
            Z = np.asarray(data, dtype=float)
            if Z.ndim != 2:
                return None, False
            nrows_z, ncols_z = Z.shape
            X, Y = np.meshgrid(np.arange(ncols_z), np.arange(nrows_z))
            fig, _ = contour_plot(X, Y, Z, title=title, figsize=_CELL_FIGSIZE)
            return fig, False

        elif pt == "distribution_plot":
            arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            fig, _ = distribution_plot(arr, title=title)
            return fig, False

        elif pt == "box_plot":
            if isinstance(data, pd.DataFrame):
                fig, _ = box_plot(data, title=title)
            else:
                fig, _ = box_plot(np.asarray(data, dtype=float).flatten(), title=title)
            return fig, False

        elif pt == "regression_plot":
            if isinstance(data, pd.DataFrame) and isinstance(x_col, str):
                fig, _ = regression_plot(x=x_col, y=y_col, data=data, title=title)
            else:
                x = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
                y = _to_1d(data, y_col) if y_col is not None else _auto_1d(data)
                fig, _ = regression_plot(x=x, y=y, title=title)
            return fig, False

        elif pt in {
            "interactive_histogram",
            "interactive_scatter",
            "interactive_line",
            "interactive_heatmap",
            "interactive_3d_surface",
        }:
            return None, True  # Plotly — not included in combined export

        return None, False

    except Exception:  # noqa: BLE001
        return None, False


def _make_combined_figure(
    cell_figs: list,
    cell_labels: list,
    nrows: int,
    ncols: int,
    fig_title: str = "",
):
    """Create a combined grid matplotlib figure from individual cell figures.

    Each cell is rendered to an RGBA image and placed into a subplot using
    ``imshow``.  Plotly cells (None entries) are left blank.
    """
    import io as _io
    import matplotlib.pyplot as plt

    combined_w = ncols * 5.0
    combined_h = nrows * 4.0 + (0.6 if fig_title else 0.0)
    combined_fig, axes_grid = plt.subplots(nrows, ncols, figsize=(combined_w, combined_h))

    # Normalise axes_grid to always be a 2-D array
    if nrows == 1 and ncols == 1:
        axes_grid = np.array([[axes_grid]])
    elif nrows == 1:
        axes_grid = axes_grid.reshape(1, -1)
    elif ncols == 1:
        axes_grid = axes_grid.reshape(-1, 1)

    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes_grid[row, col]

        fig = cell_figs[idx] if idx < len(cell_figs) else None
        lbl = cell_labels[idx] if idx < len(cell_labels) else ""

        if fig is None:
            ax.set_visible(False)
            continue

        # Render cell figure to RGBA bytes
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = plt.imread(buf)

        ax.imshow(img)
        ax.axis("off")
        if lbl:
            ax.set_title(lbl, fontsize=9, pad=3)

    if fig_title:
        combined_fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.01)

    combined_fig.tight_layout()
    return combined_fig


def _apply_axis_sharing(
    cell_figs: list,
    nrows: int,
    ncols: int,
    sharex_row: bool,
    sharey_col: bool,
) -> None:
    """Sync x-axis limits across each row and/or y-axis limits down each column.

    Operates in-place on the figures stored in *cell_figs*.
    Only matplotlib figures are affected; None / Plotly entries are skipped.
    """
    if not (sharex_row or sharey_col):
        return

    def _get_ax(fig):
        return fig.axes[0] if (fig is not None and fig.axes) else None

    if sharex_row:
        for r in range(nrows):
            row_axes = [
                _get_ax(cell_figs[r * ncols + c])
                for c in range(ncols)
                if r * ncols + c < len(cell_figs)
            ]
            row_axes = [a for a in row_axes if a is not None]
            if len(row_axes) < 2:
                continue
            xlims = [a.get_xlim() for a in row_axes]
            xmin = min(lo for lo, _ in xlims)
            xmax = max(hi for _, hi in xlims)
            for a in row_axes:
                a.set_xlim(xmin, xmax)

    if sharey_col:
        for c in range(ncols):
            col_axes = [
                _get_ax(cell_figs[r * ncols + c])
                for r in range(nrows)
                if r * ncols + c < len(cell_figs)
            ]
            col_axes = [a for a in col_axes if a is not None]
            if len(col_axes) < 2:
                continue
            ylims = [a.get_ylim() for a in col_axes]
            ymin = min(lo for lo, _ in ylims)
            ymax = max(hi for _, hi in ylims)
            for a in col_axes:
                a.set_ylim(ymin, ymax)


# ── Page layout ───────────────────────────────────────────────────────────────

st.title("Multi-Plot Dashboard")
st.caption(
    "Arrange multiple independent plots in a grid for side-by-side comparison. "
    "Each cell can use a different dataset, plot type, and column selection."
)

# Guard: need at least one dataset
summary = get_session_summary()
if summary["num_datasets"] == 0:
    st.warning("No datasets loaded. Go to **Data Upload** first.")
    st.stop()

# ── Layout selection ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 1. Layout")

layout_choice = st.radio(
    "Grid dimensions (rows × columns)",
    list(LAYOUTS.keys()),
    index=2,  # default: 2 × 1
    horizontal=True,
    key="dash_layout",
)
nrows, ncols = LAYOUTS[layout_choice]
n_cells = nrows * ncols
st.caption(f"{n_cells} plot cell(s) — {nrows} row(s) × {ncols} column(s).")

# ── Figure-level settings (M16) ───────────────────────────────────────────────
with st.expander("Figure settings", expanded=False):
    fig_title = st.text_input(
        "Figure title (optional — shown on combined export)",
        key="dash_fig_title",
    )
    _sc1, _sc2 = st.columns(2)
    sharex_row = _sc1.checkbox(
        "Share X-axis across each row",
        value=False,
        key="dash_sharex",
        help="Sync x-axis limits for all Matplotlib panels in the same row.",
    )
    sharey_col = _sc2.checkbox(
        "Share Y-axis down each column",
        value=False,
        key="dash_sharey",
        help="Sync y-axis limits for all Matplotlib panels in the same column.",
    )

# ── Cell configuration ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Cell Configuration")
st.markdown("Configure each cell below. Cells are numbered left-to-right, top-to-bottom.")

dataset_names = list(st.session_state.datasets.keys())

# Lay out cell config expanders in a 2-column UI grid (unrelated to plot grid)
cell_cfgs: list = []
ui_cols = st.columns(min(n_cells, 2))

for i in range(n_cells):
    with ui_cols[i % 2]:
        with st.expander(f"**Cell {i + 1}**", expanded=True):
            # Dataset selector — default cycles through loaded datasets
            default_ds_idx = min(i, len(dataset_names) - 1)
            ds_name = st.selectbox(
                "Dataset",
                dataset_names,
                index=default_ds_idx,
                key=f"cell_{i}_ds",
            )
            cell_data = get_dataset(ds_name)

            # Plot type selector
            pt = st.selectbox(
                "Plot type",
                list(PLOT_TYPES.keys()),
                format_func=lambda k: f"{PLOT_TYPES[k]['label']} ({PLOT_TYPES[k]['category']})",
                key=f"cell_{i}_pt",
            )

            # Column selectors (context-aware)
            x_col, y_col = _cell_col_widgets(i, pt, cell_data)

            # Optional cell title
            cell_title = st.text_input(
                "Title (optional)",
                placeholder=f"{PLOT_TYPES[pt]['label']} — {ds_name}",
                key=f"cell_{i}_title",
            )

            cell_cfgs.append(
                {
                    "dataset": ds_name,
                    "plot_type": pt,
                    "x_col": x_col,
                    "y_col": y_col,
                    "cell_title": cell_title,
                }
            )

# ── Generate ──────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("Generate Dashboard", type="primary", key="dash_generate"):
    st.markdown("## Dashboard")
    with st.spinner("Generating…"):
        # 1. Generate all cell figures first (needed for axis sharing + export)
        all_figs: list = []
        all_is_plotly: list = []
        for cfg in cell_cfgs:
            fig, is_plotly = _generate_cell_fig(cfg)
            all_figs.append(fig)
            all_is_plotly.append(is_plotly)

        # 2. Apply axis sharing across matplotlib figures
        _apply_axis_sharing(all_figs, nrows, ncols, sharex_row, sharey_col)

        # 3. Display in grid
        cell_idx = 0
        for row_idx in range(nrows):
            grid_cols = st.columns(ncols)
            for col_idx in range(ncols):
                with grid_cols[col_idx]:
                    cfg = cell_cfgs[cell_idx]
                    label = cfg["cell_title"] or (
                        f"{PLOT_TYPES[cfg['plot_type']]['label']} — {cfg['dataset']}"
                    )
                    st.caption(f"**Cell {cell_idx + 1}** · {label}")
                    fig = all_figs[cell_idx]
                    is_plotly = all_is_plotly[cell_idx]
                    if fig is None:
                        # Fallback: let _render_cell_plot show the error
                        _render_cell_plot(cfg)
                    elif is_plotly:
                        _render_cell_plot(cfg)
                    else:
                        st.pyplot(fig)
                cell_idx += 1

    # 4. Combined export (M16) ────────────────────────────────────────────────
    mpl_figs = [f for f, ip in zip(all_figs, all_is_plotly) if f is not None and not ip]
    if mpl_figs:
        st.markdown("---")
        with st.expander("Export combined figure"):
            import io as _io
            import matplotlib.pyplot as plt

            cell_labels_export = [
                (cfg["cell_title"] or f"{PLOT_TYPES[cfg['plot_type']]['label']} — {cfg['dataset']}")
                for cfg, ip in zip(cell_cfgs, all_is_plotly)
                if not ip and _generate_cell_fig(cfg)[0] is not None
            ]

            _exp_title = st.session_state.get("dash_fig_title", "")

            _ecol1, _ecol2 = st.columns(2)

            # ── Combined PNG (grid image) ────────────────────────────────────
            with _ecol1:
                st.markdown("**Combined PNG (grid image)**")
                _dpi = st.select_slider(
                    "Resolution",
                    options=[150, 200, 300],
                    value=200,
                    key="dash_export_dpi",
                )
                if st.button("Generate combined PNG", key="dash_gen_png"):
                    with st.spinner("Composing grid…"):
                        _combined = _make_combined_figure(
                            all_figs,
                            [
                                cfg["cell_title"]
                                or (f"{PLOT_TYPES[cfg['plot_type']]['label']} — {cfg['dataset']}")
                                for cfg in cell_cfgs
                            ],
                            nrows,
                            ncols,
                            fig_title=_exp_title,
                        )
                        _png_buf = _io.BytesIO()
                        _combined.savefig(
                            _png_buf,
                            format="png",
                            dpi=_dpi,
                            bbox_inches="tight",
                        )
                        plt.close(_combined)
                        _png_buf.seek(0)
                    st.download_button(
                        f"Download PNG ({_dpi} dpi)",
                        data=_png_buf,
                        file_name="dashboard.png",
                        mime="image/png",
                        key="dash_dl_png",
                    )

            # ── Multi-page PDF ───────────────────────────────────────────────
            with _ecol2:
                st.markdown("**Multi-page PDF (one panel per page)**")
                if st.button("Generate PDF", key="dash_gen_pdf"):
                    with st.spinner("Building PDF…"):
                        from matplotlib.backends.backend_pdf import PdfPages

                        _pdf_buf = _io.BytesIO()
                        with PdfPages(_pdf_buf) as _pdf:
                            if _exp_title:
                                _title_fig = plt.figure(figsize=(8, 2))
                                _title_fig.text(
                                    0.5,
                                    0.5,
                                    _exp_title,
                                    ha="center",
                                    va="center",
                                    fontsize=18,
                                    fontweight="bold",
                                )
                                _pdf.savefig(_title_fig, bbox_inches="tight")
                                plt.close(_title_fig)
                            for cfg, fig, is_plotly in zip(cell_cfgs, all_figs, all_is_plotly):
                                if fig is None or is_plotly:
                                    continue
                                cell_lbl = cfg["cell_title"] or (
                                    f"{PLOT_TYPES[cfg['plot_type']]['label']} — {cfg['dataset']}"
                                )
                                fig.suptitle(cell_lbl, fontsize=11)
                                _pdf.savefig(fig, bbox_inches="tight")
                        _pdf_buf.seek(0)
                    st.download_button(
                        "Download PDF",
                        data=_pdf_buf,
                        file_name="dashboard.pdf",
                        mime="application/pdf",
                        key="dash_dl_pdf",
                    )
