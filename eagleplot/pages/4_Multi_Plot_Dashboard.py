"""Multi-Plot Dashboard Page.

Configure a grid of independent plots (up to 4 × 4 = 16 cells) for
side-by-side visual comparison across datasets or plot types.
Each cell has its own dataset, plot type, and column selection.

Performance (roadmap D1.2): every cell is an ``st.fragment`` that holds its own
settings and plot, so changing one cell reruns and redraws only that cell. Cell
figures are kept in ``st.session_state`` keyed by the cell's configuration and are
rebuilt only when that configuration changes. The combined PNG and the multi-page
PDF are built only when downloaded (or, on Streamlit < 1.52, when "Generate" is
clicked), never on a plain rerun.
"""

import io
import pickle
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
from eagleplot.utils import figures as _figures
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

# Session-state keys
_CELLS = "dash_cells"  # cell index -> cached entry (see _build_entry)
_VERSION = "dash_fig_version"  # bumped whenever a displayed figure changes
_EXPORT_CACHE = "_dash_export_cache"  # prepared downloads (Streamlit < 1.52 only)
_FULL_RUN = "_dash_full_run"  # True while the whole page (not one fragment) runs
_FIX_PASS = "_dash_axis_fix_pass"  # guards the axis-sharing follow-up rerun


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


def _cell_label(cfg: dict) -> str:
    return cfg["cell_title"] or f"{PLOT_TYPES[cfg['plot_type']]['label']} — {cfg['dataset']}"


class _CellError(Exception):
    """The data can't be drawn as the chosen plot type (shown without a traceback)."""


def _build_cell_fig(cfg: dict, data):
    """Return ``(fig, is_plotly)`` for a cell config. Raises on error."""
    pt = cfg["plot_type"]
    x_col = cfg.get("x_col")
    y_col = cfg.get("y_col")
    title = _cell_label(cfg)

    # ── Histogram ────────────────────────────────────────────────────────────
    if pt == "histogram":
        arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
        fig, _, __ = histogram(arr, title=title, figsize=_CELL_FIGSIZE)
        return fig, False

    # ── Line Plot ────────────────────────────────────────────────────────────
    if pt in ("line_plot", "interactive_line"):
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
        if pt == "line_plot":
            fig, _ = line_plot(x, y, labels=lbl, xlabel=xlab, title=title, figsize=_CELL_FIGSIZE)
            return fig, False
        return interactive_line(x, y, labels=lbl, xlabel=xlab, title=title), True

    # ── Scatter Plot ─────────────────────────────────────────────────────────
    if pt in ("scatter_plot", "interactive_scatter"):
        x = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
        y = _to_1d(data, y_col) if y_col is not None else _auto_1d(data)
        if pt == "scatter_plot":
            fig, _ = scatter_plot(
                x,
                y,
                xlabel=str(x_col or "x"),
                ylabel=str(y_col or "y"),
                title=title,
                figsize=_CELL_FIGSIZE,
            )
            return fig, False
        fig = interactive_scatter(
            x, y, xlabel=str(x_col or "x"), ylabel=str(y_col or "y"), title=title
        )
        return fig, True

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    if pt in ("heatmap", "interactive_heatmap"):
        if isinstance(data, pd.DataFrame):
            cols = _numeric_cols(data)
            mat = data[cols].values.astype(float)
        else:
            mat = np.asarray(data, dtype=float)
        if mat.ndim != 2:
            if pt == "heatmap":
                raise _CellError("Heatmap requires a 2-D array.")
            raise _CellError("Interactive heatmap requires a 2-D array.")
        if pt == "heatmap":
            fig, _ = heatmap(mat, title=title, figsize=_CELL_FIGSIZE)
            return fig, False
        return interactive_heatmap(mat, title=title), True

    # ── Contour Plot / 3D Surface ────────────────────────────────────────────
    if pt in ("contour_plot", "interactive_3d_surface"):
        Z = np.asarray(data, dtype=float)
        if Z.ndim != 2:
            if pt == "contour_plot":
                raise _CellError("Contour plot requires a 2-D array.")
            raise _CellError("3D Surface requires a 2-D array.")
        nrows_z, ncols_z = Z.shape
        X, Y = np.meshgrid(np.arange(ncols_z), np.arange(nrows_z))
        if pt == "contour_plot":
            fig, _ = contour_plot(X, Y, Z, title=title, figsize=_CELL_FIGSIZE)
            return fig, False
        return interactive_3d_surface(X, Y, Z, title=title), True

    # ── Distribution Plot (Seaborn) ───────────────────────────────────────────
    if pt == "distribution_plot":
        arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
        fig, _ = distribution_plot(arr, title=title)
        return fig, False

    # ── Box Plot (Seaborn) ────────────────────────────────────────────────────
    if pt == "box_plot":
        if isinstance(data, pd.DataFrame):
            fig, _ = box_plot(data, title=title)
        else:
            fig, _ = box_plot(np.asarray(data, dtype=float).flatten(), title=title)
        return fig, False

    # ── Regression Plot (Seaborn) ─────────────────────────────────────────────
    if pt == "regression_plot":
        if isinstance(data, pd.DataFrame) and isinstance(x_col, str):
            fig, _ = regression_plot(x=x_col, y=y_col, data=data, title=title)
        else:
            x = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
            y = _to_1d(data, y_col) if y_col is not None else _auto_1d(data)
            fig, _ = regression_plot(x=x, y=y, title=title)
        return fig, False

    # ── Interactive Histogram (Plotly) ────────────────────────────────────────
    if pt == "interactive_histogram":
        arr = _to_1d(data, x_col) if x_col is not None else _auto_1d(data)
        return interactive_histogram(arr, title=title), True

    raise _CellError(f"Unhandled plot type: {pt}")


# ── Cell cache ────────────────────────────────────────────────────────────────


def _cells() -> dict:
    """The per-cell cache. Always the same dict object (cleared in place, never replaced),
    so on-click download builders can read the current figures from it."""
    return st.session_state.setdefault(_CELLS, {})


def _bump_version() -> None:
    st.session_state[_VERSION] = st.session_state.get(_VERSION, 0) + 1


def _exportable(entry) -> bool:
    """A Matplotlib panel without an error (Plotly panels are not in the exports)."""
    return entry is not None and entry["fig"] is not None and not entry["is_plotly"]


def _release(entry) -> None:
    """Close a replaced Matplotlib figure unless plot history still references it."""
    if not _exportable(entry):
        return
    fig = entry["fig"]
    if any(h.get("figure") is fig for h in st.session_state.get("plot_history", [])):
        return
    import matplotlib.pyplot as plt

    plt.close(fig)


def _clear_cells() -> None:
    cells = _cells()
    for entry in cells.values():
        _release(entry)
    cells.clear()


def _build_entry(cfg: dict, data, key: tuple) -> dict:
    """Draw one cell. Returns the cache entry (figure, or the error to show)."""
    import matplotlib.pyplot as plt

    entry = {
        "key": key,
        "label": _cell_label(cfg),
        "fig": None,
        "is_plotly": False,
        "error": None,
        "trace": None,
        "base": None,  # the figure's own (unshared) axis limits
        "shown": None,  # the limits it was last displayed with
        "serial": None,  # unique per built entry: keys the cached display PNG
    }
    if data is None:
        entry["error"] = f"Dataset '{cfg['dataset']}' not found."
        return entry
    open_before = set(plt.get_fignums())
    try:
        fig, is_plotly = _build_cell_fig(cfg, data)
    except _CellError as exc:
        entry["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 — shown in the cell
        entry["error"] = f"Error: {exc}"
        entry["trace"] = traceback.format_exc()
    else:
        entry["fig"], entry["is_plotly"] = fig, is_plotly
        if not is_plotly and fig.axes:
            ax = fig.axes[0]
            entry["base"] = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
    # Detach new figures from pyplot: the cached Figure objects stay drawable, but pyplot
    # no longer accumulates them (no "too many open figures") and dropped ones are freed.
    for num in set(plt.get_fignums()) - open_before:
        plt.close(num)
    return entry


def _cell_entry(i: int, cfg: dict):
    """Return ``(entry, rebuilt)``: the cached cell, rebuilt only if its config changed."""
    data = get_dataset(cfg["dataset"])
    key = (
        cfg["dataset"],
        id(data),
        cfg["plot_type"],
        cfg["x_col"],
        cfg["y_col"],
        cfg["cell_title"],
    )
    cells = _cells()
    old = cells.get(i)
    if old is not None and old["key"] == key:
        return old, False
    with st.spinner("Drawing…"):
        entry = _build_entry(cfg, data, key)
    _release(old)
    cells[i] = entry
    _bump_version()
    entry["serial"] = st.session_state[_VERSION]  # never reused within the session
    return entry, True


def _target_limits(i: int, nrows: int, ncols: int, sharex: bool, sharey: bool):
    """Axis limits cell *i* should show: its own, or synced across its row / column.

    Same rule as before D1.2: x limits are unioned over the Matplotlib panels in the
    row, y limits over those in the column, when at least two panels take part.
    """
    cells = _cells()
    entry = cells.get(i)
    if entry is None or entry["base"] is None:
        return None
    xlim, ylim = entry["base"]
    row, col = divmod(i, ncols)
    if sharex:
        bases = [cells.get(row * ncols + c) for c in range(ncols)]
        bases = [e["base"] for e in bases if e is not None and e["base"] is not None]
        if len(bases) >= 2:
            xlim = (min(b[0][0] for b in bases), max(b[0][1] for b in bases))
    if sharey:
        bases = [cells.get(r * ncols + col) for r in range(nrows)]
        bases = [e["base"] for e in bases if e is not None and e["base"] is not None]
        if len(bases) >= 2:
            ylim = (min(b[1][0] for b in bases), max(b[1][1] for b in bases))
    return (xlim, ylim)


def _stale_axes(n_cells: int, nrows: int, ncols: int, sharex: bool, sharey: bool) -> bool:
    """Whether any displayed panel no longer has the axis limits sharing calls for."""
    if not (sharex or sharey):
        return False
    cells = _cells()
    for i in range(n_cells):
        entry = cells.get(i)
        if entry is None or entry["base"] is None or entry["shown"] is None:
            continue
        if entry["shown"] != _target_limits(i, nrows, ncols, sharex, sharey):
            return True
    return False


def _show_cell(i: int, entry: dict, nrows: int, ncols: int, sharex: bool, sharey: bool):
    if entry["error"]:
        st.error(entry["error"])
        if entry["trace"]:
            with st.expander("Details"):
                st.code(entry["trace"])
    elif entry["is_plotly"]:
        _figures.show_figure(entry["fig"], key=f"dash_plotly_{i}")
    else:
        fig = entry["fig"]
        lims = _target_limits(i, nrows, ncols, sharex, sharey)
        if lims is not None:
            fig.axes[0].set_xlim(*lims[0])
            fig.axes[0].set_ylim(*lims[1])
            if lims != entry["shown"]:
                entry["shown"] = lims
                _bump_version()
        # The display PNG is encoded once per built figure and axis limits, so a page
        # rerun does not re-encode unchanged panels (D2.3). New limits invalidate it.
        _figures.show_figure(
            fig,
            alt=entry["label"],
            cache_key=f"dash_cell_{i}",
            version=(entry["serial"], entry["shown"]),
        )


# ── Combined export (M16) ─────────────────────────────────────────────────────


def _make_combined_figure(
    cell_figs: list,
    cell_labels: list,
    nrows: int,
    ncols: int,
    fig_title: str = "",
):
    """Create a combined grid matplotlib figure from individual cell figures.

    Each cell is rendered to an RGBA image and placed into a subplot using
    ``imshow``.  Plotly cells (None entries) are left blank.  The figure is not
    registered with pyplot, so it is safe to build outside the script thread.
    """
    from matplotlib.figure import Figure
    from matplotlib.image import imread

    combined_w = ncols * 5.0
    combined_h = nrows * 4.0 + (0.6 if fig_title else 0.0)
    combined_fig = Figure(figsize=(combined_w, combined_h))
    axes_grid = combined_fig.subplots(nrows, ncols, squeeze=False)

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
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = imread(buf)

        ax.imshow(img)
        ax.axis("off")
        if lbl:
            ax.set_title(lbl, fontsize=9, pad=3)

    if fig_title:
        combined_fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.01)

    combined_fig.tight_layout()
    return combined_fig


def _export_snapshot(cells: dict, n_cells: int):
    """(figures, labels) of the current cells; None for Plotly / failed panels."""
    entries = [cells.get(i) for i in range(n_cells)]
    figs = [e["fig"] if _exportable(e) else None for e in entries]
    labels = [e["label"] if e is not None else "" for e in entries]
    return figs, labels


def _combined_png(cells: dict, n_cells: int, nrows: int, ncols: int, title: str, dpi: int):
    figs, labels = _export_snapshot(cells, n_cells)
    combined = _make_combined_figure(figs, labels, nrows, ncols, fig_title=title)
    buf = io.BytesIO()
    combined.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def _report_pdf(cells: dict, n_cells: int, title: str) -> bytes:
    """Multi-page PDF: optional title page, then one Matplotlib panel per page."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    figs, labels = _export_snapshot(cells, n_cells)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        if title:
            title_fig = Figure(figsize=(8, 2))
            title_fig.text(
                0.5, 0.5, title, ha="center", va="center", fontsize=18, fontweight="bold"
            )
            pdf.savefig(title_fig, bbox_inches="tight")
        for fig, label in zip(figs, labels):
            if fig is None:
                continue
            # The page title goes on a copy: the displayed panel stays as it is.
            try:
                page = pickle.loads(pickle.dumps(fig))
            except Exception:  # noqa: BLE001 — unpicklable artist: page without title
                page = None
            if page is not None:
                page.suptitle(label, fontsize=11)
                pdf.savefig(page, bbox_inches="tight")
            else:
                pdf.savefig(fig, bbox_inches="tight")
    return buf.getvalue()


def _export_button(
    label: str,
    make,
    *,
    key: str,
    version,
    file_name: str,
    mime: str,
    prepare_label: str,
    prepare_key: str,
) -> None:
    """A download whose payload is built only when wanted (page-local helper).

    Streamlit >= 1.52 builds it when the button is clicked. Older versions need the
    bytes up front, and building the combined PNG and the PDF on every change would
    be expensive, so there a *prepare_label* button builds them once per *version*.
    (``figures.download_payload`` would instead build both whenever a panel changes.)
    """
    if _figures.LAZY_DOWNLOAD:
        st.download_button(
            label, data=make, file_name=file_name, mime=mime, key=key, on_click="ignore"
        )
        return
    cache = st.session_state.setdefault(_EXPORT_CACHE, {})
    hit = cache.get(key)
    if hit is None or hit[0] != version:
        if not st.button(prepare_label, key=prepare_key):
            return
        with st.spinner("Building…"):
            hit = (version, make())
        cache[key] = hit
    st.download_button(
        label, data=hit[1], file_name=file_name, mime=mime, key=key, on_click="ignore"
    )


# ── Fragments ─────────────────────────────────────────────────────────────────


@st.fragment
def _cell_panel(i: int, nrows: int, ncols: int, sharex: bool, sharey: bool, names: list):
    """One dashboard cell: its settings and its plot. Reruns on its own."""
    active = bool(st.session_state.get("dash_active"))
    with st.container(border=True):
        with st.expander(f"**Cell {i + 1}** settings", expanded=not active):
            # Dataset selector — default cycles through loaded datasets
            ds_name = st.selectbox(
                "Dataset",
                names,
                index=min(i, len(names) - 1),
                key=f"cell_{i}_ds",
            )
            cell_data = get_dataset(ds_name)

            pt = st.selectbox(
                "Plot type",
                list(PLOT_TYPES.keys()),
                format_func=lambda k: f"{PLOT_TYPES[k]['label']} ({PLOT_TYPES[k]['category']})",
                key=f"cell_{i}_pt",
            )

            # Column selectors (context-aware)
            x_col, y_col = _cell_col_widgets(i, pt, cell_data)

            cell_title = st.text_input(
                "Title (optional)",
                placeholder=f"{PLOT_TYPES[pt]['label']} — {ds_name}",
                key=f"cell_{i}_title",
            )

        cfg = {
            "dataset": ds_name,
            "plot_type": pt,
            "x_col": x_col,
            "y_col": y_col,
            "cell_title": cell_title,
        }
        if not active:
            return

        old = _cells().get(i)
        was_exportable = _exportable(old)
        entry, rebuilt = _cell_entry(i, cfg)
        st.caption(f"**Cell {i + 1}** · {entry['label']}")
        _show_cell(i, entry, nrows, ncols, sharex, sharey)

    if rebuilt and not st.session_state.get(_FULL_RUN):
        # Only this cell reran. Rerun the page if the change reaches beyond it: shared
        # axis limits of other panels, the export panel's contents, or (Streamlit < 1.52)
        # a prepared download that is now out of date.
        if (
            _stale_axes(nrows * ncols, nrows, ncols, sharex, sharey)
            or was_exportable != _exportable(entry)
            or (not _figures.LAZY_DOWNLOAD and st.session_state.get(_EXPORT_CACHE))
        ):
            st.rerun()


@st.fragment
def _export_panel(nrows: int, ncols: int, layout_name: str) -> None:
    """Combined-figure export. Reruns on its own; builds nothing until asked."""
    n_cells = nrows * ncols
    cells = _cells()
    if not any(_exportable(cells.get(i)) for i in range(n_cells)):
        return
    st.markdown("---")
    with st.expander("Export combined figure"):
        fig_title = st.text_input(
            "Figure title (optional — shown on combined export)",
            key="dash_fig_title",
        )
        version = (st.session_state.get(_VERSION, 0), layout_name, fig_title)

        _ecol1, _ecol2 = st.columns(2)

        # ── Combined PNG (grid image) ────────────────────────────────────────
        with _ecol1:
            st.markdown("**Combined PNG (grid image)**")
            dpi = st.select_slider(
                "Resolution",
                options=[150, 200, 300],
                value=200,
                key="dash_export_dpi",
            )
            _export_button(
                f"Download PNG ({dpi} dpi)",
                lambda: _combined_png(cells, n_cells, nrows, ncols, fig_title, dpi),
                key="dash_dl_png",
                version=version + (dpi,),
                file_name="dashboard.png",
                mime="image/png",
                prepare_label="Generate combined PNG",
                prepare_key="dash_gen_png",
            )

        # ── Multi-page PDF ───────────────────────────────────────────────────
        with _ecol2:
            st.markdown("**Multi-page PDF (one panel per page)**")
            _export_button(
                "Download PDF",
                lambda: _report_pdf(cells, n_cells, fig_title),
                key="dash_dl_pdf",
                version=version,
                file_name="dashboard.pdf",
                mime="application/pdf",
                prepare_label="Generate PDF",
                prepare_key="dash_gen_pdf",
            )
        st.caption("Plotly panels are left out of the combined PNG and the PDF.")


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
    st.caption("The combined-export figure title is set in **Export combined figure**.")

# ── Dashboard ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Dashboard")
st.markdown(
    "Configure each cell (numbered left-to-right, top-to-bottom), then click "
    "**Generate Dashboard**. After that, a cell redraws on its own when you change "
    "its settings."
)

if st.button("Generate Dashboard", type="primary", key="dash_generate"):
    st.session_state.dash_active = True
    _clear_cells()  # draw every cell afresh
    _bump_version()

dataset_names = list(st.session_state.datasets.keys())

st.session_state[_FULL_RUN] = True
try:
    for _row in range(nrows):
        _grid_cols = st.columns(ncols)
        for _col in range(ncols):
            with _grid_cols[_col]:
                _cell_panel(
                    _row * ncols + _col, nrows, ncols, sharex_row, sharey_col, dataset_names
                )
finally:
    st.session_state[_FULL_RUN] = False

if st.session_state.get("dash_active"):
    # A panel drawn early in this run may have used a neighbour's old axis limits.
    if _stale_axes(n_cells, nrows, ncols, sharex_row, sharey_col) and not st.session_state.get(
        _FIX_PASS
    ):
        st.session_state[_FIX_PASS] = True
        st.rerun()
    st.session_state[_FIX_PASS] = False
    _export_panel(nrows, ncols, layout_choice)
