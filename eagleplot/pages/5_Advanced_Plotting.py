"""Advanced Plotting Page.

Provides visualizations that go beyond the 13 core eagleplot.plotting functions
by using seaborn, plotly.graph_objects, and matplotlib directly.

Tabs
----
- Correlation Heatmap:    annotated correlation matrix (DataFrame only)
- Overlaid Distributions: histogram or KDE overlay for multiple columns
- Grouped Categorical:    box / violin / swarm with optional hue grouping
- 3D Scatter:             Plotly interactive 3-D scatter with color encoding
- HTML Export:            generate any interactive Plotly chart + download HTML

Each tab is an ``st.fragment`` (roadmap D1.2): its widgets and Generate button rerun
only that tab. The standalone HTML downloads are built when clicked, not up front.
"""

import traceback
from pathlib import Path
import sys

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.plotting import (
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
from eagleplot.utils.figures import download_payload

initialize_session_state()


# ── Constants ─────────────────────────────────────────────────────────────────

_CMAPS_DIV = ["RdBu_r", "coolwarm", "PiYG", "BrBG", "seismic"]
_CMAPS_SEQ = ["viridis", "plasma", "magma", "cividis", "Blues", "Greens"]
_CORR_METHODS = ["pearson", "spearman", "kendall"]

# Interactive plot types available for HTML export
_INTERACTIVE_TYPES = {
    "interactive_histogram": "Interactive Histogram",
    "interactive_scatter": "Interactive Scatter",
    "interactive_line": "Interactive Line",
    "interactive_heatmap": "Interactive Heatmap",
    "interactive_3d_surface": "3D Surface",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _numeric_cols(data) -> list:
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).columns.tolist()
    return []


def _all_cols(data) -> list:
    """All column names (DataFrame) or column indices (2-D array)."""
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    arr = np.asarray(data)
    if arr.ndim == 2:
        return list(range(arr.shape[1]))
    return []


def _to_1d(data, col_ref) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        return data[col_ref].dropna().values.astype(float)
    arr = np.asarray(data, dtype=float)
    if col_ref is None:
        return arr.flatten()
    return arr[:, int(col_ref)]


def _df_required(data) -> bool:
    """Return True and show error if data is not a DataFrame."""
    if not isinstance(data, pd.DataFrame):
        st.warning(
            "This feature requires a **DataFrame** (CSV, Excel, JSON, etc.). "
            "The current dataset is a NumPy array."
        )
        return False
    return True


def _html_download(fig, filename: str, key: str) -> None:
    """Render an HTML download button for a freshly generated Plotly figure.

    The standalone HTML embeds plotly.js (several MB), so it is built only when the
    button is clicked (Streamlit >= 1.52), or once per figure on older versions.
    """
    version = st.session_state.get("adv_html_version", 0) + 1
    st.session_state.adv_html_version = version
    st.download_button(
        label="Download as HTML",
        data=download_payload(
            lambda: fig.to_html(full_html=True, include_plotlyjs=True),
            cache_name="_adv_html_cache",
            version=version,
            key=key,
        ),
        file_name=filename,
        mime="text/html",
        help="Standalone HTML file — opens in any browser without Python.",
        key=key,
        on_click="ignore",
    )


# ── Page layout ───────────────────────────────────────────────────────────────

st.title("Advanced Plotting")
st.caption(
    "Annotated correlation matrices, overlaid distributions, grouped categoricals, "
    "3D scatter, and HTML export."
)

# Guard: need at least one dataset
summary = get_session_summary()
if summary["num_datasets"] == 0:
    st.warning("No datasets loaded. Go to **Data Upload** first.")
    st.stop()

# ── Dataset selector ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Dataset")

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
        key="adv_dataset",
    )
with col2:
    st.write("")
    st.write("")
    st.metric("Loaded", summary["num_datasets"])

if dataset_name:
    st.session_state.current_dataset = dataset_name

data = get_dataset(dataset_name)

if isinstance(data, pd.DataFrame):
    st.caption(f"DataFrame — {data.shape[0]} rows × {data.shape[1]} cols")
elif isinstance(data, np.ndarray):
    st.caption(f"NumPy array — shape {data.shape}, dtype {data.dtype}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _corr_tab(data, dataset_name) -> None:
    """Annotated correlation heatmap. Reruns on its own."""
    st.markdown("### Annotated Correlation Heatmap")

    if _df_required(data):
        num_cols = _numeric_cols(data)
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation matrix.")
        else:
            # Column subset
            selected_corr_cols = st.multiselect(
                "Columns to include",
                num_cols,
                default=num_cols,
                key="corr_cols",
            )
            if len(selected_corr_cols) < 2:
                st.info("Select at least 2 columns.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    corr_method = st.selectbox(
                        "Method",
                        _CORR_METHODS,
                        key="corr_method",
                    )
                with c2:
                    corr_cmap = st.selectbox(
                        "Color map",
                        _CMAPS_DIV,
                        key="corr_cmap",
                    )
                with c3:
                    corr_annot = st.checkbox("Show values", value=True, key="corr_annot")

                figw = st.slider("Figure width", 4, 16, 8, key="corr_figw")

                if st.button("Generate Correlation Heatmap", type="primary", key="corr_run"):
                    try:
                        corr = data[selected_corr_cols].corr(method=corr_method)
                        n = len(selected_corr_cols)
                        fig, ax = plt.subplots(figsize=(figw, max(4, figw * n // 8)))
                        sns.heatmap(
                            corr,
                            annot=corr_annot,
                            fmt=".2f",
                            cmap=corr_cmap,
                            center=0,
                            square=True,
                            linewidths=0.5,
                            ax=ax,
                            vmin=-1,
                            vmax=1,
                        )
                        ax.set_title(
                            f"{corr_method.capitalize()} Correlation Matrix — {dataset_name}",
                            pad=12,
                        )
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        with st.expander("Details"):
                            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Overlaid Distributions
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _dist_tab(data, dataset_name) -> None:
    """Overlaid histograms / KDEs. Reruns on its own."""
    st.markdown("### Overlaid Distributions")
    st.markdown("Compare multiple variables on the same axes.")

    # Build column options for both DataFrame and 2-D array
    if isinstance(data, pd.DataFrame):
        dist_col_options = _numeric_cols(data)
        dist_col_fmt = str
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        dist_col_options = list(range(data.shape[1]))
        dist_col_fmt = lambda i: f"Column {i}"  # noqa: E731
    else:
        dist_col_options = []
        dist_col_fmt = str

    if len(dist_col_options) < 2:
        st.warning("Need at least 2 columns. Load a DataFrame or a 2-D NumPy array.")
    else:
        dist_cols_sel = st.multiselect(
            "Select columns to overlay",
            dist_col_options,
            default=dist_col_options[: min(4, len(dist_col_options))],
            format_func=dist_col_fmt,
            key="dist_cols",
        )

        if dist_cols_sel:
            c1, c2, c3 = st.columns(3)
            with c1:
                dist_kind = st.radio(
                    "Type",
                    ["Histogram", "KDE"],
                    horizontal=True,
                    key="dist_kind",
                )
            with c2:
                dist_density = st.checkbox("Normalize (density)", value=True, key="dist_density")
            with c3:
                dist_bins = (
                    st.slider("Bins", 5, 100, 20, key="dist_bins")
                    if dist_kind == "Histogram"
                    else None
                )

            if st.button("Generate Overlaid Distributions", type="primary", key="dist_run"):
                try:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    palette = sns.color_palette("tab10", n_colors=len(dist_cols_sel))

                    for col, color in zip(dist_cols_sel, palette):
                        arr = _to_1d(data, col)
                        label = str(col) if not isinstance(col, int) else f"Column {col}"
                        if dist_kind == "Histogram":
                            ax.hist(
                                arr,
                                bins=dist_bins,
                                density=dist_density,
                                alpha=0.55,
                                color=color,
                                label=label,
                                edgecolor="white",
                                linewidth=0.5,
                            )
                        else:
                            sns.kdeplot(
                                arr,
                                ax=ax,
                                fill=True,
                                alpha=0.35,
                                color=color,
                                label=label,
                                linewidth=2,
                            )

                    ax.set_ylabel("Density" if dist_density else "Count")
                    ax.set_title(f"Overlaid {dist_kind}s — {dataset_name}", pad=10)
                    ax.legend(framealpha=0.8)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Grouped Categorical Plot
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _cat_tab(data, dataset_name) -> None:
    """Box / violin / swarm with optional hue. Reruns on its own."""
    st.markdown("### Grouped Categorical Plot")
    st.markdown(
        "Box, violin, or swarm plot with optional colour grouping (hue). Requires a **DataFrame**."
    )

    if _df_required(data):
        all_df_cols = list(data.columns)
        num_df_cols = _numeric_cols(data)

        if not num_df_cols:
            st.warning("No numeric columns available.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                cat_kind = st.selectbox(
                    "Plot kind",
                    ["Box", "Violin", "Swarm"],
                    key="cat_kind",
                )
            with c2:
                cat_palette = st.selectbox(
                    "Palette",
                    ["tab10", "Set2", "Set3", "pastel", "muted", "deep"],
                    key="cat_palette",
                )

            c1, c2, c3 = st.columns(3)
            with c1:
                cat_x = st.selectbox(
                    "X axis (grouping)",
                    [None] + all_df_cols,
                    format_func=lambda v: "— (none)" if v is None else str(v),
                    key="cat_x",
                )
            with c2:
                cat_y = st.selectbox("Y axis (numeric)", num_df_cols, key="cat_y")
            with c3:
                cat_hue = st.selectbox(
                    "Hue (optional)",
                    [None] + all_df_cols,
                    format_func=lambda v: "— (none)" if v is None else str(v),
                    key="cat_hue",
                )

            if cat_kind == "Swarm" and len(data) > 500:
                st.warning(
                    f"Dataset has {len(data):,} rows — swarm plot may be slow. "
                    "Consider filtering to ≤ 500 rows first."
                )

            if st.button("Generate Categorical Plot", type="primary", key="cat_run"):
                try:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    shared_kw = dict(
                        data=data,
                        x=cat_x,
                        y=cat_y,
                        hue=cat_hue,
                        palette=cat_palette,
                        ax=ax,
                    )
                    if cat_kind == "Box":
                        sns.boxplot(**shared_kw)
                    elif cat_kind == "Violin":
                        sns.violinplot(**shared_kw)
                    else:
                        sns.swarmplot(**shared_kw, size=4)

                    title_parts = [cat_kind, "plot of", cat_y]
                    if cat_x:
                        title_parts += ["by", cat_x]
                    if cat_hue:
                        title_parts += ["(hue:", cat_hue + ")"]
                    ax.set_title(" ".join(title_parts), pad=10)
                    if cat_hue:
                        ax.legend(title=cat_hue, bbox_to_anchor=(1.01, 1), loc="upper left")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — 3D Scatter (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _scatter3d_tab(data, dataset_name) -> None:
    """Plotly 3-D scatter. Reruns on its own."""
    st.markdown("### Interactive 3D Scatter Plot")

    # Build column options
    if isinstance(data, pd.DataFrame):
        scatter3d_options = _numeric_cols(data)
        scatter3d_fmt = str
        all_scatter_opts = list(data.columns)
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
        scatter3d_options = list(range(data.shape[1]))
        scatter3d_fmt = lambda i: f"Column {i}"  # noqa: E731
        all_scatter_opts = scatter3d_options
    else:
        scatter3d_options = []
        scatter3d_fmt = str
        all_scatter_opts = []

    if len(scatter3d_options) < 3:
        st.warning(
            "3D scatter requires at least 3 numeric columns. "
            "Load a DataFrame or a 2-D NumPy array with ≥ 3 columns."
        )
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            s3d_x = st.selectbox(
                "X axis",
                scatter3d_options,
                index=0,
                format_func=scatter3d_fmt,
                key="s3d_x",
            )
        with c2:
            s3d_y = st.selectbox(
                "Y axis",
                scatter3d_options,
                index=min(1, len(scatter3d_options) - 1),
                format_func=scatter3d_fmt,
                key="s3d_y",
            )
        with c3:
            s3d_z = st.selectbox(
                "Z axis",
                scatter3d_options,
                index=min(2, len(scatter3d_options) - 1),
                format_func=scatter3d_fmt,
                key="s3d_z",
            )

        c1, c2 = st.columns(2)
        with c1:
            s3d_color = st.selectbox(
                "Color encoding (optional)",
                [None] + list(all_scatter_opts),
                format_func=lambda v: (
                    "— (none)"
                    if v is None
                    else (str(v) if not isinstance(v, int) else f"Column {v}")
                ),
                key="s3d_color",
            )
        with c2:
            s3d_colorscale = st.selectbox(
                "Color scale",
                ["Viridis", "Plasma", "Cividis", "RdBu", "Turbo"],
                key="s3d_colorscale",
            )

        c1, c2 = st.columns(2)
        with c1:
            s3d_size = st.slider("Marker size", 2, 20, 5, key="s3d_size")
        with c2:
            s3d_opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key="s3d_opacity")

        s3d_title = st.text_input("Title (optional)", key="s3d_title", placeholder="3D Scatter")

        if st.button("Generate 3D Scatter", type="primary", key="s3d_run"):
            try:
                x_arr = _to_1d(data, s3d_x)
                y_arr = _to_1d(data, s3d_y)
                z_arr = _to_1d(data, s3d_z)

                color_arr = _to_1d(data, s3d_color) if s3d_color is not None else None

                x_lbl = str(s3d_x) if not isinstance(s3d_x, int) else f"Column {s3d_x}"
                y_lbl = str(s3d_y) if not isinstance(s3d_y, int) else f"Column {s3d_y}"
                z_lbl = str(s3d_z) if not isinstance(s3d_z, int) else f"Column {s3d_z}"

                marker = dict(
                    size=s3d_size,
                    opacity=s3d_opacity,
                    colorscale=s3d_colorscale,
                )
                if color_arr is not None:
                    marker["color"] = color_arr
                    marker["showscale"] = True
                    c_lbl = (
                        str(s3d_color) if not isinstance(s3d_color, int) else f"Column {s3d_color}"
                    )
                    marker["colorbar"] = dict(title=c_lbl)
                else:
                    marker["color"] = "#636EFA"

                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=x_arr,
                            y=y_arr,
                            z=z_arr,
                            mode="markers",
                            marker=marker,
                        )
                    ]
                )
                fig.update_layout(
                    title=s3d_title or f"3D Scatter — {dataset_name}",
                    scene=dict(
                        xaxis_title=x_lbl,
                        yaxis_title=y_lbl,
                        zaxis_title=z_lbl,
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, width="stretch")
                _html_download(fig, filename="3d_scatter.html", key="s3d_dl_html")

            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — HTML Export
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _html_export_tab(data, dataset_name) -> None:
    """Any interactive Plotly chart, downloadable as HTML. Reruns on its own."""
    st.markdown("### Export Interactive Plot as HTML")
    st.markdown(
        "Generate any Plotly interactive chart and download it as a "
        "**standalone HTML file** that works in any browser — no Python required."
    )

    # Build column options (reused from the current dataset)
    if isinstance(data, pd.DataFrame):
        exp_num_cols = _numeric_cols(data)
        exp_col_fmt = str
    elif isinstance(data, np.ndarray) and data.ndim >= 2:
        exp_num_cols = list(range(data.shape[1]))
        exp_col_fmt = lambda i: f"Column {i}"  # noqa: E731
    else:
        exp_num_cols = []
        exp_col_fmt = str

    exp_type = st.selectbox(
        "Interactive plot type",
        list(_INTERACTIVE_TYPES.keys()),
        format_func=lambda k: _INTERACTIVE_TYPES[k],
        key="exp_type",
    )

    exp_title = st.text_input(
        "Plot title", key="exp_title", placeholder=_INTERACTIVE_TYPES[exp_type]
    )

    # Column selectors — shown conditionally
    exp_x = exp_y = exp_data_col = None
    _1d_types = {"interactive_histogram"}
    _xy_types = {"interactive_scatter", "interactive_line"}
    _mat_types = {"interactive_heatmap", "interactive_3d_surface"}

    if exp_type in _1d_types:
        if exp_num_cols:
            exp_data_col = st.selectbox(
                "Data column",
                exp_num_cols,
                format_func=exp_col_fmt,
                key="exp_dcol",
            )
    elif exp_type in _xy_types:
        if len(exp_num_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                exp_x = st.selectbox(
                    "X column",
                    exp_num_cols,
                    format_func=exp_col_fmt,
                    key="exp_xcol",
                )
            with c2:
                exp_y = st.selectbox(
                    "Y column",
                    exp_num_cols,
                    index=min(1, len(exp_num_cols) - 1),
                    format_func=exp_col_fmt,
                    key="exp_ycol",
                )
    else:
        st.caption("Uses the full dataset matrix (no column selection needed).")

    if st.button("Generate & Export", type="primary", key="exp_run"):
        if not exp_num_cols and exp_type not in _mat_types:
            st.error("No numeric columns available for this dataset.")
        else:
            try:
                title = exp_title or _INTERACTIVE_TYPES[exp_type]

                if exp_type == "interactive_histogram":
                    arr = (
                        _to_1d(data, exp_data_col)
                        if exp_data_col is not None
                        else (np.asarray(data, dtype=float).flatten())
                    )
                    fig = interactive_histogram(arr, title=title)

                elif exp_type == "interactive_scatter":
                    x = _to_1d(data, exp_x) if exp_x is not None else np.array([])
                    y = _to_1d(data, exp_y) if exp_y is not None else np.array([])
                    fig = interactive_scatter(
                        x,
                        y,
                        xlabel=str(exp_x or "x"),
                        ylabel=str(exp_y or "y"),
                        title=title,
                    )

                elif exp_type == "interactive_line":
                    if exp_x is not None and exp_y is not None:
                        x = _to_1d(data, exp_x)
                        y = [_to_1d(data, exp_y)]
                        lbl = [str(exp_y)]
                    else:
                        arr = np.asarray(data, dtype=float).flatten()
                        x = np.arange(len(arr))
                        y = [arr]
                        lbl = ["Data"]
                    fig = interactive_line(x, y, labels=lbl, title=title)

                elif exp_type == "interactive_heatmap":
                    if isinstance(data, pd.DataFrame):
                        cols = _numeric_cols(data)
                        mat = data[cols].values.astype(float)
                    else:
                        mat = np.asarray(data, dtype=float)
                    if mat.ndim != 2:
                        st.error("Heatmap requires a 2-D array.")
                        return
                    fig = interactive_heatmap(mat, title=title)

                else:  # interactive_3d_surface
                    Z = np.asarray(data, dtype=float)
                    if Z.ndim != 2:
                        st.error("3D surface requires a 2-D array.")
                        return
                    nr, nc = Z.shape
                    X, Y = np.meshgrid(np.arange(nc), np.arange(nr))
                    fig = interactive_3d_surface(X, Y, Z, title=title)

                st.plotly_chart(fig, width="stretch")
                _html_download(fig, filename=f"{exp_type}.html", key="exp_dl_html")
                st.success("Plot ready — click the button above to download.")

            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    with st.expander("About HTML export"):
        st.markdown("""
The downloaded HTML file:
- Contains the full Plotly chart with zoom, pan, hover, and download toolbar
- Includes the Plotly.js library (~3 MB) — no internet connection needed to view
- Can be embedded in HTML reports or shared with collaborators who don't have Python
- Opens in Chrome, Firefox, Safari, and Edge
""")


# ── Tabs ─────────────────────────────────────────────────────────────────────
# Each tab is an st.fragment: a widget or Generate button in one tab reruns only that
# tab, and the other tabs keep what they show. Roadmap D1.2.
st.markdown("---")

tab_corr, tab_dist, tab_cat, tab_3d, tab_export = st.tabs(
    [
        "Correlation Heatmap",
        "Overlaid Distributions",
        "Grouped Categorical",
        "3D Scatter",
        "HTML Export",
    ]
)

with tab_corr:
    _corr_tab(data, dataset_name)
with tab_dist:
    _dist_tab(data, dataset_name)
with tab_cat:
    _cat_tab(data, dataset_name)
with tab_3d:
    _scatter3d_tab(data, dataset_name)
with tab_export:
    _html_export_tab(data, dataset_name)
