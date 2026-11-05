"""Dynamic plot configuration widgets for Streamlit GUI.

Provides widget factories that render Streamlit controls for each plot type
and return configuration dictionaries ready to pass to plotting functions.

Main API
--------
get_plot_config_widgets(plot_type, data) -> dict
    Render Streamlit widgets for the given plot type and dataset.
    Returns a config dict with plot parameters and ``_``-prefixed
    data-selection hints (column names or array indices).

get_plot_kwargs(config) -> dict
    Strip ``_``-prefixed keys and None values from a config dict so the
    result can be unpacked directly into a plotting function.

PLOT_TYPES : dict
    Ordered catalogue of all supported plot types with display labels
    and library categories.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

# ── Plot type catalogue ──────────────────────────────────────────────────────

PLOT_TYPES: Dict[str, Dict[str, str]] = {
    # Matplotlib (static)
    "histogram": {"label": "Histogram", "category": "Matplotlib"},
    "line_plot": {"label": "Line Plot", "category": "Matplotlib"},
    "scatter_plot": {"label": "Scatter Plot", "category": "Matplotlib"},
    "bar_chart": {"label": "Bar Chart", "category": "Matplotlib"},
    "heatmap": {"label": "Heatmap", "category": "Matplotlib"},
    "contour_plot": {"label": "Contour Plot", "category": "Matplotlib"},
    "waterfall_plot": {"label": "Waterfall Plot", "category": "Matplotlib"},
    "dual_axis_plot": {"label": "Dual Y-Axis Plot", "category": "Matplotlib"},
    "broken_axis_plot": {"label": "Broken Axis Plot", "category": "Matplotlib"},
    "z_colored_scatter": {
        "label": "Z-Colored Scatter",
        "category": "Matplotlib",
    },
    "bubble_chart": {"label": "Bubble Chart", "category": "Matplotlib"},
    "polar_plot": {"label": "Polar Plot", "category": "Matplotlib"},
    "histogram_2d": {"label": "2D Histogram", "category": "Matplotlib"},
    # Seaborn (statistical)
    "distribution_plot": {"label": "Distribution", "category": "Seaborn"},
    "box_plot": {"label": "Box / Violin Plot", "category": "Seaborn"},
    "regression_plot": {"label": "Regression Plot", "category": "Seaborn"},
    "pair_plot": {"label": "Pair Plot", "category": "Seaborn"},
    # Plotly (interactive)
    "interactive_histogram": {
        "label": "Interactive Histogram",
        "category": "Plotly",
    },
    "interactive_scatter": {"label": "Interactive Scatter", "category": "Plotly"},
    "interactive_line": {"label": "Interactive Line", "category": "Plotly"},
    "interactive_heatmap": {"label": "Interactive Heatmap", "category": "Plotly"},
    "interactive_3d_surface": {"label": "3D Surface Plot", "category": "Plotly"},
    "interactive_3d_scatter": {"label": "3D Scatter Plot", "category": "Plotly"},
    # M backlog — new plot types
    "scatter_with_regression": {"label": "Scatter + Regression", "category": "Matplotlib"},
    "residual_plot": {"label": "Residual Plot", "category": "Matplotlib"},
    "interactive_ternary": {"label": "Ternary Plot", "category": "Plotly"},
    # Specialty
    "inset_plot": {"label": "Inset Plot", "category": "Specialty"},
}


# ── Internal helpers ─────────────────────────────────────────────────────────

# Named discrete color palettes available in the GUI.
# Each entry: (display_label, list_of_hex_or_named_colors)
COLOR_PALETTES: Dict[str, List[str]] = {
    "Default": [
        "steelblue",
        "tomato",
        "seagreen",
        "darkorange",
        "mediumpurple",
        "crimson",
        "teal",
        "slategray",
    ],
    # Wong (2011) — 8-color, safe for ~8% deuteranopia/protanopia population
    "Color-Blind Safe (Wong)": [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ],
    # Okabe & Ito — widely recommended by Nature/Science style guides
    "Color-Blind Safe (Okabe-Ito)": [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ],
    # Paul Tol muted — perceptually distinct, print-friendly
    "Muted (Tol)": [
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#CC6677",
        "#882255",
    ],
    "Pastel": [
        "#AEC6CF",
        "#FFD1DC",
        "#B5EAD7",
        "#FFDAC1",
        "#C7CEEA",
        "#FF9AA2",
        "#E2F0CB",
        "#F2C4CE",
    ],
    "Vibrant": [
        "#E6194B",
        "#3CB44B",
        "#FFE119",
        "#4363D8",
        "#F58231",
        "#911EB4",
        "#42D4F4",
        "#F032E6",
    ],
}

# Convenience list for selectbox / dropdowns
COLOR_PALETTE_NAMES: List[str] = list(COLOR_PALETTES.keys())

# Default single-series color list (backward-compat alias)
_COLORS = COLOR_PALETTES["Default"]

_CMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "hot",
    "cool",
    "RdYlBu",
    "seismic",
    "bwr",
    "coolwarm",
]
_COLORSCALES = [
    "Viridis",
    "Plasma",
    "Inferno",
    "Magma",
    "Cividis",
    "Hot",
    "RdBu",
    "Seismic",
    "Earth",
    "Electric",
]

_FONT_OPTIONS = [
    "sans-serif",
    "serif",
    "monospace",
    "Arial",
    "Helvetica",
    "Times New Roman",
    "Courier New",
    "DejaVu Sans",
    "DejaVu Serif",
]

# Keys that are GUI-only style controls — must be stripped from plotting kwargs
_STYLE_KEYS = frozenset(
    {
        "grid",
        "grid_linestyle",
        "grid_which",
        "fontsize",  # legacy fallback — kept for old presets
        "fontsize_label",
        "fontsize_tick",
        "fontsize_title",
        "fontsize_legend",
        "fontfamily",
        "fontcolor",
        "linewidth",
        "legend_frameon",
        "legend_framealpha",
        "legend_position",
        "xlim_min",
        "xlim_max",
        "ylim_min",
        "ylim_max",
        "x_scale",
        "x_notation",
        "x_transform",
        "x_transform_value",
        "y_scale",
        "y_notation",
        "y_transform",
        "y_transform_value",
        "z_scale",
        "z_notation",
        "z_transform",
        "z_transform_value",
        "figure_caption",
        "color_palette",
        "cmap_reversed",
        # Plotly-specific style
        "plotly_height",
        "plotly_line_width",
    }
)


def _numeric_cols(data: Any) -> List[str]:
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).columns.tolist()
    return []


def _all_cols(data: Any) -> List[str]:
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    return []


def _is_2d_array(data: Any) -> bool:
    return isinstance(data, np.ndarray) and data.ndim == 2


_LABEL_HELP = (
    "Supports matplotlib math notation — wrap in $…$.  "
    "Examples: $\\alpha$, $x^{2}$, $x_{0}$, $\\mu$m, $\\AA$"
)


def _symbol_reference_expander() -> None:
    """Collapsible symbol reference for axis/title label inputs."""
    with st.expander("Symbol reference"):
        st.caption(
            "Copy any code below and paste it into a label field.  "
            "Wrap expressions in **$…$** — e.g. `$\\alpha$` renders as α."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                "**Greek — lowercase**  \n"
                "α `$\\alpha$` · β `$\\beta$` · γ `$\\gamma$`  \n"
                "δ `$\\delta$` · ε `$\\epsilon$` · ζ `$\\zeta$`  \n"
                "η `$\\eta$` · θ `$\\theta$` · κ `$\\kappa$`  \n"
                "λ `$\\lambda$` · μ `$\\mu$` · ν `$\\nu$`  \n"
                "ξ `$\\xi$` · π `$\\pi$` · ρ `$\\rho$`  \n"
                "σ `$\\sigma$` · τ `$\\tau$` · φ `$\\phi$`  \n"
                "χ `$\\chi$` · ψ `$\\psi$` · ω `$\\omega$`"
            )
        with col_b:
            st.markdown(
                "**Greek — uppercase**  \n"
                "Γ `$\\Gamma$` · Δ `$\\Delta$` · Θ `$\\Theta$`  \n"
                "Λ `$\\Lambda$` · Ξ `$\\Xi$` · Π `$\\Pi$`  \n"
                "Σ `$\\Sigma$` · Φ `$\\Phi$` · Ψ `$\\Psi$`  \n"
                "Ω `$\\Omega$`  \n\n"
                "**Accents**  \n"
                "x̄ `$\\bar{x}$` · ẋ `$\\dot{x}$`  \n"
                "x̃ `$\\tilde{x}$` · x̂ `$\\hat{x}$`  \n"
                "x⃗ `$\\vec{x}$`"
            )
        with col_c:
            st.markdown(
                "**Super / subscript**  \n"
                "x² `$x^{2}$` · xⁿ `$x^{n}$`  \n"
                "x₀ `$x_{0}$` · xₙ `$x_{n}$`  \n\n"
                "**Units & special**  \n"
                "Å `$\\AA$` · ° `$^{\\circ}$`  \n"
                "± `$\\pm$` · × `$\\times$`  \n"
                "≤ `$\\leq$` · ≥ `$\\geq$` · ≠ `$\\neq$`  \n"
                "∞ `$\\infty$` · ∂ `$\\partial$` · ∇ `$\\nabla$`  \n"
                "√ `$\\sqrt{x}$` · ∫ `$\\int$`  \n"
                "½ `$\\frac{1}{2}$`"
            )


def _label_widgets(
    prefix: str,
    x_default: str = "",
    y_default: str = "",
) -> Dict[str, Optional[str]]:
    """Render title + axis-label text inputs; return partial config dict.

    x_default / y_default are shown as placeholder hints and used as
    fallback values when the user leaves the field empty (so the
    selected column name becomes the default axis label).
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        title = st.text_input(
            "Title",
            key=f"{prefix}_title",
            help=_LABEL_HELP,
        )
    with col2:
        xlabel = st.text_input(
            "X Label",
            placeholder=f"Default: {x_default}" if x_default else "",
            key=f"{prefix}_xlabel",
            help=_LABEL_HELP,
        )
    with col3:
        ylabel = st.text_input(
            "Y Label",
            placeholder=f"Default: {y_default}" if y_default else "",
            key=f"{prefix}_ylabel",
            help=_LABEL_HELP,
        )
    _symbol_reference_expander()
    return {
        "title": title or None,
        "xlabel": xlabel or x_default or None,
        "ylabel": ylabel or y_default or None,
    }


def _figsize_widget(
    prefix: str,
    default: Tuple[float, float] = (8.0, 6.0),
) -> Tuple[float, float]:
    col1, col2 = st.columns(2)
    with col1:
        w = st.number_input(
            "Width (in)",
            min_value=4.0,
            max_value=20.0,
            value=float(default[0]),
            step=0.5,
            key=f"{prefix}_fw",
        )
    with col2:
        h = st.number_input(
            "Height (in)",
            min_value=3.0,
            max_value=16.0,
            value=float(default[1]),
            step=0.5,
            key=f"{prefix}_fh",
        )
    return (w, h)


def _color_widget(prefix: str, default: str = "steelblue") -> str:
    options = _COLORS + ["custom"]
    idx = options.index(default) if default in options else 0
    choice = st.selectbox(
        "Color",
        options,
        index=idx,
        key=f"{prefix}_color",
    )
    if choice == "custom":
        choice = st.color_picker(
            "Pick color",
            "#1f77b4",
            key=f"{prefix}_color_pick",
        )
    return choice


def _axis_limits_widgets(prefix: str) -> Dict:
    """Optional manual X/Y axis limit overrides."""
    config: Dict = {}
    c1, c2 = st.columns(2)
    with c1:
        if st.checkbox("Override X limits", key=f"{prefix}_use_xlim"):
            cc1, cc2 = st.columns(2)
            with cc1:
                config["xlim_min"] = st.number_input(
                    "X min",
                    value=0.0,
                    key=f"{prefix}_xmin",
                )
            with cc2:
                config["xlim_max"] = st.number_input(
                    "X max",
                    value=1.0,
                    key=f"{prefix}_xmax",
                )
    with c2:
        if st.checkbox("Override Y limits", key=f"{prefix}_use_ylim"):
            cc1, cc2 = st.columns(2)
            with cc1:
                config["ylim_min"] = st.number_input(
                    "Y min",
                    value=0.0,
                    key=f"{prefix}_ymin",
                )
            with cc2:
                config["ylim_max"] = st.number_input(
                    "Y max",
                    value=1.0,
                    key=f"{prefix}_ymax",
                )
    return config


def _grid_widgets(prefix: str) -> Dict:
    """Grid line controls (Matplotlib).

    Uses ``_shared_`` keys so the preference is consistent across all
    plot types in the same session.
    """
    c1, c2, c3 = st.columns(3)
    with c1:
        show_grid = st.checkbox(
            "Show grid",
            value=True,
            key="_shared_grid",
        )
    config: Dict = {"grid": show_grid}
    if show_grid:
        with c2:
            config["grid_linestyle"] = st.selectbox(
                "Grid style",
                ["--", "-", ":", "-."],
                key="_shared_grid_ls",
            )
        with c3:
            config["grid_which"] = st.selectbox(
                "Grid density",
                ["major", "minor", "both"],
                key="_shared_grid_which",
            )
    return config


def _appearance_widgets(
    prefix: str,
    include_linewidth: bool = False,
) -> Dict:
    """Font family, color, per-element font sizes, and optionally line width.

    All widget keys use the ``_shared_`` prefix so that settings persist
    when switching between plot types within the same session.
    """
    config: Dict = {}

    # Row 1 — font style, color, and (optional) line width
    n_cols = 3 if not include_linewidth else 4
    cols = st.columns(n_cols)
    with cols[0]:
        config["fontfamily"] = st.selectbox(
            "Font",
            _FONT_OPTIONS,
            key="_shared_fontfamily",
        )
    with cols[1]:
        config["fontcolor"] = st.color_picker(
            "Font color",
            "#262730",
            key="_shared_fontcolor",
        )
    if include_linewidth:
        with cols[2]:
            config["linewidth"] = st.slider(
                "Line width",
                0.5,
                6.0,
                1.5,
                0.5,
                key="_shared_lw",
            )

    # Row 2 — individual element font sizes
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        config["fontsize_label"] = st.number_input(
            "Label size",
            min_value=6,
            max_value=32,
            value=11,
            key="_shared_fs_label",
        )
    with s2:
        config["fontsize_tick"] = st.number_input(
            "Tick size",
            min_value=6,
            max_value=32,
            value=10,
            key="_shared_fs_tick",
        )
    with s3:
        config["fontsize_title"] = st.number_input(
            "Title size",
            min_value=6,
            max_value=36,
            value=13,
            key="_shared_fs_title",
        )
    with s4:
        config["fontsize_legend"] = st.number_input(
            "Legend size",
            min_value=6,
            max_value=32,
            value=10,
            key="_shared_fs_legend",
        )
    return config


def _legend_style_widgets(prefix: str) -> Dict:
    """Legend border, background, and position controls.

    Uses ``_shared_`` keys so the preference is consistent across all
    plot types in the same session.
    """
    c1, c2, c3 = st.columns(3)
    with c1:
        frameon = st.checkbox(
            "Legend border",
            value=True,
            key="_shared_leg_frame",
        )
    with c2:
        opaque_bg = st.checkbox(
            "Opaque background",
            value=True,
            key="_shared_leg_bg",
        )
    with c3:
        position = st.selectbox(
            "Legend position",
            [
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
                "outside right",
            ],
            key="_shared_leg_pos",
        )
    return {
        "legend_frameon": frameon,
        "legend_framealpha": 0.8 if opaque_bg else 0.0,
        "legend_position": position,
    }


def _custom_labels_widget(prefix: str, y_cols: List[str]) -> Dict:
    """Optional per-series legend label overrides.

    Renders one text-input per Y series.  Empty inputs keep the column
    name as the label.  Returns ``{'_custom_labels': [str, ...]}``.
    """
    if not y_cols:
        return {}
    with st.expander("Custom legend labels (optional)"):
        labels = []
        cols_ui = st.columns(min(len(y_cols), 4))
        for i, col in enumerate(y_cols):
            with cols_ui[i % len(cols_ui)]:
                lbl = st.text_input(
                    f"'{col}'",
                    placeholder=str(col),
                    key=f"{prefix}_clbl_{i}",
                )
                labels.append(lbl.strip() or str(col))
    return {"_custom_labels": labels}


def _custom_palette_widget(prefix: str) -> Dict:
    """Color pickers for a user-defined palette (2–8 colors).

    Uses ``_shared_`` keys so the custom colors survive plot-type switches.
    """
    defaults = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    n_colors = int(
        st.number_input(
            "Number of colors",
            min_value=2,
            max_value=8,
            value=4,
            key="_shared_custom_n",
        )
    )
    cols = st.columns(n_colors)
    colors = []
    for i in range(n_colors):
        with cols[i]:
            c = st.color_picker(
                f"C{i + 1}",
                defaults[i],
                key=f"_shared_custom_c{i}",
            )
            colors.append(c)
    return {"_custom_colors": colors}


def _palette_widget(prefix: str) -> Dict:
    """Color palette selector for multi-series plots.

    Uses a ``_shared_`` key so the selection persists across plot types.
    """
    options = COLOR_PALETTE_NAMES + ["Custom"]
    palette = st.selectbox(
        "Color palette",
        options,
        key="_shared_palette",
        help="Choose a discrete color cycle for plot series.",
    )
    config: Dict = {"color_palette": palette}
    if palette == "Custom":
        config.update(_custom_palette_widget(prefix))
    return config


_TRANSFORM_LABELS = {
    "none": "None",
    "normalize_max": "Normalize to max",
    "normalize_01": "Normalize to [0, 1]",
    "scale_by": "Scale by constant",
}


def _axis_scale_widgets(
    prefix: str,
    axis: str,
    label: str,
) -> Dict:
    """Generic scale / notation / transform block for one axis.

    Parameters
    ----------
    prefix : str
        Widget key namespace.
    axis : str
        Short axis tag used in config keys and widget keys, e.g. ``'x'``,
        ``'y'``, or ``'z'``.
    label : str
        Human-readable axis name shown in widget labels, e.g. ``'X'``,
        ``'Y'``, or ``'Color'``.
    """
    config: Dict = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        config[f"{axis}_scale"] = st.selectbox(
            f"{label} scale",
            ["linear", "log"],
            key=f"{prefix}_{axis}scale",
        )
    with c2:
        config[f"{axis}_notation"] = st.selectbox(
            f"{label} notation",
            ["default", "scientific", "engineering"],
            key=f"{prefix}_{axis}notation",
        )
    with c3:
        transform = st.selectbox(
            f"{label} transform",
            ["none", "normalize_max", "normalize_01", "scale_by"],
            format_func=lambda x: _TRANSFORM_LABELS[x],
            key=f"{prefix}_{axis}transform",
        )
        config[f"{axis}_transform"] = transform
    if transform == "scale_by":
        config[f"{axis}_transform_value"] = st.number_input(
            f"{label} scale factor",
            value=1.0,
            key=f"{prefix}_{axis}scaleval",
        )
    return config


def _x_axis_widgets(prefix: str) -> Dict:
    """X-axis scale, notation, and data-transform controls."""
    return _axis_scale_widgets(prefix, "x", "X")


def _y_axis_widgets(prefix: str) -> Dict:
    """Y-axis scale, notation, and data-transform controls."""
    return _axis_scale_widgets(prefix, "y", "Y")


def _z_axis_widgets(prefix: str) -> Dict:
    """Color-axis (Z) scale, notation, and data-transform controls."""
    return _axis_scale_widgets(prefix, "z", "Color")


def _caption_widget(prefix: str) -> Dict:
    """Optional figure caption text field."""
    caption = st.text_input(
        "Figure caption (optional)",
        key=f"{prefix}_caption",
    )
    return {"figure_caption": caption or None}


def _plotly_appearance_widgets(prefix: str) -> Dict:
    """Appearance controls for Plotly interactive plots.

    Uses ``_shared_ply_`` keys so settings persist across Plotly plot types.
    """
    config: Dict = {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        config["fontsize"] = st.slider(
            "Font size",
            8,
            24,
            13,
            key="_shared_ply_fontsize",
        )
    with c2:
        config["fontfamily"] = st.selectbox(
            "Font",
            _FONT_OPTIONS,
            key="_shared_ply_fontfamily",
        )
    with c3:
        config["fontcolor"] = st.color_picker(
            "Font color",
            "#262730",
            key="_shared_ply_fontcolor",
        )
    with c4:
        config["plotly_height"] = st.slider(
            "Chart height",
            300,
            900,
            450,
            50,
            key="_shared_ply_height",
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        show_grid = st.checkbox(
            "Show grid",
            value=True,
            key="_shared_ply_grid",
        )
        config["grid"] = show_grid
    with c2:
        config["legend_frameon"] = st.checkbox(
            "Legend border",
            value=False,
            key="_shared_ply_leg_frame",
        )
    with c3:
        config["legend_framealpha"] = (
            0.8
            if st.checkbox(
                "Opaque legend bg",
                value=True,
                key="_shared_ply_leg_bg",
            )
            else 0.0
        )
    return config


# ── Per-plot-type widget functions ───────────────────────────────────────────


def _widgets_histogram(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if not numeric:
            st.warning("No numeric columns found in dataset.")
            return {}
        config["_data_col"] = st.selectbox(
            "Data column",
            numeric,
            key=f"{prefix}_dcol",
        )
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        config["_data_col"] = None
    else:
        st.warning("Histogram requires a numeric array or a DataFrame with numeric columns.")
        return {}

    st.markdown("**Histogram parameters**")
    col1, col2, col3 = st.columns(3)
    with col1:
        bins_choice = st.selectbox(
            "Bin strategy",
            ["auto", "sqrt", "sturges", "custom"],
            key=f"{prefix}_bins_choice",
        )
        if bins_choice == "custom":
            config["bins"] = st.number_input(
                "Number of bins",
                min_value=2,
                max_value=500,
                value=30,
                key=f"{prefix}_bins_n",
            )
        else:
            config["bins"] = bins_choice
    with col2:
        config["color"] = _color_widget(prefix)
    with col3:
        config["alpha"] = st.slider(
            "Opacity",
            0.1,
            1.0,
            0.7,
            0.05,
            key=f"{prefix}_alpha",
        )
        config["density"] = st.checkbox("Normalize (density)", key=f"{prefix}_density")

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix)
    x_lbl = str(config.get("_data_col", "")) or ""
    y_lbl = "Density" if config.get("density") else "Frequency"
    config.update(_label_widgets(prefix, x_default=x_lbl, y_default=y_lbl))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_legend_style_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_line_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 2:
            st.warning("Line plot requires at least 2 numeric columns.")
            return {}
        config["_x_col"] = st.selectbox(
            "X column",
            numeric,
            index=0,
            key=f"{prefix}_xcol",
        )
        y_opts = [c for c in numeric if c != config["_x_col"]]
        config["_y_cols"] = st.multiselect(
            "Y column(s)",
            y_opts,
            default=[y_opts[0]] if y_opts else [],
            key=f"{prefix}_ycols",
        )
        if not config["_y_cols"]:
            st.info("Select at least one Y column.")
            return {}

    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            st.info("1D array: array index will be used as the X axis.")
            config["_x_col"] = None
            config["_y_cols"] = None
        elif data.ndim == 2:
            opts = [str(i) for i in range(data.shape[1])]
            config["_x_col"] = st.selectbox(
                "X column index",
                opts,
                index=0,
                key=f"{prefix}_xcol",
            )
            y_opts = [c for c in opts if c != config["_x_col"]]
            config["_y_cols"] = st.multiselect(
                "Y column index(es)",
                y_opts,
                default=[y_opts[0]] if y_opts else [],
                key=f"{prefix}_ycols",
            )
            if not config["_y_cols"]:
                st.info("Select at least one Y column.")
                return {}
        else:
            st.warning("Line plot requires a 1D or 2D array.")
            return {}
    else:
        st.warning("Unsupported data type for line plot.")
        return {}

    st.markdown("**Style**")
    col1, col2 = st.columns(2)
    with col1:
        linestyle = st.selectbox(
            "Line style",
            ["-", "--", "-.", ":"],
            key=f"{prefix}_ls",
        )
        config["linestyles"] = [linestyle]
    with col2:
        marker = st.selectbox(
            "Marker",
            ["None", "o", "s", "^", "D", "v", "*"],
            key=f"{prefix}_marker",
        )
        config["markers"] = [None if marker == "None" else marker]

    # ── Custom legend labels ──────────────────────────────────────────────────
    y_cols_lp = config.get("_y_cols") or []
    config.update(_custom_labels_widget(prefix, [str(c) for c in y_cols_lp]))

    # ── Secondary Y axis (DataFrame only) ────────────────────────────────────
    if isinstance(data, pd.DataFrame) and y_cols_lp:
        with st.expander("Secondary Y axis (optional)"):
            use_sec = st.checkbox(
                "Enable secondary Y axis",
                key=f"{prefix}_use_sec_y",
            )
            if use_sec:
                x_col_lp = config.get("_x_col")
                sec_opts = [c for c in numeric if c != x_col_lp and c not in y_cols_lp]
                if sec_opts:
                    config["_secondary_y_cols"] = st.multiselect(
                        "Secondary Y column(s)",
                        sec_opts,
                        key=f"{prefix}_sec_ycols",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        config["_secondary_y_label"] = st.text_input(
                            "Secondary Y label",
                            key=f"{prefix}_sec_ylabel",
                        )
                    with c2:
                        config["_secondary_y_scale"] = st.selectbox(
                            "Secondary Y scale",
                            ["linear", "log"],
                            key=f"{prefix}_sec_yscale",
                        )
                else:
                    st.info(
                        "All available numeric columns are already assigned to the primary Y axis."
                    )

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix)
    x_lbl = str(config.get("_x_col", "")) or ""
    y_cols = config.get("_y_cols") or []
    y_lbl = str(y_cols[0]) if y_cols else ""
    config.update(_label_widgets(prefix, x_default=x_lbl, y_default=y_lbl))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix, include_linewidth=True))
        config.update(_palette_widget(prefix))
        config.update(_legend_style_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_x_axis_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_scatter_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 2:
            st.warning("Scatter plot requires at least 2 numeric columns.")
            return {}
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox(
                "X column",
                numeric,
                index=0,
                key=f"{prefix}_xcol",
            )
        with col2:
            y_opts = [c for c in numeric if c != config["_x_col"]]
            config["_y_col"] = st.selectbox(
                "Y column",
                y_opts if y_opts else numeric,
                index=0,
                key=f"{prefix}_ycol",
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox(
                "X column index",
                opts,
                index=0,
                key=f"{prefix}_xcol",
            )
        with col2:
            y_opts = [c for c in opts if c != config["_x_col"]]
            config["_y_col"] = st.selectbox(
                "Y column index",
                y_opts if y_opts else opts,
                index=0,
                key=f"{prefix}_ycol",
            )
    else:
        st.warning("Scatter plot requires a DataFrame or 2D array with >= 2 columns.")
        return {}

    st.markdown("**Style**")
    col1, col2, col3 = st.columns(3)
    with col1:
        config["color"] = _color_widget(prefix)
    with col2:
        config["size"] = st.number_input(
            "Point size",
            min_value=1,
            max_value=500,
            value=50,
            key=f"{prefix}_size",
        )
    with col3:
        config["alpha"] = st.slider(
            "Opacity",
            0.1,
            1.0,
            0.7,
            0.05,
            key=f"{prefix}_alpha",
        )

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix)
    x_lbl = str(config.get("_x_col", "")) or ""
    y_lbl = str(config.get("_y_col", "")) or ""
    config.update(_label_widgets(prefix, x_default=x_lbl, y_default=y_lbl))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_legend_style_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_x_axis_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_heatmap(data: Any, prefix: str) -> Dict:
    config: Dict = {}

    if _is_2d_array(data):
        config["_use_numeric_df"] = False
    elif isinstance(data, pd.DataFrame):
        numeric = _numeric_cols(data)
        if len(numeric) < 2:
            st.warning("Heatmap requires a 2D array or DataFrame with >= 2 numeric columns.")
            return {}
        st.info(f"Heatmap will use all {len(numeric)} numeric columns as a matrix.")
        config["_use_numeric_df"] = True
    else:
        st.warning("Heatmap requires a 2D NumPy array or DataFrame.")
        return {}

    col1, col2, col3 = st.columns(3)
    with col1:
        config["cmap"] = st.selectbox("Colormap", _CMAPS, key=f"{prefix}_cmap")
        config["colorbar"] = st.checkbox(
            "Show colorbar",
            value=True,
            key=f"{prefix}_cbar",
        )
        config["cmap_reversed"] = st.checkbox(
            "Reverse colormap",
            key=f"{prefix}_cmap_rev",
        )
    with col2:
        use_clim = st.checkbox("Set color limits", key=f"{prefix}_use_clim")
        if use_clim:
            config["vmin"] = st.number_input("vmin", value=0.0, key=f"{prefix}_vmin")
            config["vmax"] = st.number_input("vmax", value=1.0, key=f"{prefix}_vmax")
    with col3:
        use_n = st.checkbox("Discretize colormap", key=f"{prefix}_use_ncol")
        if use_n:
            config["cmap_n_colors"] = st.number_input(
                "N colors",
                min_value=2,
                max_value=256,
                value=10,
                key=f"{prefix}_ncol",
            )

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_appearance_widgets(prefix))
        config.update(_grid_widgets(prefix))
        config.update(_x_axis_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
        config.update(_z_axis_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_contour_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}

    if _is_2d_array(data):
        config["_mode"] = "array"
        st.info("X and Y meshgrid coordinates are auto-generated from the Z array shape.")
    elif isinstance(data, pd.DataFrame):
        numeric = _numeric_cols(data)
        if len(numeric) < 3:
            st.warning(
                "Contour plot from a DataFrame requires at least 3 numeric columns (X, Y, Z)."
            )
            return {}
        config["_mode"] = "dataframe"
        c1, c2, c3 = st.columns(3)
        with c1:
            config["_x_col"] = st.selectbox("X column", numeric, index=0, key=f"{prefix}_xcol")
        with c2:
            y_opts = [c for c in numeric if c != config["_x_col"]]
            config["_y_col"] = st.selectbox(
                "Y column",
                y_opts or numeric,
                index=0,
                key=f"{prefix}_ycol",
            )
        with c3:
            z_opts = [c for c in numeric if c not in (config["_x_col"], config.get("_y_col", ""))]
            config["_z_col"] = st.selectbox(
                "Z column",
                z_opts or numeric,
                index=0,
                key=f"{prefix}_zcol",
            )
        c1, c2 = st.columns(2)
        with c1:
            config["_grid_n"] = st.number_input(
                "Grid resolution (N×N)",
                min_value=20,
                max_value=500,
                value=100,
                key=f"{prefix}_gridn",
            )
        with c2:
            config["_interp_method"] = st.selectbox(
                "Interpolation method",
                ["cubic", "linear", "nearest"],
                key=f"{prefix}_interp",
                help=(
                    "'cubic' is smoothest; 'linear' is exact on grid points; "
                    "'nearest' preserves raw values."
                ),
            )
        st.caption(
            "Structured-grid CSVs are reshaped directly; "
            "scattered XYZ data is interpolated via scipy griddata."
        )
    else:
        st.warning("Contour plot requires a 2D NumPy array or a DataFrame with X, Y, Z columns.")
        return {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        config["levels"] = st.number_input(
            "Contour levels",
            min_value=2,
            max_value=50,
            value=10,
            key=f"{prefix}_levels",
        )
    with col2:
        config["cmap"] = st.selectbox("Colormap", _CMAPS, key=f"{prefix}_cmap")
        config["cmap_reversed"] = st.checkbox(
            "Reverse colormap",
            key=f"{prefix}_cmap_rev",
        )
    with col3:
        use_n = st.checkbox("Discretize colormap", key=f"{prefix}_use_ncol")
        if use_n:
            config["cmap_n_colors"] = st.number_input(
                "N colors",
                min_value=2,
                max_value=256,
                value=10,
                key=f"{prefix}_ncol",
            )
    with col4:
        config["filled"] = st.checkbox(
            "Filled contours",
            value=True,
            key=f"{prefix}_filled",
        )
        config["colorbar"] = st.checkbox(
            "Show colorbar",
            value=True,
            key=f"{prefix}_cbar",
        )

    c1, c2 = st.columns(2)
    with c1:
        if not config.get("filled", True):
            config["linewidths"] = st.slider(
                "Line width",
                0.5,
                5.0,
                1.5,
                0.25,
                key=f"{prefix}_lw",
            )
    with c2:
        use_clim = st.checkbox("Set color limits", key=f"{prefix}_use_clim")
        if use_clim:
            cc1, cc2 = st.columns(2)
            with cc1:
                config["vmin"] = st.number_input("vmin", value=0.0, key=f"{prefix}_vmin")
            with cc2:
                config["vmax"] = st.number_input("vmax", value=1.0, key=f"{prefix}_vmax")

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_appearance_widgets(prefix))
        config.update(_grid_widgets(prefix))
        config.update(_x_axis_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
        config.update(_z_axis_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_distribution_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if not numeric:
            st.warning("No numeric columns found.")
            return {}
        config["_data_col"] = st.selectbox(
            "Data column",
            numeric,
            key=f"{prefix}_dcol",
        )
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        config["_data_col"] = None
    else:
        st.warning("Distribution plot requires a numeric array or DataFrame.")
        return {}

    col1, col2 = st.columns(2)
    with col1:
        config["kind"] = st.selectbox(
            "Plot kind",
            ["hist", "kde", "ecdf"],
            key=f"{prefix}_kind",
        )
    with col2:
        config["kde"] = st.checkbox(
            "Overlay KDE (hist only)",
            value=True,
            key=f"{prefix}_kde",
        )
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
    return config


def _widgets_box_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    all_cols = _all_cols(data)
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if not numeric:
            st.warning("No numeric columns found.")
            return {}
        categorical = [c for c in all_cols if c not in numeric]
        col1, col2, col3 = st.columns(3)
        with col1:
            x_choice = st.selectbox(
                "X (grouping) column",
                ["— none —"] + categorical + numeric,
                key=f"{prefix}_xcol",
            )
            config["x"] = None if x_choice == "— none —" else x_choice
        with col2:
            config["y"] = st.selectbox(
                "Y (value) column",
                numeric,
                key=f"{prefix}_ycol",
            )
        with col3:
            hue_choice = st.selectbox(
                "Hue (optional)",
                ["— none —"] + all_cols,
                key=f"{prefix}_hue",
            )
            config["hue"] = None if hue_choice == "— none —" else hue_choice
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        config["x"] = None
        config["y"] = None
        config["hue"] = None
        st.info("Box plot will use the numeric array as a single group.")
    else:
        st.warning("Box plot requires a numeric array or DataFrame.")
        return {}

    config["kind"] = st.selectbox(
        "Plot kind",
        ["box", "violin", "boxen"],
        key=f"{prefix}_kind",
    )
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
    return config


def _widgets_regression_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 2:
            st.warning("Regression plot requires at least 2 numeric columns.")
            return {}
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox(
                "X column",
                numeric,
                index=0,
                key=f"{prefix}_xcol",
            )
        with col2:
            y_opts = [c for c in numeric if c != config["_x_col"]]
            config["_y_col"] = st.selectbox(
                "Y column",
                y_opts if y_opts else numeric,
                index=0,
                key=f"{prefix}_ycol",
            )
        config["_use_df"] = True

    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox(
                "X column index",
                opts,
                index=0,
                key=f"{prefix}_xcol",
            )
        with col2:
            y_opts = [c for c in opts if c != config["_x_col"]]
            config["_y_col"] = st.selectbox(
                "Y column index",
                y_opts if y_opts else opts,
                index=0,
                key=f"{prefix}_ycol",
            )
        config["_use_df"] = False
    else:
        st.warning("Regression plot requires a DataFrame or 2D array with >= 2 columns.")
        return {}

    col1, col2 = st.columns(2)
    with col1:
        config["order"] = st.number_input(
            "Polynomial order",
            min_value=1,
            max_value=5,
            value=1,
            key=f"{prefix}_order",
        )
    with col2:
        config["ci"] = st.slider(
            "Confidence interval (%)",
            0,
            99,
            95,
            key=f"{prefix}_ci",
        )
    x_lbl = str(config.get("_x_col", "")) or ""
    y_lbl = str(config.get("_y_col", "")) or ""
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        # Regression has no separate label widgets — show them here
        col1, col2 = st.columns(2)
        with col1:
            xl = st.text_input(
                "X Label",
                placeholder=f"Default: {x_lbl}" if x_lbl else "",
                key=f"{prefix}_xlabel",
            )
            config["xlabel"] = xl or x_lbl or None
        with col2:
            yl = st.text_input(
                "Y Label",
                placeholder=f"Default: {y_lbl}" if y_lbl else "",
                key=f"{prefix}_ylabel",
            )
            config["ylabel"] = yl or y_lbl or None
    return config


def _widgets_interactive_histogram(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if not numeric:
            st.warning("No numeric columns found.")
            return {}
        config["_data_col"] = st.selectbox(
            "Data column",
            numeric,
            key=f"{prefix}_dcol",
        )
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        config["_data_col"] = None
    else:
        st.warning("Interactive histogram requires a numeric array or DataFrame.")
        return {}

    col1, col2, col3 = st.columns(3)
    with col1:
        config["bins"] = st.number_input(
            "Bins",
            min_value=2,
            max_value=500,
            value=30,
            key=f"{prefix}_bins",
        )
    with col2:
        config["title"] = st.text_input("Title", key=f"{prefix}_title") or None
    with col3:
        x_lbl = str(config.get("_data_col", "")) or ""
        xl = st.text_input(
            "X Label",
            placeholder=f"Default: {x_lbl}" if x_lbl else "",
            key=f"{prefix}_xlabel",
        )
        config["xlabel"] = xl or x_lbl or None
    with st.expander("Appearance"):
        config.update(_plotly_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
    return config


def _widgets_interactive_scatter(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 2:
            st.warning("Interactive scatter requires at least 2 numeric columns.")
            return {}
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox(
                "X column",
                numeric,
                index=0,
                key=f"{prefix}_xcol",
            )
        with col2:
            y_opts = [c for c in numeric if c != config["_x_col"]]
            config["_y_col"] = st.selectbox(
                "Y column",
                y_opts if y_opts else numeric,
                index=0,
                key=f"{prefix}_ycol",
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox(
                "X column index",
                opts,
                index=0,
                key=f"{prefix}_xcol",
            )
        with col2:
            y_opts = [c for c in opts if c != config["_x_col"]]
            config["_y_col"] = st.selectbox(
                "Y column index",
                y_opts if y_opts else opts,
                index=0,
                key=f"{prefix}_ycol",
            )
    else:
        st.warning("Interactive scatter requires a DataFrame or 2D array with >= 2 columns.")
        return {}

    x_lbl = str(config.get("_x_col", "")) or ""
    y_lbl = str(config.get("_y_col", "")) or ""
    col1, col2, col3 = st.columns(3)
    with col1:
        config["title"] = st.text_input("Title", key=f"{prefix}_title") or None
    with col2:
        xl = st.text_input(
            "X Label",
            placeholder=f"Default: {x_lbl}" if x_lbl else "",
            key=f"{prefix}_xlabel",
        )
        config["xlabel"] = xl or x_lbl or None
    with col3:
        yl = st.text_input(
            "Y Label",
            placeholder=f"Default: {y_lbl}" if y_lbl else "",
            key=f"{prefix}_ylabel",
        )
        config["ylabel"] = yl or y_lbl or None
    with st.expander("Appearance"):
        config.update(_plotly_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
    return config


def _widgets_interactive_line(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 2:
            st.warning("Interactive line requires at least 2 numeric columns.")
            return {}
        config["_x_col"] = st.selectbox(
            "X column",
            numeric,
            index=0,
            key=f"{prefix}_xcol",
        )
        y_opts = [c for c in numeric if c != config["_x_col"]]
        config["_y_cols"] = st.multiselect(
            "Y column(s)",
            y_opts,
            default=[y_opts[0]] if y_opts else [],
            key=f"{prefix}_ycols",
        )
        if not config["_y_cols"]:
            st.info("Select at least one Y column.")
            return {}

    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            st.info("1D array: array index as X axis.")
            config["_x_col"] = None
            config["_y_cols"] = None
        elif data.ndim == 2:
            opts = [str(i) for i in range(data.shape[1])]
            config["_x_col"] = st.selectbox(
                "X column index",
                opts,
                index=0,
                key=f"{prefix}_xcol",
            )
            y_opts = [c for c in opts if c != config["_x_col"]]
            config["_y_cols"] = st.multiselect(
                "Y column index(es)",
                y_opts,
                default=[y_opts[0]] if y_opts else [],
                key=f"{prefix}_ycols",
            )
            if not config["_y_cols"]:
                st.info("Select at least one Y column.")
                return {}
        else:
            st.warning("Interactive line requires a 1D or 2D array.")
            return {}
    else:
        st.warning("Unsupported data type for interactive line plot.")
        return {}

    x_lbl = str(config.get("_x_col", "")) or ""
    y_cols = config.get("_y_cols") or []
    y_lbl = str(y_cols[0]) if y_cols else ""
    col1, col2, col3 = st.columns(3)
    with col1:
        config["title"] = st.text_input("Title", key=f"{prefix}_title") or None
    with col2:
        xl = st.text_input(
            "X Label",
            placeholder=f"Default: {x_lbl}" if x_lbl else "",
            key=f"{prefix}_xlabel",
        )
        config["xlabel"] = xl or x_lbl or None
    with col3:
        yl = st.text_input(
            "Y Label",
            placeholder=f"Default: {y_lbl}" if y_lbl else "",
            key=f"{prefix}_ylabel",
        )
        config["ylabel"] = yl or y_lbl or None
    with st.expander("Appearance"):
        config.update(_plotly_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
    return config


def _widgets_interactive_heatmap(data: Any, prefix: str) -> Dict:
    config: Dict = {}

    if _is_2d_array(data):
        config["_use_numeric_df"] = False
    elif isinstance(data, pd.DataFrame):
        numeric = _numeric_cols(data)
        if len(numeric) < 2:
            st.warning(
                "Interactive heatmap requires a 2D array or DataFrame with >= 2 numeric columns."
            )
            return {}
        st.info(f"Heatmap will use all {len(numeric)} numeric columns.")
        config["_use_numeric_df"] = True
    else:
        st.warning("Interactive heatmap requires a 2D NumPy array or DataFrame.")
        return {}

    col1, col2 = st.columns(2)
    with col1:
        config["colorscale"] = st.selectbox(
            "Color scale",
            _COLORSCALES,
            key=f"{prefix}_cs",
        )
    with col2:
        config["title"] = st.text_input("Title", key=f"{prefix}_title") or None
    with st.expander("Appearance"):
        config.update(_plotly_appearance_widgets(prefix))
    return config


def _widgets_interactive_3d_surface(data: Any, prefix: str) -> Dict:
    config: Dict = {}

    if not _is_2d_array(data):
        st.warning("3D surface plot requires a 2D NumPy array (Z values). Load a 2D .npy file.")
        return {}

    st.info("X and Y meshgrid coordinates are auto-generated from the Z array shape.")
    col1, col2 = st.columns(2)
    with col1:
        config["colorscale"] = st.selectbox(
            "Color scale",
            _COLORSCALES,
            key=f"{prefix}_cs",
        )
    with col2:
        config["title"] = st.text_input("Title", key=f"{prefix}_title") or None
    with st.expander("Appearance"):
        config.update(_plotly_appearance_widgets(prefix))
    return config


# ── M14 widget functions ─────────────────────────────────────────────────────


def _widgets_bar_chart(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        all_cols = _all_cols(data)
        col1, col2 = st.columns(2)
        with col1:
            config["_cat_col"] = st.selectbox(
                "Category column",
                all_cols,
                index=0,
                key=f"{prefix}_catcol",
            )
        with col2:
            config["_val_col"] = st.selectbox(
                "Value column",
                numeric if numeric else all_cols,
                index=min(1, len(all_cols) - 1),
                key=f"{prefix}_valcol",
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_cat_col"] = st.selectbox(
                "Category column index",
                opts,
                index=0,
                key=f"{prefix}_catcol",
            )
        with col2:
            config["_val_col"] = st.selectbox(
                "Value column index",
                opts,
                index=min(1, data.shape[1] - 1),
                key=f"{prefix}_valcol",
            )
    else:
        st.info(
            "Bar chart will use the loaded array as values.  "
            "Integer index will be used as categories."
        )
        config["_cat_col"] = None
        config["_val_col"] = None

    st.markdown("**Chart type & grouping**")
    col1, col2 = st.columns(2)
    with col1:
        config["kind"] = st.selectbox(
            "Chart kind",
            ["simple", "grouped", "stacked"],
            key=f"{prefix}_kind",
        )
    with col2:
        if config["kind"] in ("grouped", "stacked"):
            if isinstance(data, pd.DataFrame) and numeric:
                hue_opts = [
                    c
                    for c in numeric
                    if c != config.get("_val_col") and c != config.get("_cat_col")
                ]
                config["_hue_cols"] = st.multiselect(
                    "Group columns",
                    hue_opts,
                    default=[],
                    key=f"{prefix}_huecols",
                    help=(
                        "Each selected column becomes a bar group.  "
                        "Leave empty to use a single grouped series."
                    ),
                )
            else:
                st.info("Grouped/stacked mode requires a DataFrame with multiple numeric columns.")

    st.markdown("**Style**")
    col1, col2 = st.columns(2)
    with col1:
        config["color"] = _color_widget(prefix)
    with col2:
        config["alpha"] = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key=f"{prefix}_alpha")

    # Error bars (simple mode only)
    if config.get("kind", "simple") == "simple":
        if isinstance(data, pd.DataFrame) and numeric:
            with st.expander("Error bars (optional)"):
                err_opts = ["None"] + [c for c in numeric if c != config.get("_val_col")]
                err_sel = st.selectbox(
                    "Y error column",
                    err_opts,
                    key=f"{prefix}_yerr_col",
                )
                config["_yerr_col"] = None if err_sel == "None" else err_sel

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_y_axis_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_waterfall_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 2:
            st.warning(
                "Waterfall plot requires at least 2 numeric columns (1 X + 1 or more Y traces)."
            )
            return {}
        config["_x_col"] = st.selectbox(
            "X column",
            numeric,
            index=0,
            key=f"{prefix}_xcol",
        )
        y_opts = [c for c in numeric if c != config["_x_col"]]
        config["_y_cols"] = st.multiselect(
            "Y (trace) column(s)",
            y_opts,
            default=y_opts[: min(5, len(y_opts))],
            key=f"{prefix}_ycols",
            help="Each column becomes one trace in the waterfall.",
        )
        if not config["_y_cols"]:
            st.info("Select at least one Y column.")
            return {}
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            opts = [str(i) for i in range(data.shape[1])]
            config["_x_col"] = st.selectbox(
                "X column index",
                opts,
                index=0,
                key=f"{prefix}_xcol",
            )
            y_opts2 = [c for c in opts if c != config["_x_col"]]
            config["_y_cols"] = st.multiselect(
                "Y (trace) column indices",
                y_opts2,
                default=y_opts2,
                key=f"{prefix}_ycols",
            )
        elif data.ndim == 3:
            st.info("3-D array detected: each row slice will be one trace.")
            config["_x_col"] = None
            config["_y_cols"] = None
        else:
            st.warning("Waterfall plot requires a 2-D or 3-D array.")
            return {}
    else:
        st.warning("Unsupported data type for waterfall plot.")
        return {}

    st.markdown("**Style**")
    col1, col2, col3 = st.columns(3)
    with col1:
        config["cmap"] = st.selectbox("Colormap", _CMAPS, index=0, key=f"{prefix}_cmap")
    with col2:
        config["alpha"] = st.slider("Opacity", 0.1, 1.0, 0.9, 0.05, key=f"{prefix}_alpha")
    with col3:
        offset_mode = st.selectbox(
            "Trace offset",
            ["Auto", "Manual"],
            key=f"{prefix}_offset_mode",
        )
    if offset_mode == "Manual":
        config["offset"] = st.number_input(
            "Offset value",
            value=1.0,
            key=f"{prefix}_offset",
        )
    else:
        config["offset"] = "auto"

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix, default=(10.0, 7.0))
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix, include_linewidth=True))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_dual_axis_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 3:
            st.warning("Dual Y-axis plot requires at least 3 numeric columns (1 X + 2 Y).")
            return {}
        config["_x_col"] = st.selectbox(
            "X column",
            numeric,
            index=0,
            key=f"{prefix}_xcol",
        )
        remaining = [c for c in numeric if c != config["_x_col"]]
        col1, col2 = st.columns(2)
        with col1:
            config["_y1_col"] = st.selectbox(
                "Left Y column",
                remaining,
                index=0,
                key=f"{prefix}_y1col",
            )
        with col2:
            y2_opts = [c for c in remaining if c != config["_y1_col"]]
            config["_y2_col"] = st.selectbox(
                "Right Y column",
                y2_opts if y2_opts else remaining,
                index=0,
                key=f"{prefix}_y2col",
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
        opts = [str(i) for i in range(data.shape[1])]
        config["_x_col"] = st.selectbox("X column index", opts, index=0, key=f"{prefix}_xcol")
        col1, col2 = st.columns(2)
        with col1:
            config["_y1_col"] = st.selectbox(
                "Left Y column index",
                [c for c in opts if c != config["_x_col"]],
                index=0,
                key=f"{prefix}_y1col",
            )
        with col2:
            y2_opts2 = [c for c in opts if c not in (config["_x_col"], config["_y1_col"])]
            config["_y2_col"] = st.selectbox(
                "Right Y column index",
                y2_opts2 if y2_opts2 else opts,
                index=0,
                key=f"{prefix}_y2col",
            )
    else:
        st.warning(
            "Dual Y-axis plot requires a DataFrame or 2-D array with at least 3 numeric columns."
        )
        return {}

    st.markdown("**Style**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        config["color1"] = st.color_picker("Left color", "#4682B4", key=f"{prefix}_c1")
    with col2:
        config["color2"] = st.color_picker("Right color", "#FF6347", key=f"{prefix}_c2")
    with col3:
        config["linestyle1"] = st.selectbox(
            "Left style",
            ["-", "--", "-.", ":"],
            key=f"{prefix}_ls1",
        )
    with col4:
        config["linestyle2"] = st.selectbox(
            "Right style",
            ["--", "-", "-.", ":"],
            key=f"{prefix}_ls2",
        )

    st.markdown("**Figure & labels**")
    config["figsize"] = _figsize_widget(prefix)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        config["title"] = st.text_input("Title", key=f"{prefix}_title", help=_LABEL_HELP) or None
    with col2:
        config["xlabel"] = (
            st.text_input("X Label", key=f"{prefix}_xlabel", help=_LABEL_HELP) or None
        )
    with col3:
        config["ylabel1"] = (
            st.text_input("Left Y Label", key=f"{prefix}_ylabel1", help=_LABEL_HELP) or None
        )
    with col4:
        config["ylabel2"] = (
            st.text_input("Right Y Label", key=f"{prefix}_ylabel2", help=_LABEL_HELP) or None
        )
    _symbol_reference_expander()

    with st.expander("Appearance"):
        config.update(_appearance_widgets(prefix, include_linewidth=True))
        config.update(_caption_widget(prefix))
    return config


def _widgets_z_colored_scatter(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 2:
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X column", numeric, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox(
                "Y column", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_ycol"
            )
        with col3:
            config["_z_col"] = st.selectbox(
                "Color column", numeric, index=min(2, len(numeric) - 1), key=f"{prefix}_zcol"
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X column index", opts, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox("Y column index", opts, index=1, key=f"{prefix}_ycol")
        with col3:
            config["_z_col"] = st.selectbox(
                "Color column index", opts, index=2, key=f"{prefix}_zcol"
            )
    else:
        st.warning("Z-colored scatter requires at least 3 columns.")

    st.markdown("**Appearance**")
    col1, col2 = st.columns(2)
    with col1:
        config["cmap"] = st.selectbox("Colormap", _CMAPS, index=0, key=f"{prefix}_cmap")
    with col2:
        config["alpha"] = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key=f"{prefix}_alpha")
    config["colorbar"] = st.checkbox("Show colorbar", value=True, key=f"{prefix}_colorbar")
    config["colorbar_label"] = (
        st.text_input("Colorbar label", value="", key=f"{prefix}_cblabel") or None
    )
    config["s"] = st.slider("Marker size", 5, 200, 20, 5, key=f"{prefix}_markersize")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_bubble_chart(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 3:
        all_cols = numeric
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X column", all_cols, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox(
                "Y column", all_cols, index=min(1, len(all_cols) - 1), key=f"{prefix}_ycol"
            )
        with col3:
            config["_size_col"] = st.selectbox(
                "Size column", all_cols, index=min(2, len(all_cols) - 1), key=f"{prefix}_sizecol"
            )
        z_opts = ["None"] + list(all_cols)
        z_sel = st.selectbox("Color column (optional)", z_opts, index=0, key=f"{prefix}_zcol")
        config["_z_col"] = None if z_sel == "None" else z_sel
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X col index", opts, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox("Y col index", opts, index=1, key=f"{prefix}_ycol")
        with col3:
            config["_size_col"] = st.selectbox(
                "Size col index", opts, index=2, key=f"{prefix}_sizecol"
            )
        config["_z_col"] = None
    else:
        st.warning("Bubble chart requires at least 3 columns (x, y, size).")

    st.markdown("**Appearance**")
    col1, col2 = st.columns(2)
    with col1:
        config["color"] = _color_widget(prefix)
    with col2:
        config["alpha"] = st.slider("Opacity", 0.1, 1.0, 0.6, 0.05, key=f"{prefix}_alpha")
    config["size_scale"] = st.slider("Size scale", 0.1, 10.0, 1.0, 0.1, key=f"{prefix}_sizescale")
    if config.get("_z_col"):
        config["cmap"] = st.selectbox("Colormap", _CMAPS, index=0, key=f"{prefix}_cmap")
        config["colorbar"] = st.checkbox("Show colorbar", value=True, key=f"{prefix}_colorbar")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_polar_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            config["_theta_col"] = st.selectbox(
                "Theta column (radians)", numeric, index=0, key=f"{prefix}_thetacol"
            )
        with col2:
            config["_r_col"] = st.selectbox(
                "R column", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_rcol"
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_theta_col"] = st.selectbox(
                "Theta column index", opts, index=0, key=f"{prefix}_thetacol"
            )
        with col2:
            config["_r_col"] = st.selectbox("R column index", opts, index=1, key=f"{prefix}_rcol")
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        config["_theta_col"] = None
        config["_r_col"] = None
        st.info("1D array: will use index as theta (radians) and values as r.")

    st.markdown("**Appearance**")
    col1, col2 = st.columns(2)
    with col1:
        config["color"] = _color_widget(prefix)
        config["fill"] = st.checkbox("Fill area", value=False, key=f"{prefix}_fill")
    with col2:
        config["theta_zero_location"] = st.selectbox(
            "0° at", ["N", "E", "S", "W"], index=0, key=f"{prefix}_theta0"
        )
        config["theta_direction"] = st.selectbox(
            "Direction",
            [-1, 1],
            format_func=lambda x: "Clockwise" if x == -1 else "Counter-clockwise",
            index=0,
            key=f"{prefix}_thetadir",
        )
    config["figsize"] = _figsize_widget(prefix)
    config["title"] = st.text_input("Title", value="", key=f"{prefix}_title") or None
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
    return config


def _widgets_histogram_2d(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox("X column", numeric, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox(
                "Y column", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_ycol"
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox("X col index", opts, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox("Y col index", opts, index=1, key=f"{prefix}_ycol")
    else:
        st.warning("2D histogram requires at least 2 columns.")

    st.markdown("**Settings**")
    col1, col2 = st.columns(2)
    with col1:
        config["mode"] = st.selectbox("Mode", ["hist2d", "hexbin"], index=0, key=f"{prefix}_mode")
    with col2:
        config["cmap"] = st.selectbox("Colormap", _CMAPS, index=0, key=f"{prefix}_cmap")
    if config.get("mode") == "hexbin":
        config["gridsize"] = st.slider("Grid size", 5, 80, 30, 5, key=f"{prefix}_gridsize")
    else:
        config["bins"] = st.slider("Bins", 5, 100, 30, 5, key=f"{prefix}_bins")
    config["colorbar"] = st.checkbox("Show colorbar", value=True, key=f"{prefix}_colorbar")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_pair_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    if not isinstance(data, pd.DataFrame):
        st.warning("Pair plot requires a DataFrame.")
        return config
    numeric = _numeric_cols(data)
    all_cols = _all_cols(data)
    if len(numeric) < 2:
        st.warning("Pair plot requires at least 2 numeric columns.")
        return config

    selected_vars = st.multiselect(
        "Variables to plot (leave empty for all numeric)",
        numeric,
        default=[],
        key=f"{prefix}_vars",
    )
    config["vars"] = selected_vars if selected_vars else None

    hue_opts = ["None"] + all_cols
    hue_sel = st.selectbox("Hue column", hue_opts, index=0, key=f"{prefix}_hue")
    config["hue"] = None if hue_sel == "None" else hue_sel

    col1, col2 = st.columns(2)
    with col1:
        config["kind"] = st.selectbox(
            "Off-diagonal", ["scatter", "kde", "hist", "reg"], index=0, key=f"{prefix}_kind"
        )
    with col2:
        config["diag_kind"] = st.selectbox(
            "Diagonal", ["hist", "kde", "auto"], index=0, key=f"{prefix}_diagkind"
        )

    config["title"] = st.text_input("Title", value="", key=f"{prefix}_title") or None
    return config


def _widgets_interactive_3d_scatter(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X column", numeric, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox(
                "Y column", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_ycol"
            )
        with col3:
            config["_z_col"] = st.selectbox(
                "Z column", numeric, index=min(2, len(numeric) - 1), key=f"{prefix}_zcol"
            )
        extra_opts = ["None"] + list(numeric)
        col_a, col_b = st.columns(2)
        with col_a:
            color_sel = st.selectbox(
                "Color column (optional)", extra_opts, index=0, key=f"{prefix}_ccol"
            )
            config["_color_col"] = None if color_sel == "None" else color_sel
        with col_b:
            size_sel = st.selectbox(
                "Size column (optional)", extra_opts, index=0, key=f"{prefix}_scol"
            )
            config["_size_col"] = None if size_sel == "None" else size_sel
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X col index", opts, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox("Y col index", opts, index=1, key=f"{prefix}_ycol")
        with col3:
            config["_z_col"] = st.selectbox("Z col index", opts, index=2, key=f"{prefix}_zcol")
        config["_color_col"] = None
        config["_size_col"] = None
    else:
        st.warning("3D scatter requires at least 3 columns.")

    st.markdown("**Appearance**")
    col1, col2 = st.columns(2)
    with col1:
        config["colorscale"] = st.selectbox("Colorscale", _COLORSCALES, index=0, key=f"{prefix}_cs")
        config["marker_size"] = st.slider("Marker size", 2, 20, 5, 1, key=f"{prefix}_msize")
    with col2:
        config["opacity"] = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key=f"{prefix}_opacity")
    config["title"] = st.text_input("Title", value="", key=f"{prefix}_title") or None
    config["xlabel"] = st.text_input("X label", value="", key=f"{prefix}_xl") or None
    config["ylabel"] = st.text_input("Y label", value="", key=f"{prefix}_yl") or None
    config["zlabel"] = st.text_input("Z label", value="", key=f"{prefix}_zl") or None
    with st.expander("Plotly settings"):
        config["plotly_height"] = st.slider("Height (px)", 400, 1000, 600, 50, key=f"{prefix}_ph")
    return config


def _widgets_scatter_with_regression(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox("X column", numeric, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox(
                "Y column", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_ycol"
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox("X col index", opts, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox("Y col index", opts, index=1, key=f"{prefix}_ycol")
    else:
        st.warning("Scatter + Regression requires at least 2 numeric columns.")
    col1, col2 = st.columns(2)
    with col1:
        config["color"] = _color_widget(prefix)
        config["show_ci"] = st.checkbox("Show 95% CI band", value=True, key=f"{prefix}_ci")
    with col2:
        config["alpha"] = st.slider("Opacity", 0.1, 1.0, 0.7, 0.05, key=f"{prefix}_alpha")
        config["show_equation"] = st.checkbox("Show equation", value=True, key=f"{prefix}_eq")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_residual_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X column", numeric, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_actual_col"] = st.selectbox(
                "Y actual", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_yacol"
            )
        with col3:
            config["_y_fitted_col"] = st.selectbox(
                "Y fitted", numeric, index=min(2, len(numeric) - 1), key=f"{prefix}_yfcol"
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_x_col"] = st.selectbox("X col index", opts, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_actual_col"] = st.selectbox(
                "Y actual index", opts, index=1, key=f"{prefix}_yacol"
            )
        with col3:
            config["_y_fitted_col"] = st.selectbox(
                "Y fitted index", opts, index=2, key=f"{prefix}_yfcol"
            )
    else:
        st.warning("Residual plot requires at least 3 columns: x, y_actual, y_fitted.")
    config["vs_fitted"] = st.checkbox(
        "Plot vs. fitted values (not x)", value=False, key=f"{prefix}_vsfitted"
    )
    config["show_zero_line"] = st.checkbox("Show zero line", value=True, key=f"{prefix}_zeroline")
    config["alpha"] = st.slider("Opacity", 0.1, 1.0, 0.75, 0.05, key=f"{prefix}_alpha")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_interactive_ternary(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_a_col"] = st.selectbox("A column", numeric, index=0, key=f"{prefix}_acol")
        with col2:
            config["_b_col"] = st.selectbox(
                "B column", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_bcol"
            )
        with col3:
            config["_c_col"] = st.selectbox(
                "C column", numeric, index=min(2, len(numeric) - 1), key=f"{prefix}_ccol"
            )
        extra_opts = ["None"] + numeric
        color_sel = st.selectbox(
            "Color column (optional)", extra_opts, index=0, key=f"{prefix}_colcol"
        )
        config["_color_col"] = None if color_sel == "None" else color_sel
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2, col3 = st.columns(3)
        with col1:
            config["_a_col"] = st.selectbox("A col index", opts, index=0, key=f"{prefix}_acol")
        with col2:
            config["_b_col"] = st.selectbox("B col index", opts, index=1, key=f"{prefix}_bcol")
        with col3:
            config["_c_col"] = st.selectbox("C col index", opts, index=2, key=f"{prefix}_ccol")
        config["_color_col"] = None
    else:
        st.warning("Ternary plot requires at least 3 columns (A, B, C components).")
    col1, col2, col3 = st.columns(3)
    with col1:
        config["a_label"] = st.text_input("A label", value="A", key=f"{prefix}_al")
    with col2:
        config["b_label"] = st.text_input("B label", value="B", key=f"{prefix}_bl")
    with col3:
        config["c_label"] = st.text_input("C label", value="C", key=f"{prefix}_cl")
    config["colorscale"] = st.selectbox("Colorscale", _COLORSCALES, index=0, key=f"{prefix}_cs")
    config["marker_size"] = st.slider("Marker size", 3, 20, 8, 1, key=f"{prefix}_ms")
    config["opacity"] = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05, key=f"{prefix}_op")
    config["title"] = st.text_input("Title", value="", key=f"{prefix}_title") or None
    with st.expander("Plotly settings"):
        config["plotly_height"] = st.slider("Height (px)", 400, 900, 600, 50, key=f"{prefix}_ph")
    return config


def _widgets_broken_axis_plot(data: Any, prefix: str) -> Dict:
    config: Dict = {}
    numeric = _numeric_cols(data)
    if isinstance(data, pd.DataFrame) and len(numeric) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox("X column", numeric, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox(
                "Y column", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_ycol"
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        col1, col2 = st.columns(2)
        with col1:
            config["_x_col"] = st.selectbox("X col index", opts, index=0, key=f"{prefix}_xcol")
        with col2:
            config["_y_col"] = st.selectbox("Y col index", opts, index=1, key=f"{prefix}_ycol")
    else:
        st.warning("Broken axis plot requires at least 2 columns (x and y).")

    st.markdown("**Axis breaks** — define regions to skip")
    n_breaks = st.number_input(
        "Number of breaks", min_value=1, max_value=4, value=1, step=1, key=f"{prefix}_nbreaks"
    )
    breaks = []
    for i in range(int(n_breaks)):
        c1, c2 = st.columns(2)
        with c1:
            lo = st.number_input(
                f"Break {i + 1} start", value=0.0, key=f"{prefix}_blo{i}", format="%.4g"
            )
        with c2:
            hi = st.number_input(
                f"Break {i + 1} end", value=1.0, key=f"{prefix}_bhi{i}", format="%.4g"
            )
        if hi > lo:
            breaks.append((float(lo), float(hi)))
    config["_breaks"] = breaks if breaks else [(0.0, 0.0)]
    config["axis"] = st.selectbox("Break axis", ["x", "y"], index=0, key=f"{prefix}_axis")
    col1, col2 = st.columns(2)
    with col1:
        config["color"] = _color_widget(prefix)
    with col2:
        config["linewidth"] = st.slider("Line width", 0.5, 5.0, 1.5, 0.25, key=f"{prefix}_lw")
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


def _widgets_inset_plot(data: Any, prefix: str) -> Dict:
    """Widget function for the inset_plot specialty plot type."""
    config: Dict = {}
    numeric = _numeric_cols(data)

    if isinstance(data, pd.DataFrame):
        if len(numeric) < 2:
            st.warning("Inset plot requires at least 2 numeric columns.")
            return {}
        st.markdown("**Main plot columns**")
        c1, c2 = st.columns(2)
        with c1:
            config["_x_col"] = st.selectbox(
                "X column (main)", numeric, index=0, key=f"{prefix}_xcol"
            )
        with c2:
            config["_y_col"] = st.selectbox(
                "Y column (main)", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_ycol"
            )
        st.markdown("**Inset plot columns**")
        c3, c4 = st.columns(2)
        with c3:
            config["_xi_col"] = st.selectbox(
                "X column (inset)", numeric, index=0, key=f"{prefix}_xicol"
            )
        with c4:
            config["_yi_col"] = st.selectbox(
                "Y column (inset)", numeric, index=min(1, len(numeric) - 1), key=f"{prefix}_yicol"
            )
    elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
        opts = [str(i) for i in range(data.shape[1])]
        st.markdown("**Main plot columns**")
        c1, c2 = st.columns(2)
        with c1:
            config["_x_col"] = st.selectbox(
                "X col index (main)", opts, index=0, key=f"{prefix}_xcol"
            )
        with c2:
            config["_y_col"] = st.selectbox(
                "Y col index (main)", opts, index=min(1, len(opts) - 1), key=f"{prefix}_ycol"
            )
        st.markdown("**Inset plot columns**")
        c3, c4 = st.columns(2)
        with c3:
            config["_xi_col"] = st.selectbox(
                "X col index (inset)", opts, index=0, key=f"{prefix}_xicol"
            )
        with c4:
            config["_yi_col"] = st.selectbox(
                "Y col index (inset)", opts, index=min(1, len(opts) - 1), key=f"{prefix}_yicol"
            )
    else:
        st.warning("Inset plot requires a DataFrame or 2D array with >= 2 columns.")
        return {}

    st.markdown("**Inset bounds** (axes fraction 0–1)")
    ib1, ib2, ib3, ib4 = st.columns(4)
    with ib1:
        config["_ib_x0"] = st.number_input(
            "x0", min_value=0.0, max_value=0.95, value=0.55, step=0.05, key=f"{prefix}_ibx0"
        )
    with ib2:
        config["_ib_y0"] = st.number_input(
            "y0", min_value=0.0, max_value=0.95, value=0.55, step=0.05, key=f"{prefix}_iby0"
        )
    with ib3:
        config["_ib_w"] = st.number_input(
            "width", min_value=0.05, max_value=0.9, value=0.4, step=0.05, key=f"{prefix}_ibw"
        )
    with ib4:
        config["_ib_h"] = st.number_input(
            "height", min_value=0.05, max_value=0.9, value=0.35, step=0.05, key=f"{prefix}_ibh"
        )

    st.markdown("**Indicate region on main plot** (optional)")
    use_region = st.checkbox("Shade region on main plot", value=False, key=f"{prefix}_useregion")
    if use_region:
        rc1, rc2 = st.columns(2)
        with rc1:
            config["_region_start"] = st.number_input(
                "Region start", value=0.0, key=f"{prefix}_rstart"
            )
        with rc2:
            config["_region_end"] = st.number_input("Region end", value=1.0, key=f"{prefix}_rend")

    st.markdown("**Labels**")
    config["inset_xlabel"] = (
        st.text_input("Inset X label", value="", key=f"{prefix}_ixlabel") or None
    )
    config["inset_ylabel"] = (
        st.text_input("Inset Y label", value="", key=f"{prefix}_iylabel") or None
    )
    config["figsize"] = _figsize_widget(prefix)
    config.update(_label_widgets(prefix))
    with st.expander("Appearance"):
        config.update(_grid_widgets(prefix))
        config.update(_appearance_widgets(prefix, include_linewidth=True))
        config.update(_axis_limits_widgets(prefix))
        config.update(_caption_widget(prefix))
    return config


# ── Dispatch table ───────────────────────────────────────────────────────────

_WIDGET_FUNCS: Dict[str, Any] = {
    "histogram": _widgets_histogram,
    "line_plot": _widgets_line_plot,
    "scatter_plot": _widgets_scatter_plot,
    "bar_chart": _widgets_bar_chart,
    "heatmap": _widgets_heatmap,
    "contour_plot": _widgets_contour_plot,
    "waterfall_plot": _widgets_waterfall_plot,
    "dual_axis_plot": _widgets_dual_axis_plot,
    "broken_axis_plot": _widgets_broken_axis_plot,
    "z_colored_scatter": _widgets_z_colored_scatter,
    "bubble_chart": _widgets_bubble_chart,
    "polar_plot": _widgets_polar_plot,
    "histogram_2d": _widgets_histogram_2d,
    "distribution_plot": _widgets_distribution_plot,
    "box_plot": _widgets_box_plot,
    "regression_plot": _widgets_regression_plot,
    "pair_plot": _widgets_pair_plot,
    "interactive_histogram": _widgets_interactive_histogram,
    "interactive_scatter": _widgets_interactive_scatter,
    "interactive_line": _widgets_interactive_line,
    "interactive_heatmap": _widgets_interactive_heatmap,
    "interactive_3d_surface": _widgets_interactive_3d_surface,
    "interactive_3d_scatter": _widgets_interactive_3d_scatter,
    "scatter_with_regression": _widgets_scatter_with_regression,
    "residual_plot": _widgets_residual_plot,
    "interactive_ternary": _widgets_interactive_ternary,
    "inset_plot": _widgets_inset_plot,
}


# ── Public API ───────────────────────────────────────────────────────────────


def get_plot_config_widgets(plot_type: str, data: Any) -> Dict:
    """Render Streamlit widgets for *plot_type* and return a config dict.

    Call inside a Streamlit page after loading a dataset.  Renders
    data-column selectors and plot-parameter controls appropriate for the
    chosen plot type and the actual type/shape of *data*.

    The returned dictionary contains two kinds of keys:

    * **Public keys** — valid keyword arguments for the matching plotting
      function (``title``, ``xlabel``, ``bins``, ``color``, ``figsize``, …).
    * **Private keys** (prefixed ``_``) — data-selection hints the caller
      uses to extract arrays from *data* before calling the plotting
      function (e.g. ``_x_col``, ``_y_col``, ``_data_col``).

    Use :func:`get_plot_kwargs` to strip private keys before unpacking
    into a plotting function.

    Parameters
    ----------
    plot_type : str
        One of the keys in :data:`PLOT_TYPES`.
    data : Any
        Currently loaded dataset (DataFrame, ndarray, …).

    Returns
    -------
    config : dict
        Configuration dict; empty dict if *plot_type* is unknown or *data*
        is incompatible with the selected plot type.

    Examples
    --------
    >>> config = get_plot_config_widgets('histogram', arr)
    >>> col = config.get('_data_col')      # None -> use full array
    >>> plot_data = arr if col is None else df[col]
    >>> fig, ax, info = histogram(plot_data, **get_plot_kwargs(config))
    >>> st.pyplot(fig)
    """
    if plot_type not in _WIDGET_FUNCS:
        st.error(f"Unknown plot type: '{plot_type}'")
        return {}
    return _WIDGET_FUNCS[plot_type](data, plot_type)


def get_plot_kwargs(config: Dict) -> Dict:
    """Return plotting-function kwargs extracted from *config*.

    Strips private (``_``-prefixed) keys and drops ``None`` values so the
    result can be unpacked directly into any plotting function.

    Parameters
    ----------
    config : dict
        Raw config returned by :func:`get_plot_config_widgets`.

    Returns
    -------
    kwargs : dict
        Clean dict ready for ``**kwargs`` unpacking into a plotting function.

    Examples
    --------
    >>> config = get_plot_config_widgets('scatter_plot', data)
    >>> fig, ax = scatter_plot(x, y, **get_plot_kwargs(config))
    """
    return {
        k: v
        for k, v in config.items()
        if not k.startswith("_") and v is not None and k not in _STYLE_KEYS
    }


# ── Click-to-annotate helpers (Quick Plot, Plotly mode) ──────────────────────


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return getattr(obj, key, default)


def selected_points(event: Any) -> List[Dict[str, Any]]:
    """Extract ``{"x", "y"}`` pairs from a ``st.plotly_chart`` selection event.

    Streamlit returns each selected point as a plain ``dict`` (not an object),
    so attribute access such as ``point.x`` raises ``AttributeError``. This
    helper reads both forms and skips points that carry no x/y coordinate
    (e.g. pie slices).

    Args:
        event: The value returned by ``st.plotly_chart(..., on_select="rerun")``,
            or ``None``.

    Returns:
        List of ``{"x": ..., "y": ...}`` dicts, in selection order.
    """
    if not event:
        return []
    selection = _get(event, "selection")
    if not selection:
        return []
    points = _get(selection, "points") or []
    result: List[Dict[str, Any]] = []
    for pt in points:
        x, y = _get(pt, "x"), _get(pt, "y")
        if x is None or y is None:
            continue
        result.append({"x": x, "y": y})
    return result


def format_marker_label(x: Any, y: Any) -> str:
    """Format a marked point as ``"(x, y)"``; numbers use 4 significant digits.

    Non-numeric coordinates (category names, date strings) are shown as-is
    instead of raising on the ``:.4g`` format spec.
    """

    def _fmt(v: Any) -> str:
        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            return f"{v:.4g}"
        return str(v)

    return f"({_fmt(x)}, {_fmt(y)})"
