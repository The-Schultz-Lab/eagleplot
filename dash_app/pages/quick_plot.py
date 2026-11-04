"""Quick Plot page — 26 plot types with live configuration.

Left panel: dataset / column / plot-type selection + ~50 configuration controls
Right panel: chart output (Plotly interactive or Matplotlib as PNG)

All 26 plot types from plottle.plotting are available:
  Matplotlib: histogram, line_plot, scatter_plot, bar_chart, heatmap,
              contour_plot, waterfall_plot, dual_axis_plot, broken_axis_plot,
              z_colored_scatter, bubble_chart, polar_plot, histogram_2d,
              scatter_with_regression, residual_plot, inset_plot
  Seaborn:    distribution_plot, box_plot, regression_plot, pair_plot
  Plotly:     interactive_histogram, interactive_scatter, interactive_line,
              interactive_heatmap, interactive_3d_surface, interactive_3d_scatter,
              interactive_ternary
"""

import io
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from plottle.plotting import (
    bar_chart,
    box_plot,
    broken_axis_plot,
    bubble_chart,
    contour_plot,
    distribution_plot,
    dual_axis_plot,
    heatmap,
    histogram,
    histogram_2d,
    inset_plot,
    interactive_3d_scatter,
    interactive_3d_surface,
    interactive_heatmap,
    interactive_histogram,
    interactive_line,
    interactive_scatter,
    interactive_ternary,
    line_plot,
    pair_plot,
    polar_plot,
    regression_plot,
    residual_plot,
    scatter_plot,
    scatter_with_regression,
    waterfall_plot,
    z_colored_scatter,
)

dash.register_page(__name__, path="/plot-basic", title="Quick Plot — Plottle", name="Quick Plot")

# ── Plot type registry ────────────────────────────────────────────────────────

_PLOT_TYPES = {
    "Matplotlib": [
        "Histogram",
        "Line Plot",
        "Scatter Plot",
        "Bar Chart",
        "Heatmap",
        "Contour Plot",
        "Waterfall Plot",
        "Dual Axis Plot",
        "Broken Axis Plot",
        "Z-Colored Scatter",
        "Bubble Chart",
        "Polar Plot",
        "2D Histogram",
        "Scatter with Regression",
        "Residual Plot",
        "Inset Plot",
    ],
    "Seaborn": ["Distribution Plot", "Box Plot", "Regression Plot", "Pair Plot"],
    "Plotly": [
        "Interactive Histogram",
        "Interactive Scatter",
        "Interactive Line",
        "Interactive Heatmap",
        "Interactive 3D Surface",
        "Interactive 3D Scatter",
        "Interactive Ternary",
    ],
}

_ALL_PLOT_LABELS = [pt for pts in _PLOT_TYPES.values() for pt in pts]

_COLOR_PALETTES = {
    "Default": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    "Okabe-Ito": [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ],
    "Wong": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    "Tol Muted": [
        "#332288",
        "#117733",
        "#44AA99",
        "#88CCEE",
        "#DDCC77",
        "#CC6677",
        "#AA4499",
        "#882255",
    ],
    "Pastel": ["#AEC6CF", "#FFD1DC", "#B5EAD7", "#FFDAC1", "#C7CEEA", "#F2C6DE", "#E2F0CB"],
    "Vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98FB98"],
}

_FONT_OPTIONS = [
    "sans-serif",
    "serif",
    "monospace",
    "Helvetica",
    "Arial",
    "Times New Roman",
    "DejaVu Sans",
]

_LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]
_MARKERS = ["none", "circle", "square", "diamond", "triangle-up", "x", "cross"]

# ── Layout ────────────────────────────────────────────────────────────────────


def layout(**kwargs):
    names = state.get_dataset_names()
    current = state._STATE.get("current_dataset")

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Quick Plot", className="page-title"),
                    html.P(
                        "Select a dataset, choose a plot type, and configure.",
                        className="page-caption",
                    ),
                ],
                className="page-header",
            ),
            dbc.Row(
                [
                    # Left config panel
                    dbc.Col(_config_panel(names, current), width=3, id="qp-config-col"),
                    # Right plot panel
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Generate Plot",
                                            id="qp-generate-btn",
                                            color="primary",
                                            className="w-100",
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Save to History",
                                            id="qp-save-btn",
                                            color="secondary",
                                            className="w-100",
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Clear",
                                            id="qp-clear-btn",
                                            color="outline-secondary",
                                            className="w-100",
                                        ),
                                        width=3,
                                    ),
                                ],
                                className="g-2 mb-3",
                            ),
                            html.Div(id="qp-plot-output", className="plot-container"),
                            html.Div(id="qp-feedback", className="mt-2"),
                        ],
                        width=9,
                    ),
                ],
                className="g-3",
            ),
        ]
    )


def _config_panel(names, current):
    ds_opts = [{"label": n, "value": n} for n in names]

    return html.Div(
        [
            # ── Dataset & Plot type ──────────────────────────────────────────
            html.P("DATASET", className="config-section-title"),
            dcc.Dropdown(
                id="qp-dataset",
                options=ds_opts,
                value=current,
                clearable=False,
                placeholder="Select dataset…",
            ),
            html.Div(id="qp-col-selectors", className="mt-2"),
            html.P("PLOT TYPE", className="config-section-title mt-3"),
            dcc.Dropdown(
                id="qp-plot-type",
                options=[
                    {"label": f"[{lib}] {pt}", "value": pt}
                    for lib, pts in _PLOT_TYPES.items()
                    for pt in pts
                ],
                value="Histogram",
                clearable=False,
            ),
            dcc.Checklist(
                id="qp-pub-style",
                options=[{"label": " Publication style", "value": "pub"}],
                value=[],
                className="mt-2",
                inputStyle={"marginRight": "6px"},
            ),
            # ── Title & Labels ───────────────────────────────────────────────
            html.P("LABELS", className="config-section-title mt-3"),
            dbc.Input(id="qp-title", placeholder="Title", type="text", size="sm", className="mb-1"),
            dbc.Input(
                id="qp-xlabel", placeholder="X label", type="text", size="sm", className="mb-1"
            ),
            dbc.Input(id="qp-ylabel", placeholder="Y label", type="text", size="sm"),
            # ── Style ────────────────────────────────────────────────────────
            html.P("STYLE", className="config-section-title mt-3"),
            dbc.Label("Font family"),
            dcc.Dropdown(
                id="qp-fontfamily",
                options=[{"label": f, "value": f} for f in _FONT_OPTIONS],
                value="sans-serif",
                clearable=False,
                className="mb-1",
            ),
            dbc.Label("Font size"),
            dcc.Slider(
                id="qp-fontsize",
                min=6,
                max=24,
                step=1,
                value=10,
                marks={6: "6", 12: "12", 18: "18", 24: "24"},
                tooltip={"placement": "bottom"},
            ),
            dbc.Label("Line width"),
            dcc.Slider(
                id="qp-linewidth",
                min=0.5,
                max=5.0,
                step=0.25,
                value=1.5,
                marks={1: "1", 2: "2", 3: "3", 5: "5"},
                tooltip={"placement": "bottom"},
            ),
            dbc.Label("Marker size"),
            dcc.Slider(
                id="qp-markersize",
                min=2,
                max=20,
                step=1,
                value=6,
                marks={2: "2", 10: "10", 20: "20"},
                tooltip={"placement": "bottom"},
            ),
            dbc.Label("Alpha (opacity)"),
            dcc.Slider(
                id="qp-alpha",
                min=0.1,
                max=1.0,
                step=0.05,
                value=0.85,
                marks={0.25: ".25", 0.5: ".5", 0.75: ".75", 1.0: "1"},
                tooltip={"placement": "bottom"},
            ),
            # ── Color palette ────────────────────────────────────────────────
            html.P("COLORS", className="config-section-title mt-3"),
            dcc.Dropdown(
                id="qp-palette",
                options=[{"label": k, "value": k} for k in _COLOR_PALETTES],
                value="Default",
                clearable=False,
            ),
            dbc.Label("Single color", className="mt-1"),
            dbc.Input(
                id="qp-color",
                type="color",
                value="#1f77b4",
                className="mb-1",
                style={"height": "36px", "padding": "2px"},
            ),
            # ── Grid ─────────────────────────────────────────────────────────
            html.P("GRID", className="config-section-title mt-3"),
            dbc.Checklist(
                id="qp-grid",
                options=[{"label": " Show grid", "value": "grid"}],
                value=["grid"],
                inputStyle={"marginRight": "6px"},
            ),
            dbc.Label("Grid style"),
            dcc.Dropdown(
                id="qp-grid-ls",
                options=[
                    {"label": "Solid", "value": "-"},
                    {"label": "Dashed", "value": "--"},
                    {"label": "Dotted", "value": ":"},
                    {"label": "Dash-dot", "value": "-."},
                ],
                value="--",
                clearable=False,
            ),
            # ── Legend ───────────────────────────────────────────────────────
            html.P("LEGEND", className="config-section-title mt-3"),
            dbc.Checklist(
                id="qp-legend-show",
                options=[{"label": " Show legend", "value": "leg"}],
                value=["leg"],
                inputStyle={"marginRight": "6px"},
            ),
            dcc.Dropdown(
                id="qp-legend-pos",
                options=[
                    {"label": "Best", "value": "best"},
                    {"label": "Upper right", "value": "upper right"},
                    {"label": "Upper left", "value": "upper left"},
                    {"label": "Lower right", "value": "lower right"},
                    {"label": "Lower left", "value": "lower left"},
                    {"label": "Outside right", "value": "outside"},
                ],
                value="best",
                clearable=False,
            ),
            # ── Histogram bins ───────────────────────────────────────────────
            html.P("HISTOGRAM", className="config-section-title mt-3", id="qp-hist-section"),
            dbc.Label("Bins"),
            dcc.Slider(
                id="qp-bins",
                min=5,
                max=200,
                step=5,
                value=30,
                marks={5: "5", 50: "50", 100: "100", 200: "200"},
                tooltip={"placement": "bottom"},
            ),
            # ── Figure size ──────────────────────────────────────────────────
            html.P("FIGURE SIZE", className="config-section-title mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Width"),
                            dbc.Input(
                                id="qp-figw", type="number", value=9, min=3, max=20, size="sm"
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Height"),
                            dbc.Input(
                                id="qp-figh", type="number", value=5, min=2, max=15, size="sm"
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="g-2",
            ),
            # ── Annotations ──────────────────────────────────────────────────
            html.P("ANNOTATIONS", className="config-section-title mt-3"),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dbc.Label("Type"),
                            dcc.Dropdown(
                                id="qp-ann-type",
                                options=[
                                    {"label": "Horizontal line", "value": "hline"},
                                    {"label": "Vertical line", "value": "vline"},
                                    {"label": "H span", "value": "hspan"},
                                    {"label": "V span", "value": "vspan"},
                                    {"label": "Text label", "value": "text"},
                                    {"label": "Rectangle", "value": "rectangle"},
                                    {"label": "Ellipse", "value": "ellipse"},
                                ],
                                value="hline",
                                clearable=False,
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("x"),
                                            dbc.Input(id="qp-ann-x", type="number", size="sm"),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("y"),
                                            dbc.Input(id="qp-ann-y", type="number", size="sm"),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="g-2 mt-1",
                            ),
                            dbc.Input(
                                id="qp-ann-label",
                                placeholder="Label text",
                                type="text",
                                size="sm",
                                className="mt-1",
                            ),
                            dbc.Input(
                                id="qp-ann-color",
                                type="color",
                                value="#e0a3a3",
                                style={"height": "32px"},
                                className="mt-1",
                            ),
                            dbc.Button(
                                "Add annotation",
                                id="qp-ann-add-btn",
                                color="secondary",
                                size="sm",
                                className="w-100 mt-2",
                            ),
                            html.Div(id="qp-ann-list", className="mt-2"),
                            dbc.Button(
                                "Clear all",
                                id="qp-ann-clear-btn",
                                color="outline-secondary",
                                size="sm",
                                className="w-100 mt-1",
                            ),
                        ],
                        title="Add Annotation",
                    )
                ],
                flush=True,
                start_collapsed=True,
            ),
            # ── Data transform ───────────────────────────────────────────────
            html.P("Y TRANSFORM", className="config-section-title mt-3"),
            dcc.Dropdown(
                id="qp-ytransform",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Normalize to max", "value": "normalize_max"},
                    {"label": "Normalize 0–1", "value": "normalize_01"},
                    {"label": "Scale by value", "value": "scale_by"},
                ],
                value="none",
                clearable=False,
            ),
            dbc.Input(
                id="qp-yscale-val",
                type="number",
                value=1.0,
                placeholder="Scale value",
                size="sm",
                className="mt-1",
            ),
        ],
        className="config-panel",
    )


# ── Column selectors (depend on dataset) ─────────────────────────────────────


@callback(Output("qp-col-selectors", "children"), Input("qp-dataset", "value"))
def update_col_selectors(name):
    if not name:
        return html.Div()
    data = state.get_dataset(name)
    if data is None:
        return html.Div()

    if isinstance(data, pd.DataFrame):
        all_cols = list(data.columns)
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        col_opts = [{"label": c, "value": c} for c in all_cols]
        num_opts = [{"label": c, "value": c} for c in num_cols]
        default_x = num_cols[0] if num_cols else (all_cols[0] if all_cols else None)
        default_y = num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None)
        return html.Div(
            [
                dbc.Label("X column"),
                dcc.Dropdown(
                    id="qp-xcol",
                    options=col_opts,
                    value=default_x,
                    clearable=False,
                    className="mb-1",
                ),
                dbc.Label("Y column"),
                dcc.Dropdown(
                    id="qp-ycol",
                    options=num_opts,
                    value=default_y,
                    clearable=False,
                    className="mb-1",
                ),
                dbc.Label("Z / color column (optional)"),
                dcc.Dropdown(
                    id="qp-zcol",
                    options=[{"label": "None", "value": ""}] + num_opts,
                    value="",
                    clearable=True,
                    className="mb-1",
                ),
                dbc.Label("Grouping column (optional)"),
                dcc.Dropdown(
                    id="qp-gcol",
                    options=[{"label": "None", "value": ""}] + col_opts,
                    value="",
                    clearable=True,
                ),
            ]
        )
    if isinstance(data, np.ndarray) and data.ndim == 1:
        return html.Div(
            [
                dcc.Store(id="qp-xcol", data=""),
                dcc.Store(id="qp-ycol", data=""),
                dcc.Store(id="qp-zcol", data=""),
                dcc.Store(id="qp-gcol", data=""),
                html.P(f"1D array · {data.shape[0]} values", className="text-muted-sm"),
            ]
        )
    return html.Div(
        [
            dcc.Store(id="qp-xcol", data=""),
            dcc.Store(id="qp-ycol", data=""),
            dcc.Store(id="qp-zcol", data=""),
            dcc.Store(id="qp-gcol", data=""),
            html.P(f"Array shape: {data.shape} | dtype: {data.dtype}", className="text-muted-sm"),
        ]
    )


# ── Generate plot ─────────────────────────────────────────────────────────────


@callback(
    Output("qp-plot-output", "children"),
    Output("qp-feedback", "children"),
    Input("qp-generate-btn", "n_clicks"),
    State("qp-dataset", "value"),
    State("qp-plot-type", "value"),
    State("qp-xcol", "value"),
    State("qp-ycol", "value"),
    State("qp-zcol", "value"),
    State("qp-gcol", "value"),
    State("qp-title", "value"),
    State("qp-xlabel", "value"),
    State("qp-ylabel", "value"),
    State("qp-fontfamily", "value"),
    State("qp-fontsize", "value"),
    State("qp-linewidth", "value"),
    State("qp-markersize", "value"),
    State("qp-alpha", "value"),
    State("qp-palette", "value"),
    State("qp-color", "value"),
    State("qp-grid", "value"),
    State("qp-grid-ls", "value"),
    State("qp-legend-show", "value"),
    State("qp-legend-pos", "value"),
    State("qp-bins", "value"),
    State("qp-figw", "value"),
    State("qp-figh", "value"),
    State("qp-ytransform", "value"),
    State("qp-yscale-val", "value"),
    State("qp-pub-style", "value"),
    prevent_initial_call=True,
)
def generate_plot(
    n,
    ds_name,
    plot_type,
    xcol,
    ycol,
    zcol,
    gcol,
    title,
    xlabel,
    ylabel,
    fontfamily,
    fontsize,
    linewidth,
    markersize,
    alpha,
    palette,
    color,
    grid_on,
    grid_ls,
    legend_on,
    legend_pos,
    bins,
    figw,
    figh,
    ytransform,
    yscale_val,
    pub_style,
):
    if not ds_name:
        return html.Div(
            html.P("No dataset selected.", className="plot-placeholder")
        ), dash.no_update

    data = state.get_dataset(ds_name)
    if data is None:
        return html.Div(html.P("Dataset not found.", className="plot-placeholder")), dash.no_update

    config = dict(
        title=title or "",
        xlabel=xlabel or "",
        ylabel=ylabel or "",
        fontfamily=fontfamily or "sans-serif",
        fontsize=int(fontsize or 10),
        linewidth=float(linewidth or 1.5),
        markersize=int(markersize or 6),
        alpha=float(alpha or 0.85),
        color_palette=palette or "Default",
        color=color or "#1f77b4",
        grid="grid" in (grid_on or []),
        grid_linestyle=grid_ls or "--",
        legend="leg" in (legend_on or []),
        legend_position=legend_pos or "best",
        bins=int(bins or 30),
        figsize=(float(figw or 9), float(figh or 5)),
        ytransform=ytransform or "none",
        yscale_val=float(yscale_val or 1.0),
        pub_style="pub" in (pub_style or []),
    )

    # Apply publication style overrides
    if config["pub_style"]:
        config["fontfamily"] = "serif"
        config["fontsize"] = 12
        config["linewidth"] = 1.25

    # Apply color cycle
    colors = _COLOR_PALETTES.get(palette, _COLOR_PALETTES["Default"])
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=colors)

    # Apply matplotlib style
    if config["pub_style"]:
        plt.style.use("default")
        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 12,
                "axes.linewidth": 0.8,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )
    else:
        plt.style.use("dark_background")
        matplotlib.rcParams.update(
            {
                "font.family": fontfamily or "sans-serif",
                "font.size": int(fontsize or 10),
            }
        )

    try:
        fig_output = _dispatch_plot(data, plot_type, xcol, ycol, zcol, gcol, config)
        return fig_output, html.Div()
    except Exception as exc:
        return (
            html.Div(html.P("Plot generation failed.", className="plot-placeholder")),
            dbc.Alert(
                [
                    html.Strong("Error: "),
                    str(exc),
                    dbc.Button("Details", id="qp-err-toggle", size="sm", className="ms-2"),
                ],
                color="danger",
            ),
        )


def _dispatch_plot(data, plot_type: str, xcol, ycol, zcol, gcol, cfg: dict):
    """Route to the correct plotting function and return a Dash component."""

    def _get_col(name: str):
        if not name or name == "":
            return None
        if isinstance(data, pd.DataFrame):
            return data[name].values if name in data.columns else None
        return None

    def _y_transform(arr):
        if arr is None:
            return arr
        t = cfg.get("ytransform", "none")
        if t == "normalize_max":
            mx = np.max(np.abs(arr))
            return arr / mx if mx != 0 else arr
        if t == "normalize_01":
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn) if mx != mn else arr
        if t == "scale_by":
            return arr * cfg.get("yscale_val", 1.0)
        return arr

    def _df_or_none():
        return data if isinstance(data, pd.DataFrame) else None

    def _arr_1d():
        if isinstance(data, pd.DataFrame):
            col = ycol or (list(data.select_dtypes(include=[np.number]).columns) or [None])[0]
            return data[col].dropna().values if col else np.array([])
        if isinstance(data, np.ndarray):
            return data.flatten()
        return np.array([])

    def _arr_2d():
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            return data
        if isinstance(data, pd.DataFrame):
            return data.select_dtypes(include=[np.number]).values
        return np.zeros((5, 5))

    kwargs = dict(
        title=cfg["title"] or None,
        xlabel=cfg["xlabel"] or None,
        ylabel=cfg["ylabel"] or None,
        figsize=cfg["figsize"],
        color=cfg["color"],
        alpha=cfg["alpha"],
        linewidth=cfg["linewidth"],
        grid=cfg["grid"],
    )

    # ── Plotly family (return dcc.Graph) ──────────────────────────────────────
    # prototype fix: interactive_histogram/scatter/line/heatmap/3d_surface return a
    # bare go.Figure (not (fig, info)), so the April `fig, info = ...` unpack raised.
    if plot_type == "Interactive Histogram":
        fig = interactive_histogram(
            _arr_1d(),
            bins=cfg["bins"],
            title=cfg["title"] or "Histogram",
            xlabel=cfg["xlabel"] or "",
            ylabel=cfg["ylabel"] or "Count",
        )
        return dcc.Graph(figure=_style_plotly(fig, cfg), style={"height": "500px"})

    if plot_type == "Interactive Scatter":
        x = _get_col(xcol)
        if x is None:  # prototype fix: `array or ...` raised ValueError
            x = np.arange(len(_arr_1d()))
        y = _y_transform(_arr_1d())
        fig = interactive_scatter(
            x, y, title=cfg["title"] or "", xlabel=cfg["xlabel"] or "", ylabel=cfg["ylabel"] or ""
        )
        return dcc.Graph(figure=_style_plotly(fig, cfg), style={"height": "500px"})

    if plot_type == "Interactive Line":
        x = _get_col(xcol)
        if x is None:  # prototype fix: `array or ...` raised ValueError
            x = np.arange(len(_arr_1d()))
        y = _y_transform(_arr_1d())
        fig = interactive_line(
            x, y, title=cfg["title"] or "", xlabel=cfg["xlabel"] or "", ylabel=cfg["ylabel"] or ""
        )
        return dcc.Graph(figure=_style_plotly(fig, cfg), style={"height": "500px"})

    if plot_type == "Interactive Heatmap":
        fig = interactive_heatmap(_arr_2d(), title=cfg["title"] or "Heatmap")
        return dcc.Graph(figure=_style_plotly(fig, cfg), style={"height": "500px"})

    if plot_type == "Interactive 3D Surface":
        z = _arr_2d()
        yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]  # prototype fix: needs x, y, z
        fig = interactive_3d_surface(xx, yy, z, title=cfg["title"] or "3D Surface")
        return dcc.Graph(figure=_style_plotly(fig, cfg), style={"height": "600px"})

    if plot_type == "Interactive 3D Scatter":
        df = _df_or_none()
        if df is None:
            raise ValueError("Interactive 3D Scatter requires a DataFrame.")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 3:
            raise ValueError("Need at least 3 numeric columns for 3D scatter.")
        fig, info = interactive_3d_scatter(
            df[xcol or num_cols[0]].values,
            _y_transform(df[ycol or num_cols[1]].values),
            df[zcol or num_cols[2]].values
            if (zcol and zcol in df.columns)
            else df[num_cols[2]].values,
            title=cfg["title"] or "3D Scatter",
        )
        return dcc.Graph(figure=_style_plotly(fig, cfg), style={"height": "600px"})

    if plot_type == "Interactive Ternary":
        df = _df_or_none()
        if df is None:
            raise ValueError("Interactive Ternary requires a DataFrame.")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 3:
            raise ValueError("Need at least 3 numeric columns for ternary plot.")
        fig, info = interactive_ternary(
            df[num_cols[0]].values,
            df[num_cols[1]].values,
            df[num_cols[2]].values,
            title=cfg["title"] or "Ternary",
            labels=[num_cols[0], num_cols[1], num_cols[2]],
        )
        return dcc.Graph(figure=_style_plotly(fig, cfg), style={"height": "550px"})

    # ── Matplotlib / Seaborn family (return html.Img as base64 PNG) ───────────
    return _render_mpl_plot(
        data,
        plot_type,
        xcol,
        ycol,
        zcol,
        gcol,
        cfg,
        kwargs,
        _arr_1d,
        _arr_2d,
        _get_col,
        _y_transform,
    )


def _render_mpl_plot(
    data, plot_type, xcol, ycol, zcol, gcol, cfg, kwargs, _arr_1d, _arr_2d, _get_col, _y_transform
):
    df = data if isinstance(data, pd.DataFrame) else None

    def _x():
        v = _get_col(xcol)
        return v if v is not None else np.arange(len(_arr_1d()))

    def _y():
        v = _get_col(ycol)
        return _y_transform(v if v is not None else _arr_1d())

    try:
        if plot_type == "Histogram":
            result = histogram(
                _arr_1d(),
                bins=cfg["bins"],
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("title", "xlabel", "ylabel", "figsize", "color", "alpha")
                },
            )
        elif plot_type == "Line Plot":
            # prototype fix: line_plot takes `colors` (list); `color` collided in ax.plot()
            result = line_plot(
                _x(),
                _y(),
                colors=[kwargs["color"]],
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("title", "xlabel", "ylabel", "figsize", "alpha", "linewidth")
                },
            )
        elif plot_type == "Scatter Plot":
            result = scatter_plot(
                _x(),
                _y(),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("title", "xlabel", "ylabel", "figsize", "color", "alpha")
                },
            )
        elif plot_type == "Bar Chart":
            if df is not None:
                x = _get_col(xcol)
                y = _y_transform(_get_col(ycol))
                result = bar_chart(
                    x,
                    y,
                    xlabel=xcol,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ("title", "ylabel", "figsize", "color", "alpha")
                    },
                )
            else:
                y = _arr_1d()
                result = bar_chart(
                    np.arange(len(y)),
                    y,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ("title", "xlabel", "ylabel", "figsize", "color", "alpha")
                    },
                )
        elif plot_type == "Heatmap":
            result = heatmap(
                _arr_2d(), **{k: v for k, v in kwargs.items() if k in ("title", "figsize")}
            )
        elif plot_type == "Contour Plot":
            z = _arr_2d()
            yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]  # prototype fix: needs X, Y, Z
            result = contour_plot(
                xx, yy, z, **{k: v for k, v in kwargs.items() if k in ("title", "figsize")}
            )
        elif plot_type == "Waterfall Plot":
            result = waterfall_plot(
                _y(),
                **{
                    k: v for k, v in kwargs.items() if k in ("title", "xlabel", "ylabel", "figsize")
                },
            )
        elif plot_type == "Dual Axis Plot":
            if df is not None and len(df.select_dtypes(include=[np.number]).columns) >= 3:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                x = df[xcol or num_cols[0]].values
                y1 = _y_transform(df[ycol or num_cols[1]].values)
                y2 = (
                    df[zcol or num_cols[2]].values
                    if zcol and zcol in df
                    else df[num_cols[2]].values
                )
                result = dual_axis_plot(
                    x,
                    y1,
                    y2,
                    ylabel1=kwargs["ylabel"],  # prototype fix: was `ylabel`
                    **{k: v for k, v in kwargs.items() if k in ("title", "xlabel", "figsize")},
                )
            else:
                x = _x()
                y = _y()
                result = dual_axis_plot(
                    x,
                    y,
                    y * 0.5,
                    ylabel1=kwargs["ylabel"],  # prototype fix: was `ylabel`
                    **{k: v for k, v in kwargs.items() if k in ("title", "xlabel", "figsize")},
                )
        elif plot_type == "Broken Axis Plot":
            result = broken_axis_plot(
                _x(),
                _y(),
                **{
                    k: v for k, v in kwargs.items() if k in ("title", "xlabel", "ylabel", "figsize")
                },
            )
        elif plot_type == "Z-Colored Scatter":
            x = _x()
            y = _y()
            z = _get_col(zcol) if zcol else np.sqrt(x**2 + y**2)
            result = z_colored_scatter(
                x,
                y,
                z,
                **{
                    k: v for k, v in kwargs.items() if k in ("title", "xlabel", "ylabel", "figsize")
                },
            )
        elif plot_type == "Bubble Chart":
            x = _x()
            y = _y()
            z = _get_col(zcol) if zcol else np.abs(y) + 1
            result = bubble_chart(
                x,
                y,
                z,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("title", "xlabel", "ylabel", "figsize", "alpha")
                },
            )
        elif plot_type == "Polar Plot":
            arr = _arr_1d()
            theta = np.linspace(0, 2 * np.pi, len(arr))
            result = polar_plot(
                theta, arr, **{k: v for k, v in kwargs.items() if k in ("title", "figsize")}
            )
        elif plot_type == "2D Histogram":
            x = _x()
            y = _y()
            result = histogram_2d(
                x,
                y,
                bins=cfg["bins"],
                **{
                    k: v for k, v in kwargs.items() if k in ("title", "xlabel", "ylabel", "figsize")
                },
            )
        elif plot_type == "Scatter with Regression":
            result = scatter_with_regression(
                _x(),
                _y(),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("title", "xlabel", "ylabel", "figsize", "color", "alpha")
                },
            )
        elif plot_type == "Residual Plot":
            result = residual_plot(
                _x(),
                _y(),
                **{k: v for k, v in kwargs.items() if k in ("title", "xlabel", "figsize")},
            )
        elif plot_type == "Inset Plot":
            result = inset_plot(
                _x(),
                _y(),
                **{
                    k: v for k, v in kwargs.items() if k in ("title", "xlabel", "ylabel", "figsize")
                },
            )
        elif plot_type == "Distribution Plot":
            if df is not None:
                result = distribution_plot(
                    df,
                    x=ycol or df.select_dtypes(include=[np.number]).columns[0],
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ("title", "xlabel", "ylabel", "figsize")
                    },
                )
            else:
                result = distribution_plot(
                    pd.DataFrame({"value": _arr_1d()}),
                    x="value",
                    **{k: v for k, v in kwargs.items() if k in ("title", "figsize")},
                )
        elif plot_type == "Box Plot":
            if df is not None:
                result = box_plot(
                    df,
                    y=ycol or df.select_dtypes(include=[np.number]).columns[0],
                    x=gcol if gcol else None,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ("title", "xlabel", "ylabel", "figsize")
                    },
                )
            else:
                result = box_plot(
                    pd.DataFrame({"value": _arr_1d()}),
                    y="value",
                    **{k: v for k, v in kwargs.items() if k in ("title", "figsize")},
                )
        elif plot_type == "Regression Plot":
            if df is not None:
                result = regression_plot(  # prototype fix: df goes in `data=`
                    data=df,
                    x=xcol or df.select_dtypes(include=[np.number]).columns[0],
                    y=ycol or df.select_dtypes(include=[np.number]).columns[1]
                    if len(df.select_dtypes(include=[np.number]).columns) > 1
                    else df.columns[0],
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ("title", "xlabel", "ylabel", "figsize")
                    },
                )
            else:
                result = scatter_with_regression(
                    _x(),
                    _y(),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ("title", "xlabel", "ylabel", "figsize")
                    },
                )
        elif plot_type == "Pair Plot":
            if df is not None:
                result = pair_plot(
                    df, **{k: v for k, v in kwargs.items() if k in ("title", "figsize")}
                )
            else:
                raise ValueError("Pair Plot requires a DataFrame.")
        else:
            result = histogram(_arr_1d(), bins=cfg["bins"])

    except Exception:
        raise

    # Extract the matplotlib Figure from the result tuple
    fig_obj = result[0] if isinstance(result, (tuple, list)) else result

    # Apply annotations
    try:
        ax = fig_obj.axes[0] if fig_obj.axes else None
        if ax:
            _apply_grid(ax, cfg)
            _apply_legend(ax, cfg)
    except Exception:
        pass

    return _mpl_to_img(fig_obj, cfg)


def _apply_grid(ax, cfg):
    if cfg.get("grid"):
        ax.grid(True, linestyle=cfg.get("grid_linestyle", "--"), alpha=0.4)
    else:
        ax.grid(False)


def _apply_legend(ax, cfg):
    if cfg.get("legend"):
        pos = cfg.get("legend_position", "best")
        if pos == "outside":
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        else:
            try:
                ax.legend(loc=pos)
            except Exception:
                ax.legend()


def _style_plotly(fig: go.Figure, cfg: dict) -> go.Figure:
    """Apply dark theme and palette to a Plotly figure."""
    colors = _COLOR_PALETTES.get(cfg.get("color_palette", "Default"), _COLOR_PALETTES["Default"])
    for i, trace in enumerate(fig.data):
        c = colors[i % len(colors)]
        try:
            trace.update(line={"color": c})
        except Exception:
            pass
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(27,27,27,0.8)",
        font={"family": cfg.get("fontfamily", "sans-serif"), "size": cfg.get("fontsize", 12)},
        title=cfg.get("title") or "",
    )
    return fig


def _mpl_to_img(fig, cfg: dict) -> html.Img:
    """Convert a matplotlib Figure to a base64 PNG html.Img element."""
    buf = io.BytesIO()
    dpi = 120 if not cfg.get("pub_style") else 150
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = __import__("base64").b64encode(buf.read()).decode("utf-8")
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        style={"width": "100%", "height": "auto", "borderRadius": "4px"},
    )


# ── Save to history ───────────────────────────────────────────────────────────


@callback(
    Output("qp-feedback", "children", allow_duplicate=True),
    Input("qp-save-btn", "n_clicks"),
    State("qp-dataset", "value"),
    State("qp-plot-type", "value"),
    prevent_initial_call=True,
)
def save_to_history(n, ds_name, plot_type):
    if n and ds_name:
        state.add_plot_to_history(
            {
                "type": plot_type,
                "dataset": ds_name,
                "config": {"plot_type": plot_type},
            }
        )
        return dbc.Alert("Plot saved to history.", color="success", duration=2000, dismissable=True)
    return dash.no_update


# ── Clear ─────────────────────────────────────────────────────────────────────


@callback(
    Output("qp-plot-output", "children", allow_duplicate=True),
    Input("qp-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_plot(n):
    if n:
        return html.Div(
            html.P("Click 'Generate Plot' to create a visualization.", className="plot-placeholder")
        )
    return dash.no_update


# ── Annotation management ─────────────────────────────────────────────────────

# Store annotations in module-level list (simple approach for single-user)
_ANNOTATIONS = []


@callback(
    Output("qp-ann-list", "children"),
    Input("qp-ann-add-btn", "n_clicks"),
    Input("qp-ann-clear-btn", "n_clicks"),
    State("qp-ann-type", "value"),
    State("qp-ann-x", "value"),
    State("qp-ann-y", "value"),
    State("qp-ann-label", "value"),
    State("qp-ann-color", "value"),
    prevent_initial_call=True,
)
def manage_annotations(add_n, clear_n, ann_type, x, y, label, color):
    global _ANNOTATIONS
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered == "qp-ann-clear-btn":
        _ANNOTATIONS = []
    elif triggered == "qp-ann-add-btn" and add_n:
        _ANNOTATIONS.append(
            {"type": ann_type, "x": x, "y": y, "label": label or "", "color": color or "#e0a3a3"}
        )

    if not _ANNOTATIONS:
        return html.P("No annotations added.", className="text-muted-sm")
    return html.Ul(
        [
            html.Li(
                f"{a['type']} at ({a.get('x', '')}, {a.get('y', '')}) — {a.get('label', '')}",
                style={"fontSize": "0.78rem"},
            )
            for a in _ANNOTATIONS
        ]
    )
