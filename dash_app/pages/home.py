"""Home page — Dashboard + Help tabs.

Mirrors the Streamlit Home.py landing page with:
  • Dashboard tab: session metrics, quick-action buttons
  • Help tab:      quick-start guide, file format table, tips
"""

import base64
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from plottle import __version__

dash.register_page(__name__, path="/", title="Home — Plottle", name="Home")

_LOGO_PATH = Path(__file__).resolve().parents[2] / "plottle" / "assets" / "logo.png"
_LOGO_SRC = (
    "data:image/png;base64," + base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    if _LOGO_PATH.exists()
    else None
)

# ── Layout ────────────────────────────────────────────────────────────────────


def layout(**kwargs):
    return html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(label="Dashboard", tab_id="tab-dashboard"),
                    dbc.Tab(label="Help", tab_id="tab-help"),
                ],
                id="home-tabs",
                active_tab="tab-dashboard",
            ),
            html.Div(id="home-tab-content", className="tab-content"),
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────


@callback(Output("home-tab-content", "children"), Input("home-tabs", "active_tab"))
def render_tab(tab: str):
    if tab == "tab-dashboard":
        return _dashboard_tab()
    return _help_tab()


def _dashboard_tab():
    summary = state.get_session_summary()

    # Hero row
    hero = html.Div(
        [
            html.Img(src=_LOGO_SRC, className="hero-logo") if _LOGO_SRC else html.Div(),
            html.Div(
                [
                    html.H1("Plottle", className="hero-title"),
                    html.P(
                        f"Scientific data visualization and analysis · v{__version__}",
                        className="hero-subtitle",
                    ),
                ]
            ),
        ],
        className="hero-row",
    )

    # Metric cards
    metrics = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.Div(summary["num_datasets"], className="metric-value"),
                        html.Div("Datasets", className="metric-label"),
                    ],
                    className="metric-card",
                ),
                width=4,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div(summary["num_plots"], className="metric-value"),
                        html.Div("Plots", className="metric-label"),
                    ],
                    className="metric-card",
                ),
                width=4,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Div(summary["num_analyses"], className="metric-value"),
                        html.Div("Analyses", className="metric-label"),
                    ],
                    className="metric-card",
                ),
                width=4,
            ),
        ],
        className="g-3 mb-3",
    )

    # Status / quick-action
    if summary["current_dataset"]:
        status = dbc.Alert(
            [
                html.Span("Active dataset: "),
                html.Strong(summary["current_dataset"]),
                dbc.Button(
                    "Go to Quick Plot →",
                    href="/plot-basic",
                    color="link",
                    className="ms-3 p-0",
                    style={"color": "var(--accent)"},
                ),
            ],
            color="success",
            className="mb-3",
        )
    else:
        status = dbc.Alert(
            [
                html.Span("No dataset loaded. "),
                dbc.Button(
                    "Upload Data →",
                    href="/data-upload",
                    color="link",
                    className="ms-2 p-0",
                    style={"color": "var(--accent)"},
                ),
            ],
            color="info",
            className="mb-3",
        )

    # Recent datasets list
    if summary["dataset_names"]:
        ds_list = html.Div(
            [
                html.H5("Loaded Datasets", className="mb-2"),
                html.Ul(
                    [
                        html.Li(name, style={"fontSize": "0.88rem", "color": "var(--text-muted)"})
                        for name in summary["dataset_names"]
                    ]
                ),
            ],
            className="mb-3",
        )
    else:
        ds_list = html.Div()

    return html.Div([hero, html.Hr(), metrics, status, ds_list])


def _help_tab():
    return html.Div(
        [
            html.H2("Quick Start", className="mb-3"),
            html.Ol(
                [
                    html.Li(
                        [
                            html.Strong("Upload Data"),
                            " — Go to ",
                            html.Em("Data Upload"),
                            " to load a CSV, Excel, NumPy, or other file. "
                            "Or load a built-in example dataset.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Plot"),
                            " — ",
                            html.Em("Plot → Basic"),
                            " lets you choose from 26 plot types and configure them interactively.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Analyse"),
                            " — ",
                            html.Em("Analyze → Single"),
                            " has curve fitting, statistics, signal processing, peak analysis, and more.",  # noqa: E501
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Export"),
                            " — ",
                            html.Em("Export"),
                            " saves plots as PNG/PDF/SVG and data in multiple formats.",
                        ]
                    ),
                ],
                className="mb-4",
                style={"lineHeight": "2"},
            ),
            html.H3("Supported File Formats", className="mb-2"),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("Format"), html.Th("Extensions")])),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Tabular"),
                                    html.Td("CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet"),
                                ]
                            ),
                            html.Tr([html.Td("NumPy"), html.Td(".npy, .npz")]),
                            html.Tr([html.Td("Python"), html.Td("Pickle (.pkl)")]),
                            html.Tr(
                                [
                                    html.Td("Spectroscopy"),
                                    html.Td("JCAMP-DX (.jdx/.dx), SPC (.spc), ASC (.asc)"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Scientific"),
                                    html.Td("HDF5 (.h5/.hdf5), NetCDF (.nc/.cdf), mzML"),
                                ]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                size="sm",
                className="mb-4",
            ),
            html.H3("Tips", className="mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Datasets persist across all pages during a session"  # noqa: E501
                                                ),
                                                html.Li(
                                                    "Use Export → Save Session to preserve work between restarts"  # noqa: E501
                                                ),
                                                html.Li(
                                                    "Saved sessions can be reloaded from the same page"  # noqa: E501
                                                ),
                                            ]
                                        ),
                                        title="Session management",
                                    ),
                                    dbc.AccordionItem(
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Files >50 MB or >10k rows get a downsampled preview automatically"  # noqa: E501
                                                ),
                                                html.Li(
                                                    "Interactive Plotly plots are slower than Matplotlib for large data"  # noqa: E501
                                                ),
                                                html.Li(
                                                    "Clear plot history periodically to free memory"
                                                ),
                                            ]
                                        ),
                                        title="Performance",
                                    ),
                                ],
                                flush=True,
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Histogram — single-variable distributions"
                                                ),
                                                html.Li(
                                                    "Scatter — relationship between two variables"
                                                ),
                                                html.Li("Line — time series or ordered data"),
                                                html.Li(
                                                    "Box / Violin — compare distributions across groups"  # noqa: E501
                                                ),
                                                html.Li("Heatmap — 2D matrix or correlation data"),
                                                html.Li("Ternary — 3-component compositions"),
                                            ]
                                        ),
                                        title="Choosing a plot type",
                                    ),
                                    dbc.AccordionItem(
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Apply Publication Style in Quick Plot for paper-ready figures"  # noqa: E501
                                                ),
                                                html.Li("Export at 300 DPI from Export page"),
                                                html.Li(
                                                    "Use SVG export for vector graphics in manuscripts"  # noqa: E501
                                                ),
                                            ]
                                        ),
                                        title="Publication figures",
                                    ),
                                ],
                                flush=True,
                            )
                        ],
                        width=6,
                    ),
                ],
                className="g-3 mb-4",
            ),
            html.H3("About", className="mb-2"),
            html.P(
                "Plottle is developed at North Carolina Central University for research "
                "and teaching in computational science. "
                "Built with Dash · NumPy · Pandas · Matplotlib · Seaborn · Plotly · SciPy. "
                "Version 2.0.0"
            ),
        ]
    )
