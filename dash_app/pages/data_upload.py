"""Data Upload page.

Tabs
----
Upload File     — dcc.Upload with all supported formats
Example Datasets— 10 built-in artificial datasets
Batch Import    — load all matching files from a directory
Loaded Datasets — manage / delete / preview loaded datasets
"""

import base64
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from plottle.io import downsample_for_preview, load_data
from plottle.math import calculate_statistics
from plottle.batch import batch_load_files, scan_directory

dash.register_page(__name__, path="/data-upload", title="Data Upload — Plottle", name="Data Upload")

_APP_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = _APP_ROOT / "plottle" / "example-data" / "Artificial"

_EXAMPLES = {
    "normal_distribution.csv": {
        "label": "Normal Distribution",
        "desc": "400 simulated temperature measurements (Gaussian, μ=298 K).",
        "use": "Histogram · Distribution · Interactive Histogram",
    },
    "bimodal_distribution.csv": {
        "label": "Bimodal Distribution",
        "desc": "500 absorbance values drawn from two overlapping Gaussians.",
        "use": "Distribution · Overlaid Distributions",
    },
    "sine_cosine_waves.csv": {
        "label": "Sine & Cosine Waves",
        "desc": "300-point sin, cos, and damped-sin series over 0–4π.",
        "use": "Line Plot · Interactive Line",
    },
    "scatter_correlation.csv": {
        "label": "Scatter Correlation",
        "desc": "200 (reaction time, yield, temperature) points with linear trend.",
        "use": "Scatter · Regression · Interactive Scatter",
    },
    "grouped_categorical.csv": {
        "label": "Grouped Categorical",
        "desc": "Catalyst × solvent yield data (4 catalysts × 3 solvents, 20 obs each).",
        "use": "Box / Violin / Swarm · Grouped Categorical",
    },
    "molecular_properties.csv": {
        "label": "Molecular Properties",
        "desc": "120 molecules with MW, logP, TPSA, HBD, HBA, RotBonds, pIC50.",
        "use": "Correlation Heatmap · Scatter · Regression",
    },
    "ir_spectrum.csv": {
        "label": "IR Spectrum",
        "desc": "Simulated IR transmittance spectrum from 4000–400 cm⁻¹.",
        "use": "Line Plot · Interactive Line",
    },
    "reaction_kinetics.csv": {
        "label": "Reaction Kinetics",
        "desc": "A→B→C consecutive-reaction concentration profiles over 60 min.",
        "use": "Line Plot · Interactive Line",
    },
    "gaussian_surface.npy": {
        "label": "Gaussian Surface (2D array)",
        "desc": "60×60 double-Gaussian potential energy surface.",
        "use": "Contour Plot · 3D Surface · Heatmap",
    },
    "correlation_matrix.npy": {
        "label": "Correlation Matrix (2D array)",
        "desc": "10×10 exact correlation matrix derived from random data.",
        "use": "Heatmap · Interactive Heatmap",
    },
}

_LARGE_FILE_BYTES = 50 * 1024 * 1024
_PREVIEW_MAX_ROWS = 10_000

_ACCEPTED = (
    ".npy,.npz,.csv,.xlsx,.xls,.tsv,.json,.parquet,"  # no .pkl (unsafe to unpickle uploads)
    ".jdx,.dx,.h5,.hdf5,.nc,.cdf,.spc,.asc,.mzml,.mzxml"
)

# ── Layout ────────────────────────────────────────────────────────────────────


def layout(**kwargs):
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Data Upload", className="page-title"),
                    html.P(
                        "Load datasets from file, examples, or a folder.", className="page-caption"
                    ),
                ],
                className="page-header",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="Upload File", tab_id="tab-upload"),
                    dbc.Tab(label="Example Datasets", tab_id="tab-examples"),
                    dbc.Tab(label="Batch Import", tab_id="tab-batch"),
                ],
                id="du-tabs",
                active_tab="tab-upload",
            ),
            html.Div(id="du-tab-content", className="tab-content"),
            html.Hr(),
            html.Div(id="du-datasets-section"),
            # Feedback store
            dcc.Store(id="du-feedback-store"),
        ]
    )


# ── Tab rendering ─────────────────────────────────────────────────────────────


@callback(Output("du-tab-content", "children"), Input("du-tabs", "active_tab"))
def render_tab(tab: str):
    if tab == "tab-upload":
        return _upload_tab()
    if tab == "tab-examples":
        return _examples_tab()
    return _batch_tab()


def _upload_tab():
    return html.Div(
        [
            dcc.Upload(
                id="du-upload",
                children=html.Div(
                    [
                        html.I(className="me-2"),
                        "Drag and drop or ",
                        html.A("click to select a file", style={"color": "var(--accent)"}),
                    ]
                ),
                accept=_ACCEPTED,
                style={
                    "width": "100%",
                    "height": "120px",
                    "lineHeight": "120px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "8px",
                    "borderColor": "var(--border)",
                    "textAlign": "center",
                    "background": "var(--bg-secondary)",
                    "cursor": "pointer",
                },
            ),
            html.Div(id="du-upload-output", className="mt-3"),
        ]
    )


@callback(
    Output("du-upload-output", "children"),
    Output("du-datasets-section", "children", allow_duplicate=True),
    Input("du-upload", "contents"),
    State("du-upload", "filename"),
    State("du-upload", "last_modified"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, last_modified):
    if contents is None:
        return dash.no_update, dash.no_update

    _content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    filename = Path(filename or "upload").name
    if Path(filename).suffix.lower() == ".pkl":
        return dbc.Alert("Pickle files are not accepted.", color="danger"), dash.no_update

    try:
        with tempfile.TemporaryDirectory(prefix="plottle-upload-") as tmp_dir:
            tmp_path = Path(tmp_dir) / filename
            tmp_path.write_bytes(decoded)
            data = load_data(str(tmp_path))
        file_size = len(decoded)
        is_large = isinstance(data, pd.DataFrame) and (
            file_size > _LARGE_FILE_BYTES or len(data) > _PREVIEW_MAX_ROWS
        )

        state.add_dataset(
            filename,
            data,
            metadata={"file_size": file_size, "downsampled_preview": is_large},
        )

        feedback = [
            dbc.Alert(f"Loaded '{filename}' successfully.", color="success", dismissable=True)
        ]
        if is_large:
            feedback.append(
                dbc.Alert(
                    f"Large dataset: {len(data):,} rows / {file_size / 1_048_576:.1f} MB. "
                    "Full data stored; previews are downsampled.",
                    color="warning",
                    dismissable=True,
                )
            )

        preview_data = downsample_for_preview(data, _PREVIEW_MAX_ROWS) if is_large else data
        feedback.append(html.H5("Preview", className="mt-3 mb-2"))
        feedback.append(_data_preview(preview_data, filename))

        return html.Div(feedback), _datasets_section()

    except Exception as exc:
        return html.Div(
            [
                dbc.Alert(f"Error loading file: {exc}", color="danger"),
                dbc.Collapse(
                    html.Pre(traceback.format_exc(), className="result-box"),
                    id="du-traceback-collapse",
                    is_open=False,
                ),
                dbc.Button(
                    "Show error details", id="du-traceback-btn", size="sm", color="secondary"
                ),
            ]
        ), dash.no_update


def _examples_tab():
    if not _EXAMPLE_DIR.exists():
        return dbc.Alert(
            f"Example data directory not found: {_EXAMPLE_DIR}. "
            "Run generate_examples.py in plottle/example-data/Artificial/ first.",
            color="warning",
        )

    available = [
        (fname, meta) for fname, meta in _EXAMPLES.items() if (_EXAMPLE_DIR / fname).exists()
    ]

    rows = []
    for i in range(0, len(available), 2):
        pair = available[i : i + 2]
        cols = []
        for fname, meta in pair:
            cols.append(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(meta["label"], className="example-card-title"),
                            html.Div(meta["desc"], className="example-card-desc"),
                            html.Div(f"Best for: {meta['use']}", className="example-card-use"),
                            dbc.Button(
                                "Load",
                                id={"type": "ex-load-btn", "index": fname},
                                color="primary",
                                size="sm",
                                className="w-100",
                            ),
                        ],
                        className="example-card",
                    ),
                    width=6,
                )
            )
        # pad to even
        if len(pair) < 2:
            cols.append(dbc.Col(width=6))
        rows.append(dbc.Row(cols, className="g-3 mb-2"))

    return html.Div(
        [
            html.P(
                "Load a built-in artificial dataset to explore each plot type immediately.",
                className="text-muted-sm mb-3",
            ),
            *rows,
            html.Div(id="ex-load-feedback", className="mt-2"),
        ]
    )


@callback(
    Output("ex-load-feedback", "children"),
    Output("du-datasets-section", "children", allow_duplicate=True),
    Input({"type": "ex-load-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_example(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return dash.no_update, dash.no_update

    fname = dash.callback_context.triggered_id["index"]
    fpath = _EXAMPLE_DIR / fname
    meta = _EXAMPLES.get(fname, {})

    try:
        data = load_data(str(fpath))
        state.add_dataset(fname, data, metadata={"source": "example"})
        return (
            dbc.Alert(f"Loaded {meta.get('label', fname)}", color="success", dismissable=True),
            _datasets_section(),
        )
    except Exception as exc:
        return dbc.Alert(f"Error: {exc}", color="danger"), dash.no_update


def _batch_tab():
    return html.Div(
        [
            html.P(
                "Load all matching files from a local directory at once.",
                className="text-muted-sm mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Folder path"),
                            dbc.Input(
                                id="batch-folder",
                                placeholder="C:/Users/you/data",
                                type="text",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Filename pattern (optional)"),
                            dbc.Input(id="batch-pattern", placeholder="sample_*.csv", type="text"),
                        ],
                        width=3,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Extensions"),
                        dcc.Dropdown(
                            id="batch-exts",
                            options=[
                                {"label": e, "value": e}
                                for e in [
                                    "csv",
                                    "xlsx",
                                    "xls",
                                    "tsv",
                                    "json",
                                    "parquet",
                                    "npy",
                                    "npz",
                                    "jdx",
                                    "dx",
                                    "h5",
                                    "hdf5",
                                    "nc",
                                    "cdf",
                                    "spc",
                                    "asc",
                                ]
                            ],
                            value=["csv", "xlsx", "tsv"],
                            multi=True,
                        ),
                    ],
                    width=9,
                ),
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Scan Directory",
                            id="batch-scan-btn",
                            color="secondary",
                            className="w-100",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Load All Files",
                            id="batch-load-btn",
                            color="primary",
                            className="w-100",
                        ),
                        width=3,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="batch-output"),
        ]
    )


@callback(
    Output("batch-output", "children"),
    Output("du-datasets-section", "children", allow_duplicate=True),
    Input("batch-scan-btn", "n_clicks"),
    Input("batch-load-btn", "n_clicks"),
    State("batch-folder", "value"),
    State("batch-exts", "value"),
    State("batch-pattern", "value"),
    prevent_initial_call=True,
)
def batch_action(scan_n, load_n, folder, exts, pattern):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    do_load = triggered == "batch-load-btn"

    if not folder:
        return dbc.Alert("Please enter a folder path.", color="warning"), dash.no_update

    try:
        found = scan_directory(folder, extensions=exts or None, pattern=pattern or None)
    except NotADirectoryError as e:
        return dbc.Alert(f"Not a valid directory: {e}", color="danger"), dash.no_update
    except Exception as e:
        return dbc.Alert(f"Scan error: {e}", color="danger"), dash.no_update

    if not found:
        return dbc.Alert("No matching files found.", color="info"), dash.no_update

    meta_rows = [
        {"Filename": f.name, "Extension": f.suffix, "Size (KB)": f"{f.stat().st_size / 1024:.1f}"}
        for f in found
    ]
    scan_result = html.Div(
        [
            html.P(f"{len(found)} file(s) found:"),
            dash_table.DataTable(
                data=meta_rows,
                columns=[{"name": c, "id": c} for c in meta_rows[0]],
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": "var(--bg-primary)",
                    "color": "var(--text-primary)",
                    "border": "1px solid var(--border)",
                    "fontSize": "0.82rem",
                },
                style_header={"backgroundColor": "var(--bg-secondary)", "fontWeight": "600"},
                page_size=10,
            ),
        ]
    )

    if not do_load:
        return scan_result, dash.no_update

    try:
        result = batch_load_files(found, on_error="skip")
        loaded = result["datasets"]
        errors = result["errors"]
        bmeta = result["metadata"]

        for name, data in loaded.items():
            state.add_dataset(
                name,
                data,
                metadata={"source": "batch_import", "size_bytes": bmeta[name]["size_bytes"]},
            )

        feedback = [scan_result]
        if loaded:
            feedback.append(
                dbc.Alert(
                    f"Loaded {len(loaded)} dataset(s): " + ", ".join(loaded.keys()), color="success"
                )
            )
        if errors:
            feedback.append(dbc.Alert(f"{len(errors)} file(s) failed.", color="warning"))

        return html.Div(feedback), _datasets_section()
    except Exception as e:
        return html.Div(
            [scan_result, dbc.Alert(f"Load error: {e}", color="danger")]
        ), dash.no_update


# ── Datasets section ──────────────────────────────────────────────────────────


@callback(Output("du-datasets-section", "children"), Input("du-tabs", "active_tab"))
def refresh_datasets_section(_):
    return _datasets_section()


def _datasets_section():
    summary = state.get_session_summary()
    if not summary["num_datasets"]:
        return html.Div(
            [
                html.H3("Loaded Datasets"),
                dbc.Alert(
                    "No datasets loaded yet. Upload a file above to get started.", color="info"
                ),
            ]
        )

    names = summary["dataset_names"]
    current = summary["current_dataset"] or names[0]

    return html.Div(
        [
            html.H3(f"Loaded Datasets ({summary['num_datasets']})"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="du-ds-select",
                            options=[{"label": n, "value": n} for n in names],
                            value=current,
                            clearable=False,
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Button("Delete", id="du-delete-btn", color="danger", className="w-100"),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Set Active",
                            id="du-set-active-btn",
                            color="secondary",
                            className="w-100",
                        ),
                        width=2,
                    ),
                ],
                className="mb-3 g-2",
            ),
            html.Div(id="du-ds-detail"),
        ]
    )


@callback(
    Output("du-ds-detail", "children"),
    Input("du-ds-select", "value"),
)
def show_dataset_detail(name: Optional[str]):
    if not name:
        return html.Div()
    data = state.get_dataset(name)
    meta = state.get_dataset_metadata(name)
    if data is None:
        return dbc.Alert("Dataset not found.", color="danger")

    info_items = [
        html.Div(f"Type: {meta.get('data_type', 'Unknown')}", className="dataset-card-meta"),
        html.Div(
            f"Added: {meta.get('added_time', 'Unknown')[:19].replace('T', ' ')}",
            className="dataset-card-meta",
        ),
    ]
    if "shape" in meta:
        info_items.append(html.Div(f"Shape: {meta['shape']}", className="dataset-card-meta"))
    if "columns" in meta:
        info_items.append(
            html.Div(f"Columns: {', '.join(meta['columns'])}", className="dataset-card-meta")
        )

    return html.Div(
        [
            html.Div(
                [html.Div(name, className="dataset-card-title"), *info_items],
                className="dataset-card mb-3",
            ),
            html.H5("Preview"),
            _data_preview(data, name),
            html.H5("Quick Statistics", className="mt-3"),
            _quick_stats(data, name),
        ]
    )


@callback(
    Output("du-datasets-section", "children", allow_duplicate=True),
    Input("du-delete-btn", "n_clicks"),
    State("du-ds-select", "value"),
    prevent_initial_call=True,
)
def delete_dataset(n, name):
    if n and name:
        state.delete_dataset(name)
    return _datasets_section()


@callback(
    Output("du-datasets-section", "children", allow_duplicate=True),
    Input("du-set-active-btn", "n_clicks"),
    State("du-ds-select", "value"),
    prevent_initial_call=True,
)
def set_active(n, name):
    if n and name:
        state.set_current_dataset(name)
    return _datasets_section()


# ── Helper renderers ──────────────────────────────────────────────────────────


def _data_preview(data, name: str):
    if isinstance(data, pd.DataFrame):
        preview = data.head(20)
        return dash_table.DataTable(
            data=preview.to_dict("records"),
            columns=[{"name": c, "id": c} for c in preview.columns],
            style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
            style_cell={
                "backgroundColor": "var(--bg-primary)",
                "color": "var(--text-primary)",
                "border": "1px solid var(--border)",
                "fontSize": "0.82rem",
                "maxWidth": "200px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_header={
                "backgroundColor": "var(--bg-secondary)",
                "fontWeight": "600",
                "color": "var(--text-muted)",
            },
            page_size=20,
            fixed_rows={"headers": True},
        )
    if isinstance(data, np.ndarray):
        shape_str = " × ".join(str(s) for s in data.shape)
        preview = data.flatten()[:100]
        return html.Div(
            [
                html.P(
                    f"Array shape: {shape_str} | dtype: {data.dtype}", className="text-muted-sm"
                ),
                html.Pre(str(preview), className="result-box"),
            ]
        )
    return html.Pre(str(data)[:500], className="result-box")


def _quick_stats(data, name: str):
    if isinstance(data, pd.DataFrame):
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return dbc.Alert("No numeric columns for statistics.", color="info")
        col = num_cols[0]
        try:
            stats = calculate_statistics(data[col].dropna().values)
            return dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['mean']:.4g}", className="metric-value"),
                                html.Div("Mean", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['std']:.4g}", className="metric-value"),
                                html.Div("Std Dev", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['min']:.4g}", className="metric-value"),
                                html.Div("Min", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['max']:.4g}", className="metric-value"),
                                html.Div("Max", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                ],
                className="g-2",
            )
        except Exception:
            return html.Div()
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        flat = data.flatten()
        try:
            stats = calculate_statistics(flat)
            return dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['mean']:.4g}", className="metric-value"),
                                html.Div("Mean", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['std']:.4g}", className="metric-value"),
                                html.Div("Std Dev", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['min']:.4g}", className="metric-value"),
                                html.Div("Min", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(f"{stats['max']:.4g}", className="metric-value"),
                                html.Div("Max", className="metric-label"),
                            ],
                            className="metric-card",
                        ),
                        width=3,
                    ),
                ],
                className="g-2",
            )
        except Exception:
            return html.Div()
    return html.Div()
