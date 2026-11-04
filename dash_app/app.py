"""Plottle — Dash App Entry Point.

Replaces the Streamlit ``plottle/Home.py`` with a full Dash multi-page
application that preserves all features and the NCCU dark colour scheme.

Throwaway comparison prototype (see PROTOTYPE_DASH.md). Only the Home,
Data Upload and Quick Plot pages are included.

Run
---
    # from repo root:
    python -m dash_app.app
"""

from __future__ import annotations

import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from dash_app.themes import DEFAULT_THEME, THEMES as _THEMES

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── App instance ──────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Plottle",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server  # expose Flask server for Gunicorn


# ── Logo (embed as base64 so it works regardless of static-file routing) ─────

_logo_path = _REPO_ROOT / "plottle" / "assets" / "logo.png"


def _embed_logo() -> str:
    try:
        import base64

        return "data:image/png;base64," + base64.b64encode(_logo_path.read_bytes()).decode()
    except Exception:
        return ""


_logo_img_src = _embed_logo()


# ── Sidebar helpers ───────────────────────────────────────────────────────────


def _nav_link(label: str, href: str) -> dbc.NavLink:
    return dbc.NavLink(label, href=href, active="exact", className="sidebar-link")


# ── Sidebar ───────────────────────────────────────────────────────────────────

sidebar = html.Div(
    [
        html.Div(
            html.Img(src=_logo_img_src, className="sidebar-logo-img")
            if _logo_img_src
            else html.Div(),
            className="sidebar-logo-wrap",
        ),
        html.Div(
            [
                html.Button(
                    "",
                    id=f"theme-btn-{slug}",
                    className=f"theme-swatch theme-swatch-{slug}",
                    title=label,
                    **{"data-theme-slug": slug},
                )
                for slug, label in _THEMES
            ],
            className="theme-switcher",
        ),
        html.Nav(
            [
                _nav_link("Home", "/"),
                _nav_link("Data Upload", "/data-upload"),
                _nav_link("Quick Plot", "/plot-basic"),
                html.Hr(className="sidebar-divider"),
                html.Div(
                    [
                        html.P(
                            "Scientific data visualization and analysis toolkit "
                            "developed at North Carolina Central University.",
                            className="sidebar-about-text",
                        ),
                        html.A(
                            "NCCU Schultz Lab →",
                            href="https://github.com/The-Schultz-Lab",
                            target="_blank",
                            className="sidebar-about-link",
                        ),
                    ],
                    className="sidebar-about",
                ),
            ],
        ),
    ],
    className="sidebar",
    id="sidebar",
)

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="theme-store", storage_type="local", data=DEFAULT_THEME),
        html.Div(id="_theme-target", children="", style={"display": "none"}),
        dbc.Toast(
            id="global-toast",
            header="Plottle",
            is_open=False,
            dismissable=True,
            duration=3000,
            style={"position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 9999},
        ),
        sidebar,
        html.Main(
            dash.page_container,
            className="main-content",
            id="page-content",
        ),
    ],
    className="app-wrapper",
)


# ── Theme callbacks ───────────────────────────────────────────────────────────

app.clientside_callback(
    """
    function(theme) {
        var t = theme || 'nccu-dark';
        document.documentElement.setAttribute('data-theme', t);
        document.querySelectorAll('.theme-swatch').forEach(function(b) {
            b.classList.toggle('swatch-active', b.getAttribute('data-theme-slug') === t);
        });
        return t;
    }
    """,
    Output("_theme-target", "children"),
    Input("theme-store", "data"),
)


@app.callback(
    Output("theme-store", "data"),
    [Input(f"theme-btn-{slug}", "n_clicks") for slug, _ in _THEMES],
    prevent_initial_call=True,
)
def on_swatch_click(*_):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    for slug, _ in _THEMES:
        if f"theme-btn-{slug}" == btn_id:
            return slug
    return dash.no_update


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        debug=os.environ.get("PLOTTLE_DEBUG") == "1",
        host="127.0.0.1",
        port=int(os.environ.get("PLOTTLE_DASH_PORT", "8050")),
        dev_tools_disable_version_check=True,  # no phone-home to plotly.com in debug
    )
