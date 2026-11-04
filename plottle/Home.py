"""Graphical User Interface Module - Main Entry Point.

This module provides the main entry point for the Streamlit-based GUI for
the Plottle toolkit. It provides an interactive interface for data
exploration, visualization, and analysis.

Navigation Structure
--------------------
Home
Data Upload
Plot
  Basic         — Quick Plot (27 plot types, Matplotlib/Seaborn/Plotly)
  Multiplot     — Multi-Plot Dashboard (grid layouts, axis sharing)
  Advanced
    ↳ Advanced Plotting  — Seaborn statistical + Plotly interactive
    ↳ Spectroscopy       — IR/Raman, NMR, UV-Vis, Mass Spec + NIST
    ↳ Molecular Viz      — 3D structure and vibrational modes
Analyze
  Single        — Analysis Tools (stats, curve fit, signal, peaks, …)
  Batch         — Batch Analysis + workflow presets
Export
Gallery
Help
Settings

Usage
-----
    streamlit run plottle/Home.py
"""

import base64
import sys
import streamlit as st
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_PAGES_DIR = Path(__file__).resolve().parent / "pages"

# Assets are inside the package (not at the repo root) so that they survive
# `pip install` -- see audit A-09. Resolving them relative to the repo root meant
# a pip-installed GUI came up with no logo and no NCCU branding.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_LOGO_PNG = _ASSETS_DIR / "logo.png"
_NCCU_HORIZ = _ASSETS_DIR / "nccu-horiz-logo.png"
_NCCU_WINGS = _ASSETS_DIR / "nccu-wings.png"

from plottle import __version__  # noqa: E402
from plottle.utils import initialize_session_state, get_session_summary  # noqa: E402
from plottle.utils.theming import app_css  # noqa: E402

try:
    from PIL import Image as _PILImage

    _page_icon = _PILImage.open(_LOGO_PNG) if _LOGO_PNG.exists() else "📊"
except ImportError:
    _page_icon = "📊"

# ── Page configuration (must be the first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Plottle",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            f"**Plottle** v{__version__}\n\n"
            "Scientific data visualization and analysis toolkit for research "
            "and teaching in computational science — developed at North Carolina "
            "Central University.\n\n"
            "**Institution:** NCCU Department of Chemistry and Biochemistry"
        )
    },
)

# ── App-wide styling, derived from the active theme ───────────────────────────
# Colours come from plottle.utils.theming.app_css() rather than being hardcoded, so
# the stylesheet follows whatever the user selects in Settings. Previously the NCCU
# dark palette was baked in as literals, which left the hero subtitle
# white-on-white on a light theme (A-17).
st.markdown(app_css(), unsafe_allow_html=True)

initialize_session_state()


# ── Home page content ─────────────────────────────────────────────────────────


def _home_page() -> None:
    """Render the Dashboard / Help landing page."""
    tab_dash, tab_help = st.tabs(["Dashboard", "Help"])

    with tab_dash:
        # ── Hero ──────────────────────────────────────────────────────────────
        _logo_col, _title_col = st.columns([1, 5], vertical_alignment="center")
        with _logo_col:
            if _LOGO_PNG.exists():
                _b64 = base64.b64encode(_LOGO_PNG.read_bytes()).decode()
                st.markdown(
                    f'<img src="data:image/png;base64,{_b64}" alt="Plottle logo"'
                    ' style="width:110px;height:auto;display:block;">',
                    unsafe_allow_html=True,
                )
        with _title_col:
            st.markdown(
                "<h1 style='margin-bottom: 0.1rem;'>Plottle</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='plottle-muted'>"
                "Scientific data visualization and analysis "
                f"&nbsp;&middot;&nbsp; v{__version__}"
                "</p>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Session metrics ───────────────────────────────────────────────────
        summary = get_session_summary()
        col1, col2, col3 = st.columns(3)
        col1.metric("Datasets", summary["num_datasets"])
        col2.metric("Plots", summary["num_plots"])
        col3.metric("Analyses", summary["num_analyses"])

        if summary["current_dataset"]:
            st.success(f"Active dataset: **{summary['current_dataset']}**")
            st.page_link(basic_pg, label="Go to Quick Plot →")
        else:
            st.info("No dataset loaded.")
            st.page_link(data_pg, label="Upload Data →")

    with tab_help:
        st.markdown("## Quick Start")
        st.markdown("""
        1. **Upload Data** — Go to *Data Upload* to load a CSV, Excel, NumPy, or other file.
           Or load a built-in example dataset.
        2. **Plot** — *Plot → Basic* lets you choose from 27 plot types and configure them
           interactively.
        3. **Analyse** — *Analyze → Single* has curve fitting, statistics, signal processing,
           peak analysis, and more.
        4. **Export** — *Export* saves plots as PNG/PDF/SVG and data in multiple formats.
        """)

        st.markdown("## Supported File Formats")
        st.markdown("""
        | Format | Extensions |
        |---|---|
        | Tabular | CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet |
        | NumPy | .npy, .npz |
        | Python | Pickle (.pkl) |
        | Spectroscopy | JCAMP-DX (.jdx/.dx), SPC (.spc), ASC (.asc) |
        | Scientific | HDF5 (.h5/.hdf5), NetCDF (.nc/.cdf), mzML (.mzml/.mzxml) |
        """)

        st.markdown("## Tips")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Session management"):
                st.markdown("""
                - Datasets persist across all pages during a session
                - Use *Export → Save Session* to preserve your work between restarts
                - Saved sessions can be reloaded from the same page
                """)
            with st.expander("Performance"):
                st.markdown("""
                - Files >50 MB or >10k rows get a downsampled preview automatically
                - Interactive Plotly plots are slower than Matplotlib for large data
                - Clear plot history periodically to free memory
                """)
        with col2:
            with st.expander("Choosing a plot type"):
                st.markdown("""
                - **Histogram** — single-variable distributions
                - **Scatter** — relationship between two variables
                - **Line** — time series or ordered data
                - **Box / Violin** — compare distributions across groups
                - **Heatmap** — 2D matrix or correlation data
                - **Ternary** — 3-component compositions
                """)
            with st.expander("Publication figures"):
                st.markdown("""
                - Apply *Publication Style* in the style sidebar for paper-ready figures
                - Export at 300 DPI from *Export*
                - Use SVG export for vector graphics in manuscripts
                """)

        st.markdown("## About")
        st.markdown(f"""
        **Plottle** is developed at North Carolina Central University for research and teaching
        in computational science.
        Built with Streamlit · NumPy · Pandas · Matplotlib · Seaborn · Plotly · SciPy.
        Version {__version__}
        """)


# ── Page objects ──────────────────────────────────────────────────────────────

home_pg = st.Page(_home_page, title="Home", default=True)
data_pg = st.Page(
    str(_PAGES_DIR / "1_Data_Upload.py"),
    title="Data Upload",
    url_path="data-upload",
)

# Plot group
basic_pg = st.Page(
    str(_PAGES_DIR / "2_Quick_Plot.py"),
    title="Basic",
    url_path="plot-basic",
)
multi_pg = st.Page(
    str(_PAGES_DIR / "4_Multi_Plot_Dashboard.py"),
    title="Multiplot",
    url_path="plot-multiplot",
)
adv_plt_pg = st.Page(
    str(_PAGES_DIR / "5_Advanced_Plotting.py"),
    title="Advanced Plotting",
    url_path="plot-advanced",
)
spec_pg = st.Page(
    str(_PAGES_DIR / "9_Spectroscopy.py"),
    title="Spectroscopy",
    url_path="plot-spectroscopy",
)
molvis_pg = st.Page(
    str(_PAGES_DIR / "10_Molecular_Viz.py"),
    title="Molecular Viz",
    url_path="plot-molecular-viz",
)

# Analyze group
single_pg = st.Page(
    str(_PAGES_DIR / "3_Analysis_Tools.py"),
    title="Single",
    url_path="analyze-single",
)
batch_pg = st.Page(
    str(_PAGES_DIR / "11_Batch_Analysis.py"),
    title="Batch",
    url_path="analyze-batch",
)

# Top-level pages
export_pg = st.Page(
    str(_PAGES_DIR / "6_Export_Results.py"),
    title="Export",
    url_path="export",
)
gallery_pg = st.Page(
    str(_PAGES_DIR / "7_Gallery.py"),
    title="Gallery",
    url_path="gallery",
)
help_pg = st.Page(
    str(_PAGES_DIR / "12_Help.py"),
    title="Help",
    url_path="help",
)
settings_pg = st.Page(
    str(_PAGES_DIR / "13_Settings.py"),
    title="Settings",
    url_path="settings",
)

datatools_pg = st.Page(
    str(_PAGES_DIR / "8_Data_Tools.py"),
    title="Data Tools",
    url_path="analyze-data-tools",
)

# ── Navigation (position="hidden" — custom sidebar built below) ───────────────

_PLOT_PAGES = {basic_pg, multi_pg, adv_plt_pg, spec_pg, molvis_pg}
_ANALYZE_PAGES = {single_pg, batch_pg, datatools_pg}

pg = st.navigation(
    [
        home_pg,
        data_pg,
        basic_pg,
        multi_pg,
        adv_plt_pg,
        spec_pg,
        molvis_pg,
        single_pg,
        batch_pg,
        datatools_pg,
        export_pg,
        gallery_pg,
        help_pg,
        settings_pg,
    ],
    position="hidden",
)


# ── Custom sidebar navigation ─────────────────────────────────────────────────

_ADV_HEADER = "<p class='plottle-section-label'>Advanced</p>"

with st.sidebar:
    if _LOGO_PNG.exists():
        _sb_logo_b64 = base64.b64encode(_LOGO_PNG.read_bytes()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{_sb_logo_b64}" alt="Plottle logo"'
            ' style="width:72px;height:auto;display:block;margin:0.5rem auto 1rem;">',
            unsafe_allow_html=True,
        )
    st.page_link(home_pg, label="Home")
    st.page_link(data_pg, label="Data Upload")

    with st.expander("**Plot**", expanded=pg in _PLOT_PAGES):
        st.page_link(basic_pg, label="Basic")
        st.page_link(multi_pg, label="Multiplot")
        st.markdown(_ADV_HEADER, unsafe_allow_html=True)
        st.page_link(adv_plt_pg, label="↳ Advanced Plotting")
        st.page_link(spec_pg, label="↳ Spectroscopy")
        st.page_link(molvis_pg, label="↳ Molecular Viz")

    with st.expander("**Analyze**", expanded=pg in _ANALYZE_PAGES):
        st.page_link(single_pg, label="Single")
        st.page_link(batch_pg, label="Batch")
        st.page_link(datatools_pg, label="Data Tools")

    st.divider()
    st.page_link(export_pg, label="Export")
    st.page_link(gallery_pg, label="Gallery")
    st.page_link(help_pg, label="Help")
    st.page_link(settings_pg, label="Settings")

    st.divider()
    with st.expander("About", expanded=True):
        if _NCCU_HORIZ.exists():
            _horiz_b64 = base64.b64encode(_NCCU_HORIZ.read_bytes()).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{_horiz_b64}"'
                ' alt="North Carolina Central University"'
                ' style="width:100%;height:auto;margin-bottom:0.4rem;">',
                unsafe_allow_html=True,
            )
        if _NCCU_WINGS.exists():
            _wings_b64 = base64.b64encode(_NCCU_WINGS.read_bytes()).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{_wings_b64}"'
                ' alt="NCCU Eagle Wings emblem"'
                ' style="width:60%;height:auto;display:block;margin:0 auto 0.4rem;">',
                unsafe_allow_html=True,
            )
        st.markdown(
            "**Plottle** is a scientific data visualization and analysis "
            "toolkit developed for research and teaching in computational "
            "science at North Carolina Central University."
        )
        st.markdown("[NCCU Schultz Lab on GitHub](https://github.com/The-Schultz-Lab)")


# ── Execute the active page ───────────────────────────────────────────────────

pg.run()
