"""Help & Documentation page."""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plottle.utils.session_state import initialize_session_state

initialize_session_state()

st.title("Help")
st.caption("Documentation, quick reference, and tips for using Plottle.")

tab_start, tab_plots, tab_analysis, tab_formats, tab_tips = st.tabs(
    ["Getting Started", "Plot Types", "Analysis Tools", "File Formats", "Tips & Tricks"]
)

with tab_start:
    st.markdown("""
    ## Workflow

    1. **Upload Data** — Go to *Data Upload* and upload a file, or load a built-in example dataset.
    2. **Quick Plot** — Choose from 27 plot types, configure parameters interactively
       in the sidebar.
    3. **Analysis Tools** — Statistical tests, curve fitting, signal processing,
       peak analysis, and more.
    4. **Export Results** — Download plots as PNG/PDF/SVG, data as CSV/JSON,
       or save your full session.

    ## Navigation

    All pages are listed in the left sidebar. Your session data (loaded datasets, plot history)
    persists as you navigate between pages.

    ## Example Datasets

    Go to *Data Upload → Example Datasets* to load built-in artificial datasets
    — no file needed. Each one is designed to demonstrate specific plot types.
    """)

with tab_plots:
    st.markdown("## 27 Plot Types")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Matplotlib (static)**
        - Histogram
        - Line Plot
        - Scatter Plot
        - Bar Chart
        - Heatmap
        - Contour Plot
        - Waterfall Plot
        - Dual Axis Plot
        - Broken Axis Plot
        - Inset Plot
        - Z-Colored Scatter
        - Bubble Chart
        - Polar Plot
        - 2D Histogram / Hexbin
        - Residual Plot
        - Scatter with Regression
        """)
    with col2:
        st.markdown("""
        **Seaborn (statistical)**
        - Distribution Plot
        - Box / Violin / Swarm Plot
        - Regression Plot
        - Pair Plot

        **Plotly (interactive)**
        - Interactive Histogram
        - Interactive Scatter
        - Interactive Line
        - Interactive Heatmap
        - Interactive 3D Surface
        - Interactive 3D Scatter
        - Interactive Ternary
        """)
    with col3:
        st.markdown("""
        **Choosing the right plot**

        | Goal | Plot type |
        |---|---|
        | Single distribution | Histogram |
        | Two-variable relationship | Scatter |
        | Time series | Line Plot |
        | Compare groups | Box / Violin |
        | 2D density | 2D Histogram |
        | 3-component data | Ternary |
        | 3D exploration | 3D Scatter |
        | Matrix data | Heatmap |
        | Compositions | Polar |
        """)

    st.markdown("""
    **Zoom and pan on static plots:** Click *Convert to Plotly* below any matplotlib
    figure in Quick Plot to get an interactive version.
    """)

with tab_analysis:
    st.markdown("""
    ## Analysis Tools (page 3)

    | Tab | Capabilities |
    |---|---|
    | Statistics | Descriptive stats, normality tests, distribution fitting |
    | Curve Fitting | Linear, polynomial, exponential, custom functions |
    | Signal Processing | Smoothing, filtering, FFT, baseline correction, derivatives |
    | Peak Analysis | Auto peak finding, FWHM, integration, Gaussian/Lorentzian/Voigt fitting |
    | Statistical Tests | t-tests, ANOVA (one-way, two-way), Mann-Whitney, Tukey HSD, chi-square |
    | Optimization | Function minimization, root finding |
    | Linear Algebra | Eigenvalues, linear systems, matrix decomposition |

    ## Spectroscopy (page 10)

    | Tab | Capabilities |
    |---|---|
    | IR / Raman | ATR correction, spectral subtraction, band assignment, cosmic ray removal |
    | NMR | Chemical shift calibration, line broadening, FFT, peak picking, integration |
    | UV-Vis | Beer-Lambert law, molar absorptivity, spectral overlap integral |
    | Mass Spec | m/z peak finding, centroid spectrum |

    Also includes NIST WebBook lookup (fetch IR spectra by CAS number).

    ## Data Tools (page 9)

    Formula columns, normalization, transpose, pivot/melt, filter, sort, merge,
    fill/drop NaN, resample, rolling transform.
    """)

with tab_formats:
    st.markdown("""
    ## Supported Input Formats

    | Format | Extensions | Notes |
    |---|---|---|
    | CSV | .csv | Auto-detected delimiter |
    | TSV | .tsv | Tab-separated |
    | Excel | .xlsx, .xls | First sheet by default |
    | JSON | .json | Records or columns orientation |
    | Parquet | .parquet | Apache Parquet columnar |
    | NumPy | .npy, .npz | Single array or dict of arrays |
    | Pickle | .pkl | Any Python object |
    | JCAMP-DX | .jdx, .dx | Spectroscopy (requires `jcamp`) |
    | HDF5 | .h5, .hdf5 | Hierarchical data (requires `h5py`) |
    | NetCDF | .nc, .cdf | Climate/scientific arrays (requires `xarray`) |
    | SPC | .spc | Thermo spectroscopy (requires `spc`) |
    | ASC | .asc | Generic whitespace-delimited text |
    | mzML/mzXML | .mzml, .mzxml | Mass spectrometry (requires `pymzml`) |

    ## Export Formats

    **Figures:** PNG (150/300 DPI), SVG, PDF, HTML (Plotly only)
    **Data:** CSV, JSON, NPY, PKL
    **Reports:** Multi-page PDF via Export Results page
    **Session:** Full session save/load (JSON)
    """)

with tab_tips:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Session Management")
        st.markdown("""
        - Datasets persist as you navigate between pages
        - Use *Export Results → Save Session* to preserve work between restarts
        - Reload a session from the same Export Results page
        - Plot history and analysis results are included in saved sessions
        """)

        st.markdown("### Publication Figures")
        st.markdown("""
        - Apply *Publication Style* in the Quick Plot sidebar
        - Export at 300 DPI from Export Results for print quality
        - Use SVG export for vector graphics in manuscripts
        - The colorblind-safe palette (Okabe-Ito) is available in Color Palette settings
        """)

    with col2:
        st.markdown("### Performance")
        st.markdown("""
        - Files >50 MB or >10,000 rows show a downsampled preview automatically
        - Full data is always stored; only the preview is reduced
        - Interactive Plotly plots are slower than Matplotlib for large datasets
        - Clear plot history in Export Results to free memory
        """)

        st.markdown("### Plugins")
        st.markdown("""
        - Drop `plugin_*.py` files into the `plugins/` directory
        - Plugins are auto-discovered at startup
        - See `plugins/plugin_example.py` for the plugin template
        - Plugin status is shown in Settings (page 6)
        """)
