"""Data Upload and Preview Page.

This page allows users to upload datasets in various formats, preview
the data, and view basic statistics. Uploaded datasets are stored in
session state and available throughout the app.

Supported Formats
-----------------
- NumPy arrays (.npy, .npz)
- CSV (.csv)
- Excel (.xlsx, .xls)
- TSV (.tsv)
- JSON (.json)
- Parquet (.parquet)
"""

import sys
import streamlit as st
import tempfile
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

_APP_ROOT = Path(__file__).resolve().parent.parent.parent
# Example datasets ship inside the package so this tab works after
# `pip install` -- see audit A-09.
_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "example-data" / "Artificial"

# Metadata for each built-in example file
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
        "use": "Contour Plot · 3D Surface · Heatmap · Interactive Heatmap",
    },
    "correlation_matrix.npy": {
        "label": "Correlation Matrix (2D array)",
        "desc": "10×10 exact correlation matrix derived from random data.",
        "use": "Heatmap · Interactive Heatmap",
    },
}

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.io import load_data  # noqa: E402
from eagleplot.math import calculate_statistics, check_normality  # noqa: E402
from eagleplot.utils import (  # noqa: E402
    initialize_session_state,
    add_dataset,
    get_dataset,
    delete_dataset,
    get_session_summary,
    display_dataset_card,
    display_data_preview,
)
from eagleplot.batch import scan_directory, batch_load_files  # noqa: E402
from eagleplot.io import downsample_for_preview  # noqa: E402
from eagleplot.utils.access import folder_loading_allowed_here  # noqa: E402
from eagleplot.utils.host_folders import local_folder  # noqa: E402
from eagleplot.utils.runtime import capabilities, missing_package_message  # noqa: E402

_LARGE_FILE_BYTES = 50 * 1024 * 1024  # 50 MB — warn above this
_PREVIEW_MAX_ROWS = 10_000  # downsample preview above this


@st.cache_data(show_spinner=False)
def _load_example_cached(path: str):
    """Load an example dataset from a fixed path, cached across reruns."""
    return load_data(path)


# Initialize session state
initialize_session_state()

# Page title
st.title("Data Upload")
st.caption("Load datasets from file, examples, or a folder.")

tab1, tab2, tab3 = st.tabs(["Upload File", "Example Datasets", "Batch Import"])

with tab1:
    # NOTE: `pkl` is deliberately absent. Unpickling an uploaded file executes
    # code contained in it, and a user clicking "Browse files" has not consented
    # to that. `eagleplot.io.load_pickle()` remains available in the Python API,
    # where the caller has chosen to trust the file. See TDEC-011 / audit A-16.
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=[
            "npy",
            "npz",
            "csv",
            "xlsx",
            "xls",
            "tsv",
            "json",
            "parquet",
            "jdx",
            "dx",
            "h5",
            "hdf5",
            "nc",
            "cdf",
            "spc",
            "asc",
            "mzml",
            "mzxml",
        ],
        help="Upload a data file in any supported format",
    )
    st.caption(
        "Pickle (`.pkl`) files are not accepted here — loading one runs any code it "
        "contains. Use `eagleplot.io.load_pickle()` from Python if you trust the file, "
        "or re-save your data as CSV or Parquet."
    )

    if uploaded_file is not None:
        try:
            with st.spinner(f"Loading {uploaded_file.name}..."):
                # Each upload gets its own temp directory, removed as soon as the
                # file is parsed. A shared $TMP/eagleplot/ let two sessions uploading
                # the same filename overwrite each other's data, and nothing was
                # ever deleted. The browser-supplied name is reduced to its
                # basename before it touches the filesystem. See audit A-14.
                with tempfile.TemporaryDirectory(prefix="eagleplot-upload-") as temp_dir:
                    temp_path = Path(temp_dir) / Path(uploaded_file.name).name
                    temp_path.write_bytes(uploaded_file.getvalue())
                    data = load_data(str(temp_path))
                    # np.load() returns a lazy NpzFile that keeps the file open;
                    # read every array now, before the directory is removed.
                    if isinstance(data, np.lib.npyio.NpzFile):
                        with data:
                            data = {key: data[key] for key in data.files}

                # Large-file warning: show a note and offer a downsampled preview
                _is_large = isinstance(data, pd.DataFrame) and (
                    uploaded_file.size > _LARGE_FILE_BYTES or len(data) > _PREVIEW_MAX_ROWS
                )
                if _is_large:
                    st.warning(
                        f"Large dataset: {len(data):,} rows / "
                        f"{uploaded_file.size / 1_048_576:.1f} MB. "
                        "Full data stored; previews use a downsampled view."
                    )

                # Add to session state
                add_dataset(
                    uploaded_file.name,
                    data,
                    metadata={
                        "file_size": uploaded_file.size,
                        "file_type": uploaded_file.type,
                        "downsampled_preview": _is_large,
                    },
                )

                st.toast(f"Loaded '{uploaded_file.name}' successfully", icon="✅")
                st.success(f"Successfully loaded: **{uploaded_file.name}**")

                # Show preview (downsampled for large files)
                st.markdown("### Preview")
                _preview_data = (
                    downsample_for_preview(data, _PREVIEW_MAX_ROWS) if _is_large else data
                )
                display_data_preview(_preview_data, uploaded_file.name)

        except Exception as e:
            st.error(f"Error loading file: {missing_package_message(str(e))}")
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())

with tab2:
    st.markdown(
        "Load a built-in artificial dataset to explore each plot type "
        "immediately — no file upload required."
    )

    if not _EXAMPLE_DIR.exists():
        st.warning(
            f"Example data directory not found: `{_EXAMPLE_DIR}`. "
            "Run `python generate_examples.py` from the repository root first."
        )
    else:
        st.markdown(
            """
            <style>
            /* Equal-height example dataset cards */
            [data-testid="stHorizontalBlock"] {
                align-items: stretch !important;
            }
            [data-testid="column"] {
                display: flex !important;
                flex-direction: column !important;
            }
            [data-testid="column"] > div {
                flex: 1 !important;
                display: flex !important;
                flex-direction: column !important;
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                flex: 1 !important;
                display: flex !important;
                flex-direction: column !important;
            }
            [data-testid="stVerticalBlockBorderWrapper"] > div {
                flex: 1 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        _available = [
            (fname, meta) for fname, meta in _EXAMPLES.items() if (_EXAMPLE_DIR / fname).exists()
        ]
        for i in range(0, len(_available), 2):
            _row_cols = st.columns(2)
            for j, (fname, meta) in enumerate(_available[i : i + 2]):
                fpath = _EXAMPLE_DIR / fname
                with _row_cols[j]:
                    with st.container(border=True):
                        st.markdown(f"**{meta['label']}**")
                        st.caption(meta["desc"])
                        st.caption(f"*Best for:* {meta['use']}")
                        btn_key = f"ex_load_{fname}"
                        if st.button("Load", key=btn_key, width="stretch"):
                            try:
                                ex_data = _load_example_cached(str(fpath))
                                add_dataset(
                                    fname,
                                    ex_data,
                                    metadata={
                                        "source": "example",
                                        "file_type": fpath.suffix,
                                    },
                                )
                                st.success(f"Loaded **{meta['label']}**")
                                st.rerun()
                            except Exception as _ex_err:
                                st.error(f"Error: {_ex_err}")

with tab3:
    # Reading a typed folder path is the point when EaglePlot runs on your own
    # machine, and a way to read the server's disk on a shared deployment.
    # See audit A-15 and eagleplot/utils/access.py.
    if not capabilities().local_folders:
        st.info(
            "Batch import from a folder needs files on a local disk, which this "
            "version of EaglePlot (running in your browser) cannot read. Upload files "
            "individually on the **Upload File** tab instead."
        )
    elif not folder_loading_allowed_here():
        st.info(
            "Batch import from a folder is turned off on this server, because "
            "the path would be read on the server, not your computer. Upload files "
            "individually on the **Upload File** tab instead. To enable it, start "
            "EaglePlot with the environment variable `EAGLEPLOT_ALLOW_FOLDER_LOAD=1`."
        )
    else:
        st.markdown(
            "Load all matching files from a local directory at once. "
            "Each file becomes a separate dataset."
        )
        _batch_folder = st.text_input(
            "Folder path",
            placeholder="C:/Users/you/data",
            key="batch_folder_path",
            help="Absolute path to the directory containing your data files.",
        )
        # `pkl` is deliberately absent here too — see the note on the uploader above.
        _all_exts = [
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
        _batch_exts = st.multiselect(
            "File extensions to include",
            options=_all_exts,
            default=["csv", "xlsx", "tsv"],
            key="batch_extensions",
            help="Leave blank to include all supported formats.",
        )
        _batch_pattern = st.text_input(
            "Filename pattern (optional)",
            placeholder="sample_*.csv",
            key="batch_filename_pattern",
            help="fnmatch-style pattern, e.g. 'sample_*.csv'. Leave blank to match all.",
        )

        _scan_col, _load_col = st.columns(2)
        with _scan_col:
            _do_scan = st.button("Scan Directory", key="batch_scan_btn", width="stretch")
        with _load_col:
            _do_load = st.button("Load All Files", key="batch_load_btn", width="stretch")

        if _do_scan or _do_load:
            if not _batch_folder:
                st.warning("Please enter a folder path.")
            else:
                try:
                    _exts = _batch_exts if _batch_exts else None
                    _pat = _batch_pattern.strip() if _batch_pattern.strip() else None
                    _found = scan_directory(
                        local_folder(_batch_folder), extensions=_exts, pattern=_pat
                    )

                    if not _found:
                        st.info("No matching files found in that directory.")
                    else:
                        # Always show metadata table
                        _meta_rows = []
                        for _fp in _found:
                            _size = _fp.stat().st_size
                            _meta_rows.append(
                                {
                                    "Filename": _fp.name,
                                    "Extension": _fp.suffix,
                                    "Size (KB)": f"{_size / 1024:.1f}",
                                }
                            )
                        st.markdown(f"**{len(_found)} file(s) found:**")
                        st.dataframe(pd.DataFrame(_meta_rows), hide_index=True)

                        if _do_load:
                            with st.spinner(f"Loading {len(_found)} file(s)..."):
                                _result = batch_load_files(_found, on_error="skip")

                            _loaded = _result["datasets"]
                            _errors = _result["errors"]
                            _bmeta = _result["metadata"]

                            for _name, _data in _loaded.items():
                                add_dataset(
                                    _name,
                                    _data,
                                    metadata={
                                        "source": "batch_import",
                                        "file_type": Path(_bmeta[_name]["source"]).suffix,
                                        "size_bytes": _bmeta[_name]["size_bytes"],
                                    },
                                )

                            if _loaded:
                                st.success(
                                    f"Loaded {len(_loaded)} dataset(s): "
                                    + ", ".join(f"**{n}**" for n in _loaded)
                                )
                                st.rerun()
                            if _errors:
                                with st.expander(f"{len(_errors)} file(s) failed to load"):
                                    for _ename, _emsg in _errors.items():
                                        st.error(f"`{_ename}`: {missing_package_message(_emsg)}")

                except NotADirectoryError as _e:
                    st.error(str(_e))
                except Exception as _e:
                    st.error(f"Batch import error: {_e}")


# Dataset management section
st.markdown("---")
st.markdown("## Loaded Datasets")

summary = get_session_summary()

if summary["num_datasets"] == 0:
    st.info("No datasets loaded yet. Upload a file above to get started.")
else:
    st.write(f"**{summary['num_datasets']} dataset(s) loaded**")

    # Dataset selector with delete option
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_dataset = st.selectbox(
            "Select dataset to view",
            options=list(st.session_state.datasets.keys()),
            index=(
                list(st.session_state.datasets.keys()).index(summary["current_dataset"])
                if summary["current_dataset"]
                else 0
            ),
            key="dataset_selector",
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Delete", key="delete_btn", help="Delete selected dataset"):
            if delete_dataset(selected_dataset):
                st.success(f"Deleted {selected_dataset}")
                st.rerun()
            else:
                st.error("Failed to delete dataset")

    # Update current dataset
    if selected_dataset:
        st.session_state.current_dataset = selected_dataset
        data = get_dataset(selected_dataset)
        metadata = st.session_state.dataset_metadata.get(selected_dataset, {})

        # Display dataset card (main area)
        st.markdown("---")
        display_dataset_card(selected_dataset, data, metadata)

        # ── Sidebar: Data Preview + Quick Statistics ──────────────────────
        with st.sidebar:
            st.markdown("## Data Preview")
            display_data_preview(data, selected_dataset)

            st.markdown("---")
            st.markdown("## Quick Statistics")

            if isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

                if len(numeric_cols) > 0:
                    selected_col = st.selectbox(
                        "Select column for statistics",
                        options=numeric_cols,
                        key="stats_column",
                    )

                    if st.button("Calculate Statistics", key="calc_stats"):
                        stats = calculate_statistics(data[selected_col].values)
                        st.write("**Descriptive Statistics:**")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Mean", f"{stats['mean']:.4g}")
                        mc2.metric("Median", f"{stats['median']:.4g}")
                        mc3.metric("Std Dev", f"{stats['std']:.4g}")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Min", f"{stats['min']:.4g}")
                        mc2.metric("Max", f"{stats['max']:.4g}")
                        mc3.metric("Range", f"{stats['range']:.4g}")
                        st.write("**Quartiles:**")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Q1 (25%)", f"{stats['q1']:.4g}")
                        mc2.metric("Q3 (75%)", f"{stats['q3']:.4g}")
                        mc3.metric("IQR", f"{stats['iqr']:.4g}")

                    if st.button("Test Normality", key="test_normality"):
                        result = check_normality(data[selected_col].values)
                        st.write("**Shapiro-Wilk Normality Test:**")
                        st.write(f"Statistic: {result['statistic']:.6f}")
                        st.write(f"P-value: {result['p_value']:.6f}")
                        if result["is_normal"]:
                            st.success("Normally distributed (p > 0.05)")
                        else:
                            st.warning("Not normally distributed (p ≤ 0.05)")
                else:
                    st.info("No numeric columns available for statistics")

            elif isinstance(data, np.ndarray):
                if np.issubdtype(data.dtype, np.number):
                    if st.button("Calculate Statistics", key="calc_stats"):
                        flat_data = data.flatten() if data.ndim > 1 else data
                        stats = calculate_statistics(flat_data)
                        st.write("**Descriptive Statistics:**")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Mean", f"{stats['mean']:.4g}")
                        mc2.metric("Median", f"{stats['median']:.4g}")
                        mc3.metric("Std Dev", f"{stats['std']:.4g}")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Min", f"{stats['min']:.4g}")
                        mc2.metric("Max", f"{stats['max']:.4g}")
                        mc3.metric("Range", f"{stats['range']:.4g}")

                    if st.button("Test Normality", key="test_normality"):
                        flat_data = data.flatten() if data.ndim > 1 else data
                        result = check_normality(flat_data)
                        st.write("**Shapiro-Wilk Normality Test:**")
                        st.write(f"Statistic: {result['statistic']:.6f}")
                        st.write(f"P-value: {result['p_value']:.6f}")
                        if result["is_normal"]:
                            st.success("Normally distributed (p > 0.05)")
                        else:
                            st.warning("Not normally distributed (p ≤ 0.05)")
                else:
                    st.info("Array is not numeric - statistics not applicable")
