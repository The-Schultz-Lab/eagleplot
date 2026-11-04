"""Export Results Page.

Download plots, datasets, and analysis results generated during the session.

Sections
--------
1. Plot History   — download each saved plot as PNG (150/300 dpi), SVG, PDF, or HTML
2. Dataset Export — download the current dataset as CSV or JSON
3. Analysis Export — download stored analysis results as JSON
4. Session Save/Load — persist the full session to a JSON file
"""

import io
import json
import tempfile
from pathlib import Path
import sys

import streamlit as st
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plottle.utils import initialize_session_state
from plottle.utils.session_state import (
    get_current_dataset,
    get_session_summary,
    save_session_to_file,
    load_session_from_file,
)

initialize_session_state()

st.title("Export Results")
st.caption("Download plots, datasets, and analysis results from the current session.")

# ── Helper: render a Matplotlib figure to bytes ───────────────────────────────


def _mpl_bytes(fig, fmt: str, dpi: int = 150) -> bytes:
    """Render a Matplotlib figure to bytes in the requested format."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _plotly_html(fig) -> bytes:
    """Render a Plotly figure to HTML bytes."""
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def _is_mpl(fig) -> bool:
    try:
        from matplotlib.figure import Figure

        return isinstance(fig, Figure)
    except ImportError:
        return False


def _is_plotly(fig) -> bool:
    try:
        import plotly.graph_objects as go

        return isinstance(fig, go.Figure)
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Plot History
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Plot History")

plot_history = st.session_state.get("plot_history", [])

if not plot_history:
    st.info(
        "No plots saved yet. Generate plots on the Quick Plot, "
        "Analysis Tools, Dashboard, or Advanced Plotting pages."
    )
else:
    st.markdown(f"**{len(plot_history)} plot(s) in session.**")

    for i, entry in enumerate(plot_history):
        plot_type = entry.get("type", "plot")
        dataset = entry.get("dataset", "—")
        timestamp = entry.get("timestamp", "")[:19].replace("T", " ")
        fig = entry.get("figure")

        label = f"Plot {i + 1} — {plot_type} | dataset: {dataset} | {timestamp}"

        with st.expander(label, expanded=(i == len(plot_history) - 1)):
            if fig is None:
                st.warning(
                    "Figure object not available for this entry. "
                    "Re-generate the plot to enable downloads."
                )
                continue

            if _is_mpl(fig):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.download_button(
                        "PNG 150 dpi",
                        data=_mpl_bytes(fig, "png", dpi=150),
                        file_name=f"plot_{i + 1}.png",
                        mime="image/png",
                        key=f"dl_png150_{i}",
                    )
                with c2:
                    st.download_button(
                        "PNG 300 dpi",
                        data=_mpl_bytes(fig, "png", dpi=300),
                        file_name=f"plot_{i + 1}_hires.png",
                        mime="image/png",
                        key=f"dl_png300_{i}",
                    )
                with c3:
                    st.download_button(
                        "SVG",
                        data=_mpl_bytes(fig, "svg"),
                        file_name=f"plot_{i + 1}.svg",
                        mime="image/svg+xml",
                        key=f"dl_svg_{i}",
                    )
                with c4:
                    st.download_button(
                        "PDF",
                        data=_mpl_bytes(fig, "pdf"),
                        file_name=f"plot_{i + 1}.pdf",
                        mime="application/pdf",
                        key=f"dl_pdf_{i}",
                    )

            elif _is_plotly(fig):
                st.download_button(
                    "HTML (interactive)",
                    data=_plotly_html(fig),
                    file_name=f"plot_{i + 1}.html",
                    mime="text/html",
                    key=f"dl_html_{i}",
                )
                st.caption(
                    "Plotly figures export as self-contained HTML. "
                    "PNG export requires the `kaleido` package."
                )
                try:
                    import kaleido  # noqa: F401

                    png_bytes = fig.to_image(format="png", scale=2)
                    st.download_button(
                        "PNG (2x)",
                        data=png_bytes,
                        file_name=f"plot_{i + 1}.png",
                        mime="image/png",
                        key=f"dl_plotly_png_{i}",
                    )
                except ImportError:
                    pass
            else:
                st.warning(f"Unknown figure type: {type(fig).__name__}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Dataset Export
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Dataset Export")

current_data = get_current_dataset()
current_name = st.session_state.get("current_dataset", None)

if current_data is None:
    st.info("No dataset loaded. Upload data on the Data Upload page.")
else:
    st.markdown(f"**Current dataset:** `{current_name}`")

    if isinstance(current_data, pd.DataFrame):
        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = current_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download as CSV",
                data=csv_bytes,
                file_name=f"{Path(current_name).stem}_export.csv",
                mime="text/csv",
                key="dl_dataset_csv",
            )
        with c2:
            json_bytes = current_data.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "Download as JSON",
                data=json_bytes,
                file_name=f"{Path(current_name).stem}_export.json",
                mime="application/json",
                key="dl_dataset_json",
            )

    elif isinstance(current_data, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, current_data)
        buf.seek(0)
        st.download_button(
            "Download as .npy",
            data=buf.read(),
            file_name=f"{Path(current_name).stem}_export.npy",
            mime="application/octet-stream",
            key="dl_dataset_npy",
        )

    else:
        import pickle as _pickle

        pkl_bytes = _pickle.dumps(current_data)
        st.download_button(
            "Download as .pkl",
            data=pkl_bytes,
            file_name=f"{Path(current_name).stem}_export.pkl",
            mime="application/octet-stream",
            key="dl_dataset_pkl",
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Analysis Results Export
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Analysis Results")

analysis_results = st.session_state.get("analysis_results", [])

if not analysis_results:
    st.info("No analysis results stored yet. Run analyses on the Analysis Tools page.")
else:
    st.markdown(f"**{len(analysis_results)} result(s) stored.**")

    # Build a JSON-serializable list (strip non-serializable items like arrays)
    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(x) for x in obj]
        return obj

    serializable = _make_serializable(analysis_results)
    results_json = json.dumps(serializable, indent=2).encode("utf-8")

    st.download_button(
        "Download all results as JSON",
        data=results_json,
        file_name="analysis_results.json",
        mime="application/json",
        key="dl_analysis_json",
    )

    with st.expander("Preview results"):
        for j, res in enumerate(analysis_results):
            r_type = res.get("type", "result")
            r_ts = res.get("timestamp", "")[:19].replace("T", " ")
            st.markdown(f"**{j + 1}. {r_type}** — {r_ts}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — PDF Report
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## PDF Report")
st.markdown(
    "Generate a PDF report containing all matplotlib plots, dataset previews, "
    "and analysis results from the current session."
)

# Import report generator lazily to avoid hard dependency at module load
try:
    from plottle.report import generate_pdf_report as _gen_pdf

    _report_available = True
except ImportError:
    _report_available = False

if not _report_available:
    st.warning("PDF report module not available.")
else:
    _report_title = st.text_input(
        "Report title",
        value="Plottle Analysis Report",
        key="pdf_report_title",
    )
    _report_author = st.text_input(
        "Author (optional)",
        value="",
        key="pdf_report_author",
    )
    _report_dpi = st.select_slider(
        "Figure resolution (DPI)",
        options=[72, 100, 150, 200, 300],
        value=150,
        key="pdf_report_dpi",
    )

    if st.button("Generate PDF Report", type="primary", key="pdf_report_btn"):
        _df_datasets = {
            k: v
            for k, v in st.session_state.get("datasets", {}).items()
            if isinstance(v, pd.DataFrame)
        }
        with st.spinner("Generating PDF..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as _tmp:
                _tmp_path = _tmp.name
            try:
                _gen_pdf(
                    filepath=_tmp_path,
                    title=_report_title,
                    plot_entries=st.session_state.get("plot_history", []),
                    datasets=_df_datasets,
                    analysis_results=st.session_state.get("analysis_results", []),
                    author=_report_author,
                    dpi=_report_dpi,
                )
                _pdf_bytes = Path(_tmp_path).read_bytes()
                st.download_button(
                    "Download PDF Report",
                    data=_pdf_bytes,
                    file_name="plottle_report.pdf",
                    mime="application/pdf",
                    key="dl_pdf_report",
                )
                st.success("PDF ready for download.")
            except Exception as _pdf_err:
                st.error(f"Could not generate PDF: {_pdf_err}")
            finally:
                Path(_tmp_path).unlink(missing_ok=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Session Save / Load
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Session Save / Load")
st.markdown(
    "Save the full session (datasets, plot history metadata, analysis results) "
    "to a JSON file. Note: figure objects are **not** included — only metadata."
)

tab_save, tab_load = st.tabs(["Save session", "Load session"])

with tab_save:
    summary = get_session_summary()
    st.markdown(
        f"- **Datasets:** {summary['num_datasets']}  \n"
        f"- **Plots in history:** {summary['num_plots']}  \n"
        f"- **Analysis results:** {summary['num_analyses']}"
    )

    # The download button used to be drawn inside `if st.button(...)`, so it existed
    # for one rerun only: the next interaction anywhere on the page removed it. The
    # generated bytes now live in session state and the download button is drawn on
    # every rerun while they still match the session. Generation stays an explicit
    # click because serialising large datasets on every rerun would be slow. A-32.
    _session_fingerprint = (
        tuple((name, id(obj)) for name, obj in st.session_state.datasets.items()),
        summary["current_dataset"],
        summary["num_plots"],
        summary["num_analyses"],
    )
    _cached = st.session_state.get("_session_export")
    if _cached is not None and _cached["fingerprint"] != _session_fingerprint:
        _cached = None
        st.session_state.pop("_session_export", None)

    if st.button(
        "Regenerate session file" if _cached else "Generate session file",
        type="secondary" if _cached else "primary",
        key="session_save_btn",
    ):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            save_session_to_file(tmp_path)
            _cached = {
                "fingerprint": _session_fingerprint,
                "data": Path(tmp_path).read_bytes(),
            }
            st.session_state["_session_export"] = _cached
        except Exception as exc:
            st.error(f"Could not save session: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if _cached is not None:
        st.download_button(
            "Download session.json",
            data=_cached["data"],
            file_name="plottle_session.json",
            mime="application/json",
            type="primary",
            key="dl_session_json",
        )
        st.caption(f"{len(_cached['data']) / 1024:,.1f} KB, matches the current session.")

with tab_load:
    uploaded = st.file_uploader(
        "Upload a session file (.json)",
        type=["json"],
        key="session_upload",
    )
    if uploaded is not None:
        if st.button("Restore session", type="primary", key="session_load_btn"):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                skipped = load_session_from_file(tmp_path)
                if skipped:
                    # Deliberately no st.rerun() here — a rerun would discard this
                    # warning before the user ever sees it. The data is already in
                    # session state, so navigating to another page picks it up.
                    st.warning(
                        "Session restored, but some entries could not be loaded:\n\n"
                        + "\n".join(f"- {note}" for note in skipped)
                        + "\n\nSessions saved by Plottle 2.0.1 or earlier stored arrays in a "
                        "format that is no longer accepted. Re-upload the original data files "
                        "and save the session again."
                    )
                else:
                    st.success("Session restored. Navigate to other pages to use the loaded data.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not load session: {exc}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
