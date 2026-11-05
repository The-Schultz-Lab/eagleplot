"""Batch Analysis Page.

Run statistical summaries, curve fits, and peak analysis across multiple
loaded datasets simultaneously.  Workflow presets let you save and recall
named analysis configurations that persist across app restarts via config.json.
"""

from __future__ import annotations

from pathlib import Path
import sys
import json

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.utils.figures import download_payload  # noqa: E402
from eagleplot.utils.session_state import initialize_session_state
from eagleplot.batch import (
    batch_statistics,
    batch_curve_fit,
    batch_peak_analysis,
)
from eagleplot.utils.user_settings import (
    list_workflows,
    save_workflow,
    load_workflow,
    delete_workflow,
)

initialize_session_state()

st.title("Batch Analysis")
st.caption(
    "Run the same analysis across multiple loaded datasets at once. "
    "Workflow presets let you save and reload named analysis configurations."
)

# ── helpers ───────────────────────────────────────────────────────────────────

_RESULT_OUTPUT_COLS = [
    "dataset",
    "column",
    "n",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "q1",
    "q3",
    "iqr",
    "range",
]


def _show_result(state_key: str, filename: str) -> None:
    """Show the last result of a batch tool, with a CSV download.

    Results live in session state, so they stay on screen after any later interaction,
    including clicking Download (which reruns the page). They used to exist only on the
    rerun right after Run. The CSV is built on click, or once per result (roadmap D1.2).
    """
    saved = st.session_state.get(state_key)
    if not saved:
        return
    result_df, names = saved["df"], saved["datasets"]
    st.success(f"Results for {len(names)} dataset(s): {', '.join(names)}")
    st.dataframe(result_df, width="stretch")
    st.download_button(
        "Download CSV",
        data=download_payload(
            lambda df=result_df: df.to_csv(index=False).encode("utf-8"),
            cache_name=f"_{state_key}_csv",
            version=id(result_df),
            key="csv",
        ),
        file_name=filename,
        mime="text/csv",
        key=f"{state_key}_dl",
    )


# ── top-level tabs ─────────────────────────────────────────────────────────────
tab_analysis, tab_workflows = st.tabs(["Batch Analysis", "Workflow Presets"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    # Dataset selection — only DataFrames
    all_datasets = st.session_state.get("datasets", {})
    df_datasets = {k: v for k, v in all_datasets.items() if isinstance(v, pd.DataFrame)}

    if not df_datasets:
        st.warning(
            "No tabular datasets loaded.  "
            "Please upload CSV / Excel / TSV / JSON / Parquet files on the "
            "**Data Upload** page first."
        )
        st.stop()

    selected_names = st.multiselect(
        "Select datasets to analyse",
        list(df_datasets.keys()),
        default=list(df_datasets.keys()),
        key="ba_selected_datasets",
    )

    if not selected_names:
        st.info("Select at least one dataset above to continue.")
        st.stop()

    selected_datasets: dict[str, pd.DataFrame] = {k: df_datasets[k] for k in selected_names}

    # Common columns across all selected datasets
    common_cols = set.intersection(*(set(df.columns.tolist()) for df in selected_datasets.values()))
    common_cols_list = sorted(common_cols)

    # ── sub-tabs ──────────────────────────────────────────────────────────────
    sub_stats, sub_fit, sub_peaks = st.tabs(["Statistics", "Curve Fit", "Peak Analysis"])

    # ── Statistics ────────────────────────────────────────────────────────────
    with sub_stats:
        st.subheader("Batch Statistics")
        st.markdown(
            "Compute descriptive statistics (mean, median, std, quartiles, …) "
            "for every selected dataset."
        )

        col_filter = st.multiselect(
            "Column filter (leave blank for all columns)",
            common_cols_list,
            key="ba_stats_cols",
            help="Only columns common to all selected datasets are shown here. "
            "Leave empty to include all columns in each dataset.",
        )

        if st.button("Run Statistics", key="ba_stats_run"):
            cols_arg = col_filter if col_filter else None
            try:
                result_df = batch_statistics(selected_datasets, columns=cols_arg)
                st.session_state["ba_stats_result"] = {
                    "df": result_df,
                    "datasets": list(selected_names),
                }
            except Exception as exc:
                st.error(f"Error: {exc}")
        _show_result("ba_stats_result", "batch_statistics.csv")

    # ── Curve Fit ─────────────────────────────────────────────────────────────
    with sub_fit:
        st.subheader("Batch Curve Fit")
        st.markdown("Fit the same model to the same x/y columns across all selected datasets.")

        fit_x_col = st.text_input("X column name", value="x", key="ba_fit_x")
        fit_y_col = st.text_input("Y column name", value="y", key="ba_fit_y")
        fit_type = st.selectbox(
            "Fit type",
            ["linear", "polynomial", "exponential"],
            key="ba_fit_type",
        )

        fit_degree = 2
        if fit_type == "polynomial":
            fit_degree = int(
                st.number_input(
                    "Polynomial degree",
                    min_value=2,
                    max_value=5,
                    value=2,
                    step=1,
                    key="ba_fit_degree",
                )
            )

        if st.button("Run Curve Fit", key="ba_fit_run"):
            if not fit_x_col.strip() or not fit_y_col.strip():
                st.warning("Enter both an X column name and a Y column name.")
            else:
                try:
                    result_df = batch_curve_fit(
                        selected_datasets,
                        x_col=fit_x_col.strip(),
                        y_col=fit_y_col.strip(),
                        fit_type=fit_type,
                        degree=fit_degree,
                    )
                    st.session_state["ba_fit_result"] = {
                        "df": result_df,
                        "datasets": list(selected_names),
                    }
                except Exception as exc:
                    st.error(f"Error: {exc}")
        _show_result("ba_fit_result", "batch_curve_fit.csv")

    # ── Peak Analysis ─────────────────────────────────────────────────────────
    with sub_peaks:
        st.subheader("Batch Peak Analysis")
        st.markdown(
            "Detect and characterise peaks in the same column across all selected datasets."
        )

        pa_y_col = st.text_input("Y column (signal)", value="y", key="ba_pa_y")
        pa_x_col = st.text_input(
            "X column (optional — leave blank to use row indices)",
            value="",
            key="ba_pa_x",
        )

        pa_use_height = st.checkbox("Set minimum height", value=False, key="ba_pa_use_height")
        pa_height: float | None = None
        if pa_use_height:
            pa_height = st.number_input(
                "Minimum height (absolute)",
                value=0.0,
                step=0.01,
                format="%.4f",
                key="ba_pa_height",
            )

        pa_use_prominence = st.checkbox("Set minimum prominence", value=False, key="ba_pa_use_prom")
        pa_prominence: float | None = None
        if pa_use_prominence:
            pa_prominence = st.number_input(
                "Minimum prominence (absolute)",
                value=0.0,
                step=0.01,
                format="%.4f",
                key="ba_pa_prominence",
            )

        pa_distance = int(
            st.number_input(
                "Minimum distance between peaks (samples)",
                min_value=1,
                value=1,
                step=1,
                key="ba_pa_distance",
            )
        )

        if st.button("Run Peak Analysis", key="ba_pa_run"):
            if not pa_y_col.strip():
                st.warning("Enter a Y column name.")
            else:
                try:
                    result_df = batch_peak_analysis(
                        selected_datasets,
                        y_col=pa_y_col.strip(),
                        x_col=pa_x_col.strip() if pa_x_col.strip() else None,
                        height=pa_height,
                        prominence=pa_prominence,
                        distance=pa_distance,
                    )
                    st.session_state["ba_pa_result"] = {
                        "df": result_df,
                        "datasets": list(selected_names),
                    }
                except Exception as exc:
                    st.error(f"Error: {exc}")
        _show_result("ba_pa_result", "batch_peak_analysis.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Workflow Presets
# ══════════════════════════════════════════════════════════════════════════════
with tab_workflows:
    st.subheader("Workflow Presets")
    st.info(
        "Workflows are saved in **config.json** and persist across app restarts.  "
        "A workflow is a named list of analysis steps stored as JSON."
    )

    _TEMPLATE_STEPS = [
        {"operation": "batch_statistics", "params": {"columns": []}},
        {
            "operation": "batch_curve_fit",
            "params": {"x_col": "x", "y_col": "y", "fit_type": "linear"},
        },
    ]

    # ── Load section ──────────────────────────────────────────────────────────
    st.markdown("### Load a Workflow")
    existing_workflows = list_workflows()

    if not existing_workflows:
        st.caption("No workflows saved yet.  Use the **Save** section below to create one.")
    else:
        load_name = st.selectbox("Select workflow", existing_workflows, key="wf_load_name")
        if st.button("Load workflow description", key="wf_load_btn"):
            try:
                steps = load_workflow(load_name)
                st.json(steps)
            except Exception as exc:
                st.error(f"Could not load workflow: {exc}")

    st.divider()

    # ── Save section ──────────────────────────────────────────────────────────
    st.markdown("### Save a Workflow")

    wf_save_name = st.text_input(
        "Workflow name",
        value="my_workflow",
        key="wf_save_name",
        help="Use a descriptive name; spaces and special characters are allowed.",
    )
    wf_json_text = st.text_area(
        "Workflow steps (JSON list)",
        value=json.dumps(_TEMPLATE_STEPS, indent=2),
        height=200,
        key="wf_json_text",
        help="Each step is an object with an 'operation' key and a 'params' dict.",
    )

    if st.button("Save Workflow", key="wf_save_btn"):
        name = wf_save_name.strip()
        if not name:
            st.warning("Enter a workflow name.")
        else:
            try:
                steps = json.loads(wf_json_text)
                if not isinstance(steps, list):
                    st.error("Workflow steps must be a JSON **list** (array).")
                else:
                    save_workflow(name, steps)
                    st.success(f"Workflow **{name}** saved.")
                    st.rerun()
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            except Exception as exc:
                st.error(f"Error saving workflow: {exc}")

    st.divider()

    # ── Delete section ────────────────────────────────────────────────────────
    st.markdown("### Delete a Workflow")

    existing_for_delete = list_workflows()
    if not existing_for_delete:
        st.caption("No workflows to delete.")
    else:
        del_name = st.selectbox("Select workflow to delete", existing_for_delete, key="wf_del_name")
        if st.button("Delete", key="wf_del_btn", type="primary"):
            removed = delete_workflow(del_name)
            if removed:
                st.success(f"Workflow **{del_name}** deleted.")
                st.rerun()
            else:
                st.error(f"Workflow **{del_name}** not found.")
