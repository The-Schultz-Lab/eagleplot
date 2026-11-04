"""Data Tools Page.

Non-destructive DataFrame transformations: column math, normalization,
transpose, pivot/melt, filter, sort, merge, fill/drop NaN, resample,
and rolling/cumulative operations.

All operations save the result as a new named dataset in session state
so the original data is always preserved.
"""

from pathlib import Path
import sys

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plottle.utils.session_state import (
    initialize_session_state,
    add_dataset,
)
from plottle.data_tools import (
    add_formula_column,
    normalize_column,
    transpose_dataframe,
    pivot_dataframe,
    melt_dataframe,
    filter_rows,
    sort_dataframe,
    merge_dataframes,
    fill_nan,
    drop_nan,
    resample_dataframe,
    rolling_transform,
)

initialize_session_state()

st.title("Data Tools")
st.caption("Non-destructive transformations — each operation creates a new named dataset.")

# ── dataset selector ──────────────────────────────────────────────────────────
dataset_names = list(st.session_state.datasets.keys())
if not dataset_names:
    st.warning("No dataset loaded. Please upload data on the **Data Upload** page first.")
    st.stop()

selected_name = st.selectbox("Source dataset", dataset_names, key="dt_source_ds")
data = st.session_state.datasets[selected_name]

if not isinstance(data, pd.DataFrame):
    st.error(
        "Data Tools requires a **tabular (DataFrame)** dataset.  "
        "Please upload a CSV, Excel, TSV, JSON, or Parquet file."
    )
    st.stop()

df = data
st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()


def _save_result(result_df: pd.DataFrame, suffix: str) -> None:
    """Register *result_df* in session state and show a preview."""
    new_name = f"{selected_name.rsplit('.', 1)[0]}_{suffix}"
    add_dataset(new_name, result_df)
    st.session_state["current_dataset"] = new_name
    st.success(f"Saved as **{new_name}** ({result_df.shape[0]} rows × {result_df.shape[1]} cols)")
    st.dataframe(result_df.head(20), width="stretch")


# ── tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "Column Math",
        "Normalize",
        "Transpose",
        "Pivot / Melt",
        "Filter Rows",
        "Sort",
        "Merge",
        "Fill / Drop NaN",
        "Resample",
        "Rolling / Cumulative",
    ]
)

# ── 1. Column Math ────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Column Math")
    st.markdown(
        "Compute a new column from a Python/NumPy expression.  "
        "Reference columns by name; available functions: "
        "`log`, `log10`, `log2`, `exp`, `sqrt`, `abs`, `sin`, `cos`, `tan`, "
        "`mean`, `std`, `min`, `max`, `sum`, `cumsum`, `diff`."
    )
    st.dataframe(df.head(5), width="stretch")

    new_col_name = st.text_input("New column name", value="result", key="cm_new_col")
    example_expr = f"{all_cols[0]} * 2" if all_cols else "A * 2"
    expression = st.text_input(
        "Expression",
        value=example_expr,
        help="Example: log(A) / B   or   (A - mean(A)) / std(A)",
        key="cm_expr",
    )

    if st.button("Apply", key="cm_apply"):
        try:
            result = add_formula_column(df, new_col_name, expression)
            _save_result(result, f"col_{new_col_name}")
        except Exception as exc:
            st.error(f"Error: {exc}")

# ── 2. Normalize ──────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Normalize Column")
    st.markdown("Scale a numeric column and append the result as a new column.")

    if not numeric_cols:
        st.warning("No numeric columns found in this dataset.")
    else:
        norm_col = st.selectbox("Column to normalize", numeric_cols, key="norm_col")
        norm_method = st.selectbox(
            "Method",
            ["min-max", "z-score", "pct-max", "area"],
            format_func=lambda m: {
                "min-max": "Min-Max  [0, 1]",
                "z-score": "Z-Score  (μ=0, σ=1)",
                "pct-max": "% of Max  [0, 100%]",
                "area": "Unit Area  (∫|y|dx = 1)",
            }[m],
            key="norm_method",
        )
        norm_new_col = st.text_input(
            "Output column name", value=f"{norm_col}_norm", key="norm_new_col"
        )

        if st.button("Normalize", key="norm_apply"):
            try:
                result = normalize_column(df, norm_col, norm_method, norm_new_col)
                _save_result(result, f"norm_{norm_col}")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ── 3. Transpose ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Transpose")
    st.markdown("Swap rows and columns.")
    st.dataframe(df.head(5), width="stretch")

    if st.button("Transpose", key="tp_apply"):
        try:
            result = transpose_dataframe(df)
            _save_result(result, "transposed")
        except Exception as exc:
            st.error(f"Error: {exc}")

# ── 4. Pivot / Melt ───────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Pivot / Melt")
    sub_pivot, sub_melt = st.tabs(["Pivot  (long → wide)", "Melt  (wide → long)"])

    with sub_pivot:
        st.markdown(
            "**Pivot** reshapes long-format data into wide format.  "
            "Choose the column whose unique values become new column headers."
        )
        piv_index = st.selectbox("Index column (row ID)", all_cols, key="piv_index")
        piv_columns = st.selectbox("Columns column (header values)", all_cols, key="piv_cols")
        piv_values = st.selectbox("Values column", all_cols, key="piv_vals")

        if st.button("Pivot", key="piv_apply"):
            try:
                result = pivot_dataframe(df, piv_index, piv_columns, piv_values)
                _save_result(result, "pivoted")
            except Exception as exc:
                st.error(f"Error: {exc}")

    with sub_melt:
        st.markdown(
            "**Melt** reshapes wide-format data into long format.  "
            "ID columns are kept; value columns are stacked."
        )
        id_vars = st.multiselect("ID columns (keep as-is)", all_cols, key="melt_id")
        value_vars = st.multiselect(
            "Value columns to melt (leave empty = all remaining)", all_cols, key="melt_vals"
        )
        melt_var_name = st.text_input("Variable column name", value="variable", key="melt_var")
        melt_val_name = st.text_input("Value column name", value="value", key="melt_val")

        if st.button("Melt", key="melt_apply"):
            try:
                result = melt_dataframe(
                    df,
                    id_vars=id_vars,
                    value_vars=value_vars if value_vars else None,
                    var_name=melt_var_name,
                    value_name=melt_val_name,
                )
                _save_result(result, "melted")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ── 5. Filter Rows ────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Filter Rows")
    st.markdown(
        "Keep or drop rows matching a condition using pandas `query` syntax.  "
        "Examples: `A > 0`, `B == 'control'`, `A > 0 and C < 10`."
    )
    st.caption(f"Available columns: {', '.join(all_cols)}")

    filt_condition = st.text_input(
        "Condition",
        value=f"{all_cols[0]} > 0" if all_cols else "A > 0",
        key="filt_cond",
    )
    filt_keep = st.radio("Action", ["Keep matching rows", "Drop matching rows"], key="filt_keep")

    if st.button("Apply Filter", key="filt_apply"):
        try:
            keep = filt_keep == "Keep matching rows"
            result = filter_rows(df, filt_condition, keep=keep)
            st.info(
                f"Rows before: {len(df)} → after: {len(result)} "
                f"({'kept' if keep else 'dropped'} {abs(len(df) - len(result))} rows)"
            )
            _save_result(result, "filtered")
        except Exception as exc:
            st.error(f"Error: {exc}")

# ── 6. Sort ───────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Sort")
    sort_by = st.multiselect("Sort by (in order)", all_cols, default=all_cols[:1], key="sort_by")
    sort_asc = st.radio("Direction", ["Ascending ↑", "Descending ↓"], key="sort_dir")

    if st.button("Sort", key="sort_apply"):
        if not sort_by:
            st.warning("Select at least one column to sort by.")
        else:
            try:
                result = sort_dataframe(df, sort_by, ascending=(sort_asc == "Ascending ↑"))
                _save_result(result, "sorted")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ── 7. Merge ──────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Merge / Join")
    st.markdown("Join the current dataset with a second loaded dataset on a shared key column.")

    other_options = [n for n in dataset_names if n != selected_name]
    if not other_options:
        st.info("Load at least **two** datasets to use the Merge tool.")
    else:
        right_name = st.selectbox("Right dataset", other_options, key="merge_right")
        right_df = st.session_state.datasets[right_name]

        if not isinstance(right_df, pd.DataFrame):
            st.error("Right dataset must be a DataFrame.")
        else:
            common_cols = list(set(all_cols) & set(right_df.columns.tolist()))
            if not common_cols:
                st.warning(
                    "No columns in common between the two datasets.  "
                    "Ensure both have at least one column with the same name to join on."
                )
            else:
                merge_key = st.selectbox("Join key column", common_cols, key="merge_key")
                merge_how = st.selectbox(
                    "Join type", ["inner", "left", "right", "outer"], key="merge_how"
                )

                if st.button("Merge", key="merge_apply"):
                    try:
                        result = merge_dataframes(df, right_df, on=merge_key, how=merge_how)
                        _save_result(result, "merged")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

# ── 8. Fill / Drop NaN ────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Fill / Drop NaN")

    nan_counts = df.isna().sum()
    total_nan = int(nan_counts.sum())
    st.metric("Total NaN cells", total_nan)
    if total_nan > 0:
        nan_table = nan_counts[nan_counts > 0].rename("NaN count").to_frame()
        st.dataframe(nan_table, width="content")

    nan_op = st.radio("Operation", ["Fill NaN", "Drop rows with NaN"], key="nan_op")

    if nan_op == "Fill NaN":
        fill_cols = st.multiselect(
            "Columns to fill (empty = all numeric)", numeric_cols, key="fill_cols"
        )
        fill_method = st.selectbox(
            "Fill strategy",
            ["mean", "median", "zero", "interpolate", "forward", "backward"],
            format_func=lambda m: {
                "mean": "Column mean",
                "median": "Column median",
                "zero": "Zero (0)",
                "interpolate": "Linear interpolation",
                "forward": "Forward-fill",
                "backward": "Backward-fill",
            }[m],
            key="fill_method",
        )

        if st.button("Fill NaN", key="fill_apply"):
            try:
                cols = fill_cols if fill_cols else None
                result = fill_nan(df, columns=cols, method=fill_method)
                remaining = int(result.isna().sum().sum())
                st.info(f"NaN cells before: {total_nan} → after: {remaining}")
                _save_result(result, "filled")
            except Exception as exc:
                st.error(f"Error: {exc}")

    else:
        drop_cols = st.multiselect(
            "Only check these columns (empty = any column)", all_cols, key="drop_cols"
        )

        if st.button("Drop rows with NaN", key="drop_apply"):
            try:
                cols = drop_cols if drop_cols else None
                result = drop_nan(df, columns=cols)
                st.info(f"Rows before: {len(df)} → after: {len(result)}")
                _save_result(result, "dropna")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ── 9. Resample ───────────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Resample")
    st.markdown(
        "Interpolate selected y-columns onto a new, uniformly-spaced x-grid.  "
        "Useful for aligning spectra collected at different wavelength spacings."
    )

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns (one x, one y) to resample.")
    else:
        rs_x = st.selectbox("X column (independent variable)", numeric_cols, key="rs_x")
        rs_y = st.multiselect(
            "Y columns to resample",
            [c for c in numeric_cols if c != rs_x],
            default=[c for c in numeric_cols if c != rs_x][:1],
            key="rs_y",
        )
        rs_n = st.number_input(
            "Number of output points",
            min_value=10,
            max_value=10000,
            value=min(500, len(df)),
            step=10,
            key="rs_n",
        )
        rs_method = st.selectbox(
            "Interpolation method", ["cubic", "linear", "nearest"], key="rs_method"
        )

        x_vals = df[rs_x].dropna()
        rs_xmin = st.number_input("X min", value=float(x_vals.min()), key="rs_xmin")
        rs_xmax = st.number_input("X max", value=float(x_vals.max()), key="rs_xmax")

        if st.button("Resample", key="rs_apply"):
            if not rs_y:
                st.warning("Select at least one Y column.")
            else:
                try:
                    result = resample_dataframe(
                        df,
                        rs_x,
                        rs_y,
                        n_points=int(rs_n),
                        method=rs_method,
                        x_min=rs_xmin,
                        x_max=rs_xmax,
                    )
                    _save_result(result, "resampled")
                except Exception as exc:
                    st.error(f"Error: {exc}")

# ── 10. Rolling / Cumulative ──────────────────────────────────────────────────
with tabs[9]:
    st.subheader("Rolling / Cumulative")
    st.markdown(
        "Apply a sliding-window or cumulative transform to selected columns.  "
        "The result is appended as new column(s) in a copy of the dataset."
    )

    if not numeric_cols:
        st.warning("No numeric columns found.")
    else:
        roll_cols = st.multiselect(
            "Columns to transform", numeric_cols, default=numeric_cols[:1], key="roll_cols"
        )
        roll_op = st.selectbox(
            "Operation",
            ["rolling_mean", "rolling_sum", "cumsum", "cumprod"],
            format_func=lambda m: {
                "rolling_mean": "Rolling mean",
                "rolling_sum": "Rolling sum",
                "cumsum": "Cumulative sum",
                "cumprod": "Cumulative product",
            }[m],
            key="roll_op",
        )
        roll_window = 5
        if roll_op in ("rolling_mean", "rolling_sum"):
            roll_window = int(
                st.number_input(
                    "Window size", min_value=1, max_value=len(df), value=5, step=1, key="roll_win"
                )
            )

        if st.button("Apply", key="roll_apply"):
            if not roll_cols:
                st.warning("Select at least one column.")
            else:
                try:
                    result = rolling_transform(df, roll_cols, operation=roll_op, window=roll_window)
                    _save_result(result, f"roll_{roll_op}")
                except Exception as exc:
                    st.error(f"Error: {exc}")
