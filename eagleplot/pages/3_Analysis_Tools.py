"""Analysis Tools Page.

Statistical analysis, curve fitting, optimization, and linear algebra
operations on loaded datasets, using functions from eagleplot.math.

Tabs
----
- Statistics:    calculate_statistics()
- Distribution:  check_normality(), fit_distribution()
- Curve Fitting: fit_linear(), fit_polynomial(), fit_exponential(), fit_custom()
- Optimization:  minimize_function(), find_roots()
- Linear Algebra: compute_eigenvalues(), solve_linear_system(), matrix_decomposition()
- Signal Processing, Peak Fitting, Statistical Tests

Each tool is an ``st.fragment``: its widgets and its Run button rerun only that tool,
not the whole page (roadmap D1.2).
"""

import traceback
from pathlib import Path
import sys

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from eagleplot.peaks import (
    find_peaks as pk_find_peaks,
    integrate_peaks,
    compute_fwhm,
    fit_multipeak,
)
from eagleplot.signal import (
    smooth_moving_average,
    smooth_savitzky_golay,
    smooth_gaussian,
    filter_lowpass,
    filter_highpass,
    filter_bandpass,
    filter_bandstop,
    fft as sig_fft,
    derivative as sig_derivative,
    baseline_polynomial,
    baseline_rolling_ball,
    baseline_als,
    interpolate as sig_interpolate,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.math import (
    calculate_statistics,
    check_normality,
    fit_distribution,
    fit_linear,
    fit_polynomial,
    fit_exponential,
    fit_custom,
    minimize_function,
    find_roots,
    compute_eigenvalues,
    solve_linear_system,
    matrix_decomposition,
    ttest_one_sample,
    ttest_two_sample,
    ttest_paired,
    mannwhitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    anova_oneway,
    anova_twoway,
    tukey_hsd,
    bonferroni_correction,
    pearson_correlation,
    spearman_correlation,
    chi_square_independence,
)
from eagleplot.utils import (
    initialize_session_state,
    get_dataset,
    add_analysis_result,
    get_session_summary,
)
from eagleplot.utils.figures import download_payload, show_figure

initialize_session_state()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _numeric_cols(data) -> list:
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).columns.tolist()
    return []


def _to_1d(data, col_ref=None) -> np.ndarray:
    """Extract a 1-D float array from *data*.

    Parameters
    ----------
    data : DataFrame or ndarray
        Source dataset.
    col_ref : str, int, or None
        Column name (DataFrame), integer column index (2-D ndarray), or
        ``None`` to return the whole array flattened.
    """
    if isinstance(data, pd.DataFrame):
        return data[col_ref].values.astype(float)
    arr = np.asarray(data, dtype=float)
    if col_ref is None:
        return arr.flatten()
    return arr[:, int(col_ref)]


def _to_2d(data) -> np.ndarray:
    """Return *data* as a 2-D float array (uses all numeric columns for DataFrames)."""
    if isinstance(data, pd.DataFrame):
        cols = _numeric_cols(data)
        return data[cols].values.astype(float)
    return np.asarray(data, dtype=float)


def _pick_col(data, label: str, key: str):
    """Render a column-picker widget.

    Returns
    -------
    col_ref : str | int | None
        Column name (DataFrame), integer column index (2-D ndarray), or
        ``None`` for a 1-D array (whole array is used).
    valid : bool
        False only when a DataFrame has no numeric columns.
    """
    if isinstance(data, pd.DataFrame):
        cols = _numeric_cols(data)
        if not cols:
            st.warning("No numeric columns in this dataset.")
            return None, False
        return st.selectbox(label, cols, key=key), True
    arr = np.asarray(data)
    if arr.ndim == 1:
        st.caption(f"{label}: 1-D array — all values used.")
        return None, True
    idx = st.selectbox(
        label,
        range(arr.shape[1]),
        format_func=lambda i: f"Column {i}",
        key=key,
    )
    return idx, True


def _r2_color(r2: float) -> str:
    """Return a quality label for R²."""
    if r2 >= 0.99:
        return "(excellent)"
    if r2 >= 0.90:
        return "(good)"
    return "(poor)"


def _fit_plot(x, y, predict_fn, x_label: str, y_label: str, title: str):
    """Return a matplotlib Figure with data scatter + fitted curve."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=20, alpha=0.7, color="#333333", label="Data")
    x_fit = np.linspace(x.min(), x.max(), 400)
    ax.plot(x_fit, predict_fn(x_fit), "r-", linewidth=2, label="Fit")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _result_table(result: dict) -> None:
    """Render a test result dict as a two-column Streamlit table."""
    rows = []
    for k, v in result.items():
        if isinstance(v, list):
            display = ", ".join(str(x) for x in v)
        elif isinstance(v, float):
            display = f"{v:.6g}"
        else:
            display = str(v)
        rows.append({"Parameter": k, "Value": display})
    import pandas as _pd

    st.dataframe(_pd.DataFrame(rows), width="stretch", hide_index=True)

    # Effect size display (Cohen's d or similar)
    if "effect_size" in result:
        d = result["effect_size"]
        label = result.get("effect_size_label", "Effect size")
        abs_d = abs(d)
        if abs_d < 0.2:
            magnitude = "Negligible"
            color = "gray"
        elif abs_d < 0.5:
            magnitude = "Small"
            color = "blue"
        elif abs_d < 0.8:
            magnitude = "Medium"
            color = "orange"
        else:
            magnitude = "Large"
            color = "red"
        st.markdown(
            f"**{label}:** `{d:.4f}` — "
            f":{color}[**{magnitude}**] "
            f"(|d| < 0.2 negligible · 0.2–0.5 small · 0.5–0.8 medium · ≥ 0.8 large)"
        )


def _verdict(result: dict) -> None:
    """Show a coloured reject / fail-to-reject banner."""
    if result.get("reject_null"):
        st.success(f"**Reject H₀** (p = {result['p_value']:.4g} < α = {result['alpha']:.3g})")
    else:
        st.info(f"**Fail to reject H₀** (p = {result['p_value']:.4g} ≥ α = {result['alpha']:.3g})")


# ── Page layout ───────────────────────────────────────────────────────────────

st.title("Analysis Tools")
st.caption(
    "Statistical analysis, curve fitting, optimization, and linear algebra on your loaded datasets."
)

# Guard: need at least one dataset
summary = get_session_summary()
if summary["num_datasets"] == 0:
    st.warning("No datasets loaded. Go to **Data Upload** first.")
    st.stop()

# ── Dataset selector ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Dataset")

dataset_names = list(st.session_state.datasets.keys())
default_idx = (
    dataset_names.index(summary["current_dataset"])
    if summary["current_dataset"] in dataset_names
    else 0
)

col1, col2 = st.columns([3, 1])
with col1:
    dataset_name = st.selectbox(
        "Select dataset",
        dataset_names,
        index=default_idx,
        key="at_dataset",
    )
with col2:
    st.write("")
    st.write("")
    st.metric("Loaded", summary["num_datasets"])

if dataset_name:
    st.session_state.current_dataset = dataset_name

data = get_dataset(dataset_name)

if isinstance(data, pd.DataFrame):
    st.caption(f"DataFrame — {data.shape[0]} rows × {data.shape[1]} cols")
elif isinstance(data, np.ndarray):
    st.caption(f"NumPy array — shape {data.shape}, dtype {data.dtype}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
# Every tool below runs as its own st.fragment: changing one of its settings or
# pressing its button reruns only that tool, not the whole page (roadmap D1.2).
# The dataset selector and the column pickers that several tools share stay
# outside the fragments, so changing them still refreshes every tool that uses them.
st.markdown("---")
tab_stats, tab_dist, tab_fit, tab_opt, tab_linalg, tab_smooth, tab_peaks, tab_tests = st.tabs(
    [
        "Statistics",
        "Distribution",
        "Curve Fitting",
        "Optimization",
        "Linear Algebra",
        "Signal Processing",
        "Peak Fitting",
        "Statistical Tests",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Statistics
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _statistics_panel(data, dataset_name: str) -> None:
    st.markdown("### Descriptive Statistics")

    col_ref, valid = _pick_col(data, "Select column / variable", key="stats_col")

    if valid:
        if st.button("Calculate Statistics", type="primary", key="stats_run"):
            try:
                arr = _to_1d(data, col_ref)
                s = calculate_statistics(arr)

                st.markdown("#### Results")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{s['mean']:.6g}")
                c2.metric("Median", f"{s['median']:.6g}")
                c3.metric("Std Dev", f"{s['std']:.6g}")
                c4.metric("Variance", f"{s['var']:.6g}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Min", f"{s['min']:.6g}")
                c2.metric("Max", f"{s['max']:.6g}")
                c3.metric("Range", f"{s['range']:.6g}")
                c4.metric("IQR", f"{s['iqr']:.6g}")

                c1, c2 = st.columns(2)
                c1.metric("Q1 (25%)", f"{s['q1']:.6g}")
                c2.metric("Q3 (75%)", f"{s['q3']:.6g}")

                add_analysis_result(
                    {
                        "type": "statistics",
                        "dataset": dataset_name,
                        "column": str(col_ref),
                        "results": {k: float(v) for k, v in s.items()},
                    }
                )
                st.success("Results saved to analysis history.")

            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())


with tab_stats:
    _statistics_panel(data, dataset_name)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Distribution
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _distribution_panel(data, dataset_name: str) -> None:
    st.markdown("### Distribution Analysis")

    col_ref_d, valid_d = _pick_col(data, "Select column / variable", key="dist_col")

    if valid_d:
        # ── Normality test ────────────────────────────────────────────────────
        with st.expander("Shapiro-Wilk Normality Test", expanded=True):
            if st.button("Test Normality", key="norm_run"):
                try:
                    arr = _to_1d(data, col_ref_d)
                    result = check_normality(arr)

                    c1, c2 = st.columns(2)
                    c1.metric("W statistic", f"{result['statistic']:.6f}")
                    c2.metric("p-value", f"{result['p_value']:.6f}")

                    if result["is_normal"]:
                        st.success("Data appears **normally distributed** (p > 0.05).")
                    else:
                        st.warning("Data does **not** appear normally distributed (p ≤ 0.05).")

                    add_analysis_result(
                        {
                            "type": "normality_test",
                            "dataset": dataset_name,
                            "column": str(col_ref_d),
                            "results": result,
                        }
                    )
                    st.success("Results saved to analysis history.")

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

        # ── Distribution fitting ──────────────────────────────────────────────
        with st.expander("Fit Statistical Distribution"):
            _DIST_LABELS = {
                "norm": "Normal (Gaussian)",
                "expon": "Exponential",
                "gamma": "Gamma",
                "lognorm": "Log-Normal",
                "beta": "Beta",
            }
            dist_choice = st.selectbox(
                "Distribution",
                list(_DIST_LABELS.keys()),
                format_func=lambda d: _DIST_LABELS[d],
                key="dist_choice",
            )

            if st.button("Fit Distribution", key="dist_run"):
                try:
                    arr = _to_1d(data, col_ref_d)
                    result = fit_distribution(arr, distribution=dist_choice)

                    params_fmt = ", ".join(f"{p:.4g}" for p in result["params"])
                    st.markdown(
                        f"**Fitted {_DIST_LABELS[dist_choice]} parameters:** `({params_fmt})`"
                    )

                    c1, c2 = st.columns(2)
                    c1.metric("KS statistic", f"{result['ks_statistic']:.6f}")
                    c2.metric("KS p-value", f"{result['ks_pvalue']:.6f}")

                    if result["ks_pvalue"] > 0.05:
                        st.success(
                            f"Data is consistent with a "
                            f"{_DIST_LABELS[dist_choice]} distribution (KS p > 0.05)."
                        )
                    else:
                        st.warning(
                            f"Data is NOT well-described by "
                            f"{_DIST_LABELS[dist_choice]} distribution (KS p ≤ 0.05)."
                        )

                    add_analysis_result(
                        {
                            "type": "distribution_fit",
                            "dataset": dataset_name,
                            "column": str(col_ref_d),
                            "distribution": dist_choice,
                            "results": {
                                "params": [float(p) for p in result["params"]],
                                "ks_statistic": result["ks_statistic"],
                                "ks_pvalue": result["ks_pvalue"],
                            },
                        }
                    )
                    st.success("Results saved to analysis history.")

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())


with tab_dist:
    _distribution_panel(data, dataset_name)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Curve Fitting
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _curve_fit_panel(data, dataset_name: str) -> None:
    st.markdown("### Curve Fitting")

    # Check that dataset has at least 2 numeric columns / 2 array columns
    _is_multidim = (isinstance(data, pd.DataFrame) and len(_numeric_cols(data)) >= 2) or (
        isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2
    )

    if not _is_multidim:
        st.warning(
            "Curve fitting requires at least 2 variables (X and Y). "
            "Load a DataFrame with at least 2 numeric columns, "
            "or a 2-D NumPy array with at least 2 columns."
        )
    else:
        cfit1, cfit2 = st.columns(2)
        with cfit1:
            x_col_ref, valid_x = _pick_col(data, "X variable", key="fit_xcol")
        with cfit2:
            y_col_ref, valid_y = _pick_col(data, "Y variable", key="fit_ycol")

        if valid_x and valid_y:
            fit_type = st.radio(
                "Fit type",
                ["Linear", "Polynomial", "Exponential", "Custom"],
                horizontal=True,
                key="fit_type",
            )

            poly_degree = None
            custom_expr = custom_params_str = custom_p0_str = None

            if fit_type == "Polynomial":
                poly_degree = st.number_input(
                    "Degree",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    key="fit_degree",
                )
            elif fit_type == "Custom":
                st.markdown(
                    "Define your function using `x` (array) and named parameters. "
                    "`np` is available."
                )
                custom_expr = st.text_input(
                    "Expression (e.g. `a * np.exp(-b * x) + c`)",
                    value="a * x**2 + b * x + c",
                    key="fit_expr",
                )
                custom_params_str = st.text_input(
                    "Parameter names (comma-separated)",
                    value="a, b, c",
                    key="fit_params",
                )
                custom_p0_str = st.text_input(
                    "Initial guesses (comma-separated)",
                    value="1.0, 1.0, 0.0",
                    key="fit_p0",
                )

            show_fit_plot = st.checkbox("Show fitted curve plot", value=True, key="fit_showplot")

            if st.button("Run Fit", type="primary", key="fit_run"):
                try:
                    x = _to_1d(data, x_col_ref)
                    y = _to_1d(data, y_col_ref)
                    x_label = str(x_col_ref) if x_col_ref is not None else "x"
                    y_label = str(y_col_ref) if y_col_ref is not None else "y"
                    result = None
                    equation = ""

                    # ── Linear ───────────────────────────────────────────────
                    if fit_type == "Linear":
                        result = fit_linear(x, y)
                        slope, intercept = result["slope"], result["intercept"]
                        sign = "+" if intercept >= 0 else "-"
                        equation = f"y = {slope:.6g} x {sign} {abs(intercept):.6g}"

                        st.markdown("#### Results")
                        c1, c2 = st.columns(2)
                        c1.metric("Slope (m)", f"{slope:.6g}")
                        c2.metric("Intercept (b)", f"{intercept:.6g}")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            f"R² {_r2_color(result['r_squared'])}",
                            f"{result['r_squared']:.6f}",
                        )
                        c2.metric("p-value", f"{result['p_value']:.4g}")
                        c3.metric("Std Error", f"{result['std_err']:.6g}")
                        st.info(f"**Equation:** `{equation}`")

                        if show_fit_plot:
                            fig = _fit_plot(
                                x,
                                y,
                                lambda xn: result["slope"] * xn + result["intercept"],
                                x_label,
                                y_label,
                                f"Linear Fit: {equation}",
                            )
                            show_figure(fig)
                            plt.close(fig)

                        add_analysis_result(
                            {
                                "type": "fit_linear",
                                "dataset": dataset_name,
                                "x_col": str(x_col_ref),
                                "y_col": str(y_col_ref),
                                "equation": equation,
                                "results": {
                                    k: float(v) for k, v in result.items() if k != "predict"
                                },
                            }
                        )

                    # ── Polynomial ───────────────────────────────────────────
                    elif fit_type == "Polynomial":
                        degree = int(poly_degree)
                        result = fit_polynomial(x, y, degree)
                        coeffs = result["coefficients"]

                        terms = []
                        for i, c in enumerate(coeffs):
                            power = degree - i
                            if abs(c) < 1e-15:
                                continue
                            if power == 0:
                                terms.append(f"{c:.4g}")
                            elif power == 1:
                                terms.append(f"{c:.4g}x")
                            else:
                                terms.append(f"{c:.4g}x^{power}")
                        equation = "y = " + (" + ".join(terms) if terms else "0")
                        equation = equation.replace("+ -", "- ")

                        st.markdown("#### Results")
                        st.metric(
                            f"R² {_r2_color(result['r_squared'])}",
                            f"{result['r_squared']:.6f}",
                        )
                        st.info(f"**Equation:** `{equation}`")
                        st.markdown("**Coefficients** (highest degree first):")
                        coeff_df = pd.DataFrame(
                            {
                                "Power": [degree - i for i in range(degree + 1)],
                                "Coefficient": list(coeffs),
                            }
                        )
                        st.dataframe(coeff_df, width="stretch")

                        if show_fit_plot:
                            fig = _fit_plot(
                                x,
                                y,
                                result["predict"],
                                x_label,
                                y_label,
                                f"Polynomial Fit (degree {degree})",
                            )
                            show_figure(fig)
                            plt.close(fig)

                        add_analysis_result(
                            {
                                "type": "fit_polynomial",
                                "dataset": dataset_name,
                                "x_col": str(x_col_ref),
                                "y_col": str(y_col_ref),
                                "degree": degree,
                                "equation": equation,
                                "results": {
                                    "coefficients": [float(c) for c in coeffs],
                                    "r_squared": result["r_squared"],
                                    "residuals": result["residuals"],
                                },
                            }
                        )

                    # ── Exponential ──────────────────────────────────────────
                    elif fit_type == "Exponential":
                        result = fit_exponential(x, y)
                        a, b, c_off = result["a"], result["b"], result["c"]
                        equation = f"y = {a:.4g} * exp({b:.4g} * x)" + (
                            f" + {c_off:.4g}" if c_off >= 0 else f" - {abs(c_off):.4g}"
                        )

                        st.markdown("#### Results")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("a (amplitude)", f"{a:.6g}")
                        c2.metric("b (rate)", f"{b:.6g}")
                        c3.metric("c (offset)", f"{c_off:.6g}")
                        st.metric(
                            f"R² {_r2_color(result['r_squared'])}",
                            f"{result['r_squared']:.6f}",
                        )
                        st.info(f"**Equation:** `{equation}`")

                        if show_fit_plot:
                            fig = _fit_plot(
                                x,
                                y,
                                result["predict"],
                                x_label,
                                y_label,
                                "Exponential Fit",
                            )
                            show_figure(fig)
                            plt.close(fig)

                        add_analysis_result(
                            {
                                "type": "fit_exponential",
                                "dataset": dataset_name,
                                "x_col": str(x_col_ref),
                                "y_col": str(y_col_ref),
                                "equation": equation,
                                "results": {
                                    k: float(v) for k, v in result.items() if k != "predict"
                                },
                            }
                        )

                    # ── Custom ───────────────────────────────────────────────
                    elif fit_type == "Custom":
                        param_names = [p.strip() for p in custom_params_str.split(",")]
                        p0_values = [float(v.strip()) for v in custom_p0_str.split(",")]
                        func_str = f"lambda x, {', '.join(param_names)}: {custom_expr}"
                        _ns = {"np": np, "numpy": np}
                        custom_func = eval(func_str, _ns)  # noqa: S307
                        result = fit_custom(x, y, custom_func, p0=p0_values)

                        params_fmt = ", ".join(
                            f"{n}={v:.4g}" for n, v in zip(param_names, result["parameters"])
                        )
                        equation = f"f(x) = {custom_expr}  [{params_fmt}]"

                        st.markdown("#### Results")
                        st.metric(
                            f"R² {_r2_color(result['r_squared'])}",
                            f"{result['r_squared']:.6f}",
                        )
                        st.info(f"**Fitted parameters:** `{params_fmt}`")

                        param_df = pd.DataFrame(
                            {
                                "Parameter": param_names,
                                "Value": list(result["parameters"]),
                                "Std Error": list(result["std_errors"]),
                            }
                        )
                        st.dataframe(param_df, width="stretch")

                        if show_fit_plot:
                            fig = _fit_plot(
                                x,
                                y,
                                result["predict"],
                                x_label,
                                y_label,
                                "Custom Fit",
                            )
                            show_figure(fig)
                            plt.close(fig)

                        add_analysis_result(
                            {
                                "type": "fit_custom",
                                "dataset": dataset_name,
                                "x_col": str(x_col_ref),
                                "y_col": str(y_col_ref),
                                "expression": custom_expr,
                                "results": {
                                    "parameters": [float(p) for p in result["parameters"]],
                                    "std_errors": [float(e) for e in result["std_errors"]],
                                    "r_squared": result["r_squared"],
                                },
                            }
                        )

                    if result is not None:
                        st.success("Results saved to analysis history.")

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())


with tab_fit:
    _curve_fit_panel(data, dataset_name)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Optimization
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _minimize_panel(dataset_name: str) -> None:
    st.markdown("Find **x** that minimizes scalar `f(x)` where `x` is a NumPy array.")
    min_expr = st.text_area(
        "Objective function `f(x)`",
        value="(x[0] - 2)**2 + (x[1] + 1)**2",
        height=80,
        key="min_expr",
    )
    min_x0 = st.text_input(
        "Initial guess x₀ (comma-separated)",
        value="0.0, 0.0",
        key="min_x0",
    )
    min_method = st.selectbox(
        "Method",
        ["Nelder-Mead", "BFGS", "L-BFGS-B", "Powell"],
        key="min_method",
    )

    if st.button("Minimize", key="min_run"):
        try:
            _ns = {"np": np, "numpy": np}
            obj_func = eval(f"lambda x: {min_expr}", _ns)  # noqa: S307
            x0 = np.array([float(v.strip()) for v in min_x0.split(",")])
            result = minimize_function(obj_func, x0, method=min_method)

            st.markdown("#### Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("f(x*)", f"{result['fun']:.6g}")
            c2.metric("Success", "✅" if result["success"] else "❌")
            c3.metric("Evaluations", str(result["nfev"]))
            st.write("**Optimal x:**", result["x"])
            st.caption(f"Message: {result['message']}")

            add_analysis_result(
                {
                    "type": "minimize_function",
                    "dataset": dataset_name,
                    "expression": min_expr,
                    "method": min_method,
                    "results": {
                        "x": [float(v) for v in result["x"]],
                        "fun": result["fun"],
                        "success": result["success"],
                        "nfev": result["nfev"],
                    },
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _root_panel(dataset_name: str) -> None:
    st.markdown("Find scalar `x` such that `f(x) = 0`.")
    root_expr = st.text_area(
        "Function `f(x)`",
        value="x**3 - x - 2",
        height=80,
        key="root_expr",
    )
    rc1, rc2 = st.columns(2)
    with rc1:
        root_a = st.number_input("Bracket lower bound (a)", value=-10.0, key="root_a")
    with rc2:
        root_b = st.number_input("Bracket upper bound (b)", value=10.0, key="root_b")

    if st.button("Find Root", key="root_run"):
        try:
            _ns = {"np": np, "numpy": np}
            root_func = eval(f"lambda x: {root_expr}", _ns)  # noqa: S307
            result = find_roots(root_func, bracket=(root_a, root_b))

            st.markdown("#### Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Root", f"{result['root']:.8g}")
            c2.metric("Converged", "✅" if result["converged"] else "❌")
            c3.metric("Iterations", str(result["iterations"]))
            st.caption(f"f(root) evaluated using {int(result['function_value'])} calls.")

            add_analysis_result(
                {
                    "type": "find_roots",
                    "dataset": dataset_name,
                    "expression": root_expr,
                    "results": {
                        "root": result["root"],
                        "converged": result["converged"],
                        "iterations": result["iterations"],
                    },
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


with tab_opt:
    st.markdown("### Optimization")
    st.info(
        "Enter Python expressions. Use `x` (array for minimization, scalar for "
        "root finding) and `np` (numpy). No imports needed."
    )

    # ── Function minimization ─────────────────────────────────────────────────
    with st.expander("Function Minimization", expanded=True):
        _minimize_panel(dataset_name)

    # ── Root finding ──────────────────────────────────────────────────────────
    with st.expander("Root Finding (Brent's Method)"):
        _root_panel(dataset_name)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Linear Algebra
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _eigen_panel(_mat: np.ndarray, dataset_name: str) -> None:
    _nrows, _ncols_m = _mat.shape
    show_evecs = st.checkbox("Include eigenvectors", value=False, key="eig_vecs")

    if st.button("Compute Eigenvalues", key="eig_run"):
        try:
            if _nrows != _ncols_m:
                st.error(f"Matrix must be square — got {_nrows} × {_ncols_m}.")
            else:
                result = compute_eigenvalues(_mat, eigenvectors=show_evecs)
                evals = result["eigenvalues"]

                st.markdown("#### Eigenvalues")
                edf = pd.DataFrame(
                    {
                        "Index": list(range(len(evals))),
                        "Real Part": evals.real,
                        "Imaginary Part": evals.imag,
                        "Magnitude": np.abs(evals),
                    }
                )
                st.dataframe(edf, width="stretch")

                if show_evecs and "eigenvectors" in result:
                    st.markdown("#### Eigenvectors (columns)")
                    st.dataframe(
                        pd.DataFrame(
                            result["eigenvectors"].real,
                            columns=[f"v{i}" for i in range(_ncols_m)],
                        ),
                        width="stretch",
                    )

                add_analysis_result(
                    {
                        "type": "eigenvalues",
                        "dataset": dataset_name,
                        "results": {
                            "eigenvalues_real": [float(v) for v in evals.real],
                            "eigenvalues_imag": [float(v) for v in evals.imag],
                        },
                    }
                )
                st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _solve_panel(_mat: np.ndarray, dataset_name: str) -> None:
    _nrows, _ncols_m = _mat.shape
    st.markdown("The dataset matrix is **A**. Enter **b** as a comma-separated vector.")
    b_str = st.text_input(
        "Right-hand side vector b",
        placeholder=f"{', '.join(['1.0'] * min(_nrows, 5))}{' ...' if _nrows > 5 else ''}",
        key="linalg_b",
    )

    if st.button("Solve", key="linalg_solve"):
        try:
            if _nrows != _ncols_m:
                st.error(f"A must be square — got {_nrows} × {_ncols_m}.")
            elif not b_str.strip():
                st.warning("Enter the b vector first.")
            else:
                b_vec = np.array([float(v.strip()) for v in b_str.split(",")])
                if len(b_vec) != _nrows:
                    st.error(f"b must have {_nrows} element(s), got {len(b_vec)}.")
                else:
                    result = solve_linear_system(_mat, b_vec)

                    st.markdown("#### Solution vector x")
                    sol_df = pd.DataFrame({"Component": range(len(result["x"])), "x": result["x"]})
                    st.dataframe(sol_df, width="stretch")

                    c1, c2 = st.columns(2)
                    c1.metric("Residual ‖Ax−b‖", f"{result['residual']:.4e}")
                    c2.metric("Condition number", f"{result['condition_number']:.4g}")

                    if result["condition_number"] > 1e10:
                        st.warning(
                            "High condition number — matrix is nearly singular; "
                            "solution may be inaccurate."
                        )

                    add_analysis_result(
                        {
                            "type": "solve_linear_system",
                            "dataset": dataset_name,
                            "results": {
                                "x": [float(v) for v in result["x"]],
                                "residual": result["residual"],
                                "condition_number": result["condition_number"],
                            },
                        }
                    )
                    st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _decomposition_panel(_mat: np.ndarray, dataset_name: str) -> None:
    _DECOMP_LABELS = {
        "svd": "SVD — Singular Value Decomposition",
        "qr": "QR Decomposition",
        "lu": "LU Decomposition",
        "cholesky": "Cholesky (positive-definite matrices only)",
    }
    decomp_method = st.selectbox(
        "Method",
        list(_DECOMP_LABELS.keys()),
        format_func=lambda m: _DECOMP_LABELS[m],
        key="decomp_method",
    )

    if st.button("Decompose", key="decomp_run"):
        try:
            result = matrix_decomposition(_mat, method=decomp_method)

            st.markdown(f"#### {decomp_method.upper()} Result")
            for key_name, arr_val in result.items():
                if isinstance(arr_val, np.ndarray) and arr_val.ndim == 2:
                    st.markdown(f"**{key_name}** — shape {arr_val.shape[0]} × {arr_val.shape[1]}:")
                    st.dataframe(
                        pd.DataFrame(arr_val.real),
                        width="stretch",
                    )
                elif isinstance(arr_val, np.ndarray) and arr_val.ndim == 1:
                    st.markdown(f"**{key_name}** (singular values / diagonal):")
                    st.write(arr_val.real)

            add_analysis_result(
                {
                    "type": "matrix_decomposition",
                    "dataset": dataset_name,
                    "method": decomp_method,
                    "results": {
                        k: [float(v) for v in vv.real.flatten()]
                        for k, vv in result.items()
                        if isinstance(vv, np.ndarray)
                    },
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


with tab_linalg:
    st.markdown("### Linear Algebra")
    st.info(
        "Operations use the **full dataset** as a 2-D matrix. "
        "For a DataFrame, all numeric columns are used."
    )

    try:
        _mat = _to_2d(data)
        _nrows, _ncols_m = _mat.shape
        st.caption(f"Matrix shape: {_nrows} × {_ncols_m}")
        _mat_ok = True
    except Exception as _mat_exc:
        st.error(f"Cannot convert dataset to a 2-D numeric matrix: {_mat_exc}")
        _mat_ok = False

    if _mat_ok:
        # ── Eigenvalues ───────────────────────────────────────────────────────
        with st.expander("Eigenvalues & Eigenvectors", expanded=True):
            _eigen_panel(_mat, dataset_name)

        # ── Solve Ax = b ──────────────────────────────────────────────────────
        with st.expander("Solve Linear System Ax = b"):
            _solve_panel(_mat, dataset_name)

        # ── Matrix decomposition ──────────────────────────────────────────────
        with st.expander("Matrix Decomposition"):
            _decomposition_panel(_mat, dataset_name)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Signal Processing
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _smooth_panel(data, dataset_name: str, col_ref_sp, x_col_sp) -> None:
    st.markdown("#### Smoothing")
    sp_smooth_method = st.radio(
        "Method",
        ["Moving Average", "Savitzky-Golay", "Gaussian"],
        horizontal=True,
        key="sp_smooth_method",
    )

    sp_ma_window = 11
    sp_sg_window = 11
    sp_sg_poly = 3
    sp_gauss_sigma = 2.0

    if sp_smooth_method == "Moving Average":
        sp_ma_window = st.slider("Window size (points)", 3, 101, 11, 2, key="sp_ma_window")
    elif sp_smooth_method == "Savitzky-Golay":
        c1, c2 = st.columns(2)
        with c1:
            sp_sg_window = st.slider("Window length (odd)", 5, 101, 11, 2, key="sp_sg_window")
            if sp_sg_window % 2 == 0:
                sp_sg_window += 1
        with c2:
            sp_sg_poly = int(
                st.number_input(
                    "Polynomial order",
                    min_value=1,
                    max_value=min(sp_sg_window - 1, 6),
                    value=min(3, sp_sg_window - 1),
                    key="sp_sg_poly",
                )
            )
    else:
        sp_gauss_sigma = st.number_input(
            "Sigma (std deviation, index units)",
            min_value=0.1,
            value=2.0,
            step=0.5,
            key="sp_gauss_sigma",
        )

    show_raw_sp = st.checkbox("Show original signal", value=True, key="sp_smooth_show_raw")

    if st.button("Apply Smoothing", type="primary", key="sp_smooth_run"):
        try:
            y_sp = _to_1d(data, col_ref_sp)
            x_sp = _to_1d(data, x_col_sp) if x_col_sp is not None else np.arange(len(y_sp))

            if sp_smooth_method == "Moving Average":
                y_out = smooth_moving_average(y_sp, int(sp_ma_window))
                label = f"Moving Average (window={sp_ma_window})"
            elif sp_smooth_method == "Savitzky-Golay":
                y_out = smooth_savitzky_golay(y_sp, int(sp_sg_window), int(sp_sg_poly))
                label = f"Savitzky-Golay (window={sp_sg_window}, order={sp_sg_poly})"
            else:
                y_out = smooth_gaussian(y_sp, float(sp_gauss_sigma))
                label = f"Gaussian (σ={sp_gauss_sigma})"

            fig, ax = plt.subplots(figsize=(8, 4))
            if show_raw_sp:
                ax.plot(x_sp, y_sp, alpha=0.4, lw=1, color="#888888", label="Original")
            ax.plot(x_sp, y_out, lw=2, color="#d62728", label=label)
            col_name = str(col_ref_sp) if col_ref_sp is not None else "Value"
            ax.set_xlabel(str(x_col_sp) if x_col_sp is not None else "Index")
            ax.set_ylabel(col_name)
            ax.set_title("Smoothed Signal")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            show_figure(fig)
            plt.close(fig)

            with st.expander("Smoothed data table"):
                st.dataframe(
                    pd.DataFrame({"X": x_sp, "Original": y_sp, "Smoothed": y_out}),
                    width="stretch",
                )

            add_analysis_result(
                {
                    "type": "smoothing",
                    "dataset": dataset_name,
                    "column": str(col_ref_sp),
                    "method": label,
                    "results": {"smoothed": [float(v) for v in y_out]},
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _filter_panel(data, dataset_name: str, col_ref_sp, x_col_sp) -> None:
    st.markdown("#### Butterworth Filter")
    st.caption("Zero-phase IIR filter (sosfiltfilt) — no phase distortion.")

    sp_flt_type = st.selectbox(
        "Filter type",
        ["Low-pass", "High-pass", "Band-pass", "Band-stop"],
        key="sp_flt_type",
    )
    sp_flt_order = st.slider("Filter order", 1, 10, 4, key="sp_flt_order")

    c1, c2 = st.columns(2)
    with c1:
        sp_flt_fs = st.number_input(
            "Sampling frequency (fs)",
            min_value=0.001,
            value=1.0,
            step=0.1,
            format="%.4g",
            key="sp_flt_fs",
        )

    # Defaults (overwritten by widgets below)
    sp_flt_cutoff = 0.3
    sp_flt_low = 0.1
    sp_flt_high = 0.4

    if sp_flt_type in ("Band-pass", "Band-stop"):
        col_a, col_b = st.columns(2)
        with col_a:
            sp_flt_low = st.number_input(
                "Low cut-off (Hz)",
                min_value=1e-6,
                value=0.1,
                step=0.01,
                format="%.4g",
                key="sp_flt_low",
            )
        with col_b:
            sp_flt_high = st.number_input(
                "High cut-off (Hz)",
                min_value=1e-6,
                value=0.4,
                step=0.01,
                format="%.4g",
                key="sp_flt_high",
            )
    else:
        with c2:
            sp_flt_cutoff = st.number_input(
                "Cut-off frequency (Hz)",
                min_value=1e-6,
                value=0.3,
                step=0.01,
                format="%.4g",
                key="sp_flt_cutoff",
            )

    show_raw_flt = st.checkbox("Show original signal", value=True, key="sp_flt_show_raw")

    if st.button("Apply Filter", type="primary", key="sp_flt_run"):
        try:
            y_sp = _to_1d(data, col_ref_sp)
            x_sp = _to_1d(data, x_col_sp) if x_col_sp is not None else np.arange(len(y_sp))

            if sp_flt_type == "Low-pass":
                y_out = filter_lowpass(
                    y_sp, float(sp_flt_cutoff), float(sp_flt_fs), int(sp_flt_order)
                )
                label = f"Low-pass (fc={sp_flt_cutoff} Hz, order={sp_flt_order})"
            elif sp_flt_type == "High-pass":
                y_out = filter_highpass(
                    y_sp, float(sp_flt_cutoff), float(sp_flt_fs), int(sp_flt_order)
                )
                label = f"High-pass (fc={sp_flt_cutoff} Hz, order={sp_flt_order})"
            elif sp_flt_type == "Band-pass":
                y_out = filter_bandpass(
                    y_sp,
                    float(sp_flt_low),
                    float(sp_flt_high),
                    float(sp_flt_fs),
                    int(sp_flt_order),
                )
                label = f"Band-pass ({sp_flt_low}–{sp_flt_high} Hz, order={sp_flt_order})"
            else:
                y_out = filter_bandstop(
                    y_sp,
                    float(sp_flt_low),
                    float(sp_flt_high),
                    float(sp_flt_fs),
                    int(sp_flt_order),
                )
                label = f"Band-stop ({sp_flt_low}–{sp_flt_high} Hz, order={sp_flt_order})"

            fig, ax = plt.subplots(figsize=(8, 4))
            if show_raw_flt:
                ax.plot(x_sp, y_sp, alpha=0.4, lw=1, color="#888888", label="Original")
            ax.plot(x_sp, y_out, lw=2, color="#1f77b4", label=label)
            col_name = str(col_ref_sp) if col_ref_sp is not None else "Value"
            ax.set_xlabel(str(x_col_sp) if x_col_sp is not None else "Index")
            ax.set_ylabel(col_name)
            ax.set_title("Filtered Signal")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            show_figure(fig)
            plt.close(fig)

            add_analysis_result(
                {
                    "type": "filter",
                    "dataset": dataset_name,
                    "column": str(col_ref_sp),
                    "method": label,
                    "results": {"filtered": [float(v) for v in y_out]},
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _fft_panel(data, dataset_name: str, col_ref_sp, x_col_sp) -> None:
    st.markdown("#### FFT / Power Spectrum")

    c1, c2 = st.columns(2)
    with c1:
        sp_fft_dt = st.number_input(
            "Sample spacing (dt = 1/fs)",
            min_value=1e-12,
            value=1.0,
            step=0.1,
            format="%.6g",
            key="sp_fft_dt",
            help="Set to 1.0 for index-domain data.",
        )
    with c2:
        sp_fft_plot = st.selectbox(
            "Display",
            ["Amplitude spectrum", "Power spectrum"],
            key="sp_fft_plot",
        )

    if st.button("Compute FFT", type="primary", key="sp_fft_run"):
        try:
            y_sp = _to_1d(data, col_ref_sp)
            result_fft = sig_fft(y_sp, dt=float(sp_fft_dt))

            freqs_fft = result_fft["frequencies"]
            if "Power" in sp_fft_plot:
                yplot = result_fft["power"]
                ylabel_fft = "Power"
            else:
                yplot = result_fft["amplitudes"]
                ylabel_fft = "Amplitude"

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(freqs_fft, yplot, lw=1.5, color="#2ca02c")
            ax.set_xlabel("Frequency")
            ax.set_ylabel(ylabel_fft)
            ax.set_title(f"{ylabel_fft} Spectrum")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            show_figure(fig)
            plt.close(fig)

            # Dominant frequency (skip DC at index 0)
            dom_idx = int(np.argmax(result_fft["amplitudes"][1:]) + 1)
            dom_freq = float(freqs_fft[dom_idx])
            c1, c2, c3 = st.columns(3)
            c1.metric("N points", result_fft["n"])
            c2.metric("Dominant frequency", f"{dom_freq:.4g}")
            c3.metric(
                "Period",
                f"{1.0 / dom_freq:.4g}" if dom_freq != 0 else "—",
            )

            with st.expander("Top-10 frequency components"):
                top_idx = np.argsort(result_fft["amplitudes"])[::-1][:10]
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Frequency": freqs_fft[top_idx],
                            "Amplitude": result_fft["amplitudes"][top_idx],
                            "Power": result_fft["power"][top_idx],
                            "Phase (rad)": result_fft["phases"][top_idx],
                        }
                    ),
                    width="stretch",
                )

            add_analysis_result(
                {
                    "type": "fft",
                    "dataset": dataset_name,
                    "column": str(col_ref_sp),
                    "dt": float(sp_fft_dt),
                    "results": {
                        "dominant_frequency": dom_freq,
                        "n": result_fft["n"],
                    },
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _derivative_panel(data, dataset_name: str, col_ref_sp, x_col_sp) -> None:
    st.markdown("#### Numerical Derivative")
    st.caption("Central-difference approximation via numpy.gradient.")

    sp_deriv_order = st.radio("Derivative order", [1, 2], horizontal=True, key="sp_deriv_order")
    show_raw_drv = st.checkbox("Show original signal", value=True, key="sp_deriv_show_raw")

    if st.button("Compute Derivative", type="primary", key="sp_deriv_run"):
        try:
            y_sp = _to_1d(data, col_ref_sp)
            x_arr = _to_1d(data, x_col_sp) if x_col_sp is not None else None
            x_axis = x_arr if x_arr is not None else np.arange(len(y_sp))

            dy = sig_derivative(y_sp, x=x_arr, order=int(sp_deriv_order))
            ord_label = "1st" if sp_deriv_order == 1 else "2nd"
            label = f"{ord_label} derivative"

            if show_raw_drv:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                col_name = str(col_ref_sp) if col_ref_sp is not None else "Signal"
                ax1.plot(x_axis, y_sp, lw=1.5, color="#888888")
                ax1.set_ylabel(col_name)
                ax1.set_title("Original")
                ax1.grid(True, alpha=0.3)
                ax2.plot(x_axis, dy, lw=1.5, color="#9467bd")
                ax2.set_xlabel(str(x_col_sp) if x_col_sp is not None else "Index")
                drv_sym = "dy/dx" if sp_deriv_order == 1 else "d²y/dx²"
                ax2.set_ylabel(drv_sym)
                ax2.set_title(label.capitalize())
                ax2.grid(True, alpha=0.3)
            else:
                fig, ax2 = plt.subplots(figsize=(8, 4))
                ax2.plot(x_axis, dy, lw=1.5, color="#9467bd", label=label)
                ax2.set_xlabel(str(x_col_sp) if x_col_sp is not None else "Index")
                drv_sym = "dy/dx" if sp_deriv_order == 1 else "d²y/dx²"
                ax2.set_ylabel(drv_sym)
                ax2.set_title(label.capitalize())
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            show_figure(fig)
            plt.close(fig)

            add_analysis_result(
                {
                    "type": "derivative",
                    "dataset": dataset_name,
                    "column": str(col_ref_sp),
                    "order": int(sp_deriv_order),
                    "results": {"derivative": [float(v) for v in dy]},
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _baseline_panel(data, dataset_name: str, col_ref_sp, x_col_sp) -> None:
    st.markdown("#### Baseline Correction")
    st.markdown("Estimate and subtract a background baseline from the signal.")

    sp_bl_method = st.radio(
        "Method",
        ["Polynomial", "Rolling Ball", "ALS (Asymmetric Least Squares)"],
        key="sp_bl_method",
    )

    sp_bl_degree = 3
    sp_bl_radius = 50
    sp_bl_lam = 1e5
    sp_bl_p = 0.01

    if sp_bl_method == "Polynomial":
        sp_bl_degree = st.slider("Polynomial degree", 1, 10, 3, key="sp_bl_poly_deg")
        if x_col_sp is None:
            st.info("Tip: select an X column above for physical x-axis fitting.")
    elif sp_bl_method == "Rolling Ball":
        sp_bl_radius = st.slider("Ball radius (samples)", 5, 500, 50, key="sp_bl_rb_radius")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sp_bl_lam = st.number_input(
                "Smoothness λ",
                min_value=1.0,
                value=1e5,
                step=1e4,
                format="%.2e",
                key="sp_bl_lam",
            )
        with c2:
            sp_bl_p = st.number_input(
                "Asymmetry p",
                min_value=0.001,
                max_value=0.5,
                value=0.01,
                step=0.005,
                format="%.4f",
                key="sp_bl_p",
            )

    show_raw_bl = st.checkbox("Show original signal", value=True, key="sp_bl_show_raw")

    if st.button("Correct Baseline", type="primary", key="sp_bl_run"):
        try:
            y_sp = _to_1d(data, col_ref_sp)
            x_sp = _to_1d(data, x_col_sp) if x_col_sp is not None else np.arange(len(y_sp))

            if sp_bl_method == "Polynomial":
                bl_vals, y_corr = baseline_polynomial(y_sp, x_sp, int(sp_bl_degree))
                label = f"Polynomial baseline (degree={sp_bl_degree})"
            elif sp_bl_method == "Rolling Ball":
                bl_vals, y_corr = baseline_rolling_ball(y_sp, int(sp_bl_radius))
                label = f"Rolling ball (radius={sp_bl_radius})"
            else:
                bl_vals, y_corr = baseline_als(y_sp, lam=float(sp_bl_lam), p=float(sp_bl_p))
                label = f"ALS (λ={sp_bl_lam:.0e}, p={sp_bl_p})"

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            if show_raw_bl:
                ax1.plot(x_sp, y_sp, lw=1, color="#888888", alpha=0.7, label="Original")
            ax1.plot(
                x_sp,
                bl_vals,
                lw=2,
                color="#ff7f0e",
                ls="--",
                label="Baseline",
            )
            ax1.plot(x_sp, y_corr, lw=1.5, color="#1f77b4", label="Corrected")
            col_name = str(col_ref_sp) if col_ref_sp is not None else "Signal"
            ax1.set_ylabel(col_name)
            ax1.set_title(label)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(x_sp, y_corr, lw=1.5, color="#1f77b4")
            ax2.axhline(0, color="red", lw=0.8, ls="--")
            ax2.set_xlabel(str(x_col_sp) if x_col_sp is not None else "Index")
            ax2.set_ylabel("Corrected signal")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            show_figure(fig)
            plt.close(fig)

            add_analysis_result(
                {
                    "type": "baseline_correction",
                    "dataset": dataset_name,
                    "column": str(col_ref_sp),
                    "method": label,
                    "results": {"corrected": [float(v) for v in y_corr]},
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _interpolate_panel(data, dataset_name: str, col_ref_sp, x_col_sp) -> None:
    st.markdown("#### Interpolation")
    st.markdown("Resample the signal to a new, uniform x-grid.")

    if x_col_sp is None:
        st.info(
            "No X column selected above. Select one in the "
            "'X-axis column' expander to interpolate on a physical grid. "
            "Index coordinates are used when no X column is set."
        )

    sp_interp_method = st.selectbox(
        "Method",
        ["cubic", "linear", "quadratic", "nearest"],
        key="sp_interp_method",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        sp_interp_n = int(
            st.number_input(
                "Output points",
                min_value=10,
                value=500,
                step=10,
                key="sp_interp_n",
            )
        )
    sp_use_data_bounds = st.checkbox(
        "Use data x-range as bounds", value=True, key="sp_interp_auto_bounds"
    )
    if not sp_use_data_bounds:
        with c2:
            sp_interp_start = st.number_input("X start", value=0.0, key="sp_interp_start")
        with c3:
            sp_interp_end = st.number_input("X end", value=1.0, key="sp_interp_end")

    if st.button("Interpolate", type="primary", key="sp_interp_run"):
        try:
            y_sp = _to_1d(data, col_ref_sp)
            x_sp = (
                _to_1d(data, x_col_sp)
                if x_col_sp is not None
                else np.arange(len(y_sp), dtype=float)
            )

            if sp_use_data_bounds:
                x_start = float(x_sp.min())
                x_stop = float(x_sp.max())
            else:
                x_start = float(sp_interp_start)
                x_stop = float(sp_interp_end)

            x_new = np.linspace(x_start, x_stop, sp_interp_n)
            y_new = sig_interpolate(x_sp, y_sp, x_new, method=sp_interp_method)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(
                x_sp,
                y_sp,
                s=12,
                alpha=0.5,
                color="#333333",
                label=f"Original ({len(x_sp)} pts)",
                zorder=3,
            )
            ax.plot(
                x_new,
                y_new,
                lw=1.5,
                color="#2ca02c",
                label=f"{sp_interp_method} ({sp_interp_n} pts)",
            )
            col_name = str(col_ref_sp) if col_ref_sp is not None else "Signal"
            ax.set_xlabel(str(x_col_sp) if x_col_sp is not None else "Index")
            ax.set_ylabel(col_name)
            ax.set_title("Interpolated Signal")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            show_figure(fig)
            plt.close(fig)

            with st.expander("Interpolated data (first 100 rows)"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "X_new": x_new[:100],
                            "Y_interp": y_new[:100],
                        }
                    ),
                    width="stretch",
                )

            add_analysis_result(
                {
                    "type": "interpolation",
                    "dataset": dataset_name,
                    "column": str(col_ref_sp),
                    "method": sp_interp_method,
                    "results": {
                        "n_input": int(len(x_sp)),
                        "n_output": sp_interp_n,
                        "x_range": [x_start, x_stop],
                    },
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


with tab_smooth:
    st.markdown("### Signal Processing")
    st.markdown(
        "Apply smoothing, filtering, FFT, derivatives, baseline correction, "
        "and interpolation to a 1-D signal column."
    )

    col_ref_sp, valid_sp = _pick_col(data, "Signal column (Y)", key="sp_ycol")

    # Optional X column (needed by Baseline — Polynomial and Interpolate)
    x_col_sp = None
    if isinstance(data, pd.DataFrame) and len(_numeric_cols(data)) >= 2:
        with st.expander("X-axis column (optional — defaults to index)", expanded=False):
            x_col_sp, _ = _pick_col(data, "X column", key="sp_xcol")

    if valid_sp:
        (
            sub_smooth,
            sub_filter,
            sub_fft,
            sub_deriv,
            sub_baseline,
            sub_interp,
        ) = st.tabs(["Smooth", "Filter", "FFT", "Derivative", "Baseline", "Interpolate"])

        # ── Smooth ───────────────────────────────────────────────────────────
        with sub_smooth:
            _smooth_panel(data, dataset_name, col_ref_sp, x_col_sp)

        # ── Filter ───────────────────────────────────────────────────────────
        with sub_filter:
            _filter_panel(data, dataset_name, col_ref_sp, x_col_sp)

        # ── FFT ──────────────────────────────────────────────────────────────
        with sub_fft:
            _fft_panel(data, dataset_name, col_ref_sp, x_col_sp)

        # ── Derivative ───────────────────────────────────────────────────────
        with sub_deriv:
            _derivative_panel(data, dataset_name, col_ref_sp, x_col_sp)

        # ── Baseline ─────────────────────────────────────────────────────────
        with sub_baseline:
            _baseline_panel(data, dataset_name, col_ref_sp, x_col_sp)

        # ── Interpolate ──────────────────────────────────────────────────────
        with sub_interp:
            _interpolate_panel(data, dataset_name, col_ref_sp, x_col_sp)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 7 — Peak Analysis
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _peak_find_panel(data, dataset_name: str, x_col_pk, y_col_pk) -> None:
    st.markdown("#### Auto Peak Detection")
    st.caption(
        "Adjust the thresholds below, then click **Find Peaks**. "
        "Detected peaks are highlighted on an interactive plot."
    )

    c1, c2 = st.columns(2)
    with c1:
        pk_min_height = st.number_input(
            "Min height (blank = any)",
            value=0.0,
            key="pk_height",
            help="Minimum absolute peak height.",
        )
        pk_prominence = st.number_input(
            "Min prominence",
            value=0.0,
            key="pk_prominence",
            help="How much a peak stands out from its neighbours.",
        )
    with c2:
        pk_distance = int(
            st.number_input(
                "Min distance (samples)",
                min_value=1,
                value=5,
                step=1,
                key="pk_distance",
            )
        )
        pk_width = st.number_input(
            "Min width (samples)",
            value=0.0,
            key="pk_width",
            help="Minimum peak width at half prominence.",
        )

    if st.button("Find Peaks", type="primary", key="pk_find_run"):
        try:
            x_pk = _to_1d(data, x_col_pk)
            y_pk = _to_1d(data, y_col_pk)

            result_pk = pk_find_peaks(
                y_pk,
                x=x_pk,
                height=float(pk_min_height) if pk_min_height > 0 else None,
                prominence=float(pk_prominence) if pk_prominence > 0 else None,
                distance=int(pk_distance),
                width=float(pk_width) if pk_width > 0 else None,
            )
            n_found = result_pk["n_peaks"]

            if n_found == 0:
                st.warning("No peaks found — try lowering the thresholds.")
            else:
                st.success(f"Found **{n_found}** peak(s).")

                # Compute FWHM and areas
                fwhm_result = compute_fwhm(y_pk, x_pk, result_pk)
                area_result = integrate_peaks(y_pk, x_pk, result_pk)

                # ── Plotly interactive figure ─────────────────────
                fig_pk = go.Figure()
                fig_pk.add_trace(
                    go.Scatter(
                        x=x_pk,
                        y=y_pk,
                        mode="lines",
                        name="Signal",
                        line=dict(color="#1f77b4", width=1.5),
                    )
                )
                fig_pk.add_trace(
                    go.Scatter(
                        x=result_pk["positions"],
                        y=result_pk["heights"],
                        mode="markers+text",
                        name="Peaks",
                        marker=dict(color="#d62728", size=10, symbol="triangle-down"),
                        text=[f"#{i + 1}" for i in range(n_found)],
                        textposition="top center",
                    )
                )
                x_lbl_pk = str(x_col_pk) if x_col_pk is not None else "x"
                y_lbl_pk = str(y_col_pk) if y_col_pk is not None else "y"
                fig_pk.update_layout(
                    xaxis_title=x_lbl_pk,
                    yaxis_title=y_lbl_pk,
                    title=f"Detected Peaks ({n_found})",
                    height=400,
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_pk, width="stretch")

                # ── Peak report table ─────────────────────────────
                st.markdown("#### Peak Report")
                report_df = pd.DataFrame(
                    {
                        "Peak": range(1, n_found + 1),
                        "Position": [f"{v:.4g}" for v in result_pk["positions"]],
                        "Height": [f"{v:.4g}" for v in result_pk["heights"]],
                        "Prominence": [f"{v:.4g}" for v in result_pk["prominences"]],
                        "FWHM": [f"{v:.4g}" for v in fwhm_result["fwhm"]],
                        "Area": [f"{v:.4g}" for v in area_result["areas"]],
                    }
                )
                st.dataframe(report_df, width="stretch")

                # CSV download: encoded on click (or once per report on old
                # Streamlit). Downloading does not rerun the panel, so the
                # report stays on screen.
                report_version = st.session_state.get("_at_pk_report_version", 0) + 1
                st.session_state["_at_pk_report_version"] = report_version
                st.download_button(
                    "Download peak report (CSV)",
                    data=download_payload(
                        lambda: report_df.to_csv(index=False).encode(),
                        cache_name="_at_pk_csv_cache",
                        version=report_version,
                        key="csv",
                    ),
                    file_name="peak_report.csv",
                    mime="text/csv",
                    key="pk_csv_dl",
                    on_click="ignore",
                )

                add_analysis_result(
                    {
                        "type": "peak_detection",
                        "dataset": dataset_name,
                        "x_col": str(x_col_pk),
                        "y_col": str(y_col_pk),
                        "results": {
                            "n_peaks": n_found,
                            "positions": [float(v) for v in result_pk["positions"]],
                            "heights": [float(v) for v in result_pk["heights"]],
                            "fwhm": [float(v) for v in fwhm_result["fwhm"]],
                            "areas": [float(v) for v in area_result["areas"]],
                        },
                    }
                )
                st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


@st.fragment
def _peak_fit_panel(data, dataset_name: str, x_col_pk, y_col_pk) -> None:
    st.markdown("#### Multi-Peak Model Fitting")

    c1, c2 = st.columns(2)
    with c1:
        profile_pk = st.selectbox(
            "Peak profile",
            ["Gaussian", "Lorentzian", "Pseudo-Voigt", "Voigt"],
            key="pk_profile",
        )
    with c2:
        bg_pk = st.selectbox(
            "Background",
            ["linear", "constant", "none"],
            key="pk_bg",
        )

    n_peaks_pk = int(
        st.number_input(
            "Number of peaks",
            min_value=1,
            max_value=8,
            value=1,
            step=1,
            key="pk_n",
        )
    )

    # Pre-read for default guesses
    x_pk_raw = _to_1d(data, x_col_pk)
    y_pk_raw = _to_1d(data, y_col_pk)
    x_range_pk = float(x_pk_raw.max() - x_pk_raw.min())
    default_centers = [
        float(x_pk_raw.min() + (i + 1) * x_range_pk / (n_peaks_pk + 1)) for i in range(n_peaks_pk)
    ]

    st.markdown("**Initial guesses** — one row per peak:")
    guess_rows = []
    for i in range(n_peaks_pk):
        c1, c2, c3 = st.columns(3)
        with c1:
            ctr = st.number_input(
                f"Peak {i + 1} — center",
                value=default_centers[i],
                key=f"pk_ctr_{i}",
            )
        with c2:
            amp = st.number_input(
                f"Peak {i + 1} — amplitude",
                value=float(y_pk_raw.max()),
                key=f"pk_amp_{i}",
            )
        with c3:
            wid = st.number_input(
                f"Peak {i + 1} — width (σ / Γ)",
                value=float(x_range_pk / max(n_peaks_pk * 4, 1)),
                min_value=1e-9,
                key=f"pk_wid_{i}",
            )
        guess_rows.append((float(ctr), float(amp), float(wid)))

    if st.button("Fit Peaks", type="primary", key="pk_fit_run"):
        try:
            x_pk = x_pk_raw.copy()
            y_pk = y_pk_raw.copy()
            model_name = profile_pk.lower().replace("-", "_")

            fit_result = fit_multipeak(
                y_pk,
                x_pk,
                n_peaks=n_peaks_pk,
                model=model_name,
                background=bg_pk,
                initial_guesses=guess_rows,
            )

            r2_pk = fit_result["r_squared"]
            x_lbl_pk = str(x_col_pk) if x_col_pk is not None else "x"
            y_lbl_pk = str(y_col_pk) if y_col_pk is not None else "y"

            # ── Plot ──────────────────────────────────────────────
            fig, (ax_fit, ax_res) = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
            ax_fit.scatter(x_pk, y_pk, s=10, alpha=0.5, color="#333333", label="Data")
            ax_fit.plot(
                x_pk,
                fit_result["fitted_y"],
                "r-",
                linewidth=2,
                label=f"{profile_pk} fit",
            )
            if n_peaks_pk > 1:
                for k, y_ind in enumerate(fit_result["individual_y"]):
                    ax_fit.plot(x_pk, y_ind, "--", alpha=0.6, label=f"Peak {k + 1}")
            ax_fit.set_ylabel(y_lbl_pk)
            ax_fit.set_title(f"{profile_pk} Fit  |  R² = {r2_pk:.5f}")
            ax_fit.legend()
            ax_fit.grid(True, alpha=0.3)

            ax_res.scatter(
                x_pk,
                fit_result["residuals"],
                s=8,
                alpha=0.5,
                color="#555555",
            )
            ax_res.axhline(0, color="red", linewidth=1)
            ax_res.set_xlabel(x_lbl_pk)
            ax_res.set_ylabel("Residuals")
            ax_res.grid(True, alpha=0.3)

            fig.tight_layout()
            show_figure(fig)
            plt.close(fig)

            # ── Parameter table ───────────────────────────────────
            st.markdown("#### Fitted Parameters")
            st.metric(f"R² {_r2_color(r2_pk)}", f"{r2_pk:.6f}")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Parameter": fit_result["param_names"],
                        "Value": [f"{v:.6g}" for v in fit_result["params"]],
                        "Std Error": [f"{e:.4g}" for e in fit_result["std_errors"]],
                    }
                ),
                width="stretch",
            )

            # ── Peak summary ──────────────────────────────────────
            st.markdown("#### Peak Summary")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Peak": k + 1,
                            "Center": f"{s['center']:.4g}",
                            "Amplitude": f"{s['amplitude']:.4g}",
                            "Width": f"{s['width']:.4g}",
                            "FWHM": f"{s['fwhm']:.4g}",
                        }
                        for k, s in enumerate(fit_result["peak_summaries"])
                    ]
                ),
                width="stretch",
            )

            add_analysis_result(
                {
                    "type": "peak_fitting",
                    "dataset": dataset_name,
                    "x_col": str(x_col_pk),
                    "y_col": str(y_col_pk),
                    "profile": profile_pk,
                    "n_peaks": n_peaks_pk,
                    "results": {
                        "parameters": [float(v) for v in fit_result["params"]],
                        "std_errors": [float(e) for e in fit_result["std_errors"]],
                        "r_squared": r2_pk,
                    },
                }
            )
            st.success("Results saved to analysis history.")

        except Exception as exc:
            st.error(f"Error: {exc}")
            with st.expander("Details"):
                st.code(traceback.format_exc())


with tab_peaks:
    st.markdown("### Peak Analysis")
    st.markdown(
        "Detect peaks, compute FWHM and area, and fit multi-peak models "
        "to spectral or chromatographic data."
    )

    _is_multidim_pk = (isinstance(data, pd.DataFrame) and len(_numeric_cols(data)) >= 2) or (
        isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2
    )

    if not _is_multidim_pk:
        st.warning(
            "Peak analysis requires X and Y columns. "
            "Load a DataFrame with ≥ 2 numeric columns or a 2-D NumPy array."
        )
    else:
        # Shared column selectors
        pk1, pk2 = st.columns(2)
        with pk1:
            x_col_pk, valid_xpk = _pick_col(
                data,
                "X column (wavenumber / wavelength / etc.)",
                key="pk_xcol",
            )
        with pk2:
            y_col_pk, valid_ypk = _pick_col(
                data,
                "Y column (intensity / absorbance / etc.)",
                key="pk_ycol",
            )

        if valid_xpk and valid_ypk:
            sub_find, sub_fit = st.tabs(["Find & Measure", "Fit Peaks"])

            # ── Find & Measure ────────────────────────────────────────────────
            with sub_find:
                _peak_find_panel(data, dataset_name, x_col_pk, y_col_pk)

            # ── Fit Peaks ─────────────────────────────────────────────────────
            with sub_fit:
                _peak_fit_panel(data, dataset_name, x_col_pk, y_col_pk)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 8 — Statistical Tests
# ══════════════════════════════════════════════════════════════════════════════


@st.fragment
def _statistical_tests_panel(data, dataset_name: str) -> None:
    st.markdown("### Statistical Tests")
    st.markdown(
        "Hypothesis tests, correlation analysis, and multiple-comparison "
        "corrections.  Select a test, choose columns, and inspect the result "
        "table."
    )

    _TEST_CHOICES = [
        "One-Sample t-Test",
        "Two-Sample t-Test (Student / Welch)",
        "Paired t-Test",
        "Mann-Whitney U (nonparametric 2-sample)",
        "Wilcoxon Signed-Rank (nonparametric paired)",
        "One-Way ANOVA",
        "Two-Way ANOVA",
        "Kruskal-Wallis (nonparametric ANOVA)",
        "Tukey HSD Post-Hoc",
        "Bonferroni Correction",
        "Pearson Correlation",
        "Spearman Correlation",
        "Chi-Square Independence",
    ]

    test_choice = st.selectbox("Select statistical test", _TEST_CHOICES, key="sttest_choice")
    alpha_st = st.number_input(
        "Significance level α",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.3f",
        key="sttest_alpha",
    )

    st.markdown("---")

    # ── One-sample t-test ─────────────────────────────────────────────────────
    if test_choice == "One-Sample t-Test":
        st.markdown("**H₀:** sample mean = μ₀")
        col_ref_t1, valid_t1 = _pick_col(data, "Select column", key="t1_col")
        popmean_t1 = st.number_input("Hypothesized mean (μ₀)", value=0.0, key="t1_mu")
        if valid_t1 and st.button("Run Test", type="primary", key="t1_run"):
            try:
                arr = _to_1d(data, col_ref_t1)
                result = ttest_one_sample(arr, popmean_t1, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {"type": "ttest_one_sample", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── Two-sample t-test ─────────────────────────────────────────────────────
    elif test_choice == "Two-Sample t-Test (Student / Welch)":
        st.markdown("**H₀:** mean(A) = mean(B)")
        c1, c2 = st.columns(2)
        with c1:
            col_a, valid_a = _pick_col(data, "Group A column", key="t2_col_a")
        with c2:
            col_b, valid_b = _pick_col(data, "Group B column", key="t2_col_b")
        equal_var_t2 = st.checkbox(
            "Assume equal variances (Student; uncheck for Welch)",
            value=True,
            key="t2_eq_var",
        )
        if valid_a and valid_b and st.button("Run Test", type="primary", key="t2_run"):
            try:
                a = _to_1d(data, col_a)
                b = _to_1d(data, col_b)
                result = ttest_two_sample(a, b, equal_var=equal_var_t2, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {"type": "ttest_two_sample", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── Paired t-test ─────────────────────────────────────────────────────────
    elif test_choice == "Paired t-Test":
        st.markdown("**H₀:** mean(A − B) = 0")
        c1, c2 = st.columns(2)
        with c1:
            col_a_pt, valid_a_pt = _pick_col(data, "Before / Group A", key="pt_col_a")
        with c2:
            col_b_pt, valid_b_pt = _pick_col(data, "After / Group B", key="pt_col_b")
        if valid_a_pt and valid_b_pt and st.button("Run Test", type="primary", key="pt_run"):
            try:
                a = _to_1d(data, col_a_pt)
                b = _to_1d(data, col_b_pt)
                result = ttest_paired(a, b, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {"type": "ttest_paired", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── Mann-Whitney U ─────────────────────────────────────────────────────────
    elif test_choice == "Mann-Whitney U (nonparametric 2-sample)":
        st.markdown("**H₀:** distributions of A and B are equal")
        c1, c2 = st.columns(2)
        with c1:
            col_a_mw, valid_a_mw = _pick_col(data, "Group A column", key="mw_col_a")
        with c2:
            col_b_mw, valid_b_mw = _pick_col(data, "Group B column", key="mw_col_b")
        if valid_a_mw and valid_b_mw and st.button("Run Test", type="primary", key="mw_run"):
            try:
                a = _to_1d(data, col_a_mw)
                b = _to_1d(data, col_b_mw)
                result = mannwhitney_u(a, b, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {"type": "mannwhitney_u", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── Wilcoxon signed-rank ──────────────────────────────────────────────────
    elif test_choice == "Wilcoxon Signed-Rank (nonparametric paired)":
        st.markdown("**H₀:** median difference = 0")
        c1, c2 = st.columns(2)
        with c1:
            col_a_wx, valid_a_wx = _pick_col(data, "Before / Group A", key="wx_col_a")
        with c2:
            col_b_wx, valid_b_wx = _pick_col(data, "After / Group B", key="wx_col_b")
        if valid_a_wx and valid_b_wx and st.button("Run Test", type="primary", key="wx_run"):
            try:
                a = _to_1d(data, col_a_wx)
                b = _to_1d(data, col_b_wx)
                result = wilcoxon_signed_rank(a, b, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {"type": "wilcoxon", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── One-way ANOVA ─────────────────────────────────────────────────────────
    elif test_choice == "One-Way ANOVA":
        st.markdown("**H₀:** all group means are equal")
        st.caption("Select 2–6 group columns from the loaded dataset.")
        if isinstance(data, pd.DataFrame):
            cols_all = _numeric_cols(data)
            group_cols = st.multiselect(
                "Group columns (select ≥ 2)",
                cols_all,
                key="anova1_cols",
            )
        else:
            arr_all = np.asarray(data)
            if arr_all.ndim < 2:
                st.warning("ANOVA requires a 2-D dataset with at least 2 columns.")
                group_cols = []
            else:
                n_avail = arr_all.shape[1]
                group_idx = st.multiselect(
                    "Group columns (select ≥ 2)",
                    list(range(n_avail)),
                    format_func=lambda i: f"Column {i}",
                    key="anova1_cols_np",
                )
                group_cols = group_idx  # reuse variable

        if len(group_cols) >= 2 and st.button("Run Test", type="primary", key="anova1_run"):
            try:
                if isinstance(data, pd.DataFrame):
                    arrays = [data[c].dropna().values.astype(float) for c in group_cols]
                else:
                    arrays = [np.asarray(data)[:, i].astype(float) for i in group_cols]
                result = anova_oneway(*arrays, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {
                        "type": "anova_oneway",
                        "dataset": dataset_name,
                        "groups": [str(c) for c in group_cols],
                        "results": result,
                    }
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())
        elif len(group_cols) < 2:
            st.info("Select at least 2 group columns to run the test.")

    # ── Two-Way ANOVA ─────────────────────────────────────────────────────────
    elif test_choice == "Two-Way ANOVA":
        st.markdown("**H₀:** all factor main effects and interaction are zero")
        st.caption(
            "Requires a DataFrame with one continuous response column and two "
            "categorical factor columns. Requires **statsmodels**."
        )
        try:
            import statsmodels  # noqa: F401

            _has_sm = True
        except ImportError:
            _has_sm = False

        if not _has_sm:
            st.warning(
                "Two-way ANOVA requires statsmodels. Install it with: `pip install statsmodels`"
            )
        elif not isinstance(data, pd.DataFrame):
            st.warning("Two-way ANOVA requires a tabular (DataFrame) dataset.")
        else:
            all_cols = list(data.columns)
            num_cols = _numeric_cols(data)
            col_r, col_fa, col_fb = st.columns(3)
            with col_r:
                response_col = st.selectbox("Response (numeric)", num_cols, key="anova2_resp")
            with col_fa:
                fa_col = st.selectbox("Factor A", all_cols, key="anova2_fa")
            with col_fb:
                remaining_fb = [c for c in all_cols if c != fa_col]
                fb_col = st.selectbox("Factor B", remaining_fb, key="anova2_fb")
            include_int = st.checkbox("Include A×B interaction term", value=True, key="anova2_int")
            if st.button("Run Two-Way ANOVA", type="primary", key="anova2_run"):
                try:
                    result = anova_twoway(
                        data,
                        response_col,
                        fa_col,
                        fb_col,
                        include_interaction=include_int,
                    )
                    table = result["table"]
                    st.markdown("#### ANOVA Table")
                    st.dataframe(table, width="stretch")
                    st.markdown("#### Significance Summary (α = 0.05)")

                    def _sig_badge(sig):
                        return "**Significant** (p < 0.05)" if sig else "Not significant (p ≥ 0.05)"

                    st.markdown(
                        f"- **Factor A ({fa_col}):** {_sig_badge(result['significant_a'])} "
                        f"(p = {result['p_value_a']:.4g})"
                    )
                    st.markdown(
                        f"- **Factor B ({fb_col}):** {_sig_badge(result['significant_b'])} "
                        f"(p = {result['p_value_b']:.4g})"
                    )
                    if result["significant_interaction"] is not None:
                        st.markdown(
                            f"- **Interaction ({fa_col}×{fb_col}):** "
                            f"{_sig_badge(result['significant_interaction'])} "
                            f"(p = {result['p_value_interaction']:.4g})"
                        )
                    add_analysis_result(
                        {
                            "type": "anova_twoway",
                            "dataset": dataset_name,
                            "response": response_col,
                            "factor_a": fa_col,
                            "factor_b": fb_col,
                            "interaction_included": include_int,
                        }
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

    # ── Kruskal-Wallis ────────────────────────────────────────────────────────
    elif test_choice == "Kruskal-Wallis (nonparametric ANOVA)":
        st.markdown("**H₀:** all group distributions are equal")
        if isinstance(data, pd.DataFrame):
            cols_all = _numeric_cols(data)
            group_cols_kw = st.multiselect("Group columns (select ≥ 2)", cols_all, key="kw_cols")
        else:
            arr_all = np.asarray(data)
            if arr_all.ndim < 2:
                st.warning("Kruskal-Wallis requires a 2-D dataset.")
                group_cols_kw = []
            else:
                group_cols_kw = st.multiselect(
                    "Group columns (select ≥ 2)",
                    list(range(arr_all.shape[1])),
                    format_func=lambda i: f"Column {i}",
                    key="kw_cols_np",
                )

        if len(group_cols_kw) >= 2 and st.button("Run Test", type="primary", key="kw_run"):
            try:
                if isinstance(data, pd.DataFrame):
                    arrays = [data[c].dropna().values.astype(float) for c in group_cols_kw]
                else:
                    arrays = [np.asarray(data)[:, i].astype(float) for i in group_cols_kw]
                result = kruskal_wallis(*arrays, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {
                        "type": "kruskal_wallis",
                        "dataset": dataset_name,
                        "groups": [str(c) for c in group_cols_kw],
                        "results": result,
                    }
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())
        elif len(group_cols_kw) < 2:
            st.info("Select at least 2 group columns to run the test.")

    # ── Tukey HSD ─────────────────────────────────────────────────────────────
    elif test_choice == "Tukey HSD Post-Hoc":
        st.markdown(
            "Pairwise post-hoc comparisons after a significant ANOVA. Requires scipy ≥ 1.8."
        )
        if isinstance(data, pd.DataFrame):
            cols_all = _numeric_cols(data)
            group_cols_thsd = st.multiselect(
                "Group columns (select ≥ 2)", cols_all, key="thsd_cols"
            )
        else:
            arr_all = np.asarray(data)
            if arr_all.ndim < 2:
                st.warning("Tukey HSD requires a 2-D dataset.")
                group_cols_thsd = []
            else:
                group_cols_thsd = st.multiselect(
                    "Group columns (select ≥ 2)",
                    list(range(arr_all.shape[1])),
                    format_func=lambda i: f"Column {i}",
                    key="thsd_cols_np",
                )

        if len(group_cols_thsd) >= 2 and st.button("Run Test", type="primary", key="thsd_run"):
            try:
                if isinstance(data, pd.DataFrame):
                    arrays = [data[c].dropna().values.astype(float) for c in group_cols_thsd]
                else:
                    arrays = [np.asarray(data)[:, i].astype(float) for i in group_cols_thsd]
                result = tukey_hsd(*arrays, alpha=alpha_st)
                st.markdown(
                    f"**{len(result['comparisons'])} pairwise comparisons** (α = {alpha_st})"
                )
                import pandas as _pd

                rows = []
                for cmp in result["comparisons"]:
                    gi = (
                        group_cols_thsd[cmp["group_i"]]
                        if isinstance(data, pd.DataFrame)
                        else f"Col {cmp['group_i']}"
                    )
                    gj = (
                        group_cols_thsd[cmp["group_j"]]
                        if isinstance(data, pd.DataFrame)
                        else f"Col {cmp['group_j']}"
                    )
                    rows.append(
                        {
                            "Group i": gi,
                            "Group j": gj,
                            "Mean diff": f"{cmp['mean_diff']:.4g}",
                            "p-value": f"{cmp['p_value']:.4g}",
                            "Significant": "Yes" if cmp["significant"] else "No",
                        }
                    )
                st.dataframe(_pd.DataFrame(rows), width="stretch", hide_index=True)
                add_analysis_result(
                    {
                        "type": "tukey_hsd",
                        "dataset": dataset_name,
                        "groups": [str(c) for c in group_cols_thsd],
                        "results": result,
                    }
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())
        elif len(group_cols_thsd) < 2:
            st.info("Select at least 2 group columns to run the test.")

    # ── Bonferroni ────────────────────────────────────────────────────────────
    elif test_choice == "Bonferroni Correction":
        st.markdown("Enter raw p-values (comma-separated or one per line) from multiple tests.")
        p_raw_input = st.text_area(
            "Raw p-values",
            value="0.03, 0.12, 0.001, 0.045",
            key="bonf_pvals",
        )
        if st.button("Apply Correction", type="primary", key="bonf_run"):
            try:
                import re

                p_raw = [float(v) for v in re.split(r"[,\s]+", p_raw_input.strip()) if v]
                result = bonferroni_correction(np.array(p_raw), alpha=alpha_st)
                st.metric("Adjusted α", f"{result['adjusted_alpha']:.4g}")
                import pandas as _pd

                rows = [
                    {
                        "Test #": i + 1,
                        "Raw p": f"{p:.4g}",
                        "Reject H₀": "Yes" if r else "No",
                    }
                    for i, (p, r) in enumerate(zip(result["corrected_p_values"], result["reject"]))
                ]
                st.dataframe(_pd.DataFrame(rows), width="stretch", hide_index=True)
                add_analysis_result(
                    {"type": "bonferroni", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── Pearson correlation ────────────────────────────────────────────────────
    elif test_choice == "Pearson Correlation":
        st.markdown("**H₀:** ρ = 0 (no linear correlation)")
        c1, c2 = st.columns(2)
        with c1:
            col_x_pc, valid_x_pc = _pick_col(data, "X column", key="pc_col_x")
        with c2:
            col_y_pc, valid_y_pc = _pick_col(data, "Y column", key="pc_col_y")
        if valid_x_pc and valid_y_pc and st.button("Run Test", type="primary", key="pc_run"):
            try:
                x = _to_1d(data, col_x_pc)
                y = _to_1d(data, col_y_pc)
                result = pearson_correlation(x, y, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {"type": "pearson_correlation", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── Spearman correlation ───────────────────────────────────────────────────
    elif test_choice == "Spearman Correlation":
        st.markdown("**H₀:** ρₛ = 0 (no monotonic correlation)")
        c1, c2 = st.columns(2)
        with c1:
            col_x_sc, valid_x_sc = _pick_col(data, "X column", key="sc_col_x")
        with c2:
            col_y_sc, valid_y_sc = _pick_col(data, "Y column", key="sc_col_y")
        if valid_x_sc and valid_y_sc and st.button("Run Test", type="primary", key="sc_run"):
            try:
                x = _to_1d(data, col_x_sc)
                y = _to_1d(data, col_y_sc)
                result = spearman_correlation(x, y, alpha=alpha_st)
                _verdict(result)
                _result_table(result)
                add_analysis_result(
                    {"type": "spearman_correlation", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # ── Chi-square independence ────────────────────────────────────────────────
    elif test_choice == "Chi-Square Independence":
        st.markdown(
            "**H₀:** the two categorical variables are independent. "
            "Enter the contingency table as rows separated by newlines and "
            "values separated by commas."
        )
        ct_input = st.text_area(
            "Contingency table (rows, comma-separated)",
            value="10, 20, 30\n6, 9, 17",
            key="chi2_table",
        )
        if st.button("Run Test", type="primary", key="chi2_run"):
            try:
                rows_ct = [
                    [float(v) for v in row.split(",")]
                    for row in ct_input.strip().splitlines()
                    if row.strip()
                ]
                ct_array = np.array(rows_ct)
                result = chi_square_independence(ct_array, alpha=alpha_st)
                _verdict(result)
                _result_table({k: v for k, v in result.items() if k != "expected"})
                with st.expander("Expected frequencies"):
                    import pandas as _pd

                    st.dataframe(
                        _pd.DataFrame(result["expected"]),
                        width="stretch",
                        hide_index=True,
                    )
                add_analysis_result(
                    {"type": "chi_square", "dataset": dataset_name, "results": result}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())


with tab_tests:
    _statistical_tests_panel(data, dataset_name)
