"""Analysis Tools page (roadmap D1.2): tools still work, and reruns do no wasted work."""

import ast
from pathlib import Path

import matplotlib.figure
import pandas as pd
import pytest

import eagleplot.math
import eagleplot.peaks
import eagleplot.signal
from eagleplot.utils.figures import LAZY_DOWNLOAD

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "eagleplot" / "pages" / "3_Analysis_Tools.py"
SPECTRUM = ROOT / "eagleplot" / "example-data" / "Artificial" / "ir_spectrum.csv"

#: Analysis functions whose calls are counted, by the module the page imports them from.
_COUNTED = {
    eagleplot.math: ["calculate_statistics", "fit_polynomial", "pearson_correlation"],
    eagleplot.signal: ["smooth_moving_average"],
    eagleplot.peaks: ["find_peaks", "fit_multipeak"],
}


@pytest.fixture
def work(monkeypatch):
    """Count savefig / to_csv / analysis calls made while the page runs (same process)."""
    calls = []

    def counting(label, original):
        def wrapper(*args, **kwargs):
            calls.append(label)
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        counting("savefig", matplotlib.figure.Figure.savefig),
    )
    monkeypatch.setattr(pd.DataFrame, "to_csv", counting("to_csv", pd.DataFrame.to_csv))
    for module, names in _COUNTED.items():
        for name in names:
            monkeypatch.setattr(module, name, counting(name, getattr(module, name)))
    return calls


@pytest.fixture
def page():
    """Analysis Tools with ir_spectrum.csv (wavenumber, transmittance) loaded."""
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    at.session_state["datasets"] = {"ir_spectrum.csv": pd.read_csv(SPECTRUM)}
    at.session_state["current_dataset"] = "ir_spectrum.csv"
    at.run()
    assert not at.exception
    return at


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def _has_download_button(at, key):
    # Streamlit 1.50 (the newest on Python 3.9) exposes download buttons to AppTest as
    # UnknownElement without a .key; the key is still the suffix of the element id.
    return any(b.proto.id.endswith("-" + key) for b in at.get("download_button"))


def _metric_labels(at):
    return [m.label for m in at.metric]


def _n_results(at):
    return len(at.session_state["analysis_results"])


# ── The tools still run and show their results ────────────────────────────────


class TestToolsRun:
    def test_page_renders_all_tabs(self, page):
        labels = [t.label for t in page.tabs]  # depth-first: sub-tabs follow their tab
        assert {
            "Statistics",
            "Distribution",
            "Curve Fitting",
            "Optimization",
            "Linear Algebra",
            "Signal Processing",
            "Peak Fitting",
            "Statistical Tests",
            "Smooth",
            "Interpolate",
            "Find & Measure",
            "Fit Peaks",
        } <= set(labels)
        for key in ("stats_run", "fit_run", "min_run", "eig_run", "sp_smooth_run", "pk_fit_run"):
            assert page.button(key=key) is not None

    def test_statistics(self, page, work):
        page.selectbox(key="stats_col").set_value("transmittance")
        _ok(page.button(key="stats_run").click().run())
        assert work.count("calculate_statistics") == 1
        assert {"Mean", "Median", "Std Dev", "IQR"} <= set(_metric_labels(page))
        assert page.session_state["analysis_results"][-1]["type"] == "statistics"

    def test_polynomial_fit(self, page, work):
        page.selectbox(key="fit_ycol").set_value("transmittance").run()
        page.radio(key="fit_type").set_value("Polynomial").run()
        page.number_input(key="fit_degree").set_value(3).run()
        _ok(page.button(key="fit_run").click().run())
        assert work.count("fit_polynomial") == 1
        assert work.count("savefig") == 1  # the st.pyplot of the fitted curve
        assert any(label.startswith("R²") for label in _metric_labels(page))
        last = page.session_state["analysis_results"][-1]
        assert last["type"] == "fit_polynomial" and last["degree"] == 3

    def test_minimize(self, page):
        _ok(page.button(key="min_run").click().run())
        assert "f(x*)" in _metric_labels(page)
        assert page.session_state["analysis_results"][-1]["type"] == "minimize_function"

    def test_smoothing(self, page, work):
        page.selectbox(key="sp_ycol").set_value("transmittance").run()
        page.slider(key="sp_ma_window").set_value(21).run()
        _ok(page.button(key="sp_smooth_run").click().run())
        assert work.count("smooth_moving_average") == 1
        last = page.session_state["analysis_results"][-1]
        assert last["type"] == "smoothing" and "window=21" in last["method"]

    def test_find_peaks_offers_report_without_encoding_it(self, page, work):
        page.selectbox(key="pk_ycol").set_value("transmittance").run()
        page.number_input(key="pk_prominence").set_value(0.05).run()
        _ok(page.button(key="pk_find_run").click().run())
        assert work.count("find_peaks") == 1
        assert page.session_state["analysis_results"][-1]["results"]["n_peaks"] > 0
        assert _has_download_button(page, "pk_csv_dl")
        # The CSV is built when the button is clicked (Streamlit >= 1.52), else once.
        assert work.count("to_csv") == (0 if LAZY_DOWNLOAD else 1)

    def test_peak_fit(self, page, work):
        page.selectbox(key="pk_ycol").set_value("transmittance").run()
        _ok(page.button(key="pk_fit_run").click().run())
        assert work.count("fit_multipeak") == 1
        assert page.session_state["analysis_results"][-1]["type"] == "peak_fitting"

    def test_pearson_correlation(self, page, work):
        page.selectbox(key="sttest_choice").set_value("Pearson Correlation").run()
        page.selectbox(key="pc_col_y").set_value("transmittance").run()
        _ok(page.button(key="pc_run").click().run())
        assert work.count("pearson_correlation") == 1
        assert page.session_state["analysis_results"][-1]["type"] == "pearson_correlation"


# ── Reruns do no wasted work ──────────────────────────────────────────────────


class TestRerunsDoNoWastedWork:
    def _run_everything_once(self, page):
        page.selectbox(key="fit_ycol").set_value("transmittance").run()
        page.radio(key="fit_type").set_value("Polynomial").run()
        _ok(page.button(key="fit_run").click().run())
        page.selectbox(key="sp_ycol").set_value("transmittance").run()
        _ok(page.button(key="sp_smooth_run").click().run())
        page.selectbox(key="pk_ycol").set_value("transmittance").run()
        page.number_input(key="pk_prominence").set_value(0.05).run()
        _ok(page.button(key="pk_find_run").click().run())
        _ok(page.button(key="stats_run").click().run())

    def test_plain_rerun_recomputes_and_encodes_nothing(self, page, work):
        self._run_everything_once(page)
        n_results = _n_results(page)
        work.clear()
        _ok(page.run())
        assert work == []  # no analysis, no figure rendering, no CSV encoding
        assert _n_results(page) == n_results

    def test_changing_a_setting_does_not_rerun_any_tool(self, page, work):
        self._run_everything_once(page)
        n_results = _n_results(page)
        work.clear()
        page.number_input(key="fit_degree").set_value(4).run()
        page.slider(key="sp_ma_window").set_value(31).run()
        page.number_input(key="pk_prominence").set_value(0.1).run()
        page.number_input(key="sttest_alpha").set_value(0.01).run()
        _ok(page)
        assert work == []
        assert _n_results(page) == n_results


# ── Structure: every tool is a fragment ───────────────────────────────────────


def _is_fragment_decorator(node):
    target = node.func if isinstance(node, ast.Call) else node
    return isinstance(target, ast.Attribute) and target.attr == "fragment"


def test_every_run_button_lives_in_a_fragment():
    """A tool's button and its settings must rerun only that tool, not the whole page."""
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    fragments = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(_is_fragment_decorator(d) for d in node.decorator_list)
    }
    assert len(fragments) >= 15, sorted(fragments)

    buttons_outside = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in fragments:
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in ("button", "download_button")
            ):
                buttons_outside.append(sub.lineno)
    assert buttons_outside == [], f"buttons outside a fragment at lines {buttons_outside}"
