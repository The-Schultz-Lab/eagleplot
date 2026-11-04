"""Tests for plottle/report.py — PDF report generator."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from plottle.report import (
    _analysis_result_to_figure,
    _dataframe_to_figure,
    _make_title_page,
    generate_pdf_report,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.5, 6.1], "label": ["a", "b", "c"]})


@pytest.fixture()
def large_df() -> pd.DataFrame:
    """DataFrame with more rows than the default max_rows=30."""
    return pd.DataFrame({"col": range(50), "val": np.random.default_rng(0).random(50)})


@pytest.fixture()
def wide_df() -> pd.DataFrame:
    """DataFrame with more than 10 columns."""
    return pd.DataFrame({f"c{i}": range(3) for i in range(15)})


@pytest.fixture()
def typical_result() -> dict:
    return {
        "type": "linear_fit",
        "dataset": "test.csv",
        "timestamp": "2026-03-16T12:00:00",
        "results": {
            "slope": 2.5,
            "intercept": -1.0,
            "r_squared": 0.998,
            "p_value": 1.2e-5,
        },
    }


@pytest.fixture()
def mpl_fig() -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig


@pytest.fixture()
def plotly_fig():
    try:
        import plotly.graph_objects as go
        return go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    except ImportError:
        pytest.skip("plotly not installed")


# ══════════════════════════════════════════════════════════════════════════════
# TestMakeTitlePage
# ══════════════════════════════════════════════════════════════════════════════


class TestMakeTitlePage:
    def test_returns_matplotlib_figure(self):
        fig = _make_title_page("My Report", "Alice", 3, 2, 1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_text_present_in_figure(self):
        title = "Test Report Title"
        fig = _make_title_page(title, "Bob", 1, 1, 1)
        texts = [t.get_text() for t in fig.texts]
        assert any(title in t for t in texts)
        plt.close(fig)

    def test_empty_author_string_does_not_raise(self):
        fig = _make_title_page("Report", "", 0, 0, 0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_author_appears_when_provided(self):
        fig = _make_title_page("R", "Dr. Smith", 0, 0, 0)
        texts = [t.get_text() for t in fig.texts]
        assert any("Dr. Smith" in t for t in texts)
        plt.close(fig)

    def test_summary_counts_appear_in_text(self):
        fig = _make_title_page("R", "", 7, 3, 5)
        all_text = " ".join(t.get_text() for t in fig.texts)
        assert "7" in all_text
        assert "3" in all_text
        assert "5" in all_text
        plt.close(fig)

    def test_figure_size_is_letter(self):
        fig = _make_title_page("R", "", 0, 0, 0)
        w, h = fig.get_size_inches()
        assert abs(w - 8.5) < 0.1
        assert abs(h - 11.0) < 0.1
        plt.close(fig)

    def test_zero_counts_allowed(self):
        fig = _make_title_page("Empty", "A", 0, 0, 0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TestDataframeToFigure
# ══════════════════════════════════════════════════════════════════════════════


class TestDataframeToFigure:
    def test_returns_matplotlib_figure(self, small_df):
        fig = _dataframe_to_figure(small_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_small_dataframe_no_truncation(self, small_df):
        # Should not raise and should include all rows
        fig = _dataframe_to_figure(small_df, title="Small", max_rows=30)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_dataframe_truncated(self, large_df):
        # 50 rows > max_rows=30 → title should contain a note
        fig = _dataframe_to_figure(large_df, title="Big", max_rows=30)
        assert isinstance(fig, plt.Figure)
        # Title text should mention row count
        ax = fig.axes[0]
        title_text = ax.get_title()
        assert "50" in title_text or "30" in title_text
        plt.close(fig)

    def test_wide_dataframe_column_limit(self, wide_df):
        # 15 columns > max_cols=10 → title note present
        fig = _dataframe_to_figure(wide_df, title="Wide", max_rows=30)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        title_text = ax.get_title()
        assert "15" in title_text or "10" in title_text
        plt.close(fig)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        fig = _dataframe_to_figure(df, title="Empty")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_appears_in_figure(self, small_df):
        fig = _dataframe_to_figure(small_df, title="My Dataset")
        ax = fig.axes[0]
        assert "My Dataset" in ax.get_title()
        plt.close(fig)

    def test_custom_max_rows_respected(self, large_df):
        fig = _dataframe_to_figure(large_df, max_rows=5)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        title_text = ax.get_title()
        assert "50" in title_text or "5" in title_text
        plt.close(fig)

    def test_float_values_formatted(self):
        df = pd.DataFrame({"v": [1.23456789, 9.87654321]})
        fig = _dataframe_to_figure(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TestAnalysisResultToFigure
# ══════════════════════════════════════════════════════════════════════════════


class TestAnalysisResultToFigure:
    def test_returns_matplotlib_figure(self, typical_result):
        fig = _analysis_result_to_figure(typical_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_typical_analysis_dict(self, typical_result):
        fig = _analysis_result_to_figure(typical_result)
        # Figure should have axes
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_empty_results_dict(self):
        result = {
            "type": "test",
            "dataset": "d.csv",
            "timestamp": "2026-01-01T00:00:00",
            "results": {},
        }
        fig = _analysis_result_to_figure(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_skips_callable_values_gracefully(self):
        result = {
            "type": "custom",
            "dataset": "d.csv",
            "timestamp": "2026-01-01T00:00:00",
            "results": {
                "value": 42.0,
                "func": lambda x: x,  # callable — should not raise
            },
        }
        fig = _analysis_result_to_figure(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_numpy_array_values_rendered(self):
        result = {
            "type": "eigenvalues",
            "dataset": "m.csv",
            "timestamp": "2026-01-01T00:00:00",
            "results": {"eigenvalues": np.array([1.0, 2.0, 3.0])},
        }
        fig = _analysis_result_to_figure(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_array_summarised(self):
        result = {
            "type": "svd",
            "dataset": "m.csv",
            "timestamp": "2026-01-01T00:00:00",
            "results": {"singular_values": np.arange(100, dtype=float)},
        }
        fig = _analysis_result_to_figure(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_results_key(self):
        result = {"type": "x", "dataset": "y", "timestamp": "2026-01-01T00:00:00"}
        fig = _analysis_result_to_figure(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_type_and_dataset_shown_in_title(self, typical_result):
        fig = _analysis_result_to_figure(typical_result)
        ax = fig.axes[0]
        title = ax.get_title()
        assert "linear_fit" in title
        assert "test.csv" in title
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TestGeneratePdfReport
# ══════════════════════════════════════════════════════════════════════════════


class TestGeneratePdfReport:
    def test_returns_path(self, tmp_path, small_df):
        out = tmp_path / "report.pdf"
        result = generate_pdf_report(str(out), "Test", [], {}, [])
        assert isinstance(result, Path)

    def test_output_file_exists(self, tmp_path):
        out = tmp_path / "report.pdf"
        generate_pdf_report(str(out), "Test", [], {}, [])
        assert out.exists()

    def test_output_file_nonzero_size(self, tmp_path):
        out = tmp_path / "report.pdf"
        generate_pdf_report(str(out), "Test", [], {}, [])
        assert out.stat().st_size > 0

    def test_empty_inputs_produces_valid_pdf(self, tmp_path):
        out = tmp_path / "empty.pdf"
        generate_pdf_report(str(out), "Empty Report", [], {}, [])
        # A minimal PDF starts with "%PDF"
        header = out.read_bytes()[:4]
        assert header == b"%PDF"

    def test_with_one_matplotlib_figure(self, tmp_path, mpl_fig):
        out = tmp_path / "with_mpl.pdf"
        plot_entries = [
            {
                "type": "line",
                "dataset": "d.csv",
                "timestamp": "2026-01-01T12:00:00",
                "figure": mpl_fig,
            }
        ]
        generate_pdf_report(str(out), "Plots", plot_entries, {}, [])
        assert out.stat().st_size > 0

    def test_plotly_figure_skipped_gracefully(self, tmp_path, plotly_fig):
        out = tmp_path / "with_plotly.pdf"
        plot_entries = [
            {
                "type": "scatter",
                "dataset": "d.csv",
                "timestamp": "2026-01-01T12:00:00",
                "figure": plotly_fig,
            }
        ]
        generate_pdf_report(str(out), "Plotly Skip", plot_entries, {}, [])
        assert out.exists()

    def test_none_figure_skipped_gracefully(self, tmp_path):
        out = tmp_path / "with_none.pdf"
        plot_entries = [
            {
                "type": "bar",
                "dataset": "d.csv",
                "timestamp": "2026-01-01T12:00:00",
                "figure": None,
            }
        ]
        generate_pdf_report(str(out), "None Skip", plot_entries, {}, [])
        assert out.exists()

    def test_with_one_dataframe_dataset(self, tmp_path, small_df):
        out = tmp_path / "with_df.pdf"
        generate_pdf_report(str(out), "DF", [], {"my_data": small_df}, [])
        assert out.stat().st_size > 0

    def test_non_dataframe_dataset_skipped(self, tmp_path):
        out = tmp_path / "non_df.pdf"
        datasets = {"arr": np.array([1, 2, 3]), "df": pd.DataFrame({"a": [1]})}
        generate_pdf_report(str(out), "Mixed", [], datasets, [])
        assert out.exists()

    def test_with_one_analysis_result(self, tmp_path, typical_result):
        out = tmp_path / "with_analysis.pdf"
        generate_pdf_report(str(out), "Analysis", [], {}, [typical_result])
        assert out.stat().st_size > 0

    def test_filepath_as_path_object(self, tmp_path, small_df):
        out = tmp_path / "path_obj.pdf"
        result = generate_pdf_report(out, "Path obj", [], {}, [])
        assert isinstance(result, Path)
        assert out.exists()

    def test_filepath_as_str(self, tmp_path):
        out = str(tmp_path / "str_path.pdf")
        result = generate_pdf_report(out, "Str path", [], {}, [])
        assert isinstance(result, Path)
        assert Path(out).exists()

    def test_author_field_accepted(self, tmp_path):
        out = tmp_path / "author.pdf"
        generate_pdf_report(str(out), "Report", [], {}, [], author="Jane Doe")
        assert out.exists()

    def test_custom_dpi_accepted(self, tmp_path, mpl_fig):
        out = tmp_path / "dpi300.pdf"
        plot_entries = [
            {
                "type": "line",
                "dataset": "d",
                "timestamp": "2026-01-01T00:00:00",
                "figure": mpl_fig,
            }
        ]
        generate_pdf_report(str(out), "High DPI", plot_entries, {}, [], dpi=300)
        assert out.stat().st_size > 0

    def test_multiple_plots_and_datasets(self, tmp_path, mpl_fig, small_df, large_df):
        out = tmp_path / "multi.pdf"
        fig2, ax2 = plt.subplots()
        ax2.scatter([1, 2], [3, 4])
        plot_entries = [
            {"type": "line", "dataset": "a", "timestamp": "2026-01-01T00:00:00", "figure": mpl_fig},
            {"type": "scatter", "dataset": "b", "timestamp": "2026-01-01T01:00:00", "figure": fig2},
        ]
        datasets = {"small": small_df, "large": large_df}
        generate_pdf_report(str(out), "Multi", plot_entries, datasets, [])
        assert out.stat().st_size > 0
        plt.close(fig2)

    def test_returned_path_matches_input(self, tmp_path):
        out = tmp_path / "match.pdf"
        result = generate_pdf_report(str(out), "T", [], {}, [])
        assert result == out
