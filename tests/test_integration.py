"""Integration tests for the Plottle toolkit.

Tests end-to-end pipelines that exercise multiple modules together:
- io + math:              save → load → analyse
- io + plotting:          save → load → plot → save figure
- math + plotting:        compute → visualize
- io + math + plotting:   full pipeline (load → analyse → fit → plot → save)

All tests use a temporary directory and the non-interactive Agg matplotlib
backend so they run headlessly without a display.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # must be set before any other matplotlib import
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.io import (
    save_pickle,
    load_pickle,
    save_numpy,
    load_numpy,
    save_dataframe,
    load_dataframe,
    save_data,
    load_data,
)
from plottle.math import (
    calculate_statistics,
    fit_linear,
    fit_polynomial,
)
from plottle.plotting import (
    histogram,
    line_plot,
    scatter_plot,
    heatmap,
    save_figure,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp(tmp_path):
    """Yield a temporary directory Path; cleaned up automatically by pytest."""
    return tmp_path


@pytest.fixture
def linear_data():
    """Deterministic noisy linear dataset: y = 3x + 2 + noise."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y = 3.0 * x + 2.0 + rng.normal(0, 0.5, size=x.shape)
    return x, y


@pytest.fixture
def sample_df():
    """Small deterministic DataFrame with two numeric columns."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "time": np.linspace(0, 5, 30),
            "signal": np.sin(np.linspace(0, 2 * np.pi, 30)) + rng.normal(0, 0.1, 30),
            "noise": rng.normal(0, 1, 30),
        }
    )


# ============================================================================
# Pipeline 1 — IO → Math
# ============================================================================


@pytest.mark.integration
class TestIOMathPipeline:
    def test_csv_roundtrip_then_statistics(self, tmp, sample_df):
        """Save DataFrame as CSV, reload it, compute statistics on a column."""
        path = tmp / "data.csv"
        save_dataframe(sample_df, path)
        loaded = load_dataframe(path)

        stats = calculate_statistics(loaded["signal"].values)

        assert set(stats.keys()) >= {"mean", "std", "min", "max", "q1", "q3"}
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["std"] > 0

    def test_numpy_roundtrip_then_statistics(self, tmp, linear_data):
        """Save 1-D numpy array, reload it, verify statistics are consistent."""
        x, y = linear_data
        path = tmp / "y_values.npy"
        save_numpy(y, path)
        loaded_y = load_numpy(path)

        stats_orig = calculate_statistics(y)
        stats_loaded = calculate_statistics(loaded_y)

        assert abs(stats_orig["mean"] - stats_loaded["mean"]) < 1e-10
        assert abs(stats_orig["std"] - stats_loaded["std"]) < 1e-10

    def test_pickle_roundtrip_then_statistics(self, tmp, sample_df):
        """Save DataFrame as pickle, reload it, verify statistics are preserved."""
        path = tmp / "df.pkl"
        save_pickle(sample_df, path)
        loaded = load_pickle(path)

        original_mean = calculate_statistics(sample_df["time"].values)["mean"]
        loaded_mean = calculate_statistics(loaded["time"].values)["mean"]

        assert abs(original_mean - loaded_mean) < 1e-10

    def test_universal_loader_csv_then_fit(self, tmp, linear_data):
        """Use the universal load_data / save_data interface, then fit a line."""
        x, y = linear_data
        df = pd.DataFrame({"x": x, "y": y})
        path = tmp / "linear.csv"
        save_data(df, path)
        loaded = load_data(path)

        result = fit_linear(loaded["x"].values, loaded["y"].values)

        assert "slope" in result
        assert "intercept" in result
        # True slope ≈ 3.0 — allow generous tolerance for small noisy sample
        assert abs(result["slope"] - 3.0) < 0.3


# ============================================================================
# Pipeline 2 — IO → Plotting
# ============================================================================


@pytest.mark.integration
@pytest.mark.plotting
class TestIOPlottingPipeline:
    def test_load_csv_then_histogram_then_save(self, tmp, sample_df):
        """Load CSV → histogram → save figure as PNG."""
        csv_path = tmp / "data.csv"
        png_path = tmp / "hist.png"
        save_dataframe(sample_df, csv_path)

        loaded = load_dataframe(csv_path)
        fig, ax, info = histogram(loaded["noise"].values, bins=10, title="Noise")
        save_figure(fig, png_path, dpi=72)
        plt.close(fig)

        assert png_path.exists()
        assert png_path.stat().st_size > 0

    def test_load_numpy_then_line_plot_then_save(self, tmp, linear_data):
        """Save/load 2-D numpy array → line_plot → save as SVG."""
        x, y = linear_data
        arr = np.column_stack([x, y])
        npy_path = tmp / "xy.npy"
        svg_path = tmp / "line.svg"
        save_numpy(arr, npy_path)

        loaded = load_numpy(npy_path)
        fig, ax = line_plot(loaded[:, 0], [loaded[:, 1]], xlabel="x", ylabel="y")
        save_figure(fig, svg_path)
        plt.close(fig)

        assert svg_path.exists()
        assert svg_path.stat().st_size > 0

    def test_load_csv_then_scatter_then_save(self, tmp, sample_df):
        """Load CSV → scatter_plot → save as PDF."""
        csv_path = tmp / "df.csv"
        pdf_path = tmp / "scatter.pdf"
        save_dataframe(sample_df, csv_path)

        loaded = load_dataframe(csv_path)
        fig, ax = scatter_plot(
            loaded["time"].values, loaded["signal"].values, xlabel="Time (s)", ylabel="Signal"
        )
        save_figure(fig, pdf_path)
        plt.close(fig)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0


# ============================================================================
# Pipeline 3 — Math → Plotting
# ============================================================================


@pytest.mark.integration
@pytest.mark.plotting
class TestMathPlottingPipeline:
    def test_linear_fit_then_plot(self, linear_data):
        """Fit a line to data, overlay fit on scatter plot."""
        x, y = linear_data
        result = fit_linear(x, y)
        y_fit = result["slope"] * x + result["intercept"]

        fig, ax = scatter_plot(x, y, title="Data + Linear Fit")
        ax.plot(x, y_fit, color="red", linewidth=1.5, label="Fit")
        ax.legend()
        plt.close(fig)

        assert result["r_squared"] > 0.95

    def test_polynomial_fit_then_line_plot(self, linear_data):
        """Fit a degree-2 polynomial, plot original and fitted curves."""
        x, y = linear_data
        result = fit_polynomial(x, y, degree=2)
        y_fit = result["predict"](x)

        fig, ax = line_plot(
            x,
            [y, y_fit],
            labels=["Data", "Poly Fit (deg 2)"],
            xlabel="x",
            ylabel="y",
            title="Polynomial Fit",
        )
        plt.close(fig)

        assert result["r_squared"] > 0.90

    def test_statistics_annotations_on_histogram(self, linear_data):
        """Compute stats, then annotate a histogram with mean and std lines."""
        _, y = linear_data
        stats = calculate_statistics(y)

        fig, ax, info = histogram(y, bins=15, xlabel="Value")
        ax.axvline(stats["mean"], color="red", linestyle="--", label="Mean")
        ax.axvline(stats["mean"] + stats["std"], color="orange", linestyle=":", label="+1 SD")
        ax.axvline(stats["mean"] - stats["std"], color="orange", linestyle=":", label="-1 SD")
        ax.legend()
        plt.close(fig)

        assert stats["mean"] > 0  # y = 3x + 2, so mean >> 0
        assert stats["std"] > 0

    def test_heatmap_from_computed_matrix(self):
        """Build a correlation-style matrix and render as heatmap."""
        rng = np.random.default_rng(2)
        data = rng.normal(size=(20, 4))
        df = pd.DataFrame(data, columns=["A", "B", "C", "D"])
        corr = df.corr().values  # 4×4 correlation matrix

        fig, ax = heatmap(corr, title="Correlation Matrix")
        plt.close(fig)

        assert corr.shape == (4, 4)
        # Diagonal must be 1 (self-correlation)
        np.testing.assert_allclose(np.diag(corr), np.ones(4), atol=1e-10)


# ============================================================================
# Pipeline 4 — Full pipeline: IO → Math → Plotting → Save
# ============================================================================


@pytest.mark.integration
@pytest.mark.plotting
class TestFullPipeline:
    def test_csv_analyse_fit_plot_save(self, tmp, linear_data):
        """Complete workflow: save CSV → load → analyse → fit → plot → save PNG."""
        x, y = linear_data
        df = pd.DataFrame({"time": x, "response": y})

        # 1. Persist data
        csv_path = tmp / "experiment.csv"
        save_dataframe(df, csv_path)

        # 2. Load
        loaded = load_dataframe(csv_path)
        x_l = loaded["time"].values
        y_l = loaded["response"].values

        # 3. Analyse
        stats = calculate_statistics(y_l)
        assert stats["std"] > 0

        # 4. Fit
        fit = fit_linear(x_l, y_l)
        assert fit["r_squared"] > 0.90

        # 5. Plot: data + fit + mean line
        y_fit = fit["slope"] * x_l + fit["intercept"]
        fig, ax = line_plot(
            x_l,
            [y_l, y_fit],
            labels=["Measured", "Linear fit"],
            xlabel="Time (s)",
            ylabel="Response",
            title="Full Pipeline Test",
        )
        ax.axhline(
            stats["mean"],
            color="gray",
            linestyle="--",
            alpha=0.6,
            label=f"Mean = {stats['mean']:.2f}",
        )
        ax.legend()

        # 6. Save
        png_path = tmp / "result.png"
        save_figure(fig, png_path, dpi=72)
        plt.close(fig)

        assert png_path.exists()
        assert png_path.stat().st_size > 1000  # non-trivial file

    def test_numpy_analyse_fit_plot_save_pdf(self, tmp):
        """Save / load ndarray, fit a quadratic, produce a PDF figure."""
        rng = np.random.default_rng(3)
        x = np.linspace(-3, 3, 80)
        y = 1.5 * x**2 - 2.0 * x + 0.5 + rng.normal(0, 0.3, x.shape)

        npy_path = tmp / "quadratic.npy"
        save_numpy(np.column_stack([x, y]), npy_path)

        arr = load_numpy(npy_path)
        x_l, y_l = arr[:, 0], arr[:, 1]

        result = fit_polynomial(x_l, y_l, degree=2)
        y_fit = result["predict"](x_l)

        fig, ax = scatter_plot(x_l, y_l, title="Quadratic Fit", xlabel="x", ylabel="y")
        ax.plot(x_l, y_fit, color="crimson", linewidth=2, label="Degree-2 fit")
        ax.legend()

        pdf_path = tmp / "quadratic.pdf"
        save_figure(fig, pdf_path)
        plt.close(fig)

        assert pdf_path.exists()
        assert result["r_squared"] > 0.95

    def test_multi_format_export(self, tmp, linear_data):
        """Generate one figure and export it to PNG, SVG, and PDF."""
        x, y = linear_data
        fig, ax = scatter_plot(x, y, xlabel="x", ylabel="y", title="Multi-Format Export Test")

        for fmt in ("png", "svg", "pdf"):
            out = tmp / f"plot.{fmt}"
            save_figure(fig, out, dpi=72)
            assert out.exists(), f"{fmt} file was not created"
            assert out.stat().st_size > 0, f"{fmt} file is empty"

        plt.close(fig)
