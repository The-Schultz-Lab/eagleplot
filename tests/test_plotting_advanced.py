"""Test suite for advanced plotting features (Seaborn and Plotly).

Tests all advanced plotting functions including statistical plots and
interactive visualizations.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Set matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")

# Import the module to test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.plotting import (
    distribution_plot,
    box_plot,
    regression_plot,
    interactive_histogram,
    interactive_scatter,
    interactive_line,
    interactive_heatmap,
    interactive_3d_surface,
    export_interactive,
    HAS_SEABORN,
    HAS_PLOTLY,
    pair_plot,
    interactive_3d_scatter,
    interactive_ternary,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "group": ["A"] * 50 + ["B"] * 50,
            "value": np.random.randn(100),
            "category": (["X"] * 25 + ["Y"] * 25) * 2,
        }
    )


@pytest.fixture
def sample_2d_data():
    """Sample 2D data for testing."""
    np.random.seed(42)
    return np.random.rand(10, 10)


# ============================================================================
# Seaborn Statistical Plots Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SEABORN, reason="Seaborn not installed")
class TestSeabornPlots:
    """Tests for Seaborn statistical plots."""

    def test_distribution_plot_histogram(self, sample_data):
        """Test distribution plot with histogram."""
        fig, ax = distribution_plot(sample_data, kind="hist", kde=True)

        assert fig is not None
        assert ax is not None
        assert len(ax.patches) > 0  # Has histogram bars

    def test_distribution_plot_kde(self, sample_data):
        """Test distribution plot with KDE only."""
        fig, ax = distribution_plot(sample_data, kind="kde")

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0  # Has KDE line

    def test_distribution_plot_ecdf(self, sample_data):
        """Test distribution plot with ECDF."""
        fig, ax = distribution_plot(sample_data, kind="ecdf")

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0  # Has ECDF line

    def test_distribution_plot_invalid_kind(self, sample_data):
        """Test that invalid kind raises error."""
        with pytest.raises(ValueError):
            distribution_plot(sample_data, kind="invalid")

    def test_box_plot_basic(self, sample_dataframe):
        """Test basic box plot."""
        fig, ax = box_plot(sample_dataframe, x="group", y="value")

        assert fig is not None
        assert ax is not None
        # Box plot should have artists
        assert len(ax.artists) > 0 or len(ax.patches) > 0

    def test_box_plot_with_hue(self, sample_dataframe):
        """Test box plot with hue grouping."""
        fig, ax = box_plot(sample_dataframe, x="group", y="value", hue="category")

        assert fig is not None
        assert ax is not None

    def test_violin_plot(self, sample_dataframe):
        """Test violin plot."""
        fig, ax = box_plot(sample_dataframe, x="group", y="value", kind="violin")

        assert fig is not None
        assert ax is not None
        # Violin plot should have collections
        assert len(ax.collections) > 0

    def test_boxen_plot(self, sample_dataframe):
        """Test boxen plot."""
        fig, ax = box_plot(sample_dataframe, x="group", y="value", kind="boxen")

        assert fig is not None
        assert ax is not None

    def test_box_plot_invalid_kind(self, sample_dataframe):
        """Test that invalid kind raises error."""
        with pytest.raises(ValueError):
            box_plot(sample_dataframe, x="group", y="value", kind="invalid")

    def test_regression_plot_linear(self):
        """Test linear regression plot."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.randn(50)

        fig, ax = regression_plot(x, y, order=1)

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0  # Has regression line

    def test_regression_plot_polynomial(self):
        """Test polynomial regression plot."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = x**2 + np.random.randn(50)

        fig, ax = regression_plot(x, y, order=2)

        assert fig is not None
        assert ax is not None

    def test_regression_plot_with_dataframe(self, sample_dataframe):
        """Test regression plot with DataFrame input."""
        fig, ax = regression_plot(x="value", y="value", data=sample_dataframe, order=1)

        assert fig is not None
        assert ax is not None


@pytest.mark.skipif(HAS_SEABORN, reason="Test seaborn import error")
class TestSeabornImportError:
    """Test that functions raise ImportError when seaborn is not installed."""

    def test_distribution_plot_import_error(self, sample_data):
        """Test distribution_plot raises ImportError without seaborn."""
        with pytest.raises(ImportError):
            distribution_plot(sample_data)

    def test_box_plot_import_error(self, sample_dataframe):
        """Test box_plot raises ImportError without seaborn."""
        with pytest.raises(ImportError):
            box_plot(sample_dataframe, x="group", y="value")

    def test_regression_plot_import_error(self):
        """Test regression_plot raises ImportError without seaborn."""
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        with pytest.raises(ImportError):
            regression_plot(x, y)


# ============================================================================
# Plotly Interactive Plots Tests
# ============================================================================


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not installed")
class TestPlotlyPlots:
    """Tests for Plotly interactive plots."""

    def test_interactive_histogram(self, sample_data):
        """Test interactive histogram creation."""
        fig = interactive_histogram(sample_data, bins=20, title="Test Histogram")

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "histogram"
        assert fig.layout.title.text == "Test Histogram"

    def test_interactive_histogram_custom_labels(self, sample_data):
        """Test interactive histogram with custom labels."""
        fig = interactive_histogram(sample_data, bins=30, xlabel="Custom X", ylabel="Custom Y")

        assert fig.layout.xaxis.title.text == "Custom X"
        assert fig.layout.yaxis.title.text == "Custom Y"

    def test_interactive_scatter_basic(self):
        """Test basic interactive scatter plot."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)

        fig = interactive_scatter(x, y, title="Test Scatter")

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "scatter"
        assert fig.data[0].mode == "markers"

    def test_interactive_scatter_with_color(self):
        """Test interactive scatter with color mapping."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)
        color = np.random.rand(100)

        fig = interactive_scatter(x, y, color=color)

        assert fig is not None
        assert fig.data[0].marker.color is not None

    def test_interactive_scatter_with_size(self):
        """Test interactive scatter with size mapping."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)
        size = np.random.rand(100) * 20

        fig = interactive_scatter(x, y, size=size)

        assert fig is not None
        assert fig.data[0].marker.size is not None

    def test_interactive_scatter_with_hover_data(self):
        """Test interactive scatter with hover_data (customdata branch)."""
        np.random.seed(42)
        x = np.random.rand(50)
        y = np.random.rand(50)
        hover = np.arange(50)

        fig = interactive_scatter(x, y, hover_data=hover)

        assert fig is not None
        assert fig.data[0].customdata is not None

    def test_interactive_line_single(self):
        """Test interactive line plot with single line."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        fig = interactive_line(x, y, title="Sin Wave")

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].mode == "lines"

    def test_interactive_line_multiple(self):
        """Test interactive line plot with multiple lines."""
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)

        fig = interactive_line(x, [y1, y2], labels=["sin", "cos"])

        assert fig is not None
        assert len(fig.data) == 2
        assert fig.data[0].name == "sin"
        assert fig.data[1].name == "cos"

    def test_interactive_line_custom_labels(self):
        """Test interactive line with custom axis labels."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        fig = interactive_line(x, y, xlabel="Time (s)", ylabel="Signal", title="Signal vs Time")

        assert fig.layout.xaxis.title.text == "Time (s)"
        assert fig.layout.yaxis.title.text == "Signal"

    def test_interactive_heatmap(self, sample_2d_data):
        """Test interactive heatmap creation."""
        fig = interactive_heatmap(sample_2d_data, title="Test Heatmap")

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_interactive_heatmap_with_labels(self, sample_2d_data):
        """Test interactive heatmap with axis labels."""
        x_labels = [f"X{i}" for i in range(10)]
        y_labels = [f"Y{i}" for i in range(10)]

        fig = interactive_heatmap(
            sample_2d_data, x_labels=x_labels, y_labels=y_labels, colorscale="RdBu"
        )

        assert fig is not None
        assert fig.data[0].x is not None
        assert fig.data[0].y is not None

    def test_interactive_3d_surface(self):
        """Test interactive 3D surface plot."""
        x = np.linspace(-5, 5, 30)
        y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        fig = interactive_3d_surface(X, Y, Z, title="3D Surface")

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "surface"

    def test_interactive_3d_surface_colorscale(self):
        """Test 3D surface with custom colorscale."""
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        fig = interactive_3d_surface(X, Y, Z, colorscale="Plasma")

        assert fig is not None
        # Plotly converts colorscale names to RGB tuples, so just check it's not None
        assert fig.data[0].colorscale is not None

    def test_export_interactive(self, temp_dir):
        """Test exporting Plotly figure to HTML."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        fig = interactive_line(x, y)
        output_file = temp_dir / "test_plot.html"

        export_interactive(fig, output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_export_interactive_creates_directory(self, temp_dir):
        """Test that export creates parent directories."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        fig = interactive_line(x, y)
        output_file = temp_dir / "subdir" / "nested" / "plot.html"

        export_interactive(fig, output_file)

        assert output_file.exists()


@pytest.mark.skipif(HAS_PLOTLY, reason="Test plotly import error")
class TestPlotlyImportError:
    """Test that functions raise ImportError when plotly is not installed."""

    def test_interactive_histogram_import_error(self, sample_data):
        """Test interactive_histogram raises ImportError without plotly."""
        with pytest.raises(ImportError):
            interactive_histogram(sample_data)

    def test_interactive_scatter_import_error(self):
        """Test interactive_scatter raises ImportError without plotly."""
        x = np.random.rand(10)
        y = np.random.rand(10)
        with pytest.raises(ImportError):
            interactive_scatter(x, y)

    def test_interactive_line_import_error(self):
        """Test interactive_line raises ImportError without plotly."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        with pytest.raises(ImportError):
            interactive_line(x, y)

    def test_interactive_heatmap_import_error(self, sample_2d_data):
        """Test interactive_heatmap raises ImportError without plotly."""
        with pytest.raises(ImportError):
            interactive_heatmap(sample_2d_data)

    def test_interactive_3d_surface_import_error(self):
        """Test interactive_3d_surface raises ImportError without plotly."""
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        with pytest.raises(ImportError):
            interactive_3d_surface(X, Y, Z)

    def test_export_interactive_import_error(self):
        """Test export_interactive raises ImportError without plotly."""
        # This would normally require a plotly figure, but we'll test with None
        with pytest.raises(ImportError):
            export_interactive(None, "test.html")


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SEABORN or not HAS_PLOTLY, reason="Seaborn or Plotly not installed")
class TestIntegration:
    """Integration tests combining different features."""

    def test_seaborn_and_plotly_both_work(self, sample_data):
        """Test that both Seaborn and Plotly can be used."""
        # Seaborn plot
        fig_sns, ax_sns = distribution_plot(sample_data, kind="hist")
        assert fig_sns is not None

        # Plotly plot
        fig_plotly = interactive_histogram(sample_data)
        assert fig_plotly is not None

    def test_export_workflow(self, temp_dir, sample_data):
        """Test complete workflow of creating and exporting plots."""
        # Create Plotly plot
        fig = interactive_histogram(
            sample_data, bins=30, title="Distribution", xlabel="Value", ylabel="Frequency"
        )

        # Export to HTML
        output_file = temp_dir / "distribution.html"
        export_interactive(fig, output_file)

        assert output_file.exists()

        # Check file contains expected content
        content = output_file.read_text()
        assert "plotly" in content.lower()
        assert "Distribution" in content


# ============================================================================
# M21 New Plot Types (seaborn + plotly)
# ============================================================================


class TestPairPlot:
    @pytest.mark.skipif(not HAS_SEABORN, reason="seaborn not installed")
    def test_basic(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [2, 4, 1, 3, 5],
                "C": [5, 3, 2, 4, 1],
            }
        )
        fig, ax, info = pair_plot(df)
        assert info["n_vars"] == 3
        plt.close("all")

    @pytest.mark.skipif(not HAS_SEABORN, reason="seaborn not installed")
    def test_with_hue(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6],
                "B": [2, 4, 1, 3, 5, 2],
                "cat": ["x", "y", "x", "y", "x", "y"],
            }
        )
        fig, ax, info = pair_plot(df, hue="cat")
        assert info["hue"] == "cat"
        plt.close("all")

    @pytest.mark.skipif(not HAS_SEABORN, reason="seaborn not installed")
    def test_selected_vars(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [2, 4, 1, 3, 5],
                "C": [5, 3, 2, 4, 1],
            }
        )
        fig, ax, info = pair_plot(df, vars=["A", "B"])
        assert info["n_vars"] == 2
        plt.close("all")

    def test_no_seaborn_raises(self):
        """ImportError raised when seaborn unavailable (mock HAS_SEABORN)."""
        import plottle.plotting as mp

        orig = mp.HAS_SEABORN
        mp.HAS_SEABORN = False
        try:
            with pytest.raises(ImportError, match="seaborn"):
                pair_plot(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        finally:
            mp.HAS_SEABORN = orig


class TestInteractive3DScatter:
    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_basic(self):
        x = np.linspace(0, 10, 30)
        y = np.sin(x)
        z = np.cos(x)
        fig, info = interactive_3d_scatter(x, y, z)
        assert info["n_points"] == 30

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_with_color(self):
        x = np.arange(20, dtype=float)
        y = np.arange(20, dtype=float)
        z = np.arange(20, dtype=float)
        color = np.arange(20, dtype=float)
        fig, info = interactive_3d_scatter(x, y, z, color=color, title="3D")
        assert info["n_points"] == 20

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_with_size(self):
        x = np.ones(10)
        y = np.ones(10)
        z = np.ones(10)
        size = np.linspace(3, 15, 10)
        fig, info = interactive_3d_scatter(x, y, z, size=size)
        assert info["n_points"] == 10

    def test_no_plotly_raises(self):
        """ImportError raised when plotly unavailable (mock HAS_PLOTLY)."""
        import plottle.plotting as mp

        orig = mp.HAS_PLOTLY
        mp.HAS_PLOTLY = False
        try:
            with pytest.raises(ImportError, match="plotly"):
                interactive_3d_scatter(
                    np.array([1.0]),
                    np.array([1.0]),
                    np.array([1.0]),
                )
        finally:
            mp.HAS_PLOTLY = orig


# ============================================================================
# Backlog Tests — interactive_ternary
# ============================================================================


class TestInteractiveTernary:
    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_basic(self):
        rng = np.random.default_rng(0)
        raw = rng.dirichlet([1, 1, 1], size=30)
        fig, info = interactive_ternary(raw[:, 0], raw[:, 1], raw[:, 2])
        assert info["n_points"] == 30

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_with_color(self):
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.2, 0.5, 0.3])
        c = np.array([0.3, 0.2, 0.5])
        color = np.array([1.0, 2.0, 3.0])
        fig, info = interactive_ternary(a, b, c, color=color, title="Test")
        assert info["n_points"] == 3

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_custom_labels(self):
        a, b, c = np.array([0.5]), np.array([0.3]), np.array([0.2])
        fig, info = interactive_ternary(a, b, c, a_label="Al2O3", b_label="SiO2", c_label="MgO")
        assert info["n_points"] == 1

    def test_no_plotly_raises(self):
        import plottle.plotting as mp

        orig = mp.HAS_PLOTLY
        mp.HAS_PLOTLY = False
        with pytest.raises(ImportError):
            interactive_ternary(np.array([1]), np.array([1]), np.array([1]))
        mp.HAS_PLOTLY = orig


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
