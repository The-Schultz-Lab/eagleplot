"""Unit tests for the plotting module.

This module tests all plotting functions including:
- Figure management
- Core plots (histogram, line, scatter)
- Advanced plots (heatmap, contour)
- Styling utilities
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import sys
import tempfile
import shutil

# Import functions to test
sys.path.insert(0, str(Path(__file__).parent.parent))
from plottle.plotting import (
    # Figure management
    create_figure,
    configure_axes,
    save_figure,
    # Core plots
    histogram,
    line_plot,
    scatter_plot,
    # Advanced plots
    heatmap,
    contour_plot,
    # M14 extended plot types
    bar_chart,
    waterfall_plot,
    dual_axis_plot,
    # M21 new plot types
    z_colored_scatter,
    bubble_chart,
    polar_plot,
    histogram_2d,
    # Backlog new plot types
    scatter_with_regression,
    residual_plot,
    # Specialty plot types
    inset_plot,
    # Styling
    set_style,
    get_color_palette,
    apply_publication_style,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)
    # Close all matplotlib figures
    plt.close("all")


@pytest.fixture
def sample_data():
    """Sample 1D data for testing."""
    np.random.seed(42)
    return np.random.normal(5, 2, 100)


@pytest.fixture
def sample_xy_data():
    """Sample x-y data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 3 + np.random.normal(0, 1, 50)
    return x, y


@pytest.fixture
def sample_2d_data():
    """Sample 2D data for testing."""
    np.random.seed(42)
    return np.random.rand(10, 10)


@pytest.fixture
def sample_meshgrid():
    """Sample meshgrid data for contour plots."""
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    return X, Y, Z


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Cleanup matplotlib plots after each test."""
    yield
    plt.close("all")


# ============================================================================
# Figure Management Tests
# ============================================================================


class TestFigureManagement:
    """Tests for figure management functions."""

    def test_create_figure_default(self):
        """Test creating figure with default parameters."""
        fig, ax = create_figure()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6

    def test_create_figure_custom_size(self):
        """Test creating figure with custom size."""
        fig, ax = create_figure(figsize=(10, 5), dpi=150)

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 5
        assert fig.dpi == 150

    def test_configure_axes_labels(self):
        """Test configuring axes with labels."""
        fig, ax = create_figure()
        configure_axes(ax, xlabel="X Label", ylabel="Y Label", title="Title")

        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        assert ax.get_title() == "Title"

    def test_configure_axes_limits(self):
        """Test configuring axes limits."""
        fig, ax = create_figure()
        configure_axes(ax, xlim=(0, 10), ylim=(-5, 5))

        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (-5, 5)

    def test_configure_axes_grid(self):
        """Test grid configuration."""
        fig, ax = create_figure()
        configure_axes(ax, grid=True)

        # Grid should be enabled - check via gridlines
        assert len(ax.xaxis.get_gridlines()) > 0

    def test_configure_axes_extra_kwargs(self):
        """Test configure_axes passes unrecognised kwargs via set_<key>."""
        fig, ax = create_figure()
        configure_axes(ax, facecolor="lightgrey")

        # set_facecolor goes through the kwargs loop (lines 129-130)
        assert ax.get_facecolor() is not None

    def test_save_figure_png(self, temp_dir):
        """Test saving figure as PNG."""
        fig, ax = create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        filepath = temp_dir / "test.png"
        save_figure(fig, filepath)

        assert filepath.exists()
        assert filepath.suffix == ".png"

    def test_save_figure_pdf(self, temp_dir):
        """Test saving figure as PDF."""
        fig, ax = create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        filepath = temp_dir / "test.pdf"
        save_figure(fig, filepath, format="pdf")

        assert filepath.exists()

    def test_save_figure_creates_directory(self, temp_dir):
        """Test that save_figure creates parent directories."""
        fig, ax = create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        filepath = temp_dir / "subdir" / "nested" / "test.png"
        save_figure(fig, filepath)

        assert filepath.exists()


# ============================================================================
# Core Plotting Tests
# ============================================================================


class TestHistogram:
    """Tests for histogram function."""

    def test_histogram_basic(self, sample_data):
        """Test basic histogram creation."""
        fig, ax, info = histogram(sample_data)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert isinstance(info, dict)
        assert "counts" in info
        assert "bins" in info
        assert "mean" in info
        assert "std" in info

    def test_histogram_statistics(self, sample_data):
        """Test histogram statistics."""
        fig, ax, info = histogram(sample_data)

        # Check statistics are reasonable
        assert np.isclose(info["mean"], np.mean(sample_data))
        assert np.isclose(info["std"], np.std(sample_data))

    def test_histogram_custom_bins(self, sample_data):
        """Test histogram with custom bins."""
        fig, ax, info = histogram(sample_data, bins=20)

        # Should have 20 bins
        assert len(info["counts"]) == 20

    def test_histogram_with_labels(self, sample_data):
        """Test histogram with labels."""
        fig, ax, info = histogram(
            sample_data, xlabel="Value", ylabel="Count", title="Test Histogram"
        )

        assert ax.get_xlabel() == "Value"
        assert ax.get_ylabel() == "Count"
        assert ax.get_title() == "Test Histogram"

    def test_histogram_density(self, sample_data):
        """Test histogram with density=True."""
        fig, ax, info = histogram(sample_data, density=True)

        # Y-label should change to Density
        assert ax.get_ylabel() == "Density"


class TestLinePlot:
    """Tests for line_plot function."""

    def test_line_plot_basic(self, sample_xy_data):
        """Test basic line plot."""
        x, y = sample_xy_data
        fig, ax = line_plot(x, y)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1

    def test_line_plot_multiple_lines(self, sample_xy_data):
        """Test line plot with multiple lines."""
        x, y = sample_xy_data
        y2 = y * 2

        fig, ax = line_plot(x, [y, y2])

        assert len(ax.lines) == 2

    def test_line_plot_with_labels(self, sample_xy_data):
        """Test line plot with labels and legend."""
        x, y = sample_xy_data
        y2 = y * 2

        fig, ax = line_plot(
            x, [y, y2], labels=["Line 1", "Line 2"], xlabel="X", ylabel="Y", title="Test Plot"
        )

        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_title() == "Test Plot"
        assert ax.get_legend() is not None

    def test_line_plot_custom_colors(self, sample_xy_data):
        """Test line plot with custom colors."""
        x, y = sample_xy_data
        y2 = y * 2

        fig, ax = line_plot(x, [y, y2], colors=["red", "blue"])

        assert len(ax.lines) == 2

    def test_line_plot_with_markers(self, sample_xy_data):
        """Test line plot with markers."""
        x, y = sample_xy_data

        fig, ax = line_plot(x, y, markers=["o"])

        assert len(ax.lines) == 1

    def test_line_plot_tuple_y(self, sample_xy_data):
        """Test line_plot when y is a tuple (non-ndarray, non-list branch)."""
        x, y = sample_xy_data

        fig, ax = line_plot(x, tuple(y))

        assert isinstance(fig, Figure)
        assert len(ax.lines) == 1


class TestScatterPlot:
    """Tests for scatter_plot function."""

    def test_scatter_plot_basic(self):
        """Test basic scatter plot."""
        np.random.seed(42)
        x = np.random.rand(50)
        y = np.random.rand(50)

        fig, ax = scatter_plot(x, y)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0  # Has scatter collection

    def test_scatter_plot_with_labels(self):
        """Test scatter plot with labels."""
        np.random.seed(42)
        x = np.random.rand(50)
        y = np.random.rand(50)

        fig, ax = scatter_plot(x, y, xlabel="X Data", ylabel="Y Data", title="Scatter Test")

        assert ax.get_xlabel() == "X Data"
        assert ax.get_ylabel() == "Y Data"
        assert ax.get_title() == "Scatter Test"

    def test_scatter_plot_color_mapping(self):
        """Test scatter plot with color array."""
        np.random.seed(42)
        x = np.random.rand(50)
        y = np.random.rand(50)
        c = x + y

        fig, ax = scatter_plot(x, y, color=c, colorbar=True)

        # Should have collections (scatter + colorbar)
        assert len(ax.collections) > 0

    def test_scatter_plot_size_array(self):
        """Test scatter plot with size array."""
        np.random.seed(42)
        x = np.random.rand(50)
        y = np.random.rand(50)
        s = np.random.rand(50) * 100

        fig, ax = scatter_plot(x, y, size=s)

        assert len(ax.collections) > 0


# ============================================================================
# Advanced Plotting Tests
# ============================================================================


class TestHeatmap:
    """Tests for heatmap function."""

    def test_heatmap_basic(self, sample_2d_data):
        """Test basic heatmap creation."""
        fig, ax = heatmap(sample_2d_data)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.images) > 0  # Has image

    def test_heatmap_with_labels(self, sample_2d_data):
        """Test heatmap with labels."""
        fig, ax = heatmap(sample_2d_data, xlabel="Columns", ylabel="Rows", title="Heatmap Test")

        assert ax.get_xlabel() == "Columns"
        assert ax.get_ylabel() == "Rows"
        assert ax.get_title() == "Heatmap Test"

    def test_heatmap_custom_colormap(self, sample_2d_data):
        """Test heatmap with custom colormap."""
        fig, ax = heatmap(sample_2d_data, cmap="hot")

        assert len(ax.images) > 0

    def test_heatmap_with_tick_labels(self, sample_2d_data):
        """Test heatmap with custom tick labels."""
        xlabels = [f"C{i}" for i in range(10)]
        ylabels = [f"R{i}" for i in range(10)]

        fig, ax = heatmap(sample_2d_data, xticklabels=xlabels, yticklabels=ylabels)

        assert len(ax.get_xticklabels()) == 10
        assert len(ax.get_yticklabels()) == 10

    def test_heatmap_1d_data_raises_error(self):
        """Test that 1D data raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            heatmap(data)


class TestContourPlot:
    """Tests for contour_plot function."""

    def test_contour_plot_filled(self, sample_meshgrid):
        """Test filled contour plot."""
        X, Y, Z = sample_meshgrid
        fig, ax = contour_plot(X, Y, Z, filled=True)

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) > 0  # Has contour collections

    def test_contour_plot_lines(self, sample_meshgrid):
        """Test line contour plot."""
        X, Y, Z = sample_meshgrid
        fig, ax = contour_plot(X, Y, Z, filled=False)

        assert len(ax.collections) > 0

    def test_contour_plot_with_labels(self, sample_meshgrid):
        """Test contour plot with labels."""
        X, Y, Z = sample_meshgrid
        fig, ax = contour_plot(X, Y, Z, xlabel="X Axis", ylabel="Y Axis", title="Contour Test")

        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"
        assert ax.get_title() == "Contour Test"

    def test_contour_plot_custom_levels(self, sample_meshgrid):
        """Test contour plot with custom levels."""
        X, Y, Z = sample_meshgrid
        fig, ax = contour_plot(X, Y, Z, levels=20)

        # Should create contours
        assert len(ax.collections) > 0

    def test_contour_plot_mismatched_shapes_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        X = np.random.rand(10, 10)
        Y = np.random.rand(10, 10)
        Z = np.random.rand(5, 5)  # Different shape

        with pytest.raises(ValueError):
            contour_plot(X, Y, Z)


# ============================================================================
# Styling Utilities Tests
# ============================================================================


class TestStylingUtilities:
    """Tests for styling utility functions."""

    def test_set_style_default(self):
        """Test setting default style."""
        set_style("default")
        # Should not raise error

    def test_set_style_seaborn(self):
        """Test setting seaborn style (if available)."""
        try:
            set_style("ggplot")
            # Should work if style is available
        except ValueError:
            # OK if style not available
            pass

    def test_set_style_invalid_raises_error(self):
        """Test that invalid style raises ValueError."""
        with pytest.raises(ValueError):
            set_style("nonexistent_style_12345")

    def test_get_color_palette_default(self):
        """Test getting default color palette."""
        colors = get_color_palette()

        assert isinstance(colors, list)
        assert len(colors) == 10
        # Colors should be hex codes
        assert all(c.startswith("#") for c in colors)

    def test_get_color_palette_custom(self):
        """Test getting custom color palette."""
        colors = get_color_palette("tab20", n_colors=5)

        assert len(colors) == 5
        assert all(c.startswith("#") for c in colors)

    def test_get_color_palette_continuous(self):
        """Test color palette with a continuous colormap (no .colors attribute)."""
        colors = get_color_palette("hot", n_colors=6)

        assert len(colors) == 6
        assert all(c.startswith("#") for c in colors)

    def test_apply_publication_style_single_ax(self, sample_xy_data):
        """Test applying publication style to single axes."""
        x, y = sample_xy_data
        fig, ax = line_plot(x, y)

        # Should not raise error
        apply_publication_style(fig, ax)

    def test_apply_publication_style_multiple_ax(self, sample_xy_data):
        """Test applying publication style to multiple axes."""
        x, y = sample_xy_data

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(x, y)
        ax2.plot(x, y * 2)

        # Should handle list of axes
        apply_publication_style(fig, [ax1, ax2])


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_create_and_save_histogram(self, temp_dir, sample_data):
        """Test creating and saving a histogram."""
        fig, ax, info = histogram(
            sample_data, bins=30, xlabel="Value", ylabel="Frequency", title="Test Histogram"
        )

        filepath = temp_dir / "histogram.png"
        save_figure(fig, filepath, dpi=150)

        assert filepath.exists()

    def test_create_and_save_line_plot(self, temp_dir, sample_xy_data):
        """Test creating and saving a line plot."""
        x, y = sample_xy_data

        fig, ax = line_plot(x, y, xlabel="X", ylabel="Y", title="Line Plot")

        apply_publication_style(fig, ax)

        filepath = temp_dir / "lineplot.pdf"
        save_figure(fig, filepath, format="pdf")

        assert filepath.exists()

    def test_multiple_plots_same_figure(self, sample_xy_data):
        """Test creating multiple subplots."""
        x, y = sample_xy_data

        # Create figure with subplots manually
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Use configure_axes on both
        ax1.plot(x, y)
        configure_axes(ax1, xlabel="X", ylabel="Y", title="Plot 1")

        ax2.scatter(x, y)
        configure_axes(ax2, xlabel="X", ylabel="Y", title="Plot 2")

        assert len(fig.axes) == 2

    def test_styled_scatter_with_colormap(self, temp_dir):
        """Test styled scatter plot with color mapping."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)
        c = x + y

        fig, ax = scatter_plot(
            x, y, color=c, colorbar=True, xlabel="X", ylabel="Y", title="Colored Scatter"
        )

        filepath = temp_dir / "scatter.png"
        save_figure(fig, filepath)

        assert filepath.exists()


# ============================================================================
# Error Bar Tests (M14)
# ============================================================================


class TestErrorBars:
    """Tests for error bar support added in M14."""

    def test_scatter_plot_with_yerr(self, sample_xy_data):
        """Scatter plot renders with Y error bars."""
        x, y = sample_xy_data
        yerr = 0.1 * np.ones_like(y)
        fig, ax = scatter_plot(x, y, yerr=yerr)
        assert isinstance(fig, Figure)
        # errorbar containers added
        assert len(ax.containers) > 0 or len(ax.lines) > 0

    def test_scatter_plot_with_xerr(self, sample_xy_data):
        """Scatter plot renders with X error bars."""
        x, y = sample_xy_data
        xerr = 0.05 * np.ones_like(x)
        fig, ax = scatter_plot(x, y, xerr=xerr)
        assert isinstance(fig, Figure)

    def test_scatter_plot_no_errbar(self, sample_xy_data):
        """Scatter plot without error bars is unchanged."""
        x, y = sample_xy_data
        fig, ax = scatter_plot(x, y)
        assert isinstance(fig, Figure)

    def test_line_plot_with_yerr_single_series(self, sample_xy_data):
        """Line plot with error bars for a single series."""
        x, y = sample_xy_data
        err = 0.1 * np.ones_like(y)
        fig, ax = line_plot(x, y, yerr=err)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_line_plot_with_yerr_multiple_series(self, sample_xy_data):
        """Error bars apply only to first series when given as 1-D."""
        x, y = sample_xy_data
        y2 = y * 2
        err = 0.1 * np.ones_like(y)
        fig, ax = line_plot(x, [y, y2], yerr=err)
        assert isinstance(fig, Figure)

    def test_line_plot_with_yerr_list(self, sample_xy_data):
        """Per-series error bars supplied as a list."""
        x, y = sample_xy_data
        y2 = y * 2
        err1 = 0.1 * np.ones_like(y)
        err2 = 0.2 * np.ones_like(y)
        fig, ax = line_plot(x, [y, y2], yerr=[err1, err2])
        assert isinstance(fig, Figure)


# ============================================================================
# Bar Chart Tests (M14)
# ============================================================================


class TestBarChart:
    """Tests for bar_chart function."""

    def test_simple_bar(self):
        """Simple bar chart returns (fig, ax, info)."""
        cats = ["A", "B", "C", "D"]
        vals = np.array([3.0, 5.0, 2.0, 4.0])
        fig, ax, info = bar_chart(cats, vals)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert info["kind"] == "simple"
        assert info["n_categories"] == 4

    def test_simple_bar_with_yerr(self):
        """Simple bar chart with Y error bars."""
        cats = ["A", "B", "C"]
        vals = np.array([1.0, 2.0, 3.0])
        err = np.array([0.1, 0.2, 0.15])
        fig, ax, info = bar_chart(cats, vals, yerr=err)
        assert isinstance(fig, Figure)

    def test_grouped_bar(self):
        """Grouped bar chart with hue."""
        cats = ["Q1", "Q2", "Q3"]
        vals = np.array([[1.0, 2.0, 3.0], [4.0, 3.0, 2.0]])
        hue = ["Group A", "Group B"]
        fig, ax, info = bar_chart(cats, vals, hue=hue, kind="grouped")
        assert isinstance(fig, Figure)
        assert info["kind"] == "grouped"
        assert info["n_groups"] == 2

    def test_stacked_bar(self):
        """Stacked bar chart."""
        cats = ["X", "Y", "Z"]
        vals = np.array([[1.0, 1.5, 2.0], [2.0, 1.0, 0.5]])
        hue = ["Layer 1", "Layer 2"]
        fig, ax, info = bar_chart(cats, vals, hue=hue, kind="stacked")
        assert isinstance(fig, Figure)
        assert info["kind"] == "stacked"

    def test_bar_invalid_kind(self):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown bar chart kind"):
            bar_chart(["A"], [1.0], kind="invalid")

    def test_bar_with_labels(self):
        """Bar chart respects xlabel, ylabel, title."""
        cats = ["a", "b"]
        vals = np.array([1.0, 2.0])
        fig, ax, info = bar_chart(cats, vals, xlabel="Category", ylabel="Count", title="Test")
        assert ax.get_xlabel() == "Category"
        assert ax.get_ylabel() == "Count"
        assert ax.get_title() == "Test"


# ============================================================================
# Waterfall Plot Tests (M14)
# ============================================================================


class TestWaterfallPlot:
    """Tests for waterfall_plot function."""

    def test_basic_waterfall(self):
        """Waterfall plot from 2-D array."""
        np.random.seed(0)
        x = np.linspace(0, 10, 100)
        y_mat = np.random.rand(5, 100)
        fig, ax, info = waterfall_plot(x, y_mat)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert info["n_traces"] == 5
        assert info["offset"] > 0

    def test_waterfall_auto_offset(self):
        """Auto offset falls back to 1.0 when all signals are flat."""
        x = np.linspace(0, 1, 50)
        # Each row is constant → ptp = 0 → fallback offset = 1.0
        y_mat = np.ones((3, 50)) * np.array([[1], [2], [3]])
        fig, ax, info = waterfall_plot(x, y_mat, offset="auto")
        assert info["offset"] == pytest.approx(1.0)

    def test_waterfall_manual_offset(self):
        """Manual offset is respected."""
        x = np.linspace(0, 1, 50)
        y_mat = np.random.rand(4, 50)
        fig, ax, info = waterfall_plot(x, y_mat, offset=2.5)
        assert info["offset"] == pytest.approx(2.5)

    def test_waterfall_with_labels(self):
        """Labels are set when provided."""
        x = np.linspace(0, 1, 20)
        y_mat = np.random.rand(3, 20)
        labels = ["t=0", "t=1", "t=2"]
        fig, ax, info = waterfall_plot(x, y_mat, labels=labels)
        assert isinstance(fig, Figure)

    def test_waterfall_1d_input(self):
        """1-D y input is treated as a single trace."""
        x = np.linspace(0, 1, 30)
        y = np.sin(x)
        fig, ax, info = waterfall_plot(x, y)
        assert info["n_traces"] == 1


# ============================================================================
# Dual Axis Plot Tests (M14)
# ============================================================================


class TestDualAxisPlot:
    """Tests for dual_axis_plot function."""

    def test_basic_dual_axis(self, sample_xy_data):
        """Dual axis plot returns (fig, ax, info) with secondary axis."""
        x, y = sample_xy_data
        y2 = np.exp(0.1 * x)
        fig, ax, info = dual_axis_plot(x, y, y2)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "ax2" in info
        assert info["ax2"] is not ax

    def test_dual_axis_with_labels(self, sample_xy_data):
        """Y-axis labels are applied to the correct axes."""
        x, y = sample_xy_data
        y2 = y**2
        fig, ax, info = dual_axis_plot(
            x,
            y,
            y2,
            xlabel="Time",
            ylabel1="Signal A",
            ylabel2="Signal B",
            title="Dual Axis",
        )
        assert ax.get_ylabel() == "Signal A"
        assert info["ax2"].get_ylabel() == "Signal B"
        assert ax.get_title() == "Dual Axis"

    def test_dual_axis_colors(self, sample_xy_data):
        """Custom colors are accepted without error."""
        x, y = sample_xy_data
        fig, ax, info = dual_axis_plot(
            x,
            y,
            y * 2,
            color1="navy",
            color2="darkred",
        )
        assert isinstance(fig, Figure)

    def test_dual_axis_linestyles(self, sample_xy_data):
        """Custom linestyles are accepted without error."""
        x, y = sample_xy_data
        fig, ax, info = dual_axis_plot(
            x,
            y,
            y * 0.5,
            linestyle1="--",
            linestyle2=":",
        )
        assert isinstance(fig, Figure)


# ============================================================================
# M21 New Plot Types
# ============================================================================


class TestZColoredScatter:
    def test_basic(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        z = np.cos(x)
        fig, ax, info = z_colored_scatter(x, y, z)
        assert isinstance(fig, Figure)
        assert info["n_points"] == 50
        plt.close("all")

    def test_no_colorbar(self):
        x = np.random.rand(20)
        y = np.random.rand(20)
        z = np.random.rand(20)
        fig, ax, info = z_colored_scatter(x, y, z, colorbar=False, title="Z-scatter")
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_custom_cmap(self):
        x, y, z = np.arange(10), np.arange(10), np.arange(10)
        fig, ax, info = z_colored_scatter(x, y, z, cmap="plasma")
        assert "z_min" in info
        assert "z_max" in info
        plt.close("all")

    def test_info_keys(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([0.0, 5.0, 10.0])
        _, _, info = z_colored_scatter(x, y, z)
        assert info["z_min"] == pytest.approx(0.0)
        assert info["z_max"] == pytest.approx(10.0)
        plt.close("all")


class TestBubbleChart:
    def test_basic(self):
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        sizes = np.abs(np.cos(x)) * 200 + 20
        fig, ax, info = bubble_chart(x, y, sizes)
        assert isinstance(fig, Figure)
        assert info["n_points"] == 20
        plt.close("all")

    def test_with_color(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        s = np.ones(10) * 100
        z = np.arange(10, dtype=float)
        fig, ax, info = bubble_chart(x, y, s, z=z)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_size_scale(self):
        x, y, s = np.ones(5), np.ones(5), np.ones(5) * 50
        fig, ax, info = bubble_chart(x, y, s, size_scale=2.0)
        assert info["size_min"] == pytest.approx(50.0)
        assert info["size_max"] == pytest.approx(50.0)
        plt.close("all")

    def test_info_keys(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        s = np.array([10.0, 30.0])
        _, _, info = bubble_chart(x, y, s)
        assert info["size_min"] == pytest.approx(10.0)
        assert info["size_max"] == pytest.approx(30.0)
        plt.close("all")


class TestPolarPlot:
    def test_basic(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.abs(np.sin(2 * theta))
        fig, ax, info = polar_plot(theta, r)
        assert isinstance(fig, Figure)
        assert info["n_points"] == 100
        plt.close("all")

    def test_with_fill(self):
        theta = np.linspace(0, 2 * np.pi, 50)
        r = np.ones(50)
        fig, ax, info = polar_plot(theta, r, fill=True, title="Circle")
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_rmax(self):
        theta = np.array([0.0, np.pi / 2, np.pi])
        r = np.array([1.0, 2.0, 3.0])
        _, _, info = polar_plot(theta, r)
        assert info["r_max"] == pytest.approx(3.0)
        plt.close("all")

    def test_clockwise_direction(self):
        theta = np.linspace(0, 2 * np.pi, 36)
        r = np.ones(36)
        fig, ax, info = polar_plot(theta, r, theta_direction=1)
        assert isinstance(fig, Figure)
        plt.close("all")


class TestHistogram2D:
    def test_hist2d(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        fig, ax, info = histogram_2d(x, y, bins=20)
        assert isinstance(fig, Figure)
        assert info["mode"] == "hist2d"
        assert info["n_points"] == 200
        plt.close("all")

    def test_hexbin(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 10, 300)
        y = rng.uniform(0, 10, 300)
        fig, ax, info = histogram_2d(x, y, mode="hexbin", gridsize=15)
        assert info["mode"] == "hexbin"
        plt.close("all")

    def test_no_colorbar(self):
        x = np.arange(30, dtype=float)
        y = np.arange(30, dtype=float)
        fig, ax, info = histogram_2d(x, y, colorbar=False, title="2D Hist")
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_custom_cmap(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        fig, ax, info = histogram_2d(x, y, cmap="plasma")
        assert isinstance(fig, Figure)
        plt.close("all")


# ============================================================================
# Backlog Tests — scatter_with_regression, residual_plot
# ============================================================================


class TestScatterWithRegression:
    def test_basic(self):
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.default_rng(0).normal(0, 0.5, 50)
        fig, ax, info = scatter_with_regression(x, y)
        assert isinstance(fig, Figure)
        assert "slope" in info and "r_squared" in info
        assert 0 <= info["r_squared"] <= 1
        plt.close("all")

    def test_no_ci(self):
        x, y = np.arange(20, dtype=float), np.arange(20, dtype=float) * 3
        fig, ax, info = scatter_with_regression(x, y, show_ci=False, show_equation=False)
        assert info["r_squared"] > 0.99
        plt.close("all")

    def test_info_keys(self):
        x, y = np.arange(10, dtype=float), np.arange(10, dtype=float)
        _, _, info = scatter_with_regression(x, y)
        for key in ("slope", "intercept", "r_squared", "p_value", "stderr"):
            assert key in info
        plt.close("all")


class TestResidualPlot:
    def test_basic(self):
        x = np.linspace(0, 10, 40)
        y_actual = 2 * x + np.random.default_rng(1).normal(0, 0.3, 40)
        y_fitted = 2 * x
        fig, ax, info = residual_plot(x, y_actual, y_fitted)
        assert isinstance(fig, Figure)
        assert "rmse" in info
        plt.close("all")

    def test_vs_fitted(self):
        x = np.arange(20, dtype=float)
        ya = x + 0.1
        yf = x
        fig, ax, info = residual_plot(x, ya, yf, vs_fitted=True)
        assert info["mean_residual"] == pytest.approx(0.1)
        plt.close("all")

    def test_no_zero_line(self):
        x, ya, yf = np.ones(5), np.ones(5) * 2, np.ones(5)
        fig, ax, info = residual_plot(x, ya, yf, show_zero_line=False)
        assert info["n_points"] == 5
        plt.close("all")


class TestInsetPlot:
    """Tests for the inset_plot specialty function."""

    def test_basic_returns_tuple(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        result = inset_plot(x, y, x[10:30], y[10:30])
        assert isinstance(result, tuple)
        assert len(result) == 3
        plt.close("all")

    def test_basic_info_keys(self):
        plt.close("all")
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig, ax, info = inset_plot(x, y, x[10:30], y[10:30])
        assert info["n_points"] == 50
        assert info["n_inset_points"] == 20
        assert "inset_bounds" in info
        plt.close("all")

    def test_default_inset_bounds(self):
        x = np.linspace(0, 5, 30)
        y = np.cos(x)
        fig, ax, info = inset_plot(x, y, x, y)
        assert info["inset_bounds"] == [0.55, 0.55, 0.4, 0.35]
        plt.close("all")

    def test_custom_bounds(self):
        x = np.linspace(0, 5, 30)
        y = np.cos(x)
        bounds = [0.1, 0.1, 0.3, 0.3]
        fig, ax, info = inset_plot(x, y, x, y, inset_bounds=bounds)
        assert info["inset_bounds"] == bounds
        plt.close("all")

    def test_indicate_region(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        fig, ax, info = inset_plot(x, y, x[20:40], y[20:40], indicate_region=(2.0, 4.0))
        assert fig is not None
        plt.close("all")

    def test_labels(self):
        x = np.linspace(0, 6, 60)
        y = np.cos(x)
        fig, ax, info = inset_plot(
            x,
            y,
            x,
            y,
            title="Test",
            xlabel="X",
            ylabel="Y",
            inset_xlabel="ix",
            inset_ylabel="iy",
        )
        assert ax.get_title() == "Test"
        assert ax.get_xlabel() == "X"
        plt.close("all")

    def test_returns_fig_and_axes(self):
        x = np.linspace(0, 4, 40)
        y = x**2
        fig, ax, info = inset_plot(x, y, x[:20], y[:20])
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_figsize_parameter(self):
        x = np.linspace(0, 3, 30)
        y = np.exp(x)
        fig, ax, info = inset_plot(x, y, x, y, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert abs(w - 10.0) < 0.1
        assert abs(h - 8.0) < 0.1
        plt.close("all")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
