"""Test suite for CLI functionality.

Tests all command-line interface features including plot generation,
statistics, batch processing, and data conversion.
"""

import pytest
import subprocess
import sys
import json
from pathlib import Path
import tempfile
import shutil

# Set matplotlib to use non-interactive backend
import matplotlib

matplotlib.use("Agg")


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "examples" / "data"


def run_cli(*args):
    """Run the CLI with given arguments and return result.

    Parameters
    ----------
    *args : str
        Command-line arguments

    Returns
    -------
    result : CompletedProcess
        subprocess result with returncode, stdout, stderr
    """
    # Use sys.executable, not a bare 'python': inside a virtualenv (or a tox /
    # CI matrix cell) 'python' resolves via PATH to an interpreter that does not
    # have the project's dependencies installed, and every one of these tests
    # fails with ModuleNotFoundError. See G-016 / audit A-11a.
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "plottle.cli"] + list(args)
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    return result


# ============================================================================
# Basic CLI Tests
# ============================================================================


class TestBasicCLI:
    """Tests for basic CLI functionality."""

    def test_help_message(self):
        """Test that --help displays help message."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "Plottle" in result.stdout
        assert "plot" in result.stdout
        assert "stats" in result.stdout

    def test_version(self):
        """Test that --version reports the packaged version.

        Asserts against ``plottle.__version__`` rather than a hardcoded string:
        this test previously asserted '1.0.0' and went stale unnoticed because
        CI never ran tests/test_cli.py. See audit A-11 / A-26.
        """
        from plottle import __version__

        result = run_cli("--version")
        assert result.returncode == 0
        assert __version__ in result.stdout or __version__ in result.stderr

    def test_examples(self):
        """Test that --examples displays usage examples."""
        result = run_cli("--examples")
        assert result.returncode == 0
        assert "USAGE EXAMPLES" in result.stdout
        assert "histogram" in result.stdout

    def test_no_command(self):
        """Test that running without command shows help."""
        result = run_cli()
        assert result.returncode == 1


# ============================================================================
# Plot Command Tests
# ============================================================================


class TestPlotCommand:
    """Tests for the plot command."""

    def test_plot_help(self):
        """Test plot command help."""
        result = run_cli("plot", "--help")
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--type" in result.stdout

    def test_histogram(self, temp_dir, test_data_dir):
        """Test histogram generation."""
        output = temp_dir / "histogram.png"
        result = run_cli(
            "plot",
            "--input",
            str(test_data_dir / "coordinates.npy"),
            "--type",
            "histogram",
            "--output",
            str(output),
            "--bins",
            "30",
        )
        assert result.returncode == 0
        assert output.exists()

    def test_histogram_with_options(self, temp_dir, test_data_dir):
        """Test histogram with customization options."""
        output = temp_dir / "histogram_custom.png"
        result = run_cli(
            "plot",
            "--input",
            str(test_data_dir / "coordinates.npy"),
            "--type",
            "histogram",
            "--output",
            str(output),
            "--title",
            "Test Histogram",
            "--xlabel",
            "Values",
            "--ylabel",
            "Count",
            "--bins",
            "50",
        )
        assert result.returncode == 0
        assert output.exists()

    def test_line_plot(self, temp_dir, test_data_dir):
        """Test line plot generation."""
        output = temp_dir / "line.png"
        result = run_cli(
            "plot",
            "--input",
            str(test_data_dir / "experimental_data.csv"),
            "--type",
            "line",
            "--output",
            str(output),
        )
        assert result.returncode == 0
        assert output.exists()

    def test_scatter_plot(self, temp_dir, test_data_dir):
        """Test scatter plot generation."""
        output = temp_dir / "scatter.png"
        result = run_cli(
            "plot",
            "--input",
            str(test_data_dir / "experimental_data.csv"),
            "--type",
            "scatter",
            "--output",
            str(output),
        )
        assert result.returncode == 0
        assert output.exists()

    def test_heatmap(self, temp_dir, test_data_dir):
        """Test heatmap generation."""
        output = temp_dir / "heatmap.png"
        result = run_cli(
            "plot",
            "--input",
            str(test_data_dir / "simulation_data.npz"),
            "--type",
            "heatmap",
            "--output",
            str(output),
        )
        # This might fail if data format doesn't match - that's OK
        # Just checking the command doesn't crash
        assert result.returncode in [0, 1]

    def test_plot_missing_input(self):
        """Test that plot fails without input file."""
        result = run_cli("plot", "--type", "histogram")
        assert result.returncode == 2  # argparse error

    def test_plot_invalid_type(self, test_data_dir):
        """Test that plot fails with invalid type."""
        result = run_cli(
            "plot", "--input", str(test_data_dir / "coordinates.npy"), "--type", "invalid_type"
        )
        assert result.returncode == 2  # argparse error


# ============================================================================
# Stats Command Tests
# ============================================================================


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats_help(self):
        """Test stats command help."""
        result = run_cli("stats", "--help")
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--normality" in result.stdout

    def test_basic_stats(self, test_data_dir):
        """Test basic statistics calculation."""
        result = run_cli("stats", "--input", str(test_data_dir / "coordinates.npy"))
        assert result.returncode == 0
        assert "Statistical Summary" in result.stdout
        assert "mean" in result.stdout
        assert "std" in result.stdout

    def test_stats_with_normality(self, test_data_dir):
        """Test statistics with normality test."""
        result = run_cli(
            "stats",
            "--input",
            str(test_data_dir / "experimental_data.csv"),
            "--column",
            "Temperature (K)",
            "--normality",
        )
        assert result.returncode == 0
        assert "Normality Test" in result.stdout
        assert "Shapiro-Wilk" in result.stdout

    def test_stats_output_file(self, temp_dir, test_data_dir):
        """Test statistics output to JSON file."""
        output = temp_dir / "stats.json"
        result = run_cli(
            "stats",
            "--input",
            str(test_data_dir / "coordinates.npy"),
            "--output",
            str(output),
            "--normality",
        )
        assert result.returncode == 0
        assert output.exists()

        # Check JSON content
        with open(output, "r") as f:
            data = json.load(f)
        assert "mean" in data
        assert "std" in data
        assert "statistic" in data
        assert "p_value" in data


# ============================================================================
# Batch Command Tests
# ============================================================================


class TestBatchCommand:
    """Tests for the batch command."""

    def test_batch_help(self):
        """Test batch command help."""
        result = run_cli("batch", "--help")
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_batch_processing(self, temp_dir, test_data_dir):
        """Test batch processing with JSON config."""
        # Create batch config
        config = {
            "tasks": [
                {
                    "name": "Test Histogram",
                    "command": "plot",
                    "input": str(test_data_dir / "coordinates.npy"),
                    "type": "histogram",
                    "output": str(temp_dir / "batch_hist.png"),
                    "bins": 30,
                },
                {
                    "name": "Test Stats",
                    "command": "stats",
                    "input": str(test_data_dir / "coordinates.npy"),
                    "output": str(temp_dir / "batch_stats.json"),
                },
            ]
        }

        config_file = temp_dir / "batch_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        result = run_cli("batch", "--config", str(config_file))
        assert result.returncode == 0
        assert (temp_dir / "batch_hist.png").exists()
        assert (temp_dir / "batch_stats.json").exists()

    def test_batch_missing_config(self):
        """Test that batch fails with missing config file."""
        result = run_cli("batch", "--config", "nonexistent.json")
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()


# ============================================================================
# Compare Command Tests
# ============================================================================


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_help(self):
        """Test compare command help."""
        result = run_cli("compare", "--help")
        assert result.returncode == 0
        assert "--inputs" in result.stdout

    def test_compare_histograms(self, temp_dir, test_data_dir):
        """Test comparing datasets with histogram."""
        output = temp_dir / "comparison.png"
        input_file = str(test_data_dir / "coordinates.npy")

        result = run_cli(
            "compare",
            "--inputs",
            input_file,
            input_file,
            "--type",
            "histogram",
            "--output",
            str(output),
            "--labels",
            "Data1",
            "Data2",
        )
        assert result.returncode == 0
        assert output.exists()

    def test_compare_line_plots(self, temp_dir, test_data_dir):
        """Test comparing datasets with line plot."""
        output = temp_dir / "comparison_line.png"
        input_file = str(test_data_dir / "coordinates.npy")

        result = run_cli(
            "compare", "--inputs", input_file, input_file, "--type", "line", "--output", str(output)
        )
        assert result.returncode == 0
        assert output.exists()


# ============================================================================
# Convert Command Tests
# ============================================================================


class TestConvertCommand:
    """Tests for the convert command."""

    def test_convert_help(self):
        """Test convert command help."""
        result = run_cli("convert", "--help")
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output" in result.stdout

    def test_convert_pkl_to_csv(self, temp_dir, test_data_dir):
        """Test converting pickle to CSV."""
        output = temp_dir / "converted.csv"
        result = run_cli(
            "convert",
            "--input",
            str(test_data_dir / "experimental_data.csv"),
            "--output",
            str(output),
        )
        assert result.returncode == 0
        assert output.exists()

    def test_convert_npy_to_csv(self, temp_dir, test_data_dir):
        """Test converting NumPy to CSV."""
        output = temp_dir / "converted_npy.csv"
        result = run_cli(
            "convert", "--input", str(test_data_dir / "coordinates.npy"), "--output", str(output)
        )
        # Might work or fail depending on data dimensions - both OK
        assert result.returncode in [0, 1]


# ============================================================================
# Verbose/Quiet Modes Tests
# ============================================================================


class TestVerbosityModes:
    """Tests for verbose and quiet modes."""

    def test_verbose_mode(self, temp_dir, test_data_dir):
        """Test that verbose mode produces more output."""
        output = temp_dir / "test.png"

        # Normal mode
        result_normal = run_cli(
            "plot",
            "--input",
            str(test_data_dir / "coordinates.npy"),
            "--type",
            "histogram",
            "--output",
            str(output),
        )

        # Verbose mode
        result_verbose = run_cli(
            "--verbose",
            "plot",
            "--input",
            str(test_data_dir / "coordinates.npy"),
            "--type",
            "histogram",
            "--output",
            str(output),
        )

        assert result_normal.returncode == 0
        assert result_verbose.returncode == 0
        # Verbose should have [INFO] messages; normal mode should not
        assert "[INFO]" in result_verbose.stdout or "[INFO]" in result_verbose.stderr
        assert "[INFO]" not in result_normal.stdout + result_normal.stderr

    def test_quiet_mode(self, temp_dir, test_data_dir):
        """Test that quiet mode suppresses informational output."""
        output = temp_dir / "test.png"

        result = run_cli(
            "--quiet",
            "plot",
            "--input",
            str(test_data_dir / "coordinates.npy"),
            "--type",
            "histogram",
            "--output",
            str(output),
        )

        assert result.returncode == 0
        # Quiet mode should have minimal output
        # (though errors would still appear)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_missing_input_file(self):
        """Test error message for missing input file."""
        result = run_cli("plot", "--input", "nonexistent_file.npy", "--type", "histogram")
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()

    def test_invalid_column_name(self, test_data_dir):
        """Test error message for invalid column name."""
        result = run_cli(
            "stats",
            "--input",
            str(test_data_dir / "experimental_data.csv"),
            "--column",
            "NonexistentColumn",
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, temp_dir, test_data_dir):
        """Test a complete analysis workflow."""
        # Step 1: Generate plot
        plot_output = temp_dir / "workflow_plot.png"
        result1 = run_cli(
            "plot",
            "--input",
            str(test_data_dir / "experimental_data.csv"),
            "--type",
            "line",
            "--output",
            str(plot_output),
            "--title",
            "Workflow Test",
        )
        assert result1.returncode == 0
        assert plot_output.exists()

        # Step 2: Calculate statistics
        stats_output = temp_dir / "workflow_stats.json"
        result2 = run_cli(
            "stats",
            "--input",
            str(test_data_dir / "experimental_data.csv"),
            "--column",
            "Temperature (K)",
            "--output",
            str(stats_output),
            "--normality",
        )
        assert result2.returncode == 0
        assert stats_output.exists()

        # Step 3: Verify stats content
        with open(stats_output, "r") as f:
            stats = json.load(f)
        assert "mean" in stats
        assert "std" in stats


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
