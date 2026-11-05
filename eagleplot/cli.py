"""Command-Line Interface for EaglePlot.

This module provides a comprehensive command-line interface for the EaglePlot
package, allowing users to perform data visualization, analysis, and batch processing
from the terminal.

Usage
-----
    python cli.py [command] [options]

Commands
--------
    plot        Create a plot from data file
    stats       Calculate and display statistics
    batch       Process multiple files using config
    compare     Compare multiple datasets
    convert     Convert data between formats

Examples
--------
    # Create a histogram
    $ python cli.py plot --input data.npy --type histogram --output plot.png

    # Generate statistics
    $ python cli.py stats --input data.csv --column Temperature

    # Batch processing
    $ python cli.py batch --config batch_config.json

    # Compare datasets
    $ python cli.py compare --inputs data1.csv data2.csv --type line

    # Convert data format
    $ python cli.py convert --input data.pkl --output data.csv
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

# Import module functions
from eagleplot import __version__
from eagleplot.io import load_data, save_data
from eagleplot.math import calculate_statistics, check_normality
from eagleplot.plotting import (
    histogram,
    line_plot,
    scatter_plot,
    heatmap,
    contour_plot,
    save_figure,
)

# Optional imports for advanced plotting
try:
    from eagleplot.plotting import (
        distribution_plot,
        box_plot,
        regression_plot,
        interactive_histogram,
        interactive_scatter,
        interactive_line,
        export_interactive,
    )

    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


class PlottingHelperCLI:
    """Main CLI application class."""

    def __init__(self):
        """Initialize the CLI application."""
        self.verbose = False
        self.quiet = False

    def log(self, message: str, level: str = "info"):
        """Print log messages based on verbosity settings.

        Parameters
        ----------
        message : str
            Message to print
        level : str
            Message level: 'info', 'warning', 'error'
        """
        if self.quiet and level != "error":
            return

        prefix = {"info": "[INFO]", "warning": "[WARNING]", "error": "[ERROR]"}.get(level, "")

        if self.verbose or level in ["warning", "error"]:
            print(f"{prefix} {message}")

    def load_file(self, filepath: str, column: Optional[str] = None) -> Any:
        """Load data from file.

        Parameters
        ----------
        filepath : str
            Path to data file
        column : str, optional
            Column name to extract from DataFrame

        Returns
        -------
        data : Any
            Loaded data
        """
        self.log(f"Loading data from {filepath}")
        data = load_data(filepath)

        # Extract column if specified
        if column and isinstance(data, pd.DataFrame):
            if column not in data.columns:
                raise ValueError(
                    f"Column '{column}' not found in DataFrame. "
                    f"Available columns: {list(data.columns)}"
                )
            data = data[column].values
            self.log(f"Extracted column: {column}")

        return data

    def create_plot(self, args):
        """Create a plot based on command-line arguments.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments
        """
        # Load data
        column = getattr(args, "column", None)
        data = self.load_file(args.input, column)

        # Prepare plot parameters
        plot_kwargs = {
            "title": getattr(args, "title", None),
            "xlabel": getattr(args, "xlabel", None),
            "ylabel": getattr(args, "ylabel", None),
        }

        # Add optional parameters if provided
        color = getattr(args, "color", None)
        if color:
            plot_kwargs["color"] = color
        style = getattr(args, "style", None)
        if style:
            plot_kwargs["linestyle"] = style
        marker = getattr(args, "marker", None)
        if marker:
            plot_kwargs["marker"] = marker
        xlim = getattr(args, "xlim", None)
        if xlim:
            plot_kwargs["xlim"] = tuple(map(float, xlim.split(",")))
        ylim = getattr(args, "ylim", None)
        if ylim:
            plot_kwargs["ylim"] = tuple(map(float, ylim.split(",")))

        # Create plot based on type
        self.log(f"Creating {args.type} plot")

        if args.type == "histogram":
            # Flatten multi-dimensional arrays for histogram
            if isinstance(data, np.ndarray) and data.ndim > 1:
                data = data.flatten()
            fig, ax, hist_data = histogram(data, bins=args.bins, **plot_kwargs)

        elif args.type == "line":
            if isinstance(data, pd.DataFrame):
                x = data.iloc[:, 0].values
                y = data.iloc[:, 1].values
            elif isinstance(data, np.ndarray) and data.ndim == 2:
                x = data[:, 0]
                y = data[:, 1]
            else:
                x = np.arange(len(data))
                y = data
            fig, ax = line_plot(x, y, **plot_kwargs)

        elif args.type == "scatter":
            if isinstance(data, pd.DataFrame):
                x = data.iloc[:, 0].values
                y = data.iloc[:, 1].values
            elif isinstance(data, np.ndarray) and data.ndim == 2:
                x = data[:, 0]
                y = data[:, 1]
            else:
                raise ValueError("Scatter plot requires 2D data (x, y)")
            fig, ax = scatter_plot(x, y, **plot_kwargs)

        elif args.type == "heatmap":
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError("Heatmap requires 2D array data")
            fig, ax = heatmap(data, **plot_kwargs)

        elif args.type == "contour":
            if isinstance(data, dict) and "x" in data and "y" in data and "z" in data:
                x, y, z = data["x"], data["y"], data["z"]
            elif isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] == 3:
                x, y, z = data[0], data[1], data[2]
            else:
                raise ValueError("Contour plot requires x, y, z data")
            fig, ax = contour_plot(x, y, z, **plot_kwargs)

        # Advanced plot types (if available)
        elif args.type == "distribution" and HAS_ADVANCED:
            kind = args.dist_kind if hasattr(args, "dist_kind") else "hist"
            fig, ax = distribution_plot(data, kind=kind, **plot_kwargs)

        elif args.type == "box" and HAS_ADVANCED:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Box plot requires DataFrame input")
            x_col = args.x_column if hasattr(args, "x_column") else None
            y_col = args.y_column if hasattr(args, "y_column") else None
            fig, ax = box_plot(data, x=x_col, y=y_col, **plot_kwargs)

        elif args.type == "regression" and HAS_ADVANCED:
            if isinstance(data, pd.DataFrame):
                x = data.iloc[:, 0].values
                y = data.iloc[:, 1].values
            else:
                x = data[:, 0]
                y = data[:, 1]
            order = args.order if hasattr(args, "order") else 1
            fig, ax = regression_plot(x, y, order=order, **plot_kwargs)

        # Interactive plots
        elif (
            args.type in ["interactive_histogram", "interactive_scatter", "interactive_line"]
            and HAS_ADVANCED
        ):
            if args.type == "interactive_histogram":
                fig = interactive_histogram(data, bins=args.bins, title=args.title)
            elif args.type == "interactive_scatter":
                x = data[:, 0] if data.ndim == 2 else np.arange(len(data))
                y = data[:, 1] if data.ndim == 2 else data
                fig = interactive_scatter(x, y, title=args.title)
            elif args.type == "interactive_line":
                x = np.arange(len(data))
                y = data
                fig = interactive_line(x, y, title=args.title)

            # Export interactive plot
            output_path = args.output or "plot.html"
            export_interactive(fig, output_path)
            self.log(f"Interactive plot saved to {output_path}")
            return

        else:
            raise ValueError(f"Unsupported plot type: {args.type}")

        # Save figure
        output_path = args.output or "plot.png"
        dpi = args.dpi if hasattr(args, "dpi") else 300
        save_figure(fig, output_path, dpi=dpi)
        self.log(f"Plot saved to {output_path}")

    def calculate_stats(self, args):
        """Calculate and display statistics.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments
        """
        # Load data
        column = getattr(args, "column", None)
        data = self.load_file(args.input, column)

        # Ensure data is numeric array
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=[np.number]).values.flatten()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        # Calculate statistics
        self.log("Calculating statistics...")
        stats = calculate_statistics(data)

        # Display statistics
        print("\n" + "=" * 50)
        print("Statistical Summary")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key:15s}: {value:12.6f}")
        print("=" * 50)

        # Check normality if requested
        if hasattr(args, "normality") and args.normality:
            normality_result = check_normality(data)
            print("\nNormality Test (Shapiro-Wilk):")
            print(f"  Statistic: {normality_result['statistic']:.6f}")
            print(f"  P-value: {normality_result['p_value']:.6f}")
            print(
                f"  Result: {'Normal' if normality_result['is_normal'] else 'Not Normal'}"
                " (alpha=0.05)"
            )

        # Save to file if requested
        if args.output:
            output_data = {**stats}
            if hasattr(args, "normality") and args.normality:
                output_data.update(normality_result)

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            self.log(f"Statistics saved to {args.output}")

    def batch_process(self, args):
        """Process multiple files using configuration.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments
        """
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.log(f"Loading batch configuration from {config_path}")

        # Parse config based on extension
        if config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            raise ValueError("Config file must be JSON or YAML format")

        # Process each task
        tasks = config.get("tasks", [])
        self.log(f"Processing {len(tasks)} task(s)")

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Processing task: {task.get('name', f'Task {i}')}")

            # Create args namespace from task
            task_args = argparse.Namespace(**task)

            # Execute appropriate command
            command = task.get("command", "plot")
            if command == "plot":
                self.create_plot(task_args)
            elif command == "stats":
                self.calculate_stats(task_args)
            else:
                print(f"  Warning: Unknown command '{command}', skipping")

        self.log("Batch processing complete")

    def compare_datasets(self, args):
        """Compare multiple datasets on same plot.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments
        """
        # Load all input files
        datasets = []
        labels = []

        for i, filepath in enumerate(args.inputs):
            data = self.load_file(filepath, args.column)
            datasets.append(data)

            # Generate label
            if args.labels and i < len(args.labels):
                labels.append(args.labels[i])
            else:
                labels.append(Path(filepath).stem)

        self.log(f"Comparing {len(datasets)} dataset(s)")

        # Create comparison plot
        if args.type == "histogram":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            for data, label in zip(datasets, labels):
                ax.hist(data, bins=args.bins, alpha=0.6, label=label)
            ax.set_xlabel(args.xlabel or "Value")
            ax.set_ylabel(args.ylabel or "Frequency")
            ax.set_title(args.title or "Dataset Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)

        elif args.type == "line":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            for data, label in zip(datasets, labels):
                x = np.arange(len(data))
                ax.plot(x, data, label=label, marker="o", markersize=3)
            ax.set_xlabel(args.xlabel or "Index")
            ax.set_ylabel(args.ylabel or "Value")
            ax.set_title(args.title or "Dataset Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)

        elif args.type == "scatter":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            for data, label in zip(datasets, labels):
                if data.ndim == 2:
                    ax.scatter(data[:, 0], data[:, 1], label=label, alpha=0.6)
            ax.set_xlabel(args.xlabel or "X")
            ax.set_ylabel(args.ylabel or "Y")
            ax.set_title(args.title or "Dataset Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            raise ValueError(f"Comparison plot type '{args.type}' not supported")

        # Save figure
        output_path = args.output or "comparison.png"
        dpi = args.dpi if hasattr(args, "dpi") else 300
        save_figure(fig, output_path, dpi=dpi)
        self.log(f"Comparison plot saved to {output_path}")

    def convert_data(self, args):
        """Convert data between formats.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments
        """
        # Load data
        self.log(f"Loading data from {args.input}")
        data = load_data(args.input)

        # Save in new format
        self.log(f"Converting to {Path(args.output).suffix} format")
        save_data(data, args.output)
        self.log(f"Data saved to {args.output}")

    def show_examples(self):
        """Display usage examples."""
        examples = """
PLOTTING HELPER - USAGE EXAMPLES
================================

1. Create a histogram from NumPy array:
   $ python cli.py plot --input data.npy --type histogram --bins 50 \\
       --output histogram.png --title "Data Distribution"

2. Generate line plot from CSV:
   $ python cli.py plot --input timeseries.csv --type line \\
       --xlabel "Time (s)" --ylabel "Signal" --output line.png

3. Create scatter plot with customization:
   $ python cli.py plot --input points.csv --type scatter \\
       --color blue --marker o --output scatter.png

4. Calculate statistics:
   $ python cli.py stats --input data.csv --column Temperature \\
       --output stats.json --normality

5. Batch processing with config file:
   $ python cli.py batch --config batch_plots.json

6. Compare multiple datasets:
   $ python cli.py compare --inputs data1.csv data2.csv data3.csv \\
       --type line --labels "Trial 1" "Trial 2" "Trial 3" \\
       --output comparison.png

7. Convert data format:
   $ python cli.py convert --input data.pkl --output data.csv

8. Interactive plot (Plotly):
   $ python cli.py plot --input data.npy --type interactive_histogram \\
       --output interactive.html

9. Create heatmap:
   $ python cli.py plot --input matrix.npy --type heatmap \\
       --title "Correlation Matrix" --output heatmap.png

10. Advanced: Distribution plot with Seaborn:
    $ python cli.py plot --input data.csv --column Values \\
        --type distribution --output dist.png

For more help, use: python cli.py [command] --help
        """
        print(examples)


def main():
    """Main entry point for the command-line interface.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description="EaglePlot - Scientific Data Visualization and Analysis Toolkit",
        epilog="NCCU Department of Chemistry and Biochemistry — Schultz Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"EaglePlot {__version__}")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress informational messages"
    )

    parser.add_argument("--examples", action="store_true", help="Show usage examples and exit")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================================================
    # PLOT command
    # ========================================================================
    plot_parser = subparsers.add_parser("plot", help="Create a plot from data")

    plot_parser.add_argument("--input", "-i", required=True, help="Input data file path")

    plot_parser.add_argument(
        "--output", "-o", help="Output file path (default: plot.png or plot.html)"
    )

    plot_parser.add_argument(
        "--type",
        "-t",
        required=True,
        choices=[
            "histogram",
            "line",
            "scatter",
            "heatmap",
            "contour",
            "distribution",
            "box",
            "regression",
            "interactive_histogram",
            "interactive_scatter",
            "interactive_line",
        ],
        help="Type of plot to create",
    )

    plot_parser.add_argument("--column", help="Column name to extract from DataFrame")

    # Plot customization
    plot_parser.add_argument("--title", help="Plot title")
    plot_parser.add_argument("--xlabel", help="X-axis label")
    plot_parser.add_argument("--ylabel", help="Y-axis label")
    plot_parser.add_argument("--color", help="Line/marker color")
    plot_parser.add_argument("--style", help="Line style (-, --, -., :)")
    plot_parser.add_argument("--marker", help="Marker style (o, s, ^, etc.)")
    plot_parser.add_argument("--xlim", help='X-axis limits (e.g., "0,10")')
    plot_parser.add_argument("--ylim", help='Y-axis limits (e.g., "0,100")')

    # Plot-specific options
    plot_parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for histogram (default: 30)",
    )
    plot_parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution in DPI (default: 300)"
    )
    plot_parser.add_argument(
        "--dist-kind",
        choices=["hist", "kde", "ecdf"],
        default="hist",
        help="Distribution plot kind (for --type distribution)",
    )
    plot_parser.add_argument("--x-column", help="X column name for box/scatter plots")
    plot_parser.add_argument("--y-column", help="Y column name for box/scatter plots")
    plot_parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Polynomial order for regression (default: 1)",
    )

    # ========================================================================
    # STATS command
    # ========================================================================
    stats_parser = subparsers.add_parser("stats", help="Calculate statistics")

    stats_parser.add_argument("--input", "-i", required=True, help="Input data file path")

    stats_parser.add_argument(
        "--output", "-o", help="Output file path for statistics (JSON format)"
    )

    stats_parser.add_argument("--column", help="Column name to analyze from DataFrame")

    stats_parser.add_argument("--normality", action="store_true", help="Perform normality test")

    # ========================================================================
    # BATCH command
    # ========================================================================
    batch_parser = subparsers.add_parser("batch", help="Process multiple files using config")

    batch_parser.add_argument(
        "--config", "-c", required=True, help="Configuration file (JSON or YAML)"
    )

    # ========================================================================
    # COMPARE command
    # ========================================================================
    compare_parser = subparsers.add_parser("compare", help="Compare multiple datasets")

    compare_parser.add_argument(
        "--inputs", nargs="+", required=True, help="Input data files to compare"
    )

    compare_parser.add_argument("--output", "-o", help="Output plot file (default: comparison.png)")

    compare_parser.add_argument(
        "--type",
        "-t",
        choices=["histogram", "line", "scatter"],
        default="line",
        help="Type of comparison plot (default: line)",
    )

    compare_parser.add_argument("--labels", nargs="+", help="Labels for each dataset")

    compare_parser.add_argument("--column", help="Column name to extract from DataFrames")

    compare_parser.add_argument("--title", help="Plot title")
    compare_parser.add_argument("--xlabel", help="X-axis label")
    compare_parser.add_argument("--ylabel", help="Y-axis label")
    compare_parser.add_argument("--bins", type=int, default=30, help="Number of bins for histogram")
    compare_parser.add_argument("--dpi", type=int, default=300, help="Output resolution in DPI")

    # ========================================================================
    # CONVERT command
    # ========================================================================
    convert_parser = subparsers.add_parser("convert", help="Convert data between formats")

    convert_parser.add_argument("--input", "-i", required=True, help="Input data file")

    convert_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output data file (format detected from extension)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Show examples if requested
    if args.examples:
        cli = PlottingHelperCLI()
        cli.show_examples()
        return 0

    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Initialize CLI
    cli = PlottingHelperCLI()
    cli.verbose = args.verbose
    cli.quiet = args.quiet

    # Execute command
    try:
        if args.command == "plot":
            cli.create_plot(args)
        elif args.command == "stats":
            cli.calculate_stats(args)
        elif args.command == "batch":
            cli.batch_process(args)
        elif args.command == "compare":
            cli.compare_datasets(args)
        elif args.command == "convert":
            cli.convert_data(args)
        else:
            print(f"Error: Unknown command '{args.command}'")
            return 1

        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        return 1
    except ValueError as e:
        print(f"[ERROR] Invalid value: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
