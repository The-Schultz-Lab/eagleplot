"""Examples for advanced plotting features (Seaborn and Plotly).

This script demonstrates statistical visualizations with Seaborn and
interactive plots with Plotly.

Run this script from the plottle directory:
    python examples/advanced_plotting_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path to import plottle
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set matplotlib to non-interactive backend
import matplotlib

matplotlib.use("Agg")

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
    save_figure,
    HAS_SEABORN,
    HAS_PLOTLY,
)

# Use ASCII-safe checkmarks for Windows compatibility
CHECK = "[OK]"
CROSS = "[X]"


def example_seaborn_distribution():
    """Example 1: Seaborn distribution plots."""
    if not HAS_SEABORN:
        print(f"{CROSS} Seaborn not installed - skipping distribution plots")
        return

    print("\n" + "=" * 70)
    print("Example 1: Seaborn Distribution Plots")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    skewed_data = np.random.exponential(2, 1000)

    # Create histogram with KDE
    fig1, ax1 = distribution_plot(
        normal_data, kind="hist", kde=True, bins=30, color="skyblue", edgecolor="black"
    )
    ax1.set_title("Normal Distribution with KDE")
    ax1.set_xlabel("Value")
    output_file1 = Path("examples/plots/dist_histogram_kde.png")
    output_file1.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig1, output_file1, dpi=300)
    print(f"{CHECK} Created histogram with KDE: {output_file1}")

    # Create KDE plot only
    fig2, ax2 = distribution_plot(skewed_data, kind="kde", fill=True, color="coral")
    ax2.set_title("Skewed Distribution (KDE)")
    ax2.set_xlabel("Value")
    output_file2 = Path("examples/plots/dist_kde_only.png")
    save_figure(fig2, output_file2, dpi=300)
    print(f"{CHECK} Created KDE plot: {output_file2}")

    # Create ECDF plot
    fig3, ax3 = distribution_plot(normal_data, kind="ecdf", color="green")
    ax3.set_title("Empirical CDF")
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Cumulative Probability")
    output_file3 = Path("examples/plots/dist_ecdf.png")
    save_figure(fig3, output_file3, dpi=300)
    print(f"{CHECK} Created ECDF plot: {output_file3}")


def example_seaborn_box_plots():
    """Example 2: Seaborn box and violin plots."""
    if not HAS_SEABORN:
        print(f"{CROSS} Seaborn not installed - skipping box plots")
        return

    print("\n" + "=" * 70)
    print("Example 2: Seaborn Box and Violin Plots")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Treatment": ["Control"] * 100 + ["Drug A"] * 100 + ["Drug B"] * 100,
            "Response": (
                list(np.random.normal(10, 2, 100))
                + list(np.random.normal(15, 3, 100))
                + list(np.random.normal(20, 2.5, 100))
            ),
            "Batch": (["Batch 1"] * 50 + ["Batch 2"] * 50) * 3,
        }
    )

    # Create box plot
    fig1, ax1 = box_plot(df, x="Treatment", y="Response", kind="box", palette="Set2")
    ax1.set_title("Treatment Response (Box Plot)")
    ax1.set_ylabel("Response Value")
    output_file1 = Path("examples/plots/box_plot.png")
    save_figure(fig1, output_file1, dpi=300)
    print(f"{CHECK} Created box plot: {output_file1}")

    # Create violin plot
    fig2, ax2 = box_plot(
        df, x="Treatment", y="Response", kind="violin", palette="muted", inner="quartile"
    )
    ax2.set_title("Treatment Response (Violin Plot)")
    ax2.set_ylabel("Response Value")
    output_file2 = Path("examples/plots/violin_plot.png")
    save_figure(fig2, output_file2, dpi=300)
    print(f"{CHECK} Created violin plot: {output_file2}")

    # Create box plot with hue
    fig3, ax3 = box_plot(df, x="Treatment", y="Response", hue="Batch", kind="box", palette="pastel")
    ax3.set_title("Treatment Response by Batch")
    ax3.set_ylabel("Response Value")
    output_file3 = Path("examples/plots/box_plot_hue.png")
    save_figure(fig3, output_file3, dpi=300)
    print(f"{CHECK} Created box plot with hue: {output_file3}")


def example_seaborn_regression():
    """Example 3: Seaborn regression plots."""
    if not HAS_SEABORN:
        print(f"{CROSS} Seaborn not installed - skipping regression plots")
        return

    print("\n" + "=" * 70)
    print("Example 3: Seaborn Regression Plots")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_linear = 2 * x + 3 + np.random.normal(0, 2, 100)
    y_quad = 0.5 * x**2 - 2 * x + 5 + np.random.normal(0, 3, 100)

    # Create linear regression plot
    fig1, ax1 = regression_plot(
        x, y_linear, order=1, ci=95, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}
    )
    ax1.set_title("Linear Regression with 95% CI")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    output_file1 = Path("examples/plots/regression_linear.png")
    save_figure(fig1, output_file1, dpi=300)
    print(f"{CHECK} Created linear regression plot: {output_file1}")

    # Create polynomial regression plot
    fig2, ax2 = regression_plot(
        x,
        y_quad,
        order=2,
        ci=95,
        scatter_kws={"alpha": 0.5, "color": "green"},
        line_kws={"color": "darkgreen"},
    )
    ax2.set_title("Quadratic Regression with 95% CI")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    output_file2 = Path("examples/plots/regression_quadratic.png")
    save_figure(fig2, output_file2, dpi=300)
    print(f"{CHECK} Created quadratic regression plot: {output_file2}")


def example_plotly_interactive_histogram():
    """Example 4: Plotly interactive histogram."""
    if not HAS_PLOTLY:
        print(f"{CROSS} Plotly not installed - skipping interactive histogram")
        return

    print("\n" + "=" * 70)
    print("Example 4: Plotly Interactive Histogram")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)

    # Create interactive histogram
    fig = interactive_histogram(
        data, bins=40, title="Interactive Distribution", xlabel="Value", ylabel="Frequency"
    )

    # Export to HTML
    output_file = Path("examples/plots/interactive_histogram.html")
    export_interactive(fig, output_file)
    print(f"{CHECK} Created interactive histogram: {output_file}")
    print(f"  Open {output_file} in a web browser to interact with the plot")


def example_plotly_interactive_scatter():
    """Example 5: Plotly interactive scatter plot."""
    if not HAS_PLOTLY:
        print(f"{CROSS} Plotly not installed - skipping interactive scatter")
        return

    print("\n" + "=" * 70)
    print("Example 5: Plotly Interactive Scatter Plot")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n = 200
    x = np.random.rand(n) * 10
    y = 2 * x + np.random.normal(0, 2, n)
    color = x + y  # Color by sum
    size = np.random.rand(n) * 30 + 5  # Variable sizes

    # Create interactive scatter with color and size mapping
    fig = interactive_scatter(
        x,
        y,
        color=color,
        size=size,
        title="Interactive Scatter Plot",
        xlabel="X Variable",
        ylabel="Y Variable",
    )

    # Export to HTML
    output_file = Path("examples/plots/interactive_scatter.html")
    export_interactive(fig, output_file)
    print(f"{CHECK} Created interactive scatter: {output_file}")


def example_plotly_interactive_line():
    """Example 6: Plotly interactive line plot."""
    if not HAS_PLOTLY:
        print(f"{CROSS} Plotly not installed - skipping interactive line plot")
        return

    print("\n" + "=" * 70)
    print("Example 6: Plotly Interactive Line Plot")
    print("=" * 70)

    # Generate sample data
    t = np.linspace(0, 10, 200)
    y1 = np.sin(t)
    y2 = np.cos(t)
    y3 = np.sin(t) * np.exp(-t / 10)

    # Create interactive multi-line plot
    fig = interactive_line(
        t,
        [y1, y2, y3],
        labels=["sin(t)", "cos(t)", "damped sin(t)"],
        title="Interactive Time Series",
        xlabel="Time (s)",
        ylabel="Amplitude",
    )

    # Export to HTML
    output_file = Path("examples/plots/interactive_line.html")
    export_interactive(fig, output_file)
    print(f"{CHECK} Created interactive line plot: {output_file}")


def example_plotly_interactive_heatmap():
    """Example 7: Plotly interactive heatmap."""
    if not HAS_PLOTLY:
        print(f"{CROSS} Plotly not installed - skipping interactive heatmap")
        return

    print("\n" + "=" * 70)
    print("Example 7: Plotly Interactive Heatmap")
    print("=" * 70)

    # Generate sample correlation matrix
    np.random.seed(42)
    n_vars = 10
    data = np.random.randn(100, n_vars)
    corr_matrix = np.corrcoef(data.T)

    # Create labels
    labels = [f"Var{i + 1}" for i in range(n_vars)]

    # Create interactive heatmap
    fig = interactive_heatmap(
        corr_matrix,
        x_labels=labels,
        y_labels=labels,
        title="Correlation Matrix",
        colorscale="RdBu",
        zmid=0,
    )

    # Export to HTML
    output_file = Path("examples/plots/interactive_heatmap.html")
    export_interactive(fig, output_file)
    print(f"{CHECK} Created interactive heatmap: {output_file}")


def example_plotly_3d_surface():
    """Example 8: Plotly 3D surface plot."""
    if not HAS_PLOTLY:
        print(f"{CROSS} Plotly not installed - skipping 3D surface plot")
        return

    print("\n" + "=" * 70)
    print("Example 8: Plotly 3D Surface Plot")
    print("=" * 70)

    # Generate 3D surface data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Create interactive 3D surface
    fig = interactive_3d_surface(
        X, Y, Z, title="3D Surface: sin(sqrt(x² + y²))", colorscale="Viridis"
    )

    # Export to HTML
    output_file = Path("examples/plots/interactive_3d_surface.html")
    export_interactive(fig, output_file)
    print(f"{CHECK} Created interactive 3D surface: {output_file}")


def example_practical_workflow():
    """Example 9: Practical workflow combining Seaborn and Plotly."""
    if not HAS_SEABORN or not HAS_PLOTLY:
        print(f"{CROSS} Seaborn and Plotly required - skipping practical workflow")
        return

    print("\n" + "=" * 70)
    print("Example 9: Practical Workflow - Experimental Data Analysis")
    print("=" * 70)

    # Simulate experimental data
    np.random.seed(42)
    n_samples = 150
    df = pd.DataFrame(
        {
            "Temperature": np.random.uniform(20, 80, n_samples),
            "Pressure": np.random.uniform(1, 5, n_samples),
            "Yield": np.zeros(n_samples),
            "Catalyst": np.random.choice(["A", "B", "C"], n_samples),
        }
    )

    # Yield depends on temperature and pressure
    df["Yield"] = 0.5 * df["Temperature"] + 10 * df["Pressure"] + np.random.normal(0, 5, n_samples)

    # 1. Distribution analysis with Seaborn
    fig1, ax1 = distribution_plot(df["Yield"], kind="hist", kde=True, bins=25, color="lightblue")
    ax1.set_title("Yield Distribution")
    ax1.set_xlabel("Yield (%)")
    output_file1 = Path("examples/plots/workflow_distribution.png")
    save_figure(fig1, output_file1, dpi=300)
    print(f"{CHECK} Created yield distribution plot")

    # 2. Box plot by catalyst type
    fig2, ax2 = box_plot(df, x="Catalyst", y="Yield", kind="violin", palette="Set2")
    ax2.set_title("Yield by Catalyst Type")
    ax2.set_ylabel("Yield (%)")
    output_file2 = Path("examples/plots/workflow_catalyst.png")
    save_figure(fig2, output_file2, dpi=300)
    print(f"{CHECK} Created catalyst comparison plot")

    # 3. Interactive 3D scatter for temperature-pressure-yield relationship
    fig3 = interactive_scatter(
        df["Temperature"],
        df["Pressure"],
        color=df["Yield"],
        title="Yield vs Temperature and Pressure",
        xlabel="Temperature (°C)",
        ylabel="Pressure (bar)",
    )
    output_file3 = Path("examples/plots/workflow_interactive_scatter.html")
    export_interactive(fig3, output_file3)
    print(f"{CHECK} Created interactive scatter plot")

    # 4. Regression analysis
    fig4, ax4 = regression_plot(
        x="Temperature",
        y="Yield",
        data=df,
        order=1,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
    )
    ax4.set_title("Yield vs Temperature (Linear Fit)")
    ax4.set_xlabel("Temperature (°C)")
    ax4.set_ylabel("Yield (%)")
    output_file4 = Path("examples/plots/workflow_regression.png")
    save_figure(fig4, output_file4, dpi=300)
    print(f"{CHECK} Created regression analysis plot")

    print(f"\n{CHECK} Workflow complete! Generated 4 plots:")
    print("  1. Yield distribution (Seaborn)")
    print("  2. Catalyst comparison (Seaborn)")
    print("  3. Interactive 3D exploration (Plotly)")
    print("  4. Regression analysis (Seaborn)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PLOTTING HELPER - ADVANCED PLOTTING EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates advanced plotting with Seaborn and Plotly,")
    print("including statistical visualizations and interactive plots.\n")

    if not HAS_SEABORN:
        print("⚠️  Warning: Seaborn not installed. Install with: pip install seaborn")
    if not HAS_PLOTLY:
        print("⚠️  Warning: Plotly not installed. Install with: pip install plotly")

    if not HAS_SEABORN and not HAS_PLOTLY:
        print(f"\n{CROSS} Neither Seaborn nor Plotly are installed. Exiting.")
        return

    try:
        # Run all examples
        example_seaborn_distribution()
        example_seaborn_box_plots()
        example_seaborn_regression()
        example_plotly_interactive_histogram()
        example_plotly_interactive_scatter()
        example_plotly_interactive_line()
        example_plotly_interactive_heatmap()
        example_plotly_3d_surface()
        example_practical_workflow()

        print("\n" + "=" * 70)
        print(f"All examples completed successfully! {CHECK}")
        print("=" * 70)
        print("\nGenerated plots are in: examples/plots/")
        print("- Static plots (.png): Open with any image viewer")
        print("- Interactive plots (.html): Open in web browser for full interactivity")
        print("\nKey features demonstrated:")
        print("  + Seaborn: distribution, box, violin, regression plots")
        print("  + Plotly: interactive histograms, scatter, line, heatmap, 3D surface")
        print("  + HTML export for interactive sharing")

    except Exception as e:
        print(f"\n{CROSS} Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
