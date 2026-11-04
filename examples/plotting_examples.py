"""Examples for using the plotting module.

This script demonstrates how to use the plottle plotting module
to create publication-quality scientific visualizations.

Run this script from the plottle directory:
    python examples/plotting_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path to import plottle
sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.plotting import (
    # Figure management
    configure_axes, save_figure,
    # Core plots
    histogram, line_plot, scatter_plot,
    # Advanced plots
    heatmap, contour_plot,
    # Styling
    get_color_palette, apply_publication_style
)

# Use ASCII-safe symbols for Windows compatibility
CHECK = '[OK]'
ARROW = '-->'


def example_histogram():
    """Example 1: Histogram plots."""
    print("\n" + "="*70)
    print("Example 1: Histogram Plots")
    print("="*70)

    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(100, 15, 500)

    print(f"\n{CHECK} Creating histogram of experimental data...")

    # Basic histogram
    fig, ax, info = histogram(
        data,
        bins=30,
        xlabel='Measurement Value',
        ylabel='Frequency',
        title='Distribution of Experimental Measurements',
        color='steelblue',
        alpha=0.7
    )

    print("  Data statistics:")
    print(f"    Mean: {info['mean']:.2f}")
    print(f"    Std Dev: {info['std']:.2f}")
    print(f"    Number of bins: {len(info['bins'])-1}")

    # Save figure
    output_dir = Path('examples/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir / 'histogram.png', dpi=300)
    print(f"{CHECK} Saved to {output_dir / 'histogram.png'}")

    # Density plot
    fig2, ax2, info2 = histogram(
        data,
        bins='auto',
        density=True,
        xlabel='Measurement Value',
        title='Probability Density Function',
        color='coral'
    )

    save_figure(fig2, output_dir / 'histogram_density.png', dpi=300)
    print(f"{CHECK} Saved density plot to {output_dir / 'histogram_density.png'}")


def example_line_plots():
    """Example 2: Line plots."""
    print("\n" + "="*70)
    print("Example 2: Line Plots")
    print("="*70)

    # Generate time series data
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    signal1 = np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.1, 100)
    signal2 = np.cos(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.1, 100)
    signal3 = np.sin(2 * np.pi * t) * np.exp(-0.2 * t)

    print(f"\n{CHECK} Creating time series plots...")

    # Single line plot
    fig, ax = line_plot(
        t, signal1,
        xlabel='Time (s)',
        ylabel='Signal Amplitude',
        title='Time Series Data',
        colors=['darkblue']
    )

    output_dir = Path('examples/plots')
    save_figure(fig, output_dir / 'line_single.png', dpi=300)
    print(f"{CHECK} Saved single line plot")

    # Multiple lines
    fig2, ax2 = line_plot(
        t, [signal1, signal2, signal3],
        labels=['Sin wave', 'Cos wave', 'Damped oscillation'],
        xlabel='Time (s)',
        ylabel='Amplitude',
        title='Multiple Signal Comparison',
        linestyles=['-', '--', '-.']
    )

    save_figure(fig2, output_dir / 'line_multiple.png', dpi=300)
    print(f"{CHECK} Saved multiple line plot")

    # With markers
    t_sparse = t[::10]
    signal_sparse = signal1[::10]

    fig3, ax3 = line_plot(
        t_sparse, signal_sparse,
        xlabel='Time (s)',
        ylabel='Amplitude',
        title='Data Points with Markers',
        markers=['o'],
        colors=['crimson']
    )

    save_figure(fig3, output_dir / 'line_markers.png', dpi=300)
    print(f"{CHECK} Saved line plot with markers")


def example_scatter_plots():
    """Example 3: Scatter plots."""
    print("\n" + "="*70)
    print("Example 3: Scatter Plots")
    print("="*70)

    # Generate correlated data
    np.random.seed(42)
    n_points = 200
    x = np.random.randn(n_points)
    y = 2 * x + 1 + np.random.randn(n_points) * 0.5

    print(f"\n{CHECK} Creating scatter plots...")

    # Basic scatter
    fig, ax = scatter_plot(
        x, y,
        xlabel='Variable X',
        ylabel='Variable Y',
        title='Scatter Plot: Y vs X',
        color='steelblue',
        alpha=0.6
    )

    output_dir = Path('examples/plots')
    save_figure(fig, output_dir / 'scatter_basic.png', dpi=300)
    print(f"{CHECK} Saved basic scatter plot")

    # Scatter with color mapping
    c = x + y
    fig2, ax2 = scatter_plot(
        x, y,
        color=c,
        colorbar=True,
        xlabel='X',
        ylabel='Y',
        title='Scatter Plot with Color Mapping',
        alpha=0.7
    )

    save_figure(fig2, output_dir / 'scatter_colored.png', dpi=300)
    print(f"{CHECK} Saved colored scatter plot")

    # Scatter with variable sizes
    sizes = np.random.rand(n_points) * 200 + 50
    fig3, ax3 = scatter_plot(
        x, y,
        size=sizes,
        color='coral',
        xlabel='X',
        ylabel='Y',
        title='Scatter Plot with Variable Sizes',
        alpha=0.5
    )

    save_figure(fig3, output_dir / 'scatter_sizes.png', dpi=300)
    print(f"{CHECK} Saved scatter plot with variable sizes")


def example_heatmaps():
    """Example 4: Heatmaps."""
    print("\n" + "="*70)
    print("Example 4: Heatmaps")
    print("="*70)

    # Generate correlation matrix
    np.random.seed(42)
    data = np.random.rand(10, 10)
    # Make it symmetric (like correlation matrix)
    corr_matrix = (data + data.T) / 2

    print(f"\n{CHECK} Creating heatmaps...")

    # Basic heatmap
    fig, ax = heatmap(
        corr_matrix,
        title='Correlation Matrix',
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )

    output_dir = Path('examples/plots')
    save_figure(fig, output_dir / 'heatmap_basic.png', dpi=300)
    print(f"{CHECK} Saved basic heatmap")

    # Heatmap with labels
    variables = [f'Var{i+1}' for i in range(10)]

    fig2, ax2 = heatmap(
        corr_matrix,
        xlabel='Variables',
        ylabel='Variables',
        title='Feature Correlation Heatmap',
        xticklabels=variables,
        yticklabels=variables,
        cmap='RdBu_r'
    )

    save_figure(fig2, output_dir / 'heatmap_labeled.png', dpi=300)
    print(f"{CHECK} Saved labeled heatmap")

    # Time series heatmap
    time_data = np.random.rand(20, 50)
    for i in range(20):
        time_data[i, :] += np.sin(np.linspace(0, 4*np.pi, 50)) * (i/20)

    fig3, ax3 = heatmap(
        time_data,
        xlabel='Time Points',
        ylabel='Channels',
        title='Multi-Channel Time Series',
        cmap='viridis'
    )

    save_figure(fig3, output_dir / 'heatmap_timeseries.png', dpi=300)
    print(f"{CHECK} Saved time series heatmap")


def example_contour_plots():
    """Example 5: Contour plots."""
    print("\n" + "="*70)
    print("Example 5: Contour Plots")
    print("="*70)

    # Generate 2D function
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)  # Distance from origin

    print(f"\n{CHECK} Creating contour plots...")

    # Filled contour
    fig, ax = contour_plot(
        X, Y, Z,
        xlabel='X',
        ylabel='Y',
        title='Filled Contour Plot',
        filled=True,
        levels=15,
        cmap='viridis'
    )

    output_dir = Path('examples/plots')
    save_figure(fig, output_dir / 'contour_filled.png', dpi=300)
    print(f"{CHECK} Saved filled contour plot")

    # Line contour
    fig2, ax2 = contour_plot(
        X, Y, Z,
        xlabel='X',
        ylabel='Y',
        title='Line Contour Plot',
        filled=False,
        levels=10,
        cmap='plasma'
    )

    save_figure(fig2, output_dir / 'contour_lines.png', dpi=300)
    print(f"{CHECK} Saved line contour plot")

    # Complex function (Gaussian)
    Z2 = np.exp(-(X**2 + Y**2) / 2)

    fig3, ax3 = contour_plot(
        X, Y, Z2,
        xlabel='X',
        ylabel='Y',
        title='2D Gaussian Function',
        filled=True,
        levels=20,
        cmap='hot'
    )

    save_figure(fig3, output_dir / 'contour_gaussian.png', dpi=300)
    print(f"{CHECK} Saved Gaussian contour plot")


def example_publication_styling():
    """Example 6: Publication-quality styling."""
    print("\n" + "="*70)
    print("Example 6: Publication-Quality Styling")
    print("="*70)

    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y1 = 2 * x + 3 + np.random.normal(0, 2, 50)
    y2 = -1.5 * x + 15 + np.random.normal(0, 2, 50)

    print(f"\n{CHECK} Creating publication-style plots...")

    # Get publication color palette
    colors = get_color_palette('Set1', n_colors=5)

    # Create plot
    fig, ax = line_plot(
        x, [y1, y2],
        labels=['Treatment A', 'Treatment B'],
        xlabel='Time (hours)',
        ylabel='Concentration (mM)',
        title='Time Course Analysis',
        colors=colors[:2],
        markers=['o', 's'],
        linewidth=2
    )

    # Apply publication styling
    apply_publication_style(fig, ax, fontsize=14, labelsize=12)

    output_dir = Path('examples/plots')
    save_figure(fig, output_dir / 'publication_style.png', dpi=300)
    print(f"{CHECK} Saved publication-style plot")

    # Create figure with multiple subplots
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    data = np.random.normal(0, 1, 500)
    axes[0, 0].hist(data, bins=30, color=colors[0], alpha=0.7)
    configure_axes(axes[0, 0], xlabel='Value', ylabel='Frequency',
                   title='A) Histogram')

    # Line plot
    axes[0, 1].plot(x, y1, 'o-', color=colors[1], linewidth=2)
    configure_axes(axes[0, 1], xlabel='X', ylabel='Y',
                   title='B) Line Plot')

    # Scatter
    axes[1, 0].scatter(y1, y2, color=colors[2], alpha=0.6, s=50)
    configure_axes(axes[1, 0], xlabel='Treatment A', ylabel='Treatment B',
                   title='C) Scatter Plot')

    # Heatmap-style data
    heat_data = np.random.rand(10, 10)
    im = axes[1, 1].imshow(heat_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=axes[1, 1])
    configure_axes(axes[1, 1], xlabel='X', ylabel='Y',
                   title='D) Heatmap', grid=False)

    # Apply styling to all axes
    apply_publication_style(fig2, list(axes.flat), fontsize=12)

    save_figure(fig2, output_dir / 'multi_panel.png', dpi=300)
    print(f"{CHECK} Saved multi-panel figure")


def example_practical_workflow():
    """Example 7: Practical data analysis workflow."""
    print("\n" + "="*70)
    print("Example 7: Practical Workflow - Kinetics Data Visualization")
    print("="*70)

    print("\n1. Simulating kinetics experiment...")

    # Simulate first-order decay kinetics
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    C0 = 1.0
    k = 0.3
    C_true = C0 * np.exp(-k * t)
    C_measured = C_true + np.random.normal(0, 0.02, 100)

    print(f"{CHECK} Generated kinetics data (k = {k} s^-1)")

    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    print("\n2. Creating multi-panel analysis figure...")

    # Panel A: Raw data with fit
    axes[0, 0].plot(t, C_measured, 'o', alpha=0.5, label='Measured', markersize=4)
    axes[0, 0].plot(t, C_true, '-', linewidth=2, label='True decay')
    configure_axes(
        axes[0, 0],
        xlabel='Time (s)',
        ylabel='Concentration (M)',
        title='A) Experimental Data',
        legend=True
    )

    # Panel B: Log plot for rate constant
    log_C = np.log(C_measured[C_measured > 0])
    t_positive = t[C_measured > 0]
    axes[0, 1].plot(t_positive, log_C, 'o', alpha=0.5, markersize=4)
    axes[0, 1].plot(t, np.log(C_true), '-', linewidth=2)
    configure_axes(
        axes[0, 1],
        xlabel='Time (s)',
        ylabel='ln(Concentration)',
        title='B) Linearized Plot'
    )

    # Panel C: Residuals
    residuals = C_measured - C_true
    axes[1, 0].scatter(t, residuals, alpha=0.5, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    configure_axes(
        axes[1, 0],
        xlabel='Time (s)',
        ylabel='Residuals (M)',
        title='C) Residual Analysis'
    )

    # Panel D: Residual histogram
    axes[1, 1].hist(residuals, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    configure_axes(
        axes[1, 1],
        xlabel='Residual (M)',
        ylabel='Frequency',
        title='D) Residual Distribution'
    )

    # Apply styling
    apply_publication_style(fig, list(axes.flat), fontsize=12)

    # Save
    output_dir = Path('examples/plots')
    save_figure(fig, output_dir / 'kinetics_analysis.png', dpi=300)
    print(f"{CHECK} Saved comprehensive analysis figure")

    print("\n3. Creating summary plots...")

    # Summary figure: Rate constant determination
    fig2, ax2 = line_plot(
        t_positive, log_C,
        xlabel='Time (s)',
        ylabel='ln([A])',
        title='First-Order Kinetics: Rate Constant Determination',
        markers=['o']
    )

    # Add linear fit line
    from plottle.math import fit_linear
    fit_result = fit_linear(t_positive, log_C)
    y_fit = fit_result['slope'] * t_positive + fit_result['intercept']
    ax2.plot(t_positive, y_fit, 'r-', linewidth=2,
             label=f'Linear fit: k = {-fit_result["slope"]:.3f} s$^{{-1}}$')
    ax2.legend()

    apply_publication_style(fig2, ax2)
    save_figure(fig2, output_dir / 'rate_constant.png', dpi=300)
    print(f"{CHECK} Saved rate constant plot")

    print("\n" + "="*70)
    print("Analysis complete! All plots demonstrate:")
    print("  - Data visualization best practices")
    print("  - Multi-panel figure composition")
    print("  - Residual analysis")
    print("  - Publication-quality formatting")
    print("="*70)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PLOTTING HELPER - PLOTTING MODULE EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates the plotting capabilities of the")
    print("plottle package for scientific visualization.\n")
    print(f"{CHECK} Using non-interactive backend (Agg)")
    print(f"{CHECK} All plots will be saved to examples/plots/")

    try:
        # Run all examples
        example_histogram()
        example_line_plots()
        example_scatter_plots()
        example_heatmaps()
        example_contour_plots()
        example_publication_styling()
        example_practical_workflow()

        print("\n" + "="*70)
        print(f"All examples completed successfully! {CHECK}")
        print("="*70)
        print("\nGenerated plots are saved in: examples/plots/")
        print("Total plots created: 17")
        print("\nPlots include:")
        print("  - Histograms (basic and density)")
        print("  - Line plots (single, multiple, with markers)")
        print("  - Scatter plots (basic, colored, variable sizes)")
        print("  - Heatmaps (basic, labeled, time series)")
        print("  - Contour plots (filled, lines, functions)")
        print("  - Publication-style figures")
        print("  - Multi-panel analysis workflow")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
