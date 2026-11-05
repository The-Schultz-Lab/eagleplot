"""Examples for descriptive statistics and normality testing.

Demonstrates calculate_statistics(), check_normality(), and basic data analysis
workflows using the math module.

Run from the repo root:
    python examples/statistics_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.math import calculate_statistics, check_normality
from eagleplot.plotting import histogram, save_figure

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def example_basic_statistics():
    """Example 1: Descriptive statistics on a 1-D array."""
    print("\n" + "=" * 70)
    print("Example 1: Descriptive Statistics")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=1.5, size=200)

    stats = calculate_statistics(data)

    print(f"  n        : {len(data)}")
    print(f"  mean     : {stats['mean']:.4f}")
    print(f"  median   : {stats['median']:.4f}")
    print(f"  std      : {stats['std']:.4f}")
    print(f"  variance : {stats['var']:.4f}")
    print(f"  min/max  : {stats['min']:.4f} / {stats['max']:.4f}")
    print(f"  Q1/Q3    : {stats['q1']:.4f} / {stats['q3']:.4f}")
    print(f"  IQR      : {stats['iqr']:.4f}")
    print(f"  range    : {stats['range']:.4f}")
    print(f"{CHECK} Statistics calculated")


def example_normality_test():
    """Example 2: Shapiro-Wilk normality test."""
    print("\n" + "=" * 70)
    print("Example 2: Normality Test (Shapiro-Wilk)")
    print("=" * 70)

    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    uniform_data = np.random.uniform(0, 10, 100)

    stat_n, p_n = check_normality(normal_data)
    stat_u, p_u = check_normality(uniform_data)

    print(
        f"  Normal data:  W = {stat_n:.4f}, p = {p_n:.4f}  "
        f"({'normal' if p_n > 0.05 else 'NOT normal'} at alpha=0.05)"
    )
    print(
        f"  Uniform data: W = {stat_u:.4f}, p = {p_u:.4f}  "
        f"({'normal' if p_u > 0.05 else 'NOT normal'} at alpha=0.05)"
    )
    print(f"{CHECK} Normality tests complete")


def example_statistics_with_plot():
    """Example 3: Statistics + histogram visualization."""
    print("\n" + "=" * 70)
    print("Example 3: Statistics with Visualization")
    print("=" * 70)

    np.random.seed(0)
    data = np.concatenate(
        [
            np.random.normal(3.0, 0.5, 150),
            np.random.normal(6.0, 0.8, 100),
        ]
    )

    stats = calculate_statistics(data)

    fig, ax, info = histogram(
        data,
        bins=30,
        title="Bimodal Distribution",
        xlabel="Value",
        ylabel="Frequency",
    )

    # Annotate with mean and std
    ax.axvline(
        stats["mean"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {stats['mean']:.2f}",
    )
    ax.axvline(
        stats["median"],
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {stats['median']:.2f}",
    )
    ax.legend()

    out = OUTPUT_DIR / "statistics_histogram.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Histogram with statistics annotations complete")


def example_2d_array_statistics():
    """Example 4: Statistics along axes of a 2-D array."""
    print("\n" + "=" * 70)
    print("Example 4: 2-D Array Statistics (axis-wise)")
    print("=" * 70)

    np.random.seed(1)
    matrix = np.random.rand(5, 4)

    row_stats = calculate_statistics(matrix, axis=1)
    col_stats = calculate_statistics(matrix, axis=0)

    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Row means   : {row_stats['mean']}")
    print(f"  Column stds : {col_stats['std']}")
    print(f"{CHECK} 2-D axis-wise statistics complete")


if __name__ == "__main__":
    print("EaglePlot — Statistics Examples")
    print("=" * 70)
    example_basic_statistics()
    example_normality_test()
    example_statistics_with_plot()
    example_2d_array_statistics()
    print("\nAll statistics examples complete.")
