"""Examples for Plotly interactive plots.

Demonstrates all five interactive plot types and HTML export.
Open the generated HTML files in any browser for full interactivity.

Run from the repo root:
    python examples/plotly_interactive_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.plotting import (
    interactive_histogram,
    interactive_scatter,
    interactive_line,
    interactive_heatmap,
    interactive_3d_surface,
    export_interactive,
    HAS_PLOTLY,
)

if not HAS_PLOTLY:
    print("Plotly is not installed. Run: pip install plotly")
    sys.exit(0)

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def example_interactive_histogram():
    """Example 1: Interactive histogram."""
    print("\n" + "=" * 70)
    print("Example 1: Interactive Histogram")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.normal(0, 1, 500)

    fig = interactive_histogram(
        data,
        bins=30,
        title="Normal Distribution",
        xlabel="Value",
        ylabel="Count",
    )

    out = OUTPUT_DIR / "interactive_histogram.html"
    export_interactive(fig, out)
    print(f"  Saved: {out}")
    print(f"{CHECK} Interactive histogram complete")


def example_interactive_scatter():
    """Example 2: Interactive scatter with color and size mapping."""
    print("\n" + "=" * 70)
    print("Example 2: Interactive Scatter Plot")
    print("=" * 70)

    np.random.seed(0)
    n = 150
    x = np.random.uniform(-5, 5, n)
    y = np.sin(x) + np.random.normal(0, 0.3, n)
    color = np.abs(y)  # color encodes |y|
    size = 5 + 15 * color  # point size scales with |y|

    fig = interactive_scatter(
        x,
        y,
        color=color,
        size=size,
        title="sin(x) + noise",
        xlabel="x",
        ylabel="y",
    )

    out = OUTPUT_DIR / "interactive_scatter.html"
    export_interactive(fig, out)
    print(f"  Saved: {out}")
    print(f"{CHECK} Interactive scatter complete")


def example_interactive_line():
    """Example 3: Interactive multi-series line plot."""
    print("\n" + "=" * 70)
    print("Example 3: Interactive Multi-Series Line Plot")
    print("=" * 70)

    t = np.linspace(0, 4 * np.pi, 300)
    y_sin = np.sin(t)
    y_cos = np.cos(t)
    y_damped = np.exp(-t / 6) * np.sin(t)

    fig = interactive_line(
        t,
        [y_sin, y_cos, y_damped],
        labels=["sin(t)", "cos(t)", "damped sin"],
        title="Trigonometric Functions",
        xlabel="t (radians)",
        ylabel="Amplitude",
    )

    out = OUTPUT_DIR / "interactive_line.html"
    export_interactive(fig, out)
    print(f"  Saved: {out}")
    print(f"{CHECK} Interactive line plot complete")


def example_interactive_heatmap():
    """Example 4: Interactive heatmap — correlation matrix."""
    print("\n" + "=" * 70)
    print("Example 4: Interactive Heatmap")
    print("=" * 70)

    np.random.seed(42)
    n = 5
    labels = [f"Var {i + 1}" for i in range(n)]
    raw = np.random.randn(100, n)
    raw[:, 1] = 0.8 * raw[:, 0] + 0.2 * raw[:, 1]  # induce correlation
    corr = np.corrcoef(raw.T)

    fig = interactive_heatmap(
        corr,
        x_labels=labels,
        y_labels=labels,
        colorscale="RdBu",
        title="Correlation Matrix",
    )

    out = OUTPUT_DIR / "interactive_heatmap.html"
    export_interactive(fig, out)
    print(f"  Saved: {out}")
    print(f"{CHECK} Interactive heatmap complete")


def example_interactive_3d_surface():
    """Example 5: Interactive 3D surface — potential energy surface."""
    print("\n" + "=" * 70)
    print("Example 5: Interactive 3D Surface Plot")
    print("=" * 70)

    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = interactive_3d_surface(
        X,
        Y,
        Z,
        title="sin(r) Surface",
        colorscale="Viridis",
    )

    out = OUTPUT_DIR / "interactive_3d_surface.html"
    export_interactive(fig, out)
    print(f"  Saved: {out}")
    print(f"{CHECK} 3D surface plot complete")


if __name__ == "__main__":
    print("EaglePlot — Plotly Interactive Examples")
    print("=" * 70)
    example_interactive_histogram()
    example_interactive_scatter()
    example_interactive_line()
    example_interactive_heatmap()
    example_interactive_3d_surface()
    print("\nAll Plotly examples complete.")
    print("Open .html files in OUTPUT_DIR to view interactive plots.")
