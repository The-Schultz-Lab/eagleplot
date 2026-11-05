"""Examples for multi-panel and complex figure layouts.

Demonstrates creating subplots, shared axes, insets, and complex
grid layouts using Matplotlib alongside the EaglePlot API.

Run from the repo root:
    python examples/multi_panel_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.plotting import (
    save_figure,
)

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def example_2x2_grid():
    """Example 1: 2x2 grid of independent plots."""
    print("\n" + "=" * 70)
    print("Example 1: 2x2 Grid Layout")
    print("=" * 70)

    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Signal Analysis Dashboard", fontsize=14)

    axes[0, 0].plot(x, np.sin(x), "b-")
    axes[0, 0].set_title("sin(x)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    axes[0, 1].plot(x, np.cos(x), "r-")
    axes[0, 1].set_title("cos(x)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")

    data_noise = np.random.normal(0, 1, 500)
    axes[1, 0].hist(data_noise, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1, 0].set_title("Normal Distribution")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Count")

    corr = np.corrcoef(np.random.randn(4, 50))
    im = axes[1, 1].imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 1].set_title("Correlation Matrix")
    fig.colorbar(im, ax=axes[1, 1])

    fig.tight_layout()
    out = OUTPUT_DIR / "multi_panel_2x2.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} 2x2 grid complete")


def example_shared_axes():
    """Example 2: Shared x-axis — time series with multiple y-scales."""
    print("\n" + "=" * 70)
    print("Example 2: Shared X-Axis (Dual Y-Scale)")
    print("=" * 70)

    np.random.seed(0)
    t = np.linspace(0, 10, 300)
    temperature = 298 + 20 * np.sin(t) + np.random.randn(300) * 0.5
    pressure = 1.0 + 0.02 * np.sin(t + 1.2) + np.random.randn(300) * 0.002

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(t, temperature, "b-", linewidth=1.5, label="Temperature (K)")
    ax2.plot(t, pressure, "r--", linewidth=1.5, label="Pressure (atm)")

    ax1.set_xlabel("Time (ns)", fontsize=11)
    ax1.set_ylabel("Temperature (K)", color="blue", fontsize=11)
    ax2.set_ylabel("Pressure (atm)", color="red", fontsize=11)
    ax1.set_title("MD Simulation: Temperature and Pressure")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    out = OUTPUT_DIR / "shared_xaxis.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Shared x-axis plot complete")


def example_gridspec_layout():
    """Example 3: GridSpec for asymmetric panel sizes."""
    print("\n" + "=" * 70)
    print("Example 3: GridSpec Asymmetric Layout")
    print("=" * 70)

    np.random.seed(3)
    x = np.linspace(-3, 3, 200)
    data = np.random.normal(0, 1, 500)

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Large left panel (spans two rows)
    ax_main = fig.add_subplot(gs[:, 0:2])
    ax_main.scatter(
        np.random.randn(200),
        np.random.randn(200),
        c=np.random.rand(200),
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    ax_main.set_title("Scatter Plot (main)")
    ax_main.set_xlabel("PC 1")
    ax_main.set_ylabel("PC 2")

    # Top-right: normal curve
    ax_top = fig.add_subplot(gs[0, 2])
    ax_top.plot(x, np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), "k-", linewidth=2)
    ax_top.set_title("Normal PDF")
    ax_top.set_xlabel("x")

    # Bottom-right: histogram
    ax_bot = fig.add_subplot(gs[1, 2])
    ax_bot.hist(data, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    ax_bot.set_title("Samples")
    ax_bot.set_xlabel("Value")

    out = OUTPUT_DIR / "gridspec_layout.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} GridSpec layout complete")


if __name__ == "__main__":
    print("EaglePlot — Multi-Panel Figure Examples")
    print("=" * 70)
    example_2x2_grid()
    example_shared_axes()
    example_gridspec_layout()
    print("\nAll multi-panel examples complete.")
