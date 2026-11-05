"""Examples for publication-quality figure styling.

Demonstrates apply_publication_style(), get_color_palette(), set_style(),
and creating figures that meet journal submission standards.

Run from the repo root:
    python examples/publication_style_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.plotting import (
    line_plot,
    scatter_plot,
    apply_publication_style,
    get_color_palette,
    set_style,
    create_figure,
    configure_axes,
    save_figure,
)
from eagleplot.utils.plot_config import COLOR_PALETTE_NAMES

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def example_publication_style():
    """Example 1: Apply publication style to a line plot."""
    print("\n" + "=" * 70)
    print("Example 1: Publication-Style Line Plot")
    print("=" * 70)

    np.random.seed(42)
    x = np.linspace(400, 800, 200)
    colors = get_color_palette("Color-Blind Safe (Wong)", n_colors=3)

    y1 = 0.8 * np.exp(-((x - 520) ** 2) / (2 * 30**2))
    y2 = 0.6 * np.exp(-((x - 580) ** 2) / (2 * 25**2))
    y3 = 0.4 * np.exp(-((x - 660) ** 2) / (2 * 35**2))

    fig, ax = line_plot(
        x,
        [y1, y2, y3],
        labels=["Sample A", "Sample B", "Sample C"],
        xlabel="Wavelength (nm)",
        ylabel="Absorbance (a.u.)",
        title="UV-Vis Absorption Spectra",
        colors=colors,
    )

    apply_publication_style(fig, ax)

    out = OUTPUT_DIR / "publication_style.png"
    save_figure(fig, out, dpi=300)
    print(f"  Saved at 300 dpi: {out}")
    print(f"{CHECK} Publication style applied")


def example_color_palettes():
    """Example 2: Compare available color palettes."""
    print("\n" + "=" * 70)
    print("Example 2: Color Palettes")
    print("=" * 70)

    print("  Available palettes:")
    for name in COLOR_PALETTE_NAMES:
        colors = get_color_palette(name, n_colors=6)
        print(f"    {name}: {colors[:3]} ...")

    # Demonstrate color-blind-safe palette
    palette = get_color_palette("Color-Blind Safe (Okabe-Ito)", n_colors=5)
    np.random.seed(0)
    x = np.linspace(0, 2 * np.pi, 100)

    fig, ax = create_figure(figsize=(8, 4))
    for i, color in enumerate(palette):
        ax.plot(x, np.sin(x + i * np.pi / 5), color=color, linewidth=2, label=f"Series {i + 1}")
    configure_axes(ax, xlabel="x", ylabel="y", title="Color-Blind-Safe Palette Demo", grid=True)
    ax.legend(loc="upper right", ncol=3)

    out = OUTPUT_DIR / "color_palettes.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Color palettes demonstrated")


def example_matplotlib_styles():
    """Example 3: Compare Matplotlib built-in styles."""
    print("\n" + "=" * 70)
    print("Example 3: Matplotlib Style Gallery")
    print("=" * 70)

    styles = ["default", "seaborn-v0_8-whitegrid", "ggplot"]
    np.random.seed(1)
    x = np.linspace(0, 10, 50)
    y = x + np.random.randn(50)

    for style in styles:
        try:
            set_style(style)
            fig, ax = scatter_plot(x, y, xlabel="x", ylabel="y", title=f"Style: {style}")
            clean_name = style.replace("-", "_").replace(".", "_")
            out = OUTPUT_DIR / f"style_{clean_name}.png"
            save_figure(fig, out, dpi=150)
            print(f"  Saved: {out}")
        except Exception as e:
            print(f"  Skipped {style}: {e}")
        finally:
            set_style("default")  # reset

    print(f"{CHECK} Style comparison complete")


def example_two_column_figure():
    """Example 4: Two-panel figure for journal submission."""
    print("\n" + "=" * 70)
    print("Example 4: Two-Panel Journal Figure")
    print("=" * 70)

    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    data = np.random.normal(5, 1.5, 300)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A — line plot
    axes[0].plot(x, np.sin(x), "b-", linewidth=1.5)
    axes[0].set_xlabel("Time (ns)", fontsize=10)
    axes[0].set_ylabel("Signal (mV)", fontsize=10)
    axes[0].set_title("(a) Time Trace", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Panel B — histogram
    axes[1].hist(data, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Energy (kcal/mol)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("(b) Energy Distribution", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    out = OUTPUT_DIR / "two_panel_figure.png"
    save_figure(fig, out, dpi=300)
    print(f"  Saved at 300 dpi: {out}")
    print(f"{CHECK} Two-panel figure complete")


if __name__ == "__main__":
    print("EaglePlot — Publication Style Examples")
    print("=" * 70)
    example_publication_style()
    example_color_palettes()
    example_matplotlib_styles()
    example_two_column_figure()
    print("\nAll publication style examples complete.")
