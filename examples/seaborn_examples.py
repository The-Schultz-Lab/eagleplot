"""Examples for Seaborn statistical plots.

Demonstrates distribution_plot(), box_plot(), and regression_plot()
with pandas DataFrames.

Run from the repo root:
    python examples/seaborn_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.plotting import (
    distribution_plot,
    box_plot,
    regression_plot,
    save_figure,
    HAS_SEABORN,
)

if not HAS_SEABORN:
    print("Seaborn is not installed. Run: pip install seaborn")
    sys.exit(0)

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def make_sample_df():
    """Build a DataFrame simulating spectral peak heights across conditions."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "solvent": ["Water"] * 20 + ["DMSO"] * 20 + ["Ethanol"] * 20,
            "intensity": np.concatenate(
                [
                    np.random.normal(100, 12, 20),
                    np.random.normal(130, 18, 20),
                    np.random.normal(115, 10, 20),
                ]
            ),
            "wavelength": np.tile(np.random.uniform(490, 530, 20), 3),
            "concentration": np.concatenate(
                [
                    np.random.uniform(0.1, 1.0, 20),
                    np.random.uniform(0.1, 1.0, 20),
                    np.random.uniform(0.1, 1.0, 20),
                ]
            ),
        }
    )


def example_distribution_plots():
    """Example 1: Distribution plots — histogram, KDE, and ECDF."""
    print("\n" + "=" * 70)
    print("Example 1: Distribution Plots")
    print("=" * 70)

    df = make_sample_df()
    data = df["intensity"].values

    for kind in ("hist", "kde", "ecdf"):
        fig, ax = distribution_plot(
            data,
            kind=kind,
            title=f"Intensity Distribution ({kind.upper()})",
            xlabel="Intensity (a.u.)",
        )
        out = OUTPUT_DIR / f"distribution_{kind}.png"
        save_figure(fig, out, dpi=150)
        print(f"  Saved: {out}")

    print(f"{CHECK} Distribution plots complete")


def example_box_and_violin():
    """Example 2: Box and violin plots comparing groups."""
    print("\n" + "=" * 70)
    print("Example 2: Box and Violin Plots")
    print("=" * 70)

    df = make_sample_df()

    fig, ax = box_plot(df, x="solvent", y="intensity", kind="box", title="Intensity by Solvent")
    out = OUTPUT_DIR / "box_plot.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")

    fig, ax = box_plot(
        df,
        x="solvent",
        y="intensity",
        kind="violin",
        title="Intensity Distribution by Solvent (Violin)",
    )
    out = OUTPUT_DIR / "violin_plot.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")

    print(f"{CHECK} Box and violin plots complete")


def example_regression_plot():
    """Example 3: Regression plot — concentration vs. intensity."""
    print("\n" + "=" * 70)
    print("Example 3: Regression Plot")
    print("=" * 70)

    df = make_sample_df()

    fig, ax = regression_plot(
        x="concentration",
        y="intensity",
        data=df,
        order=1,
        title="Intensity vs. Concentration (Linear Regression)",
        xlabel="Concentration (mM)",
        ylabel="Intensity (a.u.)",
    )

    out = OUTPUT_DIR / "regression_plot.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Regression plot complete")


if __name__ == "__main__":
    print("EaglePlot — Seaborn Statistical Plot Examples")
    print("=" * 70)
    example_distribution_plots()
    example_box_and_violin()
    example_regression_plot()
    print("\nAll Seaborn examples complete.")
