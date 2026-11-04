"""Complete Beer-Lambert calibration curve workflow.

End-to-end example: load calibration data, fit a linear model, compute
unknown concentrations from measured absorbances, and export a
publication-ready figure.

Run from the repo root:
    python examples/calibration_workflow.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.math import fit_linear
from plottle.plotting import scatter_plot, apply_publication_style, save_figure

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def generate_calibration_data():
    """Return a realistic calibration DataFrame (synthetic data)."""
    np.random.seed(42)
    concentrations = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    epsilon_l = 2.18  # molar absorptivity * path length
    absorbances = epsilon_l * concentrations + np.random.normal(0, 0.015, len(concentrations))
    return pd.DataFrame({"concentration": concentrations, "absorbance": absorbances})


def run_calibration_workflow():
    """Full Beer-Lambert calibration workflow."""
    print("\n" + "=" * 70)
    print("Beer-Lambert Calibration Workflow")
    print("=" * 70)

    # 1. Load calibration data
    df = generate_calibration_data()
    print(f"\nStep 1 — Calibration data ({len(df)} points):")
    print(
        f"  Concentration range: {df['concentration'].min():.2f} – "
        f"{df['concentration'].max():.2f} mM"
    )
    print(f"  Absorbance range   : {df['absorbance'].min():.4f} – {df['absorbance'].max():.4f}")
    print(f"  {CHECK} Data loaded")

    # 2. Fit Beer-Lambert: A = epsilon * l * c
    result = fit_linear(df["concentration"].values, df["absorbance"].values)
    epsilon_l = result["slope"]
    print("\nStep 2 — Linear Fit:")
    print(f"  slope  (epsilon*l) : {epsilon_l:.4f} mM^-1")
    print(f"  intercept          : {result['intercept']:.4f}")
    print(f"  R^2                : {result['r_squared']:.6f}")
    print(f"  p-value            : {result['p_value']:.2e}")
    print(f"  {CHECK} Fit complete (R^2 = {result['r_squared']:.4f})")

    # 3. Determine unknown concentrations
    unknown_absorbances = np.array([0.31, 0.67, 1.05, 1.48])
    unknown_concentrations = (unknown_absorbances - result["intercept"]) / epsilon_l

    print("\nStep 3 — Unknown Concentrations:")
    for A, c in zip(unknown_absorbances, unknown_concentrations):
        print(f"  A = {A:.2f}  -->  c = {c:.4f} mM")
    print(f"  {CHECK} Concentrations determined")

    # 4. Publication-quality figure
    c_fit = np.linspace(0, 1.6, 200)
    A_fit = result["slope"] * c_fit + result["intercept"]

    fig, ax = scatter_plot(
        df["concentration"].values,
        df["absorbance"].values,
        xlabel="Concentration (mM)",
        ylabel="Absorbance",
        title=f"Beer-Lambert Calibration (R\u00b2 = {result['r_squared']:.4f})",
        color="steelblue",
    )
    ax.plot(c_fit, A_fit, "k-", linewidth=2, label="Linear fit")
    ax.scatter(
        unknown_concentrations,
        unknown_absorbances,
        marker="*",
        s=120,
        color="red",
        zorder=5,
        label="Unknowns",
    )
    ax.legend()

    apply_publication_style(fig, ax)

    out = OUTPUT_DIR / "beer_lambert_calibration.png"
    save_figure(fig, out, dpi=300)
    print(f"\nStep 4 — Figure saved at 300 dpi: {out}")
    print(f"  {CHECK} Workflow complete")

    return result, unknown_concentrations


if __name__ == "__main__":
    print("Plottle — Beer-Lambert Calibration Workflow")
    result, unknowns = run_calibration_workflow()
    print("\nSummary:")
    print(f"  epsilon*l = {result['slope']:.4f} mM^-1")
    print(f"  Unknown concentrations: {unknowns.round(4)} mM")
