"""Examples for curve fitting functions.

Demonstrates fit_linear(), fit_polynomial(), fit_exponential(), and fit_custom()
with synthetic scientific datasets.

Run from the repo root:
    python examples/curve_fitting_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.math import fit_linear, fit_polynomial, fit_exponential, fit_custom
from plottle.plotting import scatter_plot, save_figure

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def example_linear_fit():
    """Example 1: Linear fit — Beer-Lambert calibration."""
    print("\n" + "=" * 70)
    print("Example 1: Linear Fit (Beer-Lambert Calibration)")
    print("=" * 70)

    np.random.seed(42)
    concentration = np.array([0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    absorbance = 2.1 * concentration + 0.05 + np.random.normal(0, 0.02, len(concentration))

    result = fit_linear(concentration, absorbance)

    print(f"  slope (epsilon*l) : {result['slope']:.4f}")
    print(f"  intercept         : {result['intercept']:.4f}")
    print(f"  R^2               : {result['r_squared']:.6f}")
    print(f"  p-value           : {result['p_value']:.2e}")

    fig, ax = scatter_plot(
        concentration,
        absorbance,
        xlabel="Concentration (mM)",
        ylabel="Absorbance",
        title=f"Beer-Lambert: R^2 = {result['r_squared']:.4f}",
    )
    x_fit = np.linspace(0, 1.6, 100)
    y_fit = result["slope"] * x_fit + result["intercept"]
    ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Linear fit")
    ax.legend()

    out = OUTPUT_DIR / "linear_fit.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Linear fit complete")


def example_polynomial_fit():
    """Example 2: Polynomial fit — potential energy curve."""
    print("\n" + "=" * 70)
    print("Example 2: Polynomial Fit (Potential Energy Curve)")
    print("=" * 70)

    np.random.seed(0)
    r = np.linspace(1.5, 4.0, 40)
    V = 10 * (r - 2.5) ** 2 - 5 + np.random.normal(0, 0.3, len(r))

    result = fit_polynomial(r, V, degree=2)

    print(f"  degree     : {result['degree']}")
    print(f"  R^2        : {result['r_squared']:.6f}")
    print(f"  coeff[0]   : {result['coefficients'][0]:.4f}  (quadratic)")
    print(f"  coeff[1]   : {result['coefficients'][1]:.4f}  (linear)")

    x_fit = np.linspace(1.5, 4.0, 200)
    y_fit = result["predict"](x_fit)

    fig, ax = scatter_plot(
        r, V, xlabel="r (Angstrom)", ylabel="Energy (kcal/mol)", title="Polynomial Fit to PE Curve"
    )
    ax.plot(x_fit, y_fit, "r-", linewidth=2, label=f"Degree-{result['degree']} fit")
    ax.legend()

    out = OUTPUT_DIR / "polynomial_fit.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Polynomial fit complete")


def example_exponential_fit():
    """Example 3: Exponential fit — fluorescence decay."""
    print("\n" + "=" * 70)
    print("Example 3: Exponential Fit (Fluorescence Decay)")
    print("=" * 70)

    np.random.seed(1)
    time = np.linspace(0, 10, 60)
    tau = 2.5
    intensity = 1000 * np.exp(-time / tau) + 10 + np.random.normal(0, 5, len(time))

    result = fit_exponential(time, intensity)

    print(f"  a (amplitude) : {result['a']:.2f}")
    print(f"  b (decay)     : {result['b']:.4f}")
    print(f"  R^2           : {result['r_squared']:.6f}")

    x_fit = np.linspace(0, 10, 200)
    y_fit = result["predict"](x_fit)

    fig, ax = scatter_plot(
        time,
        intensity,
        xlabel="Time (ns)",
        ylabel="Intensity (a.u.)",
        title="Exponential Decay Fit",
    )
    ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Exponential fit")
    ax.legend()

    out = OUTPUT_DIR / "exponential_fit.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Exponential fit complete")


def example_custom_fit():
    """Example 4: Custom function fit — Gaussian peak."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Fit (Gaussian Peak)")
    print("=" * 70)

    def gaussian(x, amplitude, center, width):
        return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))

    np.random.seed(2)
    x = np.linspace(-5, 5, 80)
    y = gaussian(x, amplitude=3.0, center=0.5, width=1.2) + np.random.normal(0, 0.05, len(x))

    result = fit_custom(x, y, func=gaussian, p0=[2.5, 0.0, 1.0])

    p = result["parameters"]
    print(f"  amplitude : {p['amplitude']:.4f}  (true: 3.0)")
    print(f"  center    : {p['center']:.4f}  (true: 0.5)")
    print(f"  width     : {p['width']:.4f}  (true: 1.2)")
    print(f"  R^2       : {result['r_squared']:.6f}")

    x_fit = np.linspace(-5, 5, 300)
    y_fit = result["predict"](x_fit)

    fig, ax = scatter_plot(
        x, y, xlabel="Chemical Shift (ppm)", ylabel="Intensity", title="Custom Gaussian Fit"
    )
    ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Gaussian fit")
    ax.legend()

    out = OUTPUT_DIR / "custom_fit_gaussian.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Custom fit complete")


if __name__ == "__main__":
    print("Plottle — Curve Fitting Examples")
    print("=" * 70)
    example_linear_fit()
    example_polynomial_fit()
    example_exponential_fit()
    example_custom_fit()
    print("\nAll curve fitting examples complete.")
