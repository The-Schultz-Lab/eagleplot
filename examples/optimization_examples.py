"""Examples for optimization and root-finding functions.

Demonstrates minimize_function() and find_roots() with common scientific problems.

Run from the repo root:
    python examples/optimization_examples.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.math import minimize_function, find_roots
from plottle.plotting import line_plot, save_figure

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def example_minimize_parabola():
    """Example 1: Minimize a simple quadratic (known minimum)."""
    print("\n" + "=" * 70)
    print("Example 1: Minimize a Parabola")
    print("=" * 70)

    def parabola(x):
        return (x - 3.0) ** 2 + 1.5

    result = minimize_function(parabola, x0=0.0)

    print(f"  x_opt   : {result['x_opt']:.6f}  (true: 3.0)")
    print(f"  f(x)    : {result['fun']:.6f}  (true: 1.5)")
    print(f"  success : {result['success']}")
    print(f"{CHECK} Parabola minimized")


def example_minimize_rosenbrock():
    """Example 2: Minimize the 2-D Rosenbrock function."""
    print("\n" + "=" * 70)
    print("Example 2: Rosenbrock Function Minimization")
    print("=" * 70)

    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    result = minimize_function(rosenbrock, x0=[-1.0, 1.0], method="Nelder-Mead")

    print(f"  x_opt   : ({result['x_opt'][0]:.6f}, {result['x_opt'][1]:.6f})  (true: [1, 1])")
    print(f"  f(x)    : {result['fun']:.2e}")
    print(f"  success : {result['success']}")
    print(f"{CHECK} Rosenbrock minimized")


def example_find_roots_polynomial():
    """Example 3: Find roots of a polynomial."""
    print("\n" + "=" * 70)
    print("Example 3: Root Finding — Cubic Polynomial")
    print("=" * 70)

    # f(x) = x^3 - 6x^2 + 11x - 6  has roots at x=1, x=2, x=3
    def f(x):
        return x**3 - 6 * x**2 + 11 * x - 6

    brackets = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
    true_roots = [1.0, 2.0, 3.0]

    x = np.linspace(0, 4, 300)
    fig, ax = line_plot(x, f(x), xlabel="x", ylabel="f(x)", title="Roots of x^3 - 6x^2 + 11x - 6")
    ax.axhline(0, color="gray", linewidth=0.8)

    for bracket, true_root in zip(brackets, true_roots):
        result = find_roots(f, bracket=bracket)
        print(f"  root in {bracket}: x = {result['root']:.6f}  (true: {true_root:.1f})")
        ax.axvline(result["root"], color="red", linestyle="--", linewidth=1, alpha=0.7)

    out = OUTPUT_DIR / "roots_polynomial.png"
    save_figure(fig, out, dpi=150)
    print(f"  Saved: {out}")
    print(f"{CHECK} Root finding complete")


def example_find_roots_transcendental():
    """Example 4: Root of a transcendental equation."""
    print("\n" + "=" * 70)
    print("Example 4: Root of sin(x) = x/3")
    print("=" * 70)

    def f(x):
        return np.sin(x) - x / 3.0

    result = find_roots(f, bracket=(1.0, 3.0))

    print(f"  root     : {result['root']:.6f}")
    print(f"  f(root)  : {f(result['root']):.2e}")
    print(f"  converged: {result['converged']}")
    print(f"{CHECK} Transcendental root found")


if __name__ == "__main__":
    print("Plottle — Optimization Examples")
    print("=" * 70)
    example_minimize_parabola()
    example_minimize_rosenbrock()
    example_find_roots_polynomial()
    example_find_roots_transcendental()
    print("\nAll optimization examples complete.")
