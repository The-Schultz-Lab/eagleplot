"""Examples for using the math module.

This script demonstrates how to use the plottle math module
for statistical analysis, curve fitting, optimization, and linear algebra.

Run this script from the plottle directory:
    python examples/math_examples.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import plottle
sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.math import (
    # Statistics
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_statistics,
    # Distribution analysis
    check_normality,
    fit_distribution,
    # Curve fitting
    fit_polynomial,
    fit_linear,
    fit_exponential,
    fit_custom,
    # Optimization
    minimize_function,
    find_roots,
    # Linear algebra
    compute_eigenvalues,
    solve_linear_system,
    matrix_decomposition,
)

# Use ASCII-safe symbols for Windows compatibility
CHECK = "[OK]"
ARROW = "-->"


def example_basic_statistics():
    """Example 1: Basic statistical analysis."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Statistical Analysis")
    print("=" * 70)

    # Simulate experimental measurements
    np.random.seed(42)
    measurements = np.random.normal(100, 5, 50)  # 50 measurements around 100

    print(f"\n{CHECK} Generated 50 experimental measurements")
    print(f"  First 10 values: {measurements[:10].round(2)}")

    # Calculate individual statistics
    mean = calculate_mean(measurements)
    median = calculate_median(measurements)
    std = calculate_std(measurements, ddof=1)

    print(f"\n{CHECK} Individual statistics:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Median: {median:.2f}")
    print(f"  Std Dev: {std:.2f}")

    # Calculate comprehensive statistics
    stats = calculate_statistics(measurements)

    print(f"\n{CHECK} Comprehensive statistics:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")
    print(f"  Q1 (25%): {stats['q1']:.2f}")
    print(f"  Q3 (75%): {stats['q3']:.2f}")
    print(f"  IQR: {stats['iqr']:.2f}")
    print(f"  Range: {stats['range']:.2f}")


def example_distribution_analysis():
    """Example 2: Distribution analysis and fitting."""
    print("\n" + "=" * 70)
    print("Example 2: Distribution Analysis")
    print("=" * 70)

    # Generate normal distribution data
    np.random.seed(42)
    normal_data = np.random.normal(5, 2, 1000)

    print(f"\n{CHECK} Testing normality of data...")
    result = check_normality(normal_data)

    print(f"  Shapiro-Wilk statistic: {result['statistic']:.4f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Is normal? {result['is_normal']}")

    # Fit normal distribution
    print(f"\n{CHECK} Fitting normal distribution...")
    fit_result = fit_distribution(normal_data, "norm")

    mu, sigma = fit_result["params"]
    print("  Fitted parameters:")
    print(f"    Mean (mu): {mu:.2f}")
    print(f"    Std (sigma): {sigma:.2f}")
    print(f"  KS test p-value: {fit_result['ks_pvalue']:.4f}")
    print(f"  {ARROW} Good fit: p-value > 0.05")

    # Generate exponential data and fit
    print(f"\n{CHECK} Testing exponential distribution...")
    exp_data = np.random.exponential(2, 1000)
    exp_result = fit_distribution(exp_data, "expon")

    print("  Fitted exponential distribution")
    print(f"  KS test p-value: {exp_result['ks_pvalue']:.4f}")


def example_curve_fitting():
    """Example 3: Curve fitting techniques."""
    print("\n" + "=" * 70)
    print("Example 3: Curve Fitting")
    print("=" * 70)

    # Generate linear data with noise
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = 2.5 * x + 3.0
    y_noisy = y_true + np.random.normal(0, 1, 50)

    print(f"\n{CHECK} Generated noisy linear data: y = 2.5x + 3.0")

    # Fit linear model
    linear_result = fit_linear(x, y_noisy)

    print(f"\n{CHECK} Linear regression results:")
    print(f"  Slope: {linear_result['slope']:.3f}")
    print(f"  Intercept: {linear_result['intercept']:.3f}")
    print(f"  R-squared: {linear_result['r_squared']:.4f}")
    print(f"  P-value: {linear_result['p_value']:.2e}")

    # Fit polynomial
    poly_result = fit_polynomial(x, y_noisy, degree=1)

    print(f"\n{CHECK} Polynomial fit (degree 1):")
    print(f"  Coefficients: {poly_result['coefficients']}")
    print(f"  R-squared: {poly_result['r_squared']:.4f}")

    # Predict new values
    x_new = np.array([5.0, 7.5, 10.0])
    y_pred = poly_result["predict"](x_new)

    print(f"\n{CHECK} Predictions at new x values:")
    for xi, yi in zip(x_new, y_pred):
        print(f"  x={xi:.1f} {ARROW} y={yi:.2f}")

    # Generate exponential data
    print(f"\n{CHECK} Fitting exponential data: y = 2*exp(0.5*x) + 1")
    x_exp = np.linspace(0, 2, 50)
    y_exp = 2 * np.exp(0.5 * x_exp) + 1 + np.random.normal(0, 0.1, 50)

    exp_fit = fit_exponential(x_exp, y_exp)

    print("  Fitted parameters:")
    print(f"    a = {exp_fit['a']:.3f}")
    print(f"    b = {exp_fit['b']:.3f}")
    print(f"    c = {exp_fit['c']:.3f}")
    print(f"  R-squared: {exp_fit['r_squared']:.4f}")

    # Custom function fitting (Gaussian)
    print(f"\n{CHECK} Fitting custom Gaussian function")

    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    x_gauss = np.linspace(-5, 5, 100)
    y_gauss = gaussian(x_gauss, 10, 0, 1) + np.random.normal(0, 0.5, 100)

    gauss_fit = fit_custom(x_gauss, y_gauss, gaussian, p0=[8, 0, 1])

    amp, mu, sigma = gauss_fit["parameters"]
    print("  Fitted Gaussian parameters:")
    print(f"    Amplitude: {amp:.2f}")
    print(f"    Mean (mu): {mu:.2f}")
    print(f"    Std (sigma): {sigma:.2f}")
    print(f"  R-squared: {gauss_fit['r_squared']:.4f}")


def example_optimization():
    """Example 4: Optimization and root finding."""
    print("\n" + "=" * 70)
    print("Example 4: Optimization and Root Finding")
    print("=" * 70)

    # Minimize the Rosenbrock function
    print(f"\n{CHECK} Minimizing Rosenbrock function")
    print("  f(x,y) = (1-x)^2 + 100*(y-x^2)^2")

    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    result = minimize_function(rosenbrock, x0=[0, 0], method="BFGS")

    print("  Starting point: [0, 0]")
    print(f"  Optimal point: [{result['x'][0]:.4f}, {result['x'][1]:.4f}]")
    print(f"  Minimum value: {result['fun']:.6f}")
    print(f"  Success: {result['success']}")
    print(f"  Function evaluations: {result['nfev']}")

    # Minimize a simple quadratic
    print(f"\n{CHECK} Minimizing quadratic function")
    print("  f(x,y) = (x-3)^2 + (y+2)^2")

    def quadratic(x):
        return (x[0] - 3) ** 2 + (x[1] + 2) ** 2

    result2 = minimize_function(quadratic, x0=[0, 0])

    print(f"  Optimal point: [{result2['x'][0]:.4f}, {result2['x'][1]:.4f}]")
    print(f"  {ARROW} Expected minimum at [3, -2]")

    # Find roots
    print(f"\n{CHECK} Finding roots of function")
    print("  f(x) = x^2 - 4")

    def func(x):
        return x**2 - 4

    root_result = find_roots(func, bracket=(0, 3))

    print(f"  Root found: x = {root_result['root']:.6f}")
    print(f"  {ARROW} Expected root at x = 2")
    print(f"  Iterations: {root_result['iterations']}")
    print(f"  Converged: {root_result['converged']}")

    # Find root of trigonometric function
    print(f"\n{CHECK} Finding root of sin(x)")

    trig_result = find_roots(np.sin, bracket=(2, 4))

    print(f"  Root found: x = {trig_result['root']:.6f}")
    print(f"  {ARROW} Expected root at x = pi = {np.pi:.6f}")


def example_linear_algebra():
    """Example 5: Linear algebra operations."""
    print("\n" + "=" * 70)
    print("Example 5: Linear Algebra")
    print("=" * 70)

    # Compute eigenvalues
    print(f"\n{CHECK} Computing eigenvalues and eigenvectors")
    A = np.array([[1, 2], [2, 1]])

    print("  Matrix A:")
    print(f"    [[{A[0, 0]}, {A[0, 1]}]")
    print(f"     [{A[1, 0]}, {A[1, 1]}]]")

    eigen_result = compute_eigenvalues(A)

    eigenvals = np.sort(eigen_result["eigenvalues"].real)
    print(f"\n  Eigenvalues: [{eigenvals[0]:.2f}, {eigenvals[1]:.2f}]")
    print(f"  {ARROW} Expected: [-1, 3]")

    # Solve linear system
    print(f"\n{CHECK} Solving linear system Ax = b")
    A_sys = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])

    print("  System:")
    print("    3x + 1y = 9")
    print("    1x + 2y = 8")

    sol_result = solve_linear_system(A_sys, b)

    print(f"\n  Solution: x = {sol_result['x']}")
    print(f"  {ARROW} Expected: [2, 3]")
    print(f"  Residual: {sol_result['residual']:.2e}")
    print(f"  Condition number: {sol_result['condition_number']:.2f}")

    # Matrix decomposition - SVD
    print(f"\n{CHECK} Matrix decomposition (SVD)")
    M = np.random.rand(5, 3)

    svd_result = matrix_decomposition(M, method="svd")

    print(f"  Original matrix shape: {M.shape}")
    print(f"  U shape: {svd_result['U'].shape}")
    print(f"  S shape: {svd_result['S'].shape}")
    print(f"  Vh shape: {svd_result['Vh'].shape}")

    # Reconstruct matrix
    U, S, Vh = svd_result["U"], svd_result["S"], svd_result["Vh"]
    M_reconstructed = U[:, : len(S)] @ np.diag(S) @ Vh
    reconstruction_error = np.linalg.norm(M - M_reconstructed)

    print(f"  Reconstruction error: {reconstruction_error:.2e}")
    print(f"  {ARROW} Should be near zero")

    # QR decomposition
    print(f"\n{CHECK} QR decomposition")
    qr_result = matrix_decomposition(M, method="qr")

    Q, R = qr_result["Q"], qr_result["R"]
    print(f"  Q shape: {Q.shape}")
    print(f"  R shape: {R.shape}")

    M_qr = Q @ R
    qr_error = np.linalg.norm(M - M_qr)
    print(f"  Reconstruction error: {qr_error:.2e}")


def example_practical_workflow():
    """Example 6: Practical data analysis workflow."""
    print("\n" + "=" * 70)
    print("Example 6: Practical Workflow - Kinetics Data Analysis")
    print("=" * 70)

    print("\n1. Simulating chemical kinetics experiment...")

    # Simulate concentration vs time data for first-order reaction
    # C(t) = C0 * exp(-k*t)
    np.random.seed(42)
    t = np.linspace(0, 10, 50)  # Time in seconds
    C0 = 1.0  # Initial concentration
    k_true = 0.3  # True rate constant
    C_true = C0 * np.exp(-k_true * t)
    C_measured = C_true + np.random.normal(0, 0.02, 50)  # Add noise

    print(f"{CHECK} Generated kinetics data")
    print(f"  True rate constant k = {k_true} s^-1")
    print(f"  Initial concentration C0 = {C0} M")
    print(f"  {len(t)} time points from {t[0]} to {t[-1]} s")

    # Analyze the data
    print("\n2. Statistical analysis of concentration data...")

    stats = calculate_statistics(C_measured)

    print(f"{CHECK} Concentration statistics:")
    print(f"  Mean: {stats['mean']:.4f} M")
    print(f"  Std: {stats['std']:.4f} M")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}] M")

    # Fit exponential decay model
    print("\n3. Fitting exponential decay model...")

    # Define first-order decay function
    def first_order_decay(t, C0, k):
        return C0 * np.exp(-k * t)

    fit_result = fit_custom(t, C_measured, first_order_decay, p0=[1.0, 0.5])

    C0_fit, k_fit = fit_result["parameters"]
    C0_err, k_err = fit_result["std_errors"]

    print(f"{CHECK} Fitted parameters:")
    print(f"  C0 = {C0_fit:.4f} +/- {C0_err:.4f} M")
    print(f"  k = {k_fit:.4f} +/- {k_err:.4f} s^-1")
    print(f"  R-squared: {fit_result['r_squared']:.4f}")
    print(f"  {ARROW} True k = {k_true} s^-1 (within error!)")

    # Calculate half-life
    t_half = np.log(2) / k_fit
    t_half_err = t_half * (k_err / k_fit)  # Error propagation

    print(f"\n{CHECK} Derived properties:")
    print(f"  Half-life: {t_half:.2f} +/- {t_half_err:.2f} s")

    # Predict concentrations at new time points
    print("\n4. Predicting future concentrations...")

    t_future = np.array([12, 15, 20])
    C_future = fit_result["predict"](t_future)

    print(f"{CHECK} Concentration predictions:")
    for ti, Ci in zip(t_future, C_future):
        print(f"  t = {ti:2.0f} s {ARROW} C = {Ci:.4f} M")

    # Check residuals
    print("\n5. Residual analysis...")

    C_fitted = fit_result["predict"](t)
    residuals = C_measured - C_fitted

    residual_stats = calculate_statistics(residuals)

    print(f"{CHECK} Residual statistics:")
    print(f"  Mean: {residual_stats['mean']:.2e} (should be ~0)")
    print(f"  Std: {residual_stats['std']:.4f}")
    print(f"  Range: [{residual_stats['min']:.4f}, {residual_stats['max']:.4f}]")

    # Test normality of residuals
    normality = check_normality(residuals)

    print(f"\n{CHECK} Testing residual normality:")
    print(f"  Is normal: {normality['is_normal']}")
    print(f"  P-value: {normality['p_value']:.4f}")
    print(f"  {ARROW} Good fit should have normally distributed residuals")

    # Determine reaction progress at specific time
    print("\n6. Reaction progress analysis...")

    t_target = 5.0
    C_at_5s = fit_result["predict"](np.array([t_target]))[0]
    conversion = (C0_fit - C_at_5s) / C0_fit * 100

    print(f"{CHECK} At t = {t_target} s:")
    print(f"  Concentration: {C_at_5s:.4f} M")
    print(f"  Conversion: {conversion:.1f}%")

    print("\n" + "=" * 70)
    print("Analysis complete! This workflow demonstrated:")
    print("  - Statistical analysis of experimental data")
    print("  - Non-linear curve fitting")
    print("  - Parameter estimation with uncertainties")
    print("  - Prediction and extrapolation")
    print("  - Residual analysis and quality assessment")
    print("=" * 70)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PLOTTING HELPER - MATH MODULE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the mathematical capabilities of the")
    print("plottle package for scientific data analysis.\n")

    try:
        # Run all examples
        example_basic_statistics()
        example_distribution_analysis()
        example_curve_fitting()
        example_optimization()
        example_linear_algebra()
        example_practical_workflow()

        print("\n" + "=" * 70)
        print(f"All examples completed successfully! {CHECK}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
