"""Unit tests for the math module.

This module tests all mathematical and statistical functions including:
- Basic statistics
- Distribution analysis
- Curve fitting
- Optimization
- Linear algebra
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Import functions to test
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
    # ANOVA
    anova_twoway,
)


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Sample 1D data for testing."""
    np.random.seed(42)
    return np.random.normal(5, 2, 100)


@pytest.fixture
def sample_2d_data():
    """Sample 2D data for testing."""
    np.random.seed(42)
    return np.random.rand(10, 5)


@pytest.fixture
def linear_data():
    """Perfect linear relationship data."""
    x = np.linspace(0, 10, 50)
    y = 2 * x + 3
    return x, y


@pytest.fixture
def noisy_linear_data():
    """Linear data with noise."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 3 + np.random.normal(0, 1, 50)
    return x, y


# ============================================================================
# Basic Statistics Tests
# ============================================================================


class TestBasicStatistics:
    """Tests for basic statistical functions."""

    def test_calculate_mean(self, sample_data):
        """Test mean calculation."""
        mean = calculate_mean(sample_data)
        assert isinstance(mean, (float, np.floating))
        assert np.isclose(mean, np.mean(sample_data))

    def test_calculate_mean_2d(self, sample_2d_data):
        """Test mean with 2D data and axis parameter."""
        mean_all = calculate_mean(sample_2d_data)
        mean_axis0 = calculate_mean(sample_2d_data, axis=0)
        mean_axis1 = calculate_mean(sample_2d_data, axis=1)

        assert isinstance(mean_all, (float, np.floating))
        assert mean_axis0.shape == (5,)
        assert mean_axis1.shape == (10,)

    def test_calculate_median(self, sample_data):
        """Test median calculation."""
        median = calculate_median(sample_data)
        assert isinstance(median, (float, np.floating))
        assert np.isclose(median, np.median(sample_data))

    def test_calculate_std(self, sample_data):
        """Test standard deviation calculation."""
        std_pop = calculate_std(sample_data, ddof=0)
        std_sample = calculate_std(sample_data, ddof=1)

        assert std_sample > std_pop  # Sample std should be larger
        assert np.isclose(std_pop, np.std(sample_data, ddof=0))

    def test_calculate_statistics(self, sample_data):
        """Test comprehensive statistics."""
        stats = calculate_statistics(sample_data)

        required_keys = ["mean", "median", "std", "var", "min", "max", "q1", "q3", "iqr", "range"]
        assert all(key in stats for key in required_keys)

        # Verify relationships
        assert stats["min"] < stats["mean"] < stats["max"]
        assert stats["q1"] < stats["median"] < stats["q3"]
        assert np.isclose(stats["iqr"], stats["q3"] - stats["q1"])
        assert np.isclose(stats["range"], stats["max"] - stats["min"])


# ============================================================================
# Distribution Analysis Tests
# ============================================================================


class TestDistributionAnalysis:
    """Tests for distribution analysis functions."""

    def test_check_normality_normal_data(self):
        """Test normality test with normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = check_normality(data)

        assert "statistic" in result
        assert "p_value" in result
        assert "is_normal" in result
        assert isinstance(result["is_normal"], bool)
        # Large normal sample should pass normality test
        assert result["is_normal"] is True

    def test_check_normality_uniform_data(self):
        """Test normality test with non-normal data."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 1000)
        result = check_normality(data)

        # Uniform distribution should fail normality test
        assert result["is_normal"] is False

    def test_fit_distribution_normal(self):
        """Test fitting normal distribution."""
        np.random.seed(42)
        data = np.random.normal(5, 2, 1000)
        result = fit_distribution(data, "norm")

        assert "params" in result
        assert "distribution" in result
        assert result["distribution"] == "norm"

        # Check fitted parameters are close to true values
        mu, sigma = result["params"]
        assert np.isclose(mu, 5, atol=0.2)
        assert np.isclose(sigma, 2, atol=0.2)

    def test_fit_distribution_exponential(self):
        """Test fitting exponential distribution."""
        np.random.seed(42)
        data = np.random.exponential(2, 1000)
        result = fit_distribution(data, "expon")

        assert result["distribution"] == "expon"
        assert "ks_statistic" in result
        assert "ks_pvalue" in result

    def test_fit_distribution_invalid(self):
        """Test fitting with invalid distribution name."""
        data = np.random.rand(100)
        with pytest.raises(ValueError):
            fit_distribution(data, "invalid_dist")


# ============================================================================
# Curve Fitting Tests
# ============================================================================


class TestCurveFitting:
    """Tests for curve fitting functions."""

    def test_fit_polynomial_degree1(self, linear_data):
        """Test polynomial fit with degree 1 (linear)."""
        x, y = linear_data
        result = fit_polynomial(x, y, degree=1)

        assert "coefficients" in result
        assert "r_squared" in result
        assert len(result["coefficients"]) == 2

        # Perfect fit should have R² = 1
        assert np.isclose(result["r_squared"], 1.0, atol=1e-10)

        # Check coefficients (slope=2, intercept=3)
        slope, intercept = result["coefficients"]
        assert np.isclose(slope, 2.0, atol=1e-10)
        assert np.isclose(intercept, 3.0, atol=1e-10)

    def test_fit_polynomial_prediction(self, noisy_linear_data):
        """Test polynomial fit prediction function."""
        x, y = noisy_linear_data
        result = fit_polynomial(x, y, degree=1)

        # Test prediction
        x_new = np.array([5.0, 10.0, 15.0])
        y_pred = result["predict"](x_new)

        assert len(y_pred) == len(x_new)
        assert all(isinstance(val, (float, np.floating)) for val in y_pred)

    def test_fit_linear(self, noisy_linear_data):
        """Test linear regression."""
        x, y = noisy_linear_data
        result = fit_linear(x, y)

        required_keys = ["slope", "intercept", "r_value", "r_squared", "p_value", "std_err"]
        assert all(key in result for key in required_keys)

        # Check reasonable fit (R² should be high for noisy linear data)
        assert result["r_squared"] > 0.8

        # Slope should be close to 2
        assert np.isclose(result["slope"], 2.0, atol=0.5)

    def test_fit_exponential(self):
        """Test exponential fit."""
        np.random.seed(42)
        x = np.linspace(0, 2, 50)
        y_true = 2 * np.exp(0.5 * x) + 1
        y = y_true + np.random.normal(0, 0.1, 50)

        result = fit_exponential(x, y)

        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "r_squared" in result

        # Check parameters are reasonable
        assert np.isclose(result["a"], 2, atol=0.5)
        assert np.isclose(result["b"], 0.5, atol=0.2)
        assert np.isclose(result["c"], 1, atol=0.5)

    def test_fit_exponential_predict(self):
        """Test that the predict function returned by fit_exponential works."""
        x = np.linspace(0, 2, 50)
        y = 2 * np.exp(0.5 * x) + 1

        result = fit_exponential(x, y)
        y_pred = result["predict"](np.array([0.0, 1.0, 2.0]))

        assert len(y_pred) == 3
        assert np.isclose(y_pred[0], 3.0, atol=0.1)  # 2*exp(0)+1 = 3

    def test_fit_exponential_convergence_failure(self):
        """Test RuntimeError is raised when exponential fit cannot converge."""
        from unittest.mock import patch

        x = np.linspace(0, 1, 10)
        y = np.ones(10)

        with patch("plottle.math.optimize.curve_fit", side_effect=RuntimeError("did not converge")):
            with pytest.raises(RuntimeError, match="Exponential fit failed"):
                fit_exponential(x, y)

    def test_fit_custom_gaussian(self):
        """Test custom function fitting with Gaussian."""

        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        np.random.seed(42)
        x = np.linspace(-5, 5, 100)
        y_true = gaussian(x, 10, 0, 1)
        y = y_true + np.random.normal(0, 0.5, 100)

        result = fit_custom(x, y, gaussian, p0=[8, 0, 1])

        assert "parameters" in result
        assert "covariance" in result
        assert "std_errors" in result
        assert "r_squared" in result

        amp, mu, sigma = result["parameters"]
        assert np.isclose(amp, 10, atol=2)
        assert np.isclose(mu, 0, atol=0.5)
        assert np.isclose(sigma, 1, atol=0.5)

    def test_fit_custom_predict(self):
        """Test that the predict function returned by fit_custom works."""

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 5, 50)
        y = 3 * x + 1

        result = fit_custom(x, y, linear, p0=[1, 0])
        y_pred = result["predict"](np.array([0.0, 1.0, 2.0]))

        assert len(y_pred) == 3
        assert np.isclose(y_pred[0], 1.0, atol=0.1)
        assert np.isclose(y_pred[1], 4.0, atol=0.1)

    def test_fit_custom_convergence_failure(self):
        """Test RuntimeError is raised when custom fit cannot converge."""
        from unittest.mock import patch

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 1, 10)
        y = x

        with patch("plottle.math.optimize.curve_fit", side_effect=RuntimeError("did not converge")):
            with pytest.raises(RuntimeError, match="Custom fit failed"):
                fit_custom(x, y, linear, p0=[1, 0])


# ============================================================================
# Optimization Tests
# ============================================================================


class TestOptimization:
    """Tests for optimization functions."""

    def test_minimize_function_rosenbrock(self):
        """Test minimization with Rosenbrock function."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = minimize_function(rosenbrock, x0=[0, 0], method="BFGS")

        assert "x" in result
        assert "fun" in result
        assert "success" in result

        # Rosenbrock minimum is at (1, 1)
        assert np.allclose(result["x"], [1, 1], atol=1e-4)
        assert np.isclose(result["fun"], 0, atol=1e-6)

    def test_minimize_function_simple(self):
        """Test minimization with simple quadratic."""

        def quadratic(x):
            return (x[0] - 2) ** 2 + (x[1] + 3) ** 2

        result = minimize_function(quadratic, x0=[0, 0])

        assert result["success"] is True
        assert np.isclose(result["x"][0], 2, atol=0.01)
        assert np.isclose(result["x"][1], -3, atol=0.01)

    def test_find_roots_simple(self):
        """Test root finding for simple function."""

        def func(x):
            return x**2 - 4

        result = find_roots(func, bracket=(0, 3))

        assert "root" in result
        assert "converged" in result
        assert result["converged"] is True

        # Root should be at x=2
        assert np.isclose(result["root"], 2.0, atol=1e-6)

    def test_find_roots_trigonometric(self):
        """Test root finding for trigonometric function."""

        def func(x):
            return np.sin(x)

        result = find_roots(func, bracket=(2, 4))

        # Root should be at x=π
        assert np.isclose(result["root"], np.pi, atol=1e-6)

    def test_find_roots_no_sign_change(self):
        """Test ValueError when bracket does not contain a sign change."""

        def func(x):
            return x**2 + 1  # Always positive

        with pytest.raises(ValueError):
            find_roots(func, bracket=(0, 3))


# ============================================================================
# Linear Algebra Tests
# ============================================================================


class TestLinearAlgebra:
    """Tests for linear algebra functions."""

    def test_compute_eigenvalues_2x2(self):
        """Test eigenvalue computation for 2x2 matrix."""
        A = np.array([[1, 2], [2, 1]])
        result = compute_eigenvalues(A)

        assert "eigenvalues" in result
        assert "eigenvectors" in result

        # Known eigenvalues: 3 and -1
        eigenvals = np.sort(result["eigenvalues"].real)
        assert np.isclose(eigenvals[0], -1, atol=1e-10)
        assert np.isclose(eigenvals[1], 3, atol=1e-10)

    def test_compute_eigenvalues_no_vectors(self):
        """Test eigenvalue computation without eigenvectors."""
        A = np.array([[4, 2], [1, 3]])
        result = compute_eigenvalues(A, eigenvectors=False)

        assert "eigenvalues" in result
        assert "eigenvectors" not in result

    def test_compute_eigenvalues_nonsquare(self):
        """Test that non-square matrix raises error."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            compute_eigenvalues(A)

    def test_solve_linear_system_simple(self):
        """Test solving simple linear system."""
        A = np.array([[3, 1], [1, 2]])
        b = np.array([9, 8])

        result = solve_linear_system(A, b)

        assert "x" in result
        assert "residual" in result
        assert "condition_number" in result

        # Known solution: x = [2, 3]
        assert np.allclose(result["x"], [2, 3])
        assert result["residual"] < 1e-10

    def test_solve_linear_system_3x3(self):
        """Test solving 3x3 linear system."""
        A = np.array([[1, 2, 3], [2, 5, 3], [1, 0, 8]])
        b = np.array([1, 2, 3])

        result = solve_linear_system(A, b)

        # Verify solution
        x = result["x"]
        assert np.allclose(A @ x, b)

    def test_matrix_decomposition_svd(self):
        """Test SVD decomposition."""
        A = np.random.rand(5, 3)
        result = matrix_decomposition(A, method="svd")

        assert "U" in result
        assert "S" in result
        assert "Vh" in result

        # Reconstruct matrix
        U, S, Vh = result["U"], result["S"], result["Vh"]
        A_reconstructed = U[:, : len(S)] @ np.diag(S) @ Vh
        assert np.allclose(A, A_reconstructed)

    def test_matrix_decomposition_qr(self):
        """Test QR decomposition."""
        A = np.random.rand(5, 3)
        result = matrix_decomposition(A, method="qr")

        assert "Q" in result
        assert "R" in result

        # Reconstruct matrix
        Q, R = result["Q"], result["R"]
        A_reconstructed = Q @ R
        assert np.allclose(A, A_reconstructed)

    def test_matrix_decomposition_lu(self):
        """Test LU decomposition."""
        A = np.random.rand(4, 4)
        result = matrix_decomposition(A, method="lu")

        assert "P" in result
        assert "L" in result
        assert "U" in result

        # Reconstruct matrix
        P, L, U = result["P"], result["L"], result["U"]
        A_reconstructed = P @ L @ U
        assert np.allclose(A, A_reconstructed)

    def test_matrix_decomposition_cholesky(self):
        """Test Cholesky decomposition."""
        # Create positive definite matrix
        A = np.array([[4, 2], [2, 3]])
        result = matrix_decomposition(A, method="cholesky")

        assert "L" in result

        # Reconstruct matrix
        L = result["L"]
        A_reconstructed = L @ L.T
        assert np.allclose(A, A_reconstructed)

    def test_matrix_decomposition_invalid_method(self):
        """Test invalid decomposition method."""
        A = np.random.rand(3, 3)
        with pytest.raises(ValueError):
            matrix_decomposition(A, method="invalid")

    def test_matrix_decomposition_cholesky_nonsquare(self):
        """Test Cholesky decomposition with non-square matrix raises ValueError."""
        A = np.random.rand(3, 4)
        with pytest.raises(ValueError, match="square"):
            matrix_decomposition(A, method="cholesky")


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self):
        """Test that empty arrays are handled properly."""
        data = np.array([])
        # NumPy returns nan for empty arrays with a warning
        with pytest.warns(RuntimeWarning):
            mean = calculate_mean(data)
            assert np.isnan(mean)

    def test_single_value(self):
        """Test with single value."""
        data = np.array([5.0])
        mean = calculate_mean(data)
        assert mean == 5.0

        std = calculate_std(data, ddof=0)
        assert std == 0.0

    def test_constant_data(self):
        """Test with constant data."""
        data = np.ones(100)
        stats = calculate_statistics(data)

        assert stats["mean"] == 1.0
        assert stats["std"] == 0.0
        assert stats["range"] == 0.0


# ============================================================================
# Two-Way ANOVA Tests
# ============================================================================


class TestTwoWayANOVA:
    """Tests for anova_twoway (requires statsmodels)."""

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_basic(self):
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 30
        a = np.tile(["A1", "A2", "A3"], 10)
        b = np.repeat(["B1", "B2"], 15)
        y = rng.normal(0, 1, n)
        df = pd.DataFrame({"response": y, "FactorA": a, "FactorB": b})
        result = anova_twoway(df, "response", "FactorA", "FactorB")
        assert "table" in result
        assert "significant_a" in result
        assert isinstance(result["significant_a"], bool)

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_no_interaction(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "y": rng.normal(0, 1, 20),
                "A": np.tile(["a", "b"], 10),
                "B": np.tile(["x", "y", "x", "y"], 5),
            }
        )
        result = anova_twoway(df, "y", "A", "B", include_interaction=False)
        assert result["interaction_included"] is False
        assert result["significant_interaction"] is None

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_result_has_required_keys(self):
        import pandas as pd

        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "response": rng.normal(5, 1, 24),
                "A": np.tile(["low", "high"], 12),
                "B": np.tile(["x", "y", "z", "x", "y", "z"], 4),
            }
        )
        result = anova_twoway(df, "response", "A", "B")
        for key in (
            "table",
            "factor_a",
            "factor_b",
            "interaction_included",
            "significant_a",
            "significant_b",
            "significant_interaction",
            "p_value_a",
            "p_value_b",
            "p_value_interaction",
        ):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.skipif(not _has_statsmodels(), reason="statsmodels not installed")
    def test_p_values_are_floats(self):
        import pandas as pd

        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "y": rng.normal(0, 1, 20),
                "A": np.tile(["a", "b"], 10),
                "B": np.tile(["x", "y", "x", "y"], 5),
            }
        )
        result = anova_twoway(df, "y", "A", "B")
        assert isinstance(result["p_value_a"], float)
        assert isinstance(result["p_value_b"], float)

    def test_no_statsmodels_raises(self):
        import unittest.mock as mock
        import pandas as pd

        df = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5, 6],
                "A": ["a", "b", "a", "b", "a", "b"],
                "B": ["x", "x", "y", "y", "x", "y"],
            }
        )
        with mock.patch.dict(
            "sys.modules",
            {
                "statsmodels": None,
                "statsmodels.formula": None,
                "statsmodels.formula.api": None,
                "statsmodels.stats": None,
                "statsmodels.stats.anova": None,
            },
        ):
            with pytest.raises(ImportError, match="statsmodels"):
                anova_twoway(df, "y", "A", "B")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
