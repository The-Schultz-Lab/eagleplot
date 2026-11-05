"""Mathematical Operations Module.

This module provides mathematical and statistical functions for
scientific computing, including numerical operations, statistics,
and optimization routines.

Features
--------
- Numerical array operations (NumPy)
- Statistical analysis and distributions (SciPy)
- Optimization and fitting algorithms
- Linear algebra operations

Examples
--------
>>> from eagleplot.math import calculate_mean, fit_polynomial
>>> mean_value = calculate_mean(data)
>>> coefficients = fit_polynomial(x, y, degree=2)
"""

import numpy as np
from scipy import stats, optimize, linalg
from typing import Dict, Tuple, Callable, Optional, Union, Any

# ============================================================================
# Basic Statistics
# ============================================================================


def calculate_mean(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculate the mean of data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to compute the mean. If None, compute over all data.

    Returns
    -------
    mean : float or np.ndarray
        Mean value(s)

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> calculate_mean(data)
    3.0
    """
    return np.mean(data, axis=axis)


def calculate_median(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Calculate the median of data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to compute the median. If None, compute over all data.

    Returns
    -------
    median : float or np.ndarray
        Median value(s)

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> calculate_median(data)
    3.0
    """
    return np.median(data, axis=axis)


def calculate_std(
    data: np.ndarray, axis: Optional[int] = None, ddof: int = 0
) -> Union[float, np.ndarray]:
    """Calculate the standard deviation of data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to compute the std. If None, compute over all data.
    ddof : int, default 0
        Delta degrees of freedom (0 for population, 1 for sample)

    Returns
    -------
    std : float or np.ndarray
        Standard deviation value(s)

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> calculate_std(data, ddof=1)
    1.5811388300841898
    """
    return np.std(data, axis=axis, ddof=ddof)


def calculate_statistics(
    data: np.ndarray, axis: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate comprehensive statistics for data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to compute statistics. If None, compute over all data.

    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'mean': Mean value
        - 'median': Median value
        - 'std': Standard deviation (sample)
        - 'var': Variance (sample)
        - 'min': Minimum value
        - 'max': Maximum value
        - 'q1': First quartile (25th percentile)
        - 'q3': Third quartile (75th percentile)
        - 'iqr': Interquartile range
        - 'range': Data range (max - min)

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> stats = calculate_statistics(data)
    >>> print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    Mean: 3.0, Std: 1.5811388300841898
    """
    return {
        "mean": np.mean(data, axis=axis),
        "median": np.median(data, axis=axis),
        "std": np.std(data, axis=axis, ddof=1),
        "var": np.var(data, axis=axis, ddof=1),
        "min": np.min(data, axis=axis),
        "max": np.max(data, axis=axis),
        "q1": np.percentile(data, 25, axis=axis),
        "q3": np.percentile(data, 75, axis=axis),
        "iqr": np.percentile(data, 75, axis=axis) - np.percentile(data, 25, axis=axis),
        "range": np.ptp(data, axis=axis),  # peak-to-peak (max - min)
    }


# ============================================================================
# Distribution Analysis
# ============================================================================


def check_normality(data: np.ndarray) -> Dict[str, float]:
    """Test if data follows a normal distribution using Shapiro-Wilk test.

    Parameters
    ----------
    data : np.ndarray
        1D array of data to test

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'statistic': Test statistic
        - 'p_value': P-value
        - 'is_normal': Boolean (True if p > 0.05)

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = check_normality(data)
    >>> print(f"Is normal: {result['is_normal']}")
    """
    data = np.asarray(data).flatten()
    statistic, p_value = stats.shapiro(data)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_normal": bool(p_value > 0.05),
    }


def fit_distribution(data: np.ndarray, distribution: str = "norm") -> Dict[str, Any]:
    """Fit a statistical distribution to data.

    Parameters
    ----------
    data : np.ndarray
        1D array of data to fit
    distribution : str, default 'norm'
        Distribution name. Options: 'norm', 'expon', 'gamma', 'lognorm', 'beta'

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'params': Fitted distribution parameters
        - 'distribution': Distribution name
        - 'ks_statistic': Kolmogorov-Smirnov test statistic
        - 'ks_pvalue': KS test p-value

    Raises
    ------
    ValueError
        If distribution name is not supported

    Examples
    --------
    >>> data = np.random.normal(5, 2, 1000)
    >>> result = fit_distribution(data, 'norm')
    >>> print(f"Mean: {result['params'][0]:.2f}, Std: {result['params'][1]:.2f}")
    """
    data = np.asarray(data).flatten()

    # Get distribution from scipy.stats
    dist_map = {
        "norm": stats.norm,
        "expon": stats.expon,
        "gamma": stats.gamma,
        "lognorm": stats.lognorm,
        "beta": stats.beta,
    }

    if distribution not in dist_map:
        raise ValueError(
            f"Unsupported distribution: {distribution}. Supported: {list(dist_map.keys())}"
        )

    dist = dist_map[distribution]

    # Fit distribution
    params = dist.fit(data)

    # Perform Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(data, lambda x: dist.cdf(x, *params))

    return {
        "params": params,
        "distribution": distribution,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }


# ============================================================================
# Curve Fitting
# ============================================================================


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> Dict[str, Any]:
    """Fit polynomial to data and return coefficients with quality metrics.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    degree : int
        Polynomial degree

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'coefficients': Polynomial coefficients (highest degree first)
        - 'degree': Polynomial degree
        - 'r_squared': R² coefficient of determination
        - 'residuals': Sum of squared residuals
        - 'predict': Function to predict y values for new x

    Examples
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2*x + 3 + np.random.normal(0, 1, 50)
    >>> result = fit_polynomial(x, y, degree=1)
    >>> print(f"R²: {result['r_squared']:.3f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)

    # Calculate predictions and R²
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Create prediction function
    def predict(x_new):
        return np.polyval(coeffs, x_new)

    return {
        "coefficients": coeffs,
        "degree": degree,
        "r_squared": float(r_squared),
        "residuals": float(ss_res),
        "predict": predict,
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit linear model y = mx + b and return parameters with statistics.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'slope': Slope (m)
        - 'intercept': Intercept (b)
        - 'r_value': Correlation coefficient
        - 'r_squared': R² coefficient of determination
        - 'p_value': P-value for hypothesis test
        - 'std_err': Standard error of the estimate

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 5, 4, 5])
    >>> result = fit_linear(x, y)
    >>> print(f"y = {result['slope']:.2f}x + {result['intercept']:.2f}")
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fit exponential model y = a * exp(b * x) + c.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable (must be positive)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'a': Amplitude parameter
        - 'b': Exponential rate parameter
        - 'c': Offset parameter
        - 'r_squared': R² coefficient of determination
        - 'predict': Function to predict y values for new x

    Examples
    --------
    >>> x = np.linspace(0, 2, 50)
    >>> y = 2 * np.exp(0.5 * x) + 1 + np.random.normal(0, 0.1, 50)
    >>> result = fit_exponential(x, y)
    >>> print(f"a={result['a']:.2f}, b={result['b']:.2f}, c={result['c']:.2f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Define exponential function
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Initial guess
    p0 = [1.0, 0.1, 0.0]

    try:
        # Fit using curve_fit
        popt, _ = optimize.curve_fit(exp_func, x, y, p0=p0, maxfev=10000)

        # Calculate R²
        y_pred = exp_func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Create prediction function
        def predict(x_new):
            return exp_func(x_new, *popt)

        return {
            "a": float(popt[0]),
            "b": float(popt[1]),
            "c": float(popt[2]),
            "r_squared": float(r_squared),
            "predict": predict,
        }

    except RuntimeError as e:
        raise RuntimeError(
            "Exponential fit failed to converge. Try adjusting initial data or ensure y > 0."
        ) from e


def fit_custom(
    x: np.ndarray, y: np.ndarray, func: Callable, p0: Optional[list] = None
) -> Dict[str, Any]:
    """Fit custom function to data using non-linear least squares.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    func : callable
        Function to fit. Must have signature ``func(x, *params)``
    p0 : list, optional
        Initial guess for parameters

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'parameters': Optimized parameters
        - 'covariance': Covariance matrix
        - 'std_errors': Standard errors of parameters
        - 'r_squared': R² coefficient of determination
        - 'predict': Function to predict y values for new x

    Examples
    --------
    >>> def gaussian(x, amp, mu, sigma):
    ...     return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
    >>> x = np.linspace(-5, 5, 100)
    >>> y = gaussian(x, 10, 0, 1) + np.random.normal(0, 0.5, 100)
    >>> result = fit_custom(x, y, gaussian, p0=[8, 0, 1])
    """
    x = np.asarray(x)
    y = np.asarray(y)

    try:
        # Fit using curve_fit
        popt, pcov = optimize.curve_fit(func, x, y, p0=p0)

        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))

        # Calculate R²
        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Create prediction function
        def predict(x_new):
            return func(x_new, *popt)

        return {
            "parameters": popt,
            "covariance": pcov,
            "std_errors": perr,
            "r_squared": float(r_squared),
            "predict": predict,
        }

    except RuntimeError as e:
        raise RuntimeError(
            "Custom fit failed to converge. "
            "Try adjusting initial guess (p0) or check your function."
        ) from e


# ============================================================================
# Optimization
# ============================================================================


def minimize_function(
    func: Callable,
    x0: np.ndarray,
    method: str = "Nelder-Mead",
    bounds: Optional[list] = None,
) -> Dict[str, Any]:
    """Minimize a scalar function using scipy.optimize.minimize.

    Parameters
    ----------
    func : callable
        Objective function to minimize
    x0 : np.ndarray
        Initial guess
    method : str, default 'Nelder-Mead'
        Optimization method. Options: 'Nelder-Mead', 'BFGS', 'L-BFGS-B', etc.
    bounds : list of tuples, optional
        Bounds for variables (for methods that support bounds)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'x': Optimal parameters
        - 'fun': Function value at optimum
        - 'success': Whether optimization succeeded
        - 'message': Termination message
        - 'nfev': Number of function evaluations

    Examples
    --------
    >>> def rosenbrock(x):
    ...     return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    >>> result = minimize_function(rosenbrock, x0=[0, 0], method='BFGS')
    >>> print(f"Minimum at: {result['x']}")
    """
    x0 = np.asarray(x0)

    # Perform optimization
    res = optimize.minimize(func, x0, method=method, bounds=bounds)

    return {
        "x": res.x,
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev) if hasattr(res, "nfev") else None,
    }


def find_roots(func: Callable, bracket: Tuple[float, float]) -> Dict[str, Any]:
    """Find root of a function in given bracket using Brent's method.

    Parameters
    ----------
    func : callable
        Function for which to find root
    bracket : tuple of (a, b)
        Bracketing interval [a, b] where func(a) and func(b) have opposite signs

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'root': Root of the function
        - 'function_value': Function value at root
        - 'iterations': Number of iterations
        - 'converged': Whether root finding converged

    Examples
    --------
    >>> func = lambda x: x**2 - 4
    >>> result = find_roots(func, bracket=(0, 3))
    >>> print(f"Root: {result['root']:.4f}")
    """
    try:
        root, res = optimize.brentq(func, bracket[0], bracket[1], full_output=True)

        return {
            "root": float(root),
            "function_value": float(res.function_calls),
            "iterations": int(res.iterations),
            "converged": bool(res.converged),
        }

    except ValueError as e:
        raise ValueError(
            "Root finding failed. Ensure func(a) and func(b) have opposite signs."
        ) from e


# ============================================================================
# Linear Algebra
# ============================================================================


def compute_eigenvalues(matrix: np.ndarray, eigenvectors: bool = True) -> Dict[str, np.ndarray]:
    """Compute eigenvalues and optionally eigenvectors of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix
    eigenvectors : bool, default True
        Whether to compute eigenvectors

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'eigenvalues': Array of eigenvalues
        - 'eigenvectors': Matrix of eigenvectors (if requested)

    Examples
    --------
    >>> A = np.array([[1, 2], [2, 1]])
    >>> result = compute_eigenvalues(A)
    >>> print(f"Eigenvalues: {result['eigenvalues']}")
    """
    matrix = np.asarray(matrix)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    if eigenvectors:
        eigenvals, eigenvecs = linalg.eig(matrix)
        return {"eigenvalues": eigenvals, "eigenvectors": eigenvecs}
    else:
        eigenvals = linalg.eigvals(matrix)
        return {"eigenvalues": eigenvals}


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Solve linear system Ax = b.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix
    b : np.ndarray
        Right-hand side vector

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'x': Solution vector
        - 'residual': Residual norm
        - 'condition_number': Condition number of A

    Examples
    --------
    >>> A = np.array([[3, 1], [1, 2]])
    >>> b = np.array([9, 8])
    >>> result = solve_linear_system(A, b)
    >>> print(f"Solution: {result['x']}")
    """
    A = np.asarray(A)
    b = np.asarray(b)

    # Solve system
    x = linalg.solve(A, b)

    # Calculate residual
    residual = linalg.norm(A @ x - b)

    # Calculate condition number
    cond = np.linalg.cond(A)

    return {"x": x, "residual": float(residual), "condition_number": float(cond)}


def matrix_decomposition(matrix: np.ndarray, method: str = "svd") -> Dict[str, np.ndarray]:
    """Perform matrix decomposition.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    method : str, default 'svd'
        Decomposition method. Options: 'svd', 'qr', 'lu', 'cholesky'

    Returns
    -------
    result : dict
        Dictionary with decomposition components (depends on method)
        - 'svd': {'U', 'S', 'Vh'}
        - 'qr': {'Q', 'R'}
        - 'lu': {'P', 'L', 'U'}
        - 'cholesky': {'L'}

    Examples
    --------
    >>> A = np.random.rand(5, 3)
    >>> result = matrix_decomposition(A, method='svd')
    >>> U, S, Vh = result['U'], result['S'], result['Vh']
    """
    matrix = np.asarray(matrix)

    if method == "svd":
        U, S, Vh = linalg.svd(matrix)
        return {"U": U, "S": S, "Vh": Vh}

    elif method == "qr":
        Q, R = linalg.qr(matrix)
        return {"Q": Q, "R": R}

    elif method == "lu":
        P, L, U = linalg.lu(matrix)
        return {"P": P, "L": L, "U": U}

    elif method == "cholesky":
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Cholesky decomposition requires square matrix")
        L = linalg.cholesky(matrix, lower=True)
        return {"L": L}

    else:
        raise ValueError(f"Unsupported method: {method}. Supported: 'svd', 'qr', 'lu', 'cholesky'")


# ============================================================================
# Statistical Tests (M11)
# ============================================================================


def ttest_one_sample(data: np.ndarray, popmean: float, alpha: float = 0.05) -> Dict[str, Any]:
    """One-sample t-test against a known population mean.

    Parameters
    ----------
    data : np.ndarray
        Sample data.
    popmean : float
        Hypothesized population mean (H₀: μ = popmean).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        t_statistic, p_value, df, reject_null, ci_low, ci_high, mean, std, n.
    """
    data = np.asarray(data).flatten()
    t_stat, p_value = stats.ttest_1samp(data, popmean)
    n = len(data)
    df = n - 1
    se = stats.sem(data)
    ci = stats.t.interval(1 - alpha, df, loc=float(np.mean(data)), scale=se)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": int(df),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)),
        "n": int(n),
    }


def ttest_two_sample(
    a: np.ndarray,
    b: np.ndarray,
    equal_var: bool = True,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Independent two-sample t-test.

    Parameters
    ----------
    a, b : np.ndarray
        Two independent samples.
    equal_var : bool, default True
        If True, Student's t-test (equal variances). If False, Welch's t-test.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        t_statistic, p_value, df, reject_null, effect_size_cohens_d,
        mean_a, mean_b, std_a, std_b, n_a, n_b, test_type.
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    cohens_d = (
        (float(np.mean(a)) - float(np.mean(b))) / float(pooled_std) if pooled_std > 0 else 0.0
    )
    df: Optional[int] = int(na + nb - 2) if equal_var else None
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": df,
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "effect_size_cohens_d": float(cohens_d),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a, ddof=1)),
        "std_b": float(np.std(b, ddof=1)),
        "n_a": int(na),
        "n_b": int(nb),
        "test_type": "Student" if equal_var else "Welch",
    }


def ttest_paired(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Paired t-test.

    Parameters
    ----------
    a, b : np.ndarray
        Two matched samples (must be the same length).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        t_statistic, p_value, df, reject_null, mean_diff, std_diff, n.
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    t_stat, p_value = stats.ttest_rel(a, b)
    diff = a - b
    n = len(diff)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": int(n - 1),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "n": int(n),
    }


def mannwhitney_u(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Mann-Whitney U test (nonparametric two-sample test).

    Parameters
    ----------
    a, b : np.ndarray
        Two independent samples.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        u_statistic, p_value, reject_null, n_a, n_b.
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Wilcoxon signed-rank test (nonparametric paired test).

    Parameters
    ----------
    a, b : np.ndarray
        Two matched samples (must be the same length).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        statistic, p_value, reject_null, n.
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    stat, p_value = stats.wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "n": int(len(a)),
    }


def kruskal_wallis(*groups: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Kruskal-Wallis H test (nonparametric one-way ANOVA equivalent).

    Parameters
    ----------
    groups : np.ndarray
        Two or more independent samples (variadic positional arguments).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        h_statistic, p_value, df, reject_null, n_groups, group_sizes.
    """
    arrays = [np.asarray(g).flatten() for g in groups]
    h_stat, p_value = stats.kruskal(*arrays)
    return {
        "h_statistic": float(h_stat),
        "p_value": float(p_value),
        "df": int(len(arrays) - 1),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "n_groups": int(len(arrays)),
        "group_sizes": [int(len(g)) for g in arrays],
    }


def anova_oneway(*groups: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """One-way analysis of variance (ANOVA).

    Parameters
    ----------
    groups : np.ndarray
        Two or more independent samples (variadic positional arguments).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        f_statistic, p_value, df_between, df_within, reject_null,
        eta_squared, n_groups, group_sizes.
    """
    arrays = [np.asarray(g).flatten() for g in groups]
    f_stat, p_value = stats.f_oneway(*arrays)
    k = len(arrays)
    N = sum(len(g) for g in arrays)
    grand_mean = float(np.mean(np.concatenate(arrays)))
    ss_between = float(sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in arrays))
    ss_total = float(sum(float(np.sum((g - grand_mean) ** 2)) for g in arrays))
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "df_between": int(k - 1),
        "df_within": int(N - k),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "eta_squared": float(eta_squared),
        "n_groups": int(k),
        "group_sizes": [int(len(g)) for g in arrays],
    }


def tukey_hsd(*groups: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Tukey HSD post-hoc pairwise comparisons (requires scipy >= 1.8).

    Parameters
    ----------
    groups : np.ndarray
        Two or more independent samples (variadic positional arguments).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        comparisons (list of dicts with group_i, group_j, mean_diff,
        p_value, significant), alpha, n_groups.
    """
    from scipy.stats import tukey_hsd as _tukey_hsd

    arrays = [np.asarray(g).flatten() for g in groups]
    res = _tukey_hsd(*arrays)
    k = len(arrays)
    comparisons = []
    for i in range(k):
        for j in range(i + 1, k):
            p_val = float(res.pvalue[i, j])
            comparisons.append(
                {
                    "group_i": int(i),
                    "group_j": int(j),
                    "mean_diff": float(np.mean(arrays[i])) - float(np.mean(arrays[j])),
                    "p_value": p_val,
                    "significant": bool(p_val < alpha),
                }
            )
    return {
        "comparisons": comparisons,
        "alpha": float(alpha),
        "n_groups": int(k),
    }


def anova_twoway(
    data: "Any",
    response: str,
    factor_a: str,
    factor_b: str,
    include_interaction: bool = True,
) -> Dict[str, Any]:
    """Two-way ANOVA with optional interaction term.

    Uses ``statsmodels`` OLS (Type II sums of squares via
    ``statsmodels.stats.anova.anova_lm``).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the response and both factor columns.
    response : str
        Name of the continuous response variable column.
    factor_a : str
        Name of the first categorical factor column.
    factor_b : str
        Name of the second categorical factor column.
    include_interaction : bool, default True
        Whether to include the A×B interaction term.

    Returns
    -------
    dict
        Keys: ``table`` (pd.DataFrame — ANOVA table with F, p-value, df, SS),
        ``factor_a`` (str), ``factor_b`` (str), ``interaction_included`` (bool),
        ``significant_a`` (bool), ``significant_b`` (bool),
        ``significant_interaction`` (bool or None).

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'y': [5,6,7,8,4,5], 'A': ['a','a','b','b','a','b'],
    ...                    'B': ['x','y','x','y','x','y']})
    >>> result = anova_twoway(df, 'y', 'A', 'B')
    >>> result['table'].index.tolist()  # doctest: +SKIP
    ['A', 'B', 'A:B', 'Residual']
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for two-way ANOVA. Install it with: pip install statsmodels"
        ) from exc

    if include_interaction:
        formula = f"Q('{response}') ~ C(Q('{factor_a}')) * C(Q('{factor_b}'))"
    else:
        formula = f"Q('{response}') ~ C(Q('{factor_a}')) + C(Q('{factor_b}'))"

    model = smf.ols(formula, data=data).fit()
    table = anova_lm(model, typ=2)

    # Rename index for readability
    rename_map = {}
    for idx in table.index:
        if factor_a in idx and factor_b in idx:
            rename_map[idx] = f"{factor_a}:{factor_b}"
        elif factor_a in idx:
            rename_map[idx] = factor_a
        elif factor_b in idx:
            rename_map[idx] = factor_b
    table = table.rename(index=rename_map)

    alpha = 0.05
    p_a = float(table.loc[factor_a, "PR(>F)"]) if factor_a in table.index else 1.0
    p_b = float(table.loc[factor_b, "PR(>F)"]) if factor_b in table.index else 1.0
    interaction_key = f"{factor_a}:{factor_b}"
    p_int = float(table.loc[interaction_key, "PR(>F)"]) if interaction_key in table.index else None

    return {
        "table": table,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "interaction_included": include_interaction,
        "significant_a": p_a < alpha,
        "significant_b": p_b < alpha,
        "significant_interaction": (p_int < alpha) if p_int is not None else None,
        "p_value_a": p_a,
        "p_value_b": p_b,
        "p_value_interaction": p_int,
    }


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Bonferroni multiple-comparison correction.

    Parameters
    ----------
    p_values : np.ndarray
        Array of raw p-values from multiple tests.
    alpha : float, default 0.05
        Family-wise significance level.

    Returns
    -------
    result : dict
        adjusted_alpha, corrected_p_values, reject, n_tests, original_alpha.
    """
    p_values = np.asarray(p_values).flatten()
    m = len(p_values)
    adjusted_alpha = alpha / m
    reject = p_values < adjusted_alpha
    return {
        "adjusted_alpha": float(adjusted_alpha),
        "corrected_p_values": [float(p) for p in p_values],
        "reject": [bool(r) for r in reject],
        "n_tests": int(m),
        "original_alpha": float(alpha),
    }


def pearson_correlation(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Pearson correlation coefficient and significance test.

    Parameters
    ----------
    x, y : np.ndarray
        Two paired numeric arrays of equal length.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        r, p_value, reject_null, alpha, ci_low, ci_high, n.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    r, p_value = stats.pearsonr(x, y)
    n = len(x)
    # Fisher z-transform 95% CI
    z = np.arctanh(float(r))
    se_z = 1.0 / np.sqrt(n - 3)
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_low = float(np.tanh(z - z_crit * se_z))
    ci_high = float(np.tanh(z + z_crit * se_z))
    return {
        "r": float(r),
        "p_value": float(p_value),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(n),
    }


def spearman_correlation(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Spearman rank correlation and significance test.

    Parameters
    ----------
    x, y : np.ndarray
        Two paired numeric arrays of equal length.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        rho, p_value, reject_null, alpha, n.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    rho, p_value = stats.spearmanr(x, y)
    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "n": int(len(x)),
    }


def chi_square_independence(contingency_table: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Chi-square test of independence.

    Parameters
    ----------
    contingency_table : np.ndarray
        2-D observed frequency table.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    result : dict
        chi2, p_value, df, reject_null, alpha, expected.
    """
    contingency_table = np.asarray(contingency_table)
    chi2, p_value, df, expected = stats.chi2_contingency(contingency_table)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "df": int(df),
        "reject_null": bool(p_value < alpha),
        "alpha": float(alpha),
        "expected": expected.tolist(),
    }
