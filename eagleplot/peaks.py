"""Peak Analysis Module.

Auto-detection, integration, FWHM computation, and model fitting for
spectral / chromatographic peaks.

All functions accept NumPy arrays and return plain dicts.

Functions
---------
find_peaks        -- detect peaks with scipy.signal.find_peaks
integrate_peaks   -- trapezoidal area under each peak
compute_fwhm      -- FWHM in x-axis units for each peak
fit_peak          -- fit one peak + background model
fit_multipeak     -- fit N simultaneous peaks + background
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as _sp_find_peaks
from scipy.signal import peak_prominences, peak_widths
from scipy.special import voigt_profile

# ─── Internal profile functions ───────────────────────────────────────────────


def _gaussian(x: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _lorentzian(x: np.ndarray, center: float, amplitude: float, gamma: float) -> np.ndarray:
    return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2)


def _pseudo_voigt(
    x: np.ndarray, center: float, amplitude: float, sigma: float, eta: float
) -> np.ndarray:
    """Linear combination: η·Lorentzian + (1−η)·Gaussian."""
    return eta * _lorentzian(x, center, amplitude, sigma) + (1 - eta) * _gaussian(
        x, center, amplitude, sigma
    )


def _voigt_peak(
    x: np.ndarray, center: float, amplitude: float, sigma: float, gamma: float
) -> np.ndarray:
    """Voigt profile normalised so that the peak value equals *amplitude*."""
    norm = voigt_profile(0.0, sigma, gamma)
    if norm == 0:
        return np.zeros_like(x)
    return amplitude * voigt_profile(x - center, sigma, gamma) / norm


def _build_model(model: str):
    """Return (profile_fn, n_params_per_peak) for the requested model name."""
    model = model.lower()
    if model == "gaussian":
        return _gaussian, 3
    if model == "lorentzian":
        return _lorentzian, 3
    if model in ("pseudo_voigt", "pseudo-voigt"):
        return _pseudo_voigt, 4
    if model == "voigt":
        return _voigt_peak, 4
    raise ValueError(f"Unknown model '{model}'. Choose: gaussian, lorentzian, pseudo_voigt, voigt.")


# ─── Public API ───────────────────────────────────────────────────────────────


def find_peaks(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
    width: Optional[float] = None,
    distance: Optional[int] = None,
) -> Dict[str, Any]:
    """Detect peaks in a 1-D signal.

    Thin wrapper around :func:`scipy.signal.find_peaks` that also computes
    prominences, widths, and converts sample-domain quantities to x-axis units
    when *x* is provided.

    Parameters
    ----------
    y : np.ndarray
        1-D signal values.
    x : np.ndarray, optional
        Corresponding x-axis values. When supplied, peak positions and widths
        are returned in x-axis units as well as sample indices.
    height : float, optional
        Minimum peak height.
    prominence : float, optional
        Minimum peak prominence.
    width : float, optional
        Minimum peak width (in samples).
    distance : int, optional
        Minimum number of samples between peaks.

    Returns
    -------
    dict
        Keys:

        - ``indices``        : sample indices of detected peaks
        - ``positions``      : x-values at peak positions (or indices if no *x*)
        - ``heights``        : y-values at peak positions
        - ``prominences``    : peak prominences
        - ``widths_samples`` : FWHM widths in samples
        - ``widths``         : FWHM widths in x-axis units (= widths_samples if no *x*)
        - ``left_ips``       : left interpolated half-maximum sample positions
        - ``right_ips``      : right interpolated half-maximum sample positions
        - ``n_peaks``        : number of detected peaks
    """
    y = np.asarray(y, dtype=float)
    kwargs: Dict[str, Any] = {}
    if height is not None:
        kwargs["height"] = height
    if prominence is not None:
        kwargs["prominence"] = prominence
    if width is not None:
        kwargs["width"] = width
    if distance is not None:
        kwargs["distance"] = distance

    indices, _ = _sp_find_peaks(y, **kwargs)
    if len(indices) == 0:
        empty = np.array([], dtype=float)
        return {
            "indices": np.array([], dtype=int),
            "positions": empty,
            "heights": empty,
            "prominences": empty,
            "widths_samples": empty,
            "widths": empty,
            "left_ips": empty,
            "right_ips": empty,
            "n_peaks": 0,
        }

    prominences_arr, _, _ = peak_prominences(y, indices)
    widths_samples, _, left_ips, right_ips = peak_widths(y, indices, rel_height=0.5)

    if x is not None:
        x = np.asarray(x, dtype=float)
        positions = x[indices]
        # Convert sample-domain widths to x-axis units via linear interpolation
        left_x = np.interp(left_ips, np.arange(len(x)), x)
        right_x = np.interp(right_ips, np.arange(len(x)), x)
        widths_x = right_x - left_x
    else:
        positions = indices.astype(float)
        widths_x = widths_samples

    return {
        "indices": indices,
        "positions": positions,
        "heights": y[indices],
        "prominences": prominences_arr,
        "widths_samples": widths_samples,
        "widths": widths_x,
        "left_ips": left_ips,
        "right_ips": right_ips,
        "n_peaks": len(indices),
    }


def integrate_peaks(
    y: np.ndarray,
    x: np.ndarray,
    peaks: Dict[str, Any],
    bounds: str = "auto",
) -> Dict[str, Any]:
    """Compute the area under each detected peak.

    Uses trapezoidal integration on the background-subtracted signal. The
    background under each peak is approximated as a straight line between the
    left and right base points (as determined by scipy's ``peak_prominences``).

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    x : np.ndarray
        Corresponding x-axis values.
    peaks : dict
        Output of :func:`find_peaks`.
    bounds : str
        Currently only ``'auto'`` is supported (prominence-based bases).

    Returns
    -------
    dict
        Keys: ``areas``, ``left_bases`` (x values), ``right_bases`` (x values).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    indices = peaks["indices"]

    if len(indices) == 0:
        empty = np.array([], dtype=float)
        return {"areas": empty, "left_bases": empty, "right_bases": empty}

    _, left_b, right_b = peak_prominences(y, indices)
    left_b = left_b.astype(int)
    right_b = right_b.astype(int)

    areas = []
    left_xs = []
    right_xs = []
    for l_idx, r_idx in zip(left_b, right_b):
        x_seg = x[l_idx : r_idx + 1]
        y_seg = y[l_idx : r_idx + 1]
        baseline = np.linspace(y[l_idx], y[r_idx], len(x_seg))
        area = np.trapezoid(np.maximum(y_seg - baseline, 0.0), x_seg)
        areas.append(area)
        left_xs.append(x[l_idx])
        right_xs.append(x[r_idx])

    return {
        "areas": np.array(areas),
        "left_bases": np.array(left_xs),
        "right_bases": np.array(right_xs),
    }


def compute_fwhm(
    y: np.ndarray,
    x: np.ndarray,
    peaks: Dict[str, Any],
) -> Dict[str, Any]:
    """Return FWHM values in x-axis units for each peak.

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    x : np.ndarray
        Corresponding x-axis values.
    peaks : dict
        Output of :func:`find_peaks`.

    Returns
    -------
    dict
        Keys: ``fwhm`` (x-axis units), ``fwhm_samples``, ``centers`` (x values).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    indices = peaks["indices"]

    if len(indices) == 0:
        empty = np.array([], dtype=float)
        return {"fwhm": empty, "fwhm_samples": empty, "centers": empty}

    widths_samples, _, left_ips, right_ips = peak_widths(y, indices, rel_height=0.5)
    left_x = np.interp(left_ips, np.arange(len(x)), x)
    right_x = np.interp(right_ips, np.arange(len(x)), x)
    fwhm_x = right_x - left_x

    return {
        "fwhm": fwhm_x,
        "fwhm_samples": widths_samples,
        "centers": x[indices],
    }


def fit_peak(
    y: np.ndarray,
    x: np.ndarray,
    center_guess: float,
    model: str = "gaussian",
    background: str = "linear",
) -> Dict[str, Any]:
    """Fit a single-peak model (+ optional background) to a 1-D signal.

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    x : np.ndarray
        Corresponding x-axis values.
    center_guess : float
        Initial guess for the peak center (in x-axis units).
    model : str
        Peak profile: ``'gaussian'``, ``'lorentzian'``, ``'pseudo_voigt'``,
        or ``'voigt'``.
    background : str
        Background model: ``'none'``, ``'constant'``, or ``'linear'``.

    Returns
    -------
    dict
        Keys: ``params``, ``param_names``, ``std_errors``, ``fitted_y``,
        ``residuals``, ``r_squared``, ``model``, ``background``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    profile_fn, n_pp = _build_model(model)

    amp_guess = float(y.max() - y.min())
    width_guess = float((x.max() - x.min()) / 10.0)

    if model in ("gaussian", "lorentzian"):
        p0_peak = [center_guess, amp_guess, width_guess]
        peak_names = ["center", "amplitude", "width"]
    else:
        p0_peak = [center_guess, amp_guess, width_guess, 0.5]
        peak_names = ["center", "amplitude", "width", "eta_or_gamma"]

    if background == "none":
        p0 = p0_peak
        bg_names: List[str] = []

        def full_model(xv: np.ndarray, *p: float) -> np.ndarray:
            return profile_fn(xv, *p[:n_pp])

    elif background == "constant":
        p0 = p0_peak + [float(y.min())]
        bg_names = ["bg_offset"]

        def full_model(xv: np.ndarray, *p: float) -> np.ndarray:  # type: ignore[misc]
            return profile_fn(xv, *p[:n_pp]) + p[n_pp]

    else:  # linear
        slope_guess = (float(y[-1]) - float(y[0])) / (float(x[-1]) - float(x[0]) + 1e-12)
        p0 = p0_peak + [float(y.min()), slope_guess]
        bg_names = ["bg_offset", "bg_slope"]

        def full_model(xv: np.ndarray, *p: float) -> np.ndarray:  # type: ignore[misc]
            return profile_fn(xv, *p[:n_pp]) + p[n_pp] + p[n_pp + 1] * xv

    popt, pcov = curve_fit(full_model, x, y, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    fitted_y = full_model(x, *popt)
    residuals = y - fitted_y
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "params": popt,
        "param_names": peak_names + bg_names,
        "std_errors": perr,
        "fitted_y": fitted_y,
        "residuals": residuals,
        "r_squared": r2,
        "model": model,
        "background": background,
    }


def fit_multipeak(
    y: np.ndarray,
    x: np.ndarray,
    n_peaks: int,
    model: str = "gaussian",
    background: str = "linear",
    initial_guesses: Optional[List[Tuple[float, float, float]]] = None,
) -> Dict[str, Any]:
    """Fit N simultaneous peak profiles (+ optional shared background) to data.

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    x : np.ndarray
        Corresponding x-axis values.
    n_peaks : int
        Number of peaks to fit.
    model : str
        Peak profile: ``'gaussian'``, ``'lorentzian'``, ``'pseudo_voigt'``,
        or ``'voigt'``.
    background : str
        Background model applied to the summed peaks: ``'none'``,
        ``'constant'``, or ``'linear'``.
    initial_guesses : list of (center, amplitude, width), optional
        One tuple per peak. If ``None``, guesses are estimated automatically
        using :func:`find_peaks`.

    Returns
    -------
    dict
        Keys:

        - ``params``        : all optimised parameters (peaks then background)
        - ``param_names``   : corresponding parameter names
        - ``std_errors``    : standard errors from covariance diagonal
        - ``fitted_y``      : summed model evaluated at *x*
        - ``individual_y``  : list of per-peak curves (without background)
        - ``residuals``     : y − fitted_y
        - ``r_squared``
        - ``model``
        - ``background``
        - ``n_peaks``
        - ``peak_summaries`` : list of dicts with center, amplitude, width, fwhm per peak
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    profile_fn, n_pp = _build_model(model)

    # ── Auto-estimate guesses when not provided ───────────────────────────────
    if initial_guesses is None:
        detected = find_peaks(y, x=x, prominence=(y.max() - y.min()) * 0.05)
        if detected["n_peaks"] >= n_peaks:
            # Sort by prominence, take top n_peaks
            order = np.argsort(detected["prominences"])[::-1][:n_peaks]
            guesses = [
                (
                    float(detected["positions"][i]),
                    float(detected["heights"][i]),
                    float(detected["widths"][i]) / 2.0 or (x.max() - x.min()) / (4 * n_peaks),
                )
                for i in order
            ]
        else:
            # Distribute evenly
            x_range = float(x.max() - x.min())
            amp_guess = float(y.max() - y.min())
            guesses = [
                (
                    float(x.min()) + (i + 1) * x_range / (n_peaks + 1),
                    amp_guess,
                    x_range / (4 * n_peaks),
                )
                for i in range(n_peaks)
            ]
    else:
        guesses = list(initial_guesses)

    # ── Build parameter vector p0 ─────────────────────────────────────────────
    p0_peaks: List[float] = []
    peak_param_names: List[str] = []
    base_names = (
        ["center", "amplitude", "sigma"]
        if model in ("gaussian", "lorentzian")
        else ["center", "amplitude", "sigma", "eta"]
    )
    for i, (ctr, amp, wid) in enumerate(guesses):
        if n_pp == 3:
            p0_peaks.extend([ctr, amp, abs(wid)])
        else:
            p0_peaks.extend([ctr, amp, abs(wid), 0.5])
        peak_param_names += [f"peak{i + 1}_{nm}" for nm in base_names]

    if background == "none":
        p0 = p0_peaks
        bg_names: List[str] = []

        def full_model(xv: np.ndarray, *p: float) -> np.ndarray:
            return sum(profile_fn(xv, *p[n_pp * k : n_pp * (k + 1)]) for k in range(n_peaks))

    elif background == "constant":
        p0 = p0_peaks + [float(y.min())]
        bg_names = ["bg_offset"]

        def full_model(xv: np.ndarray, *p: float) -> np.ndarray:  # type: ignore[misc]
            peaks_sum = sum(profile_fn(xv, *p[n_pp * k : n_pp * (k + 1)]) for k in range(n_peaks))
            return peaks_sum + p[n_peaks * n_pp]

    else:  # linear
        slope_guess = (float(y[-1]) - float(y[0])) / (float(x[-1]) - float(x[0]) + 1e-12)
        p0 = p0_peaks + [float(y.min()), slope_guess]
        bg_names = ["bg_offset", "bg_slope"]

        def full_model(xv: np.ndarray, *p: float) -> np.ndarray:  # type: ignore[misc]
            peaks_sum = sum(profile_fn(xv, *p[n_pp * k : n_pp * (k + 1)]) for k in range(n_peaks))
            off = n_peaks * n_pp
            return peaks_sum + p[off] + p[off + 1] * xv

    popt, pcov = curve_fit(full_model, x, y, p0=p0, maxfev=30000)
    perr = np.sqrt(np.diag(pcov))
    fitted_y = full_model(x, *popt)
    residuals = y - fitted_y
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Per-peak curves and summaries ─────────────────────────────────────────
    individual_y = [profile_fn(x, *popt[n_pp * k : n_pp * (k + 1)]) for k in range(n_peaks)]

    peak_summaries = []
    for k in range(n_peaks):
        p_k = popt[n_pp * k : n_pp * (k + 1)]
        center_k, amp_k, width_k = float(p_k[0]), float(p_k[1]), float(p_k[2])
        # FWHM formula for Gaussian/Lorentzian; approximate for others
        if model == "gaussian":
            fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(width_k)
        elif model == "lorentzian":
            fwhm = 2.0 * abs(width_k)
        else:
            fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(width_k)
        peak_summaries.append(
            {"center": center_k, "amplitude": amp_k, "width": abs(width_k), "fwhm": fwhm}
        )

    return {
        "params": popt,
        "param_names": peak_param_names + bg_names,
        "std_errors": perr,
        "fitted_y": fitted_y,
        "individual_y": individual_y,
        "residuals": residuals,
        "r_squared": r2,
        "model": model,
        "background": background,
        "n_peaks": n_peaks,
        "peak_summaries": peak_summaries,
    }
