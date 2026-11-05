"""Signal Processing Module.

Smoothing, filtering (Butterworth IIR), frequency analysis (FFT),
derivatives, baseline correction, and interpolation for 1-D signals.

All functions accept and return NumPy arrays.

Functions
---------
Smoothing:
    smooth_moving_average, smooth_savitzky_golay, smooth_gaussian

Filtering:
    filter_lowpass, filter_highpass, filter_bandpass, filter_bandstop

Frequency domain:
    fft, ifft, power_spectrum

Derivatives:
    derivative

Baseline correction:
    baseline_polynomial, baseline_rolling_ball, baseline_als

Interpolation:
    interpolate
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft as _scipy_fft
from scipy.fft import fftfreq
from scipy.fft import ifft as _scipy_ifft
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ─── Smoothing ────────────────────────────────────────────────────────────────


def smooth_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Smooth a signal using a centred rolling mean.

    Parameters
    ----------
    y : np.ndarray
        1-D input signal.
    window : int
        Number of points in the rolling window (must be ≥ 1).

    Returns
    -------
    np.ndarray
        Smoothed signal (same length as *y*).
    """
    import pandas as pd

    y = np.asarray(y, dtype=float)
    if window < 1:
        raise ValueError("window must be ≥ 1")
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def smooth_savitzky_golay(y: np.ndarray, window: int, poly_order: int) -> np.ndarray:
    """Smooth a signal using the Savitzky-Golay algorithm.

    Parameters
    ----------
    y : np.ndarray
        1-D input signal.
    window : int
        Window length (must be odd and > poly_order).
    poly_order : int
        Polynomial order for the local least-squares fit.

    Returns
    -------
    np.ndarray
        Smoothed signal (same length as *y*).
    """
    y = np.asarray(y, dtype=float)
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if poly_order >= window:
        raise ValueError("poly_order must be less than window")
    return savgol_filter(y, window_length=window, polyorder=poly_order)


def smooth_gaussian(y: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a signal by convolution with a Gaussian kernel.

    Parameters
    ----------
    y : np.ndarray
        1-D input signal.
    sigma : float
        Standard deviation of the Gaussian in index units.

    Returns
    -------
    np.ndarray
        Smoothed signal (same length as *y*).
    """
    y = np.asarray(y, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return gaussian_filter1d(y, sigma=sigma)


# ─── Filtering ────────────────────────────────────────────────────────────────


def _butterworth(
    y: np.ndarray,
    btype: str,
    cutoff: Any,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth filter (second-order sections, sosfiltfilt)."""
    y = np.asarray(y, dtype=float)
    nyq = fs / 2.0
    if isinstance(cutoff, (list, tuple)):
        wn = [cutoff[0] / nyq, cutoff[1] / nyq]
    else:
        wn = cutoff / nyq
    sos = butter(order, wn, btype=btype, output="sos")
    return sosfiltfilt(sos, y)


def filter_lowpass(y: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    y : np.ndarray
        1-D input signal.
    cutoff : float
        Cut-off frequency in the same units as *fs*.
    fs : float
        Sampling frequency.
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    return _butterworth(y, "lowpass", cutoff, fs, order)


def filter_highpass(y: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    y : np.ndarray
        1-D input signal.
    cutoff : float
        Cut-off frequency in the same units as *fs*.
    fs : float
        Sampling frequency.
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    return _butterworth(y, "highpass", cutoff, fs, order)


def filter_bandpass(
    y: np.ndarray, low: float, high: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    y : np.ndarray
        1-D input signal.
    low : float
        Lower cut-off frequency.
    high : float
        Upper cut-off frequency.
    fs : float
        Sampling frequency.
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    return _butterworth(y, "bandpass", [low, high], fs, order)


def filter_bandstop(
    y: np.ndarray, low: float, high: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-stop (notch) filter.

    Parameters
    ----------
    y : np.ndarray
        1-D input signal.
    low : float
        Lower cut-off frequency.
    high : float
        Upper cut-off frequency.
    fs : float
        Sampling frequency.
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    return _butterworth(y, "bandstop", [low, high], fs, order)


# ─── Frequency domain ─────────────────────────────────────────────────────────


def fft(y: np.ndarray, dt: float = 1.0) -> Dict[str, Any]:
    """Compute the real-input FFT and return a summary dictionary.

    Parameters
    ----------
    y : np.ndarray
        1-D real signal.
    dt : float
        Sample spacing (1 / sampling_frequency). Default 1.0.

    Returns
    -------
    dict
        Keys:

        - ``frequencies`` : one-sided frequency array
        - ``amplitudes``  : one-sided amplitude array (two-sided correction applied)
        - ``power``       : amplitudes²
        - ``phases``      : phase angles (radians)
        - ``spectrum``    : full complex spectrum (pass to :func:`ifft`)
        - ``dt``          : sample spacing used
        - ``n``           : number of input samples
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    freqs = fftfreq(n, d=dt)
    spectrum = _scipy_fft(y)
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    pos_spectrum = spectrum[pos_mask]
    amplitudes = np.abs(pos_spectrum) * 2.0 / n
    if n % 2 == 0:
        amplitudes[-1] /= 2.0  # Nyquist bin is not doubled
    phases = np.angle(pos_spectrum)
    power = amplitudes**2
    return {
        "frequencies": pos_freqs,
        "amplitudes": amplitudes,
        "power": power,
        "phases": phases,
        "spectrum": spectrum,
        "dt": dt,
        "n": n,
    }


def ifft(spectrum: np.ndarray) -> np.ndarray:
    """Reconstruct a real signal from a complex frequency spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        Full complex spectrum as returned in ``fft(y)["spectrum"]``.

    Returns
    -------
    np.ndarray
        Reconstructed real signal.
    """
    return np.real(_scipy_ifft(spectrum))


def power_spectrum(y: np.ndarray, dt: float = 1.0) -> Dict[str, Any]:
    """Compute the one-sided power spectrum of a real signal.

    Parameters
    ----------
    y : np.ndarray
        1-D real signal.
    dt : float
        Sample spacing. Default 1.0.

    Returns
    -------
    dict
        Keys: ``frequencies``, ``power``.
    """
    result = fft(y, dt=dt)
    return {"frequencies": result["frequencies"], "power": result["power"]}


# ─── Derivatives ──────────────────────────────────────────────────────────────


def derivative(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    order: int = 1,
) -> np.ndarray:
    """Compute a numerical derivative using numpy.gradient (central differences).

    Parameters
    ----------
    y : np.ndarray
        1-D signal values.
    x : np.ndarray, optional
        Corresponding x-axis values. If ``None``, unit spacing is assumed.
    order : int
        Derivative order (1 or 2). Default 1.

    Returns
    -------
    np.ndarray
        d^order y / dx^order, same length as *y*.
    """
    y = np.asarray(y, dtype=float)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if x is not None:
        x = np.asarray(x, dtype=float)
        dy = np.gradient(y, x)
        if order == 2:
            dy = np.gradient(dy, x)
    else:
        dy = np.gradient(y)
        if order == 2:
            dy = np.gradient(dy)
    return dy


# ─── Baseline correction ──────────────────────────────────────────────────────


def baseline_polynomial(
    y: np.ndarray,
    x: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a polynomial baseline and subtract it from the signal.

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    x : np.ndarray
        Corresponding x-axis values.
    degree : int
        Polynomial degree.

    Returns
    -------
    (baseline, corrected) : tuple of np.ndarray
        ``baseline`` is the fitted polynomial; ``corrected = y - baseline``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    coeffs = np.polyfit(x, y, degree)
    baseline = np.polyval(coeffs, x)
    return baseline, y - baseline


def baseline_rolling_ball(
    y: np.ndarray,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate baseline using the rolling-ball (morphological opening) algorithm.

    Implemented as 1-D morphological opening: erosion (minimum filter)
    followed by dilation (maximum filter).

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    radius : int
        Ball radius in samples. Larger values → smoother baseline.

    Returns
    -------
    (baseline, corrected) : tuple of np.ndarray
        ``baseline`` is the rolling-ball estimate; ``corrected = y - baseline``.
    """
    y = np.asarray(y, dtype=float)
    size = 2 * int(radius) + 1
    eroded = minimum_filter1d(y, size=size)
    baseline = maximum_filter1d(eroded, size=size)
    return baseline, y - baseline


def baseline_als(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    max_iter: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Asymmetric least squares (ALS) baseline correction.

    Fits a smooth baseline by penalising deviations asymmetrically — peaks
    above the baseline are down-weighted to push the baseline toward the minima.

    Parameters
    ----------
    y : np.ndarray
        Signal values.
    lam : float
        Smoothness parameter (λ). Larger → smoother baseline. Default 1e5.
    p : float
        Asymmetry parameter (fraction of weight assigned to points above the
        baseline). Typically 0.001–0.1. Default 0.01.
    max_iter : int
        Maximum iterations. Default 10.

    Returns
    -------
    (baseline, corrected) : tuple of np.ndarray
        ``baseline`` is the ALS estimate; ``corrected = y - baseline``.

    References
    ----------
    Eilers & Boelens (2005), *Baseline Correction with Asymmetric Least
    Squares Smoothing*.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    D_sp = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), dtype=float, format="csc")
    DtD = lam * (D_sp.T @ D_sp)
    w = np.ones(n)
    z = y.copy()
    for _ in range(max_iter):
        W = diags(w, 0, dtype=float, format="csc")
        z = spsolve(W + DtD, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z, y - z


# ─── Interpolation ────────────────────────────────────────────────────────────


def interpolate(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """Interpolate a signal to a new x-grid.

    Parameters
    ----------
    x : np.ndarray
        Original x-axis values (must be monotonically increasing).
    y : np.ndarray
        Corresponding signal values.
    x_new : np.ndarray
        New x-axis values at which to evaluate the interpolant.
    method : str
        One of ``'linear'``, ``'quadratic'``, ``'cubic'``, ``'nearest'``,
        ``'previous'``. Default ``'cubic'``.

    Returns
    -------
    np.ndarray
        Interpolated y values at *x_new*.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if method == "cubic":
        return CubicSpline(x, y)(x_new)
    f = interp1d(x, y, kind=method, fill_value="extrapolate")
    return f(x_new)
