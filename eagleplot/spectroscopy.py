"""Spectroscopy Analysis Module.

Conversion, correction, and analysis utilities for IR/Raman, UV-Vis, NMR,
and mass spectrometry data.

All functions accept and return NumPy arrays or plain dicts. No Streamlit
imports are present in this module.

Functions
---------
IR / Raman:
    absorbance_to_transmittance, transmittance_to_absorbance,
    atr_correction, spectral_subtraction, remove_cosmic_rays,
    assign_bands

UV-Vis:
    beer_lambert, molar_absorptivity_series, spectral_overlap_integral

NMR:
    calibrate_ppm_axis, apply_line_broadening, zero_fill, nmr_fft,
    pick_nmr_peaks, integrate_nmr_regions

Mass Spec:
    find_mz_peaks, centroid_spectrum
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats
from scipy.fft import fft as _scipy_fft
from scipy.fft import fftfreq as _fftfreq
from scipy.interpolate import interp1d
from scipy.signal import find_peaks as _find_peaks

# ─── IR / Raman ───────────────────────────────────────────────────────────────


def absorbance_to_transmittance(absorbance: np.ndarray) -> np.ndarray:
    """Convert absorbance A to percent transmittance T(%) = 100 × 10^(-A).

    Parameters
    ----------
    absorbance:
        Array of absorbance values.

    Returns
    -------
    np.ndarray
        Percent transmittance array.
    """
    absorbance = np.asarray(absorbance, dtype=float)
    return 100.0 * np.power(10.0, -absorbance)


def transmittance_to_absorbance(transmittance: np.ndarray) -> np.ndarray:
    """Convert percent transmittance T(%) to absorbance A = 2 - log10(T).

    Clips input to a minimum of 1e-10 to avoid log(0).

    Parameters
    ----------
    transmittance:
        Array of percent transmittance values (0–100 range expected).

    Returns
    -------
    np.ndarray
        Absorbance array.

    Raises
    ------
    ValueError
        If any input value is <= 0 before clipping check.
    """
    transmittance = np.asarray(transmittance, dtype=float)
    if np.any(transmittance <= 0.0):
        raise ValueError("transmittance values must be > 0; got values <= 0.")
    t_clipped = np.clip(transmittance, 1e-10, None)
    return 2.0 - np.log10(t_clipped)


def atr_correction(
    wavenumbers: np.ndarray,
    absorbance: np.ndarray,
    n_atr: float = 1.5,
    angle_deg: float = 45.0,
    n_sample: float = 1.0,
) -> np.ndarray:
    """ATR depth-of-penetration correction.

    Divides each absorbance value by the effective dp(ν) (depth of
    penetration) to approximate a transmission spectrum::

        dp(ν) = λ / (2π × n_atr × √(sin²(θ) - (n_sample/n_atr)²))
        λ = 10000 / ν  (µm, since ν is in cm⁻¹)

    Parameters
    ----------
    wavenumbers:
        Wavenumber axis in cm⁻¹.
    absorbance:
        Absorbance values matching the wavenumber axis.
    n_atr:
        Refractive index of the ATR crystal (default 1.5, ZnSe).
    angle_deg:
        Angle of incidence in degrees (default 45°).
    n_sample:
        Refractive index of the sample (default 1.0).

    Returns
    -------
    np.ndarray
        Corrected absorbance array.

    Raises
    ------
    ValueError
        If sin(angle_deg) <= n_sample / n_atr (below or at critical angle).
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    absorbance = np.asarray(absorbance, dtype=float)

    angle_rad = np.deg2rad(angle_deg)
    sin_theta = np.sin(angle_rad)
    ratio = n_sample / n_atr

    if sin_theta <= ratio:
        raise ValueError(
            f"Angle {angle_deg}° is at or below the critical angle for total "
            f"internal reflection (sin(θ)={sin_theta:.4f} <= n_sample/n_atr="
            f"{ratio:.4f}). Increase the angle or change refractive indices."
        )

    sin2_theta = sin_theta**2
    ratio_sq = ratio**2
    lambda_um = 10000.0 / wavenumbers  # cm⁻¹ → µm
    dp = lambda_um / (2.0 * np.pi * n_atr * np.sqrt(sin2_theta - ratio_sq))
    corrected = np.where(dp > 0, absorbance / dp, absorbance)
    return corrected


def spectral_subtraction(
    wavenumbers: np.ndarray,
    spectrum_a: np.ndarray,
    spectrum_b: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract scale*B from A.

    Parameters
    ----------
    wavenumbers:
        Wavenumber axis shared by both spectra.
    spectrum_a:
        Primary spectrum.
    spectrum_b:
        Spectrum to subtract (after scaling).
    scale:
        Scaling factor applied to spectrum_b before subtraction.

    Returns
    -------
    np.ndarray
        Difference spectrum (same length as spectrum_a).

    Raises
    ------
    ValueError
        If array lengths do not match wavenumbers.
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    spectrum_a = np.asarray(spectrum_a, dtype=float)
    spectrum_b = np.asarray(spectrum_b, dtype=float)

    if len(spectrum_a) != len(wavenumbers) or len(spectrum_b) != len(wavenumbers):
        raise ValueError(
            "spectrum_a and spectrum_b must both have the same length as wavenumbers. "
            f"Got wavenumbers={len(wavenumbers)}, A={len(spectrum_a)}, B={len(spectrum_b)}."
        )
    return spectrum_a - scale * spectrum_b


def remove_cosmic_rays(
    y: np.ndarray,
    threshold_sigma: float = 5.0,
    window: int = 5,
) -> np.ndarray:
    """Remove cosmic ray spikes from Raman spectra using local z-score detection.

    For each point, the local median and standard deviation are computed from
    the `window` neighboring points on each side (excluding the point itself).
    If |y[i] - local_median| > threshold_sigma * local_std, y[i] is replaced
    with the local median.

    Parameters
    ----------
    y:
        Input Raman spectrum.
    threshold_sigma:
        Number of local standard deviations above which a point is flagged.
    window:
        Number of neighboring points on each side to use for local statistics.

    Returns
    -------
    np.ndarray
        Cleaned copy of y (same shape, original not mutated).
    """
    y = np.asarray(y, dtype=float)
    result = y.copy()
    n = len(y)

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbors = np.concatenate([y[lo:i], y[i + 1 : hi]])
        if len(neighbors) == 0:
            continue
        local_median = np.median(neighbors)
        local_std = np.std(neighbors)
        if local_std == 0.0:
            continue
        if abs(y[i] - local_median) > threshold_sigma * local_std:
            result[i] = local_median

    return result


# ─── Functional Group Lookup ──────────────────────────────────────────────────

FUNCTIONAL_GROUPS: dict = {
    "O-H stretch (free)": (3580.0, 3650.0),
    "O-H stretch (H-bonded)": (3200.0, 3580.0),
    "N-H stretch (primary amine)": (3300.0, 3500.0),
    "N-H stretch (secondary/amide)": (3100.0, 3350.0),
    "C-H stretch (sp3)": (2850.0, 3000.0),
    "C-H stretch (sp2/aromatic)": (3000.0, 3100.0),
    "C-H stretch (alkyne)": (3250.0, 3350.0),
    "C=O stretch (ester)": (1730.0, 1750.0),
    "C=O stretch (aldehyde)": (1710.0, 1740.0),
    "C=O stretch (ketone)": (1700.0, 1725.0),
    "C=O stretch (conjugated/amide)": (1630.0, 1700.0),
    "C=C stretch (alkene)": (1620.0, 1680.0),
    "Aromatic C=C stretch": (1500.0, 1600.0),
    "C≡C stretch (alkyne)": (2100.0, 2260.0),
    "C≡N stretch (nitrile)": (2200.0, 2260.0),
    "N-O asymm stretch (nitro)": (1540.0, 1570.0),
    "N-O symm stretch (nitro)": (1350.0, 1380.0),
    "C-O-C stretch (ether/ester)": (1050.0, 1260.0),
    "C-O stretch (alcohol)": (1000.0, 1200.0),
    "Aromatic C-H oop bend": (700.0, 900.0),
    "C-Cl stretch": (600.0, 800.0),
    "C-Br stretch": (500.0, 680.0),
}


def assign_bands(wavenumber: float) -> list[str]:
    """Return list of functional group names whose range includes wavenumber.

    Parameters
    ----------
    wavenumber:
        A single wavenumber value in cm⁻¹.

    Returns
    -------
    list[str]
        Names of matching functional groups; empty list if no match.
    """
    matches: list[str] = []
    for name, (low, high) in FUNCTIONAL_GROUPS.items():
        if low <= wavenumber <= high:
            matches.append(name)
    return matches


# ─── UV-Vis ───────────────────────────────────────────────────────────────────


def beer_lambert(
    absorbance: float,
    epsilon: float | None = None,
    concentration: float | None = None,
    path_length: float = 1.0,
) -> dict:
    """Beer-Lambert calculator: A = ε × c × l.

    Exactly one of epsilon or concentration must be None (the unknown to
    solve for).

    Parameters
    ----------
    absorbance:
        Measured absorbance (dimensionless).
    epsilon:
        Molar absorptivity (L mol⁻¹ cm⁻¹). Pass None to solve for ε.
    concentration:
        Concentration (mol L⁻¹). Pass None to solve for c.
    path_length:
        Optical path length in cm (default 1.0).

    Returns
    -------
    dict
        Keys: absorbance, epsilon, concentration, path_length, solved_for.

    Raises
    ------
    ValueError
        If both or neither of epsilon/concentration are None, or if known
        values are <= 0.
    """
    if epsilon is None and concentration is None:
        raise ValueError("Exactly one of epsilon or concentration must be None. Both are None.")
    if epsilon is not None and concentration is not None:
        raise ValueError("Exactly one of epsilon or concentration must be None. Neither is None.")
    if path_length <= 0.0:
        raise ValueError(f"path_length must be > 0; got {path_length}.")

    if epsilon is None:
        # solve for epsilon
        if concentration <= 0.0:
            raise ValueError(f"concentration must be > 0; got {concentration}.")
        solved_epsilon = absorbance / (concentration * path_length)
        solved_concentration = concentration
        solved_for = "epsilon"
    else:
        # solve for concentration
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0; got {epsilon}.")
        solved_epsilon = epsilon
        solved_concentration = absorbance / (epsilon * path_length)
        solved_for = "concentration"

    return {
        "absorbance": absorbance,
        "epsilon": solved_epsilon,
        "concentration": solved_concentration,
        "path_length": path_length,
        "solved_for": solved_for,
    }


def molar_absorptivity_series(
    concentrations: np.ndarray,
    absorbances: np.ndarray,
    path_length: float = 1.0,
) -> dict:
    """Determine molar absorptivity from a Beer-Lambert calibration series.

    Performs linear regression of absorbances vs concentrations × path_length.

    Parameters
    ----------
    concentrations:
        Array of concentration values (mol L⁻¹).
    absorbances:
        Corresponding absorbance measurements.
    path_length:
        Optical path length in cm (default 1.0).

    Returns
    -------
    dict
        Keys: epsilon, r_squared, slope, intercept, fitted_absorbances,
        residuals, linearity_ok (bool: r_squared >= 0.999).

    Raises
    ------
    ValueError
        If arrays have different lengths or fewer than 2 points.
    """
    concentrations = np.asarray(concentrations, dtype=float)
    absorbances = np.asarray(absorbances, dtype=float)

    if len(concentrations) != len(absorbances):
        raise ValueError(
            f"concentrations and absorbances must have the same length; "
            f"got {len(concentrations)} and {len(absorbances)}."
        )
    if len(concentrations) < 2:
        raise ValueError("At least 2 data points are required for molar absorptivity series.")

    x = concentrations * path_length
    result = _stats.linregress(x, absorbances)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue**2)
    fitted = slope * x + intercept
    residuals = absorbances - fitted

    return {
        "epsilon": slope,
        "r_squared": r_squared,
        "slope": slope,
        "intercept": intercept,
        "fitted_absorbances": fitted,
        "residuals": residuals,
        "linearity_ok": r_squared >= 0.999,
    }


def spectral_overlap_integral(
    wavelengths_d: np.ndarray,
    emission_d: np.ndarray,
    wavelengths_a: np.ndarray,
    absorption_a: np.ndarray,
) -> dict:
    """Compute spectral overlap integral J for FRET.

    J = ∫ F_D(λ) × ε_A(λ) × λ⁴ dλ

    Both spectra are interpolated onto a common wavelength grid (500 points
    over the overlapping wavelength range). Donor emission is normalized
    (∫F_D dλ = 1) before computing J.

    Parameters
    ----------
    wavelengths_d:
        Wavelength axis for donor emission spectrum (nm).
    emission_d:
        Donor fluorescence emission intensity.
    wavelengths_a:
        Wavelength axis for acceptor absorption spectrum (nm).
    absorption_a:
        Acceptor molar absorptivity (L mol⁻¹ cm⁻¹).

    Returns
    -------
    dict
        Keys: J, wavelengths_common, integrand, donor_norm, acceptor_interp.
        Returns J=0.0 if spectra do not overlap.
    """
    wavelengths_d = np.asarray(wavelengths_d, dtype=float)
    emission_d = np.asarray(emission_d, dtype=float)
    wavelengths_a = np.asarray(wavelengths_a, dtype=float)
    absorption_a = np.asarray(absorption_a, dtype=float)

    overlap_lo = max(wavelengths_d.min(), wavelengths_a.min())
    overlap_hi = min(wavelengths_d.max(), wavelengths_a.max())

    if overlap_lo >= overlap_hi:
        empty = np.array([])
        return {
            "J": 0.0,
            "wavelengths_common": empty,
            "integrand": empty,
            "donor_norm": empty,
            "acceptor_interp": empty,
        }

    wl_common = np.linspace(overlap_lo, overlap_hi, 500)

    interp_d = interp1d(wavelengths_d, emission_d, bounds_error=False, fill_value=0.0)
    interp_a = interp1d(wavelengths_a, absorption_a, bounds_error=False, fill_value=0.0)

    donor_raw = interp_d(wl_common)
    donor_raw = np.clip(donor_raw, 0.0, None)
    acceptor = interp_a(wl_common)
    acceptor = np.clip(acceptor, 0.0, None)

    norm_factor = np.trapezoid(donor_raw, wl_common)
    if norm_factor > 0.0:
        donor_norm = donor_raw / norm_factor
    else:
        donor_norm = donor_raw

    integrand = donor_norm * acceptor * wl_common**4
    J = float(np.trapezoid(integrand, wl_common))

    return {
        "J": J,
        "wavelengths_common": wl_common,
        "integrand": integrand,
        "donor_norm": donor_norm,
        "acceptor_interp": acceptor,
    }


# ─── NMR ──────────────────────────────────────────────────────────────────────


def calibrate_ppm_axis(
    hz_array: np.ndarray,
    spectrometer_freq_mhz: float,
    reference_hz: float = 0.0,
) -> np.ndarray:
    """Convert Hz axis to ppm: ppm = (hz - reference_hz) / spectrometer_freq_mhz.

    Parameters
    ----------
    hz_array:
        Frequency axis in Hz.
    spectrometer_freq_mhz:
        Spectrometer operating frequency in MHz.
    reference_hz:
        Reference frequency offset in Hz (default 0.0, e.g., TMS).

    Returns
    -------
    np.ndarray
        Chemical shift axis in ppm (same shape as hz_array).
    """
    hz_array = np.asarray(hz_array, dtype=float)
    return (hz_array - reference_hz) / spectrometer_freq_mhz


def apply_line_broadening(
    fid: np.ndarray,
    dt: float,
    lb: float,
    mode: str = "lorentzian",
) -> np.ndarray:
    """Apply line broadening to an NMR FID.

    Parameters
    ----------
    fid:
        Free induction decay array (real or complex).
    dt:
        Dwell time — seconds per point.
    lb:
        Line broadening factor in Hz.
    mode:
        ``'lorentzian'``: multiply by exp(-π × lb × t).
        ``'gaussian'``: multiply by exp(-0.5 × (π × lb × t)²).

    Returns
    -------
    np.ndarray
        Broadened FID (same shape as input).

    Raises
    ------
    ValueError
        If mode is not ``'lorentzian'`` or ``'gaussian'``.
    """
    fid = np.asarray(fid, dtype=complex if np.iscomplexobj(fid) else float)
    t = np.arange(len(fid), dtype=float) * dt

    if mode == "lorentzian":
        window = np.exp(-np.pi * lb * t)
    elif mode == "gaussian":
        window = np.exp(-0.5 * (np.pi * lb * t) ** 2)
    else:
        raise ValueError(f"mode must be 'lorentzian' or 'gaussian'; got '{mode}'.")

    return fid * window


def zero_fill(fid: np.ndarray, n_points: int) -> np.ndarray:
    """Zero-fill FID to n_points total.

    Parameters
    ----------
    fid:
        Input FID array.
    n_points:
        Target total number of points (must be >= len(fid)).

    Returns
    -------
    np.ndarray
        Zero-padded array of length n_points.

    Raises
    ------
    ValueError
        If n_points < len(fid).
    """
    fid = np.asarray(fid, dtype=complex if np.iscomplexobj(fid) else float)
    if n_points < len(fid):
        raise ValueError(f"n_points ({n_points}) must be >= len(fid) ({len(fid)}).")
    pad_width = n_points - len(fid)
    return np.pad(fid, (0, pad_width), mode="constant", constant_values=0)


def nmr_fft(fid: np.ndarray, dt: float) -> dict:
    """FFT a real or complex FID.

    Uses ``scipy.fft.fft``. The real part of the transform is returned as the
    spectrum.

    Parameters
    ----------
    fid:
        Input FID (real or complex).
    dt:
        Dwell time in seconds per point.

    Returns
    -------
    dict
        Keys: spectrum (real part, ndarray), frequencies_hz (ndarray),
        n_points (int).
    """
    fid = np.asarray(fid, dtype=complex if np.iscomplexobj(fid) else float)
    n = len(fid)
    spectrum_complex = _scipy_fft(fid)
    spectrum_real = np.real(spectrum_complex)
    frequencies_hz = _fftfreq(n, d=dt)
    return {
        "spectrum": spectrum_real,
        "frequencies_hz": frequencies_hz,
        "n_points": n,
    }


def pick_nmr_peaks(
    ppm: np.ndarray,
    spectrum: np.ndarray,
    threshold_fraction: float = 0.01,
    min_separation_ppm: float = 0.05,
) -> dict:
    """Peak picking on a 1D NMR spectrum.

    Uses ``scipy.signal.find_peaks``. Works whether ppm is increasing or
    decreasing.

    Parameters
    ----------
    ppm:
        Chemical shift axis in ppm.
    spectrum:
        Spectral intensity array.
    threshold_fraction:
        Minimum peak height as a fraction of the tallest peak.
    min_separation_ppm:
        Minimum separation between adjacent peaks in ppm units.

    Returns
    -------
    dict
        Keys: ppm_positions, intensities, n_peaks.
    """
    ppm = np.asarray(ppm, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    max_intensity = spectrum.max() if spectrum.size > 0 else 0.0
    height = threshold_fraction * max_intensity

    ppm_step = float(np.abs(np.mean(np.diff(ppm)))) if len(ppm) > 1 else 1.0
    min_distance = max(1, int(np.round(min_separation_ppm / ppm_step)))

    peaks, _ = _find_peaks(spectrum, height=height, distance=min_distance)

    return {
        "ppm_positions": ppm[peaks],
        "intensities": spectrum[peaks],
        "n_peaks": len(peaks),
    }


def integrate_nmr_regions(
    ppm: np.ndarray,
    spectrum: np.ndarray,
    regions: list,
) -> dict:
    """Integrate spectrum over a list of (ppm_low, ppm_high) regions.

    Parameters
    ----------
    ppm:
        Chemical shift axis in ppm (may be ascending or descending).
    spectrum:
        Spectral intensity array.
    regions:
        List of (ppm_low, ppm_high) tuples defining integration windows.

    Returns
    -------
    dict
        Keys: regions, integrals, normalized_ratios.
        Ratios are normalized so the smallest non-zero integral equals 1.0.
        If all integrals are zero, ratios is a list of zeros.
    """
    ppm = np.asarray(ppm, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    integrals: list[float] = []
    for low, high in regions:
        # handle ascending and descending ppm axes
        mask = (ppm >= min(low, high)) & (ppm <= max(low, high))
        if mask.sum() < 2:
            integrals.append(0.0)
        else:
            integrals.append(float(np.trapezoid(spectrum[mask], ppm[mask])))

    integrals_arr = np.array(integrals)
    nonzero = integrals_arr[integrals_arr != 0.0]
    if nonzero.size > 0:
        min_val = float(np.abs(nonzero).min())
        normalized = (integrals_arr / min_val).tolist()
    else:
        normalized = [0.0] * len(integrals)

    return {
        "regions": list(regions),
        "integrals": integrals,
        "normalized_ratios": normalized,
    }


# ─── Mass Spec ────────────────────────────────────────────────────────────────


def find_mz_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    min_intensity_fraction: float = 0.01,
    min_separation: float = 0.3,
) -> dict:
    """Detect peaks in a mass spectrum.

    Parameters
    ----------
    mz:
        m/z axis array.
    intensity:
        Intensity array (same length as mz).
    min_intensity_fraction:
        Minimum peak height as a fraction of the base peak intensity.
    min_separation:
        Minimum m/z separation between adjacent peaks.

    Returns
    -------
    dict
        Keys: mz_positions, intensities, relative_intensities (% of base
        peak), base_peak_mz, base_peak_intensity, n_peaks.
        Returns empty arrays and n_peaks=0 if intensity is all zero.
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    empty_result = {
        "mz_positions": np.array([]),
        "intensities": np.array([]),
        "relative_intensities": np.array([]),
        "base_peak_mz": np.nan,
        "base_peak_intensity": 0.0,
        "n_peaks": 0,
    }

    if intensity.max() == 0.0:
        return empty_result

    base_intensity = float(intensity.max())
    height_threshold = min_intensity_fraction * base_intensity

    mz_step = float(np.abs(np.mean(np.diff(mz)))) if len(mz) > 1 else 1.0
    min_distance = max(1, int(np.round(min_separation / mz_step)))

    peaks, _ = _find_peaks(intensity, height=height_threshold, distance=min_distance)

    if len(peaks) == 0:
        return empty_result

    mz_pos = mz[peaks]
    int_pos = intensity[peaks]
    rel_int = (int_pos / base_intensity) * 100.0
    base_idx = int(np.argmax(int_pos))

    return {
        "mz_positions": mz_pos,
        "intensities": int_pos,
        "relative_intensities": rel_int,
        "base_peak_mz": float(mz_pos[base_idx]),
        "base_peak_intensity": float(int_pos[base_idx]),
        "n_peaks": len(peaks),
    }


def centroid_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    width: float = 0.05,
) -> dict:
    """Centroid a profile-mode mass spectrum.

    Groups contiguous points where intensity > 0 and computes the
    intensity-weighted centroid m/z for each group.

    Parameters
    ----------
    mz:
        m/z axis array.
    intensity:
        Intensity array (same length as mz).
    width:
        Minimum m/z gap between groups (not currently used for grouping, but
        reserved for future use). Grouping is based on contiguous nonzero runs.

    Returns
    -------
    dict
        Keys: mz_centroids, intensities, n_peaks.
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    above = intensity > 0.0
    centroids: list[float] = []
    peak_intensities: list[float] = []

    i = 0
    n = len(intensity)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            # group is [i, j)
            seg_mz = mz[i:j]
            seg_int = intensity[i:j]
            total = seg_int.sum()
            if total > 0.0:
                centroid = float(np.sum(seg_mz * seg_int) / total)
                centroids.append(centroid)
                peak_intensities.append(float(total))
            i = j
        else:
            i += 1

    return {
        "mz_centroids": np.array(centroids),
        "intensities": np.array(peak_intensities),
        "n_peaks": len(centroids),
    }
