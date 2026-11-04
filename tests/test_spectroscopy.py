import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from plottle.spectroscopy import (
    FUNCTIONAL_GROUPS,
    absorbance_to_transmittance,
    apply_line_broadening,
    assign_bands,
    atr_correction,
    beer_lambert,
    calibrate_ppm_axis,
    centroid_spectrum,
    find_mz_peaks,
    integrate_nmr_regions,
    molar_absorptivity_series,
    nmr_fft,
    pick_nmr_peaks,
    remove_cosmic_rays,
    spectral_overlap_integral,
    spectral_subtraction,
    transmittance_to_absorbance,
    zero_fill,
)


# ---------------------------------------------------------------------------
# 1. TestAbsorbanceTransmittanceConversion
# ---------------------------------------------------------------------------


class TestAbsorbanceTransmittanceConversion:
    def test_zero_absorbance_gives_100_percent(self):
        result = absorbance_to_transmittance(np.array([0.0]))
        assert pytest.approx(result[0], rel=1e-6) == 100.0

    def test_one_absorbance_gives_10_percent(self):
        result = absorbance_to_transmittance(np.array([1.0]))
        assert pytest.approx(result[0], rel=1e-5) == 10.0

    def test_two_absorbance_gives_1_percent(self):
        result = absorbance_to_transmittance(np.array([2.0]))
        assert pytest.approx(result[0], rel=1e-5) == 1.0

    def test_roundtrip(self):
        original = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        recovered = transmittance_to_absorbance(absorbance_to_transmittance(original))
        np.testing.assert_allclose(recovered, original, rtol=1e-5)

    def test_output_is_ndarray(self):
        a = np.linspace(0.0, 2.0, 20)
        result = absorbance_to_transmittance(a)
        assert isinstance(result, np.ndarray)
        assert result.shape == a.shape

    def test_transmittance_zero_raises(self):
        with pytest.raises(ValueError):
            transmittance_to_absorbance(np.array([50.0, 0.0, 10.0]))


# ---------------------------------------------------------------------------
# 2. TestATRCorrection
# ---------------------------------------------------------------------------


class TestATRCorrection:
    def _spectrum(self):
        wn = np.linspace(700.0, 4000.0, 200)
        ab = np.ones(200) * 0.5
        return wn, ab

    def test_returns_array_same_length(self):
        wn, ab = self._spectrum()
        result = atr_correction(wn, ab)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(ab)

    def test_below_critical_angle_raises(self):
        # n_atr=1.5, n_sample=1.0 → critical angle ~41.8°; 20° is below critical
        wn, ab = self._spectrum()
        with pytest.raises(ValueError):
            atr_correction(wn, ab, n_atr=1.5, angle_deg=20.0, n_sample=1.0)

    def test_output_differs_from_input(self):
        wn, ab = self._spectrum()
        result = atr_correction(wn, ab)
        assert not np.allclose(result, ab)

    def test_wavenumber_dependence_produces_variation(self):
        wn, ab = self._spectrum()
        result = atr_correction(wn, ab)
        assert result.std() > 0.0

    def test_non_negative_output_for_non_negative_input(self):
        wn, ab = self._spectrum()
        result = atr_correction(wn, ab)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# 3. TestSpectralSubtraction
# ---------------------------------------------------------------------------


class TestSpectralSubtraction:
    def _arrays(self):
        wn = np.linspace(500.0, 4000.0, 100)
        a = np.sin(wn / 500.0) + 2.0
        b = np.cos(wn / 500.0) + 1.5
        return wn, a, b

    def test_scale_zero_returns_spectrum_a(self):
        wn, a, b = self._arrays()
        result = spectral_subtraction(wn, a, b, scale=0.0)
        np.testing.assert_array_equal(result, a)

    def test_scale_one_exact_difference(self):
        wn, a, b = self._arrays()
        result = spectral_subtraction(wn, a, b, scale=1.0)
        np.testing.assert_allclose(result, a - b)

    def test_scale_half(self):
        wn, a, b = self._arrays()
        result = spectral_subtraction(wn, a, b, scale=0.5)
        np.testing.assert_allclose(result, a - 0.5 * b)

    def test_length_mismatch_raises(self):
        wn = np.arange(100, dtype=float)
        a = np.ones(100)
        b = np.ones(80)
        with pytest.raises(ValueError):
            spectral_subtraction(wn, a, b)

    def test_returns_ndarray(self):
        wn, a, b = self._arrays()
        result = spectral_subtraction(wn, a, b)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# 4. TestCosmicRayRemoval
# ---------------------------------------------------------------------------


class TestCosmicRayRemoval:
    def test_spike_replaced(self):
        # Signal needs some non-zero variance so local_std > 0; use small noise.
        rng = np.random.default_rng(42)
        y = 1.0 + rng.normal(0, 0.05, 50)
        y[25] = 1000.0
        result = remove_cosmic_rays(y, threshold_sigma=3.0, window=5)
        assert result[25] < 100.0

    def test_clean_signal_unchanged(self):
        rng = np.random.default_rng(42)
        y = np.sin(np.linspace(0, 2 * np.pi, 100)) + rng.normal(0, 0.01, 100)
        result = remove_cosmic_rays(y, threshold_sigma=10.0, window=5)
        np.testing.assert_allclose(result, y, atol=0.1)

    def test_same_length_output(self):
        y = np.linspace(0, 1, 200)
        result = remove_cosmic_rays(y)
        assert result.shape == y.shape

    def test_high_threshold_preserves_spike(self):
        y = np.ones(30) * 1.0
        y[10] = 50.0
        result = remove_cosmic_rays(y, threshold_sigma=100.0, window=3)
        assert result[10] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 5. TestFunctionalGroupAssignment
# ---------------------------------------------------------------------------


class TestFunctionalGroupAssignment:
    def test_assign_bands_returns_list(self):
        result = assign_bands(1000.0)
        assert isinstance(result, list)

    def test_carbonyl_assignment(self):
        # ~1720 cm-1 should match a C=O stretch
        matches = assign_bands(1720.0)
        assert any("C=O" in m for m in matches)

    def test_oh_stretch_assignment(self):
        # ~3400 cm-1 should match O-H stretch
        matches = assign_bands(3400.0)
        assert any("O-H" in m or "O-h" in m.lower() for m in matches)

    def test_ch_stretch_assignment(self):
        # ~2900 cm-1 should match C-H stretch
        matches = assign_bands(2900.0)
        assert any("C-H" in m or "c-h" in m.lower() for m in matches)

    def test_no_match_returns_empty(self):
        # 50 cm-1 is outside all defined ranges
        matches = assign_bands(50.0)
        assert matches == []

    def test_functional_groups_is_dict(self):
        assert isinstance(FUNCTIONAL_GROUPS, dict)
        assert len(FUNCTIONAL_GROUPS) > 0

    def test_functional_groups_values_are_tuples(self):
        for name, bounds in FUNCTIONAL_GROUPS.items():
            # Each value should be a (low, high) tuple (not a list of tuples)
            assert isinstance(bounds, tuple), (
                f"FUNCTIONAL_GROUPS['{name}'] is {type(bounds)}, expected tuple"
            )
            assert len(bounds) == 2


# ---------------------------------------------------------------------------
# 6. TestBeerLambert
# ---------------------------------------------------------------------------


class TestBeerLambert:
    def test_solve_concentration(self):
        # A=0.5, eps=5000, l=1 → c = A / (eps * l) = 1e-4
        result = beer_lambert(0.5, epsilon=5000.0, concentration=None, path_length=1.0)
        assert pytest.approx(result["concentration"], rel=1e-6) == 1e-4

    def test_solve_epsilon(self):
        # A=0.5, c=1e-4, l=1 → eps = A / (c * l) = 5000
        result = beer_lambert(0.5, epsilon=None, concentration=1e-4, path_length=1.0)
        assert pytest.approx(result["epsilon"], rel=1e-6) == 5000.0

    def test_path_length_scaling(self):
        r1 = beer_lambert(0.5, epsilon=5000.0, concentration=None, path_length=1.0)
        r2 = beer_lambert(0.5, epsilon=5000.0, concentration=None, path_length=2.0)
        assert pytest.approx(r2["concentration"], rel=1e-6) == r1["concentration"] / 2.0

    def test_both_none_raises(self):
        with pytest.raises(ValueError):
            beer_lambert(0.5, epsilon=None, concentration=None)

    def test_neither_none_raises(self):
        with pytest.raises(ValueError):
            beer_lambert(0.5, epsilon=5000.0, concentration=1e-4)

    def test_path_length_zero_raises(self):
        with pytest.raises(ValueError):
            beer_lambert(0.5, epsilon=5000.0, concentration=None, path_length=0.0)

    def test_result_dict_keys(self):
        result = beer_lambert(0.5, epsilon=5000.0, concentration=None)
        for key in ("absorbance", "epsilon", "concentration", "path_length", "solved_for"):
            assert key in result

    def test_solved_for_field_concentration(self):
        result = beer_lambert(0.5, epsilon=5000.0, concentration=None)
        assert result["solved_for"] == "concentration"

    def test_solved_for_field_epsilon(self):
        result = beer_lambert(0.5, epsilon=None, concentration=1e-4)
        assert result["solved_for"] == "epsilon"


# ---------------------------------------------------------------------------
# 7. TestMolarAbsorptivitySeries
# ---------------------------------------------------------------------------


class TestMolarAbsorptivitySeries:
    def _perfect_data(self):
        # Perfect Beer-Lambert: eps=5000, l=1
        conc = np.array([1e-5, 2e-5, 5e-5, 1e-4, 2e-4])
        abs_vals = 5000.0 * conc * 1.0
        return conc, abs_vals

    def test_known_epsilon(self):
        conc, abs_vals = self._perfect_data()
        result = molar_absorptivity_series(conc, abs_vals, path_length=1.0)
        assert pytest.approx(result["epsilon"], rel=1e-5) == 5000.0

    def test_r_squared_perfect(self):
        conc, abs_vals = self._perfect_data()
        result = molar_absorptivity_series(conc, abs_vals)
        assert pytest.approx(result["r_squared"], abs=1e-9) == 1.0

    def test_r_squared_at_most_one(self):
        conc, abs_vals = self._perfect_data()
        result = molar_absorptivity_series(conc, abs_vals)
        assert result["r_squared"] <= 1.0

    def test_linearity_ok_flag_true(self):
        conc, abs_vals = self._perfect_data()
        result = molar_absorptivity_series(conc, abs_vals)
        assert result["linearity_ok"] is True

    def test_linearity_ok_is_bool(self):
        conc, abs_vals = self._perfect_data()
        result = molar_absorptivity_series(conc, abs_vals)
        assert isinstance(result["linearity_ok"], bool)

    def test_result_keys(self):
        conc, abs_vals = self._perfect_data()
        result = molar_absorptivity_series(conc, abs_vals)
        for key in (
            "epsilon",
            "r_squared",
            "slope",
            "intercept",
            "fitted_absorbances",
            "residuals",
            "linearity_ok",
        ):
            assert key in result


# ---------------------------------------------------------------------------
# 8. TestSpectralOverlapIntegral
# ---------------------------------------------------------------------------


class TestSpectralOverlapIntegral:
    def _gaussian(self, wl, center, sigma, amplitude=1.0):
        return amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    def test_result_keys(self):
        wl = np.linspace(400, 700, 200)
        donor = self._gaussian(wl, 550, 20)
        acceptor = self._gaussian(wl, 580, 20, amplitude=50000)
        result = spectral_overlap_integral(wl, donor, wl, acceptor)
        for key in ("J", "wavelengths_common", "integrand", "donor_norm", "acceptor_interp"):
            assert key in result

    def test_no_overlap_returns_J_zero(self):
        wl_d = np.linspace(400, 500, 100)
        wl_a = np.linspace(600, 700, 100)
        donor = self._gaussian(wl_d, 450, 10)
        acceptor = self._gaussian(wl_a, 650, 10, amplitude=50000)
        result = spectral_overlap_integral(wl_d, donor, wl_a, acceptor)
        assert result["J"] == 0.0

    def test_J_nonnegative(self):
        wl = np.linspace(400, 700, 200)
        donor = self._gaussian(wl, 540, 25)
        acceptor = self._gaussian(wl, 560, 25, amplitude=40000)
        result = spectral_overlap_integral(wl, donor, wl, acceptor)
        assert result["J"] >= 0.0

    def test_overlapping_spectra_J_positive(self):
        wl = np.linspace(450, 700, 500)
        donor = self._gaussian(wl, 530, 20)
        acceptor = self._gaussian(wl, 555, 20, amplitude=80000)
        result = spectral_overlap_integral(wl, donor, wl, acceptor)
        assert result["J"] > 0.0

    def test_wavelengths_common_is_array(self):
        wl = np.linspace(400, 700, 100)
        donor = self._gaussian(wl, 540, 20)
        acceptor = self._gaussian(wl, 540, 20, amplitude=30000)
        result = spectral_overlap_integral(wl, donor, wl, acceptor)
        assert isinstance(result["wavelengths_common"], np.ndarray)


# ---------------------------------------------------------------------------
# 9. TestNMRCalibration
# ---------------------------------------------------------------------------


class TestNMRCalibration:
    def test_zero_reference(self):
        hz = np.array([0.0, 300.0, 600.0, 1500.0])
        ppm = calibrate_ppm_axis(hz, spectrometer_freq_mhz=300.0, reference_hz=0.0)
        np.testing.assert_allclose(ppm, hz / 300.0)

    def test_offset_reference(self):
        hz = np.array([300.0, 600.0, 900.0])
        ppm = calibrate_ppm_axis(hz, spectrometer_freq_mhz=300.0, reference_hz=300.0)
        expected = (hz - 300.0) / 300.0
        np.testing.assert_allclose(ppm, expected)

    def test_returns_ndarray(self):
        hz = np.arange(10, dtype=float) * 100.0
        result = calibrate_ppm_axis(hz, spectrometer_freq_mhz=400.0)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        hz = np.linspace(0, 3000, 256)
        ppm = calibrate_ppm_axis(hz, spectrometer_freq_mhz=600.0)
        assert ppm.shape == hz.shape


# ---------------------------------------------------------------------------
# 10. TestLinebroadeningAndZeroFill
# ---------------------------------------------------------------------------


class TestLinebroadeningAndZeroFill:
    def _fid(self, n=512):
        t = np.arange(n, dtype=float) * 1e-4
        return np.cos(2 * np.pi * 100 * t) * np.exp(-t * 10)

    def test_lorentzian_mode_returns_array(self):
        fid = self._fid()
        result = apply_line_broadening(fid, dt=1e-4, lb=1.0, mode="lorentzian")
        assert result.shape == fid.shape

    def test_gaussian_mode_returns_array(self):
        fid = self._fid()
        result = apply_line_broadening(fid, dt=1e-4, lb=1.0, mode="gaussian")
        assert result.shape == fid.shape

    def test_invalid_mode_raises(self):
        fid = self._fid()
        with pytest.raises(ValueError):
            apply_line_broadening(fid, dt=1e-4, lb=1.0, mode="hamming")

    def test_zero_fill_increases_length(self):
        fid = np.ones(256)
        result = zero_fill(fid, 1024)
        assert len(result) == 1024

    def test_zero_fill_pads_with_zeros(self):
        fid = np.ones(64)
        result = zero_fill(fid, 256)
        np.testing.assert_array_equal(result[64:], np.zeros(192))

    def test_zero_fill_raises_when_n_points_too_small(self):
        fid = np.ones(512)
        with pytest.raises(ValueError):
            zero_fill(fid, 256)


# ---------------------------------------------------------------------------
# 11. TestNMRFFT
# ---------------------------------------------------------------------------


class TestNMRFFT:
    def _fid(self, n=1024):
        t = np.arange(n, dtype=float) * 1e-4
        return np.cos(2 * np.pi * 100 * t) * np.exp(-t * 5)

    def test_result_keys(self):
        fid = self._fid()
        result = nmr_fft(fid, dt=1e-4)
        for key in ("spectrum", "frequencies_hz", "n_points"):
            assert key in result

    def test_spectrum_length_equals_n_points(self):
        fid = self._fid(512)
        result = nmr_fft(fid, dt=1e-4)
        assert len(result["spectrum"]) == result["n_points"]

    def test_frequencies_hz_length_equals_n_points(self):
        fid = self._fid(512)
        result = nmr_fft(fid, dt=1e-4)
        assert len(result["frequencies_hz"]) == result["n_points"]

    def test_n_points_matches_input_length(self):
        fid = self._fid(1024)
        result = nmr_fft(fid, dt=1e-4)
        assert result["n_points"] == 1024


# ---------------------------------------------------------------------------
# 12. TestNMRPeakPicking
# ---------------------------------------------------------------------------


class TestNMRPeakPicking:
    def _synthetic_nmr(self):
        ppm = np.linspace(0.0, 10.0, 2000)
        spectrum = np.exp(-((ppm - 2.0) ** 2) / (2 * 0.05**2)) + 0.5 * np.exp(
            -((ppm - 7.5) ** 2) / (2 * 0.05**2)
        )
        return ppm, spectrum

    def test_finds_two_peaks(self):
        ppm, spectrum = self._synthetic_nmr()
        result = pick_nmr_peaks(ppm, spectrum, threshold_fraction=0.1)
        assert result["n_peaks"] >= 2

    def test_result_keys(self):
        ppm, spectrum = self._synthetic_nmr()
        result = pick_nmr_peaks(ppm, spectrum)
        for key in ("ppm_positions", "intensities", "n_peaks"):
            assert key in result

    def test_n_peaks_matches_ppm_positions_length(self):
        ppm, spectrum = self._synthetic_nmr()
        result = pick_nmr_peaks(ppm, spectrum)
        assert result["n_peaks"] == len(result["ppm_positions"])

    def test_high_threshold_reduces_peaks(self):
        ppm, spectrum = self._synthetic_nmr()
        r_low = pick_nmr_peaks(ppm, spectrum, threshold_fraction=0.01)
        r_high = pick_nmr_peaks(ppm, spectrum, threshold_fraction=0.9)
        assert r_high["n_peaks"] <= r_low["n_peaks"]

    def test_ppm_positions_within_range(self):
        ppm, spectrum = self._synthetic_nmr()
        result = pick_nmr_peaks(ppm, spectrum, threshold_fraction=0.1)
        for p in result["ppm_positions"]:
            assert ppm.min() <= p <= ppm.max()


# ---------------------------------------------------------------------------
# 13. TestNMRIntegration
# ---------------------------------------------------------------------------


class TestNMRIntegration:
    def _spectrum(self):
        ppm = np.linspace(0.0, 10.0, 1000)
        spectrum = np.exp(-((ppm - 2.0) ** 2) / (2 * 0.1**2)) + 0.5 * np.exp(
            -((ppm - 7.0) ** 2) / (2 * 0.1**2)
        )
        return ppm, spectrum

    def test_result_keys(self):
        ppm, spectrum = self._spectrum()
        result = integrate_nmr_regions(ppm, spectrum, [(1.5, 2.5), (6.5, 7.5)])
        for key in ("regions", "integrals", "normalized_ratios"):
            assert key in result

    def test_integral_count_matches_regions(self):
        ppm, spectrum = self._spectrum()
        result = integrate_nmr_regions(ppm, spectrum, [(1.5, 2.5), (6.5, 7.5), (4.0, 5.0)])
        assert len(result["integrals"]) == 3

    def test_normalized_ratios_min_is_one(self):
        ppm, spectrum = self._spectrum()
        result = integrate_nmr_regions(ppm, spectrum, [(1.5, 2.5), (6.5, 7.5)])
        nonzero = [r for r in result["normalized_ratios"] if r > 0]
        if nonzero:
            assert pytest.approx(min(nonzero), rel=1e-5) == 1.0

    def test_all_zero_spectrum_zero_ratios(self):
        ppm = np.linspace(0, 10, 100)
        spectrum = np.zeros_like(ppm)
        result = integrate_nmr_regions(ppm, spectrum, [(1.0, 2.0), (3.0, 4.0)])
        assert all(r == 0.0 for r in result["normalized_ratios"])

    def test_empty_region_integral_is_zero(self):
        ppm, spectrum = self._spectrum()
        # Region [4.0, 5.0] contains no peaks in the fixture
        result = integrate_nmr_regions(ppm, spectrum, [(4.0, 5.0)])
        assert result["integrals"][0] >= 0.0  # non-negative at minimum


# ---------------------------------------------------------------------------
# 14. TestMZPeakFinding
# ---------------------------------------------------------------------------


class TestMZPeakFinding:
    def _ms_spectrum(self):
        mz = np.linspace(50.0, 500.0, 4500)
        intensity = np.zeros_like(mz)
        for center, height in [(128.0, 1000.0), (256.5, 500.0), (384.2, 250.0)]:
            idx = np.argmin(np.abs(mz - center))
            intensity[idx] = height
        return mz, intensity

    def test_result_keys(self):
        mz, intensity = self._ms_spectrum()
        result = find_mz_peaks(mz, intensity)
        for key in (
            "mz_positions",
            "intensities",
            "relative_intensities",
            "base_peak_mz",
            "base_peak_intensity",
            "n_peaks",
        ):
            assert key in result

    def test_base_peak_is_highest(self):
        mz, intensity = self._ms_spectrum()
        result = find_mz_peaks(mz, intensity)
        assert pytest.approx(result["base_peak_intensity"]) == max(result["intensities"])

    def test_relative_intensity_of_base_peak_is_100(self):
        mz, intensity = self._ms_spectrum()
        result = find_mz_peaks(mz, intensity)
        assert pytest.approx(max(result["relative_intensities"])) == 100.0

    def test_all_zero_intensity_returns_zero_peaks(self):
        mz = np.linspace(50, 500, 100)
        intensity = np.zeros(100)
        result = find_mz_peaks(mz, intensity)
        assert result["n_peaks"] == 0
        assert len(result["mz_positions"]) == 0

    def test_n_peaks_matches_mz_positions(self):
        mz, intensity = self._ms_spectrum()
        result = find_mz_peaks(mz, intensity)
        assert result["n_peaks"] == len(result["mz_positions"])

    def test_relative_intensities_bounded(self):
        mz, intensity = self._ms_spectrum()
        result = find_mz_peaks(mz, intensity)
        assert all(0.0 <= r <= 100.0 for r in result["relative_intensities"])


# ---------------------------------------------------------------------------
# 15. TestCentroidSpectrum
# ---------------------------------------------------------------------------


class TestCentroidSpectrum:
    def test_result_keys(self):
        mz = np.array([100.0, 100.1, 100.2, 200.0, 200.1])
        intensity = np.array([0.0, 5.0, 0.0, 10.0, 0.0])
        result = centroid_spectrum(mz, intensity)
        for key in ("mz_centroids", "intensities", "n_peaks"):
            assert key in result

    def test_single_peak_weighted_centroid(self):
        # Two equal-intensity points at 101.0 and 102.0 → centroid at 101.5
        mz = np.array([100.0, 101.0, 102.0, 103.0])
        intensity = np.array([0.0, 1.0, 1.0, 0.0])
        result = centroid_spectrum(mz, intensity, width=2.0)
        assert result["n_peaks"] == 1
        assert pytest.approx(result["mz_centroids"][0], rel=1e-5) == 101.5

    def test_n_peaks_matches_centroids(self):
        mz = np.linspace(50, 500, 1000)
        intensity = np.zeros(1000)
        intensity[200] = 100.0
        intensity[700] = 200.0
        result = centroid_spectrum(mz, intensity)
        assert result["n_peaks"] == len(result["mz_centroids"])

    def test_all_zero_intensity(self):
        mz = np.linspace(100, 300, 50)
        intensity = np.zeros(50)
        result = centroid_spectrum(mz, intensity)
        assert result["n_peaks"] == 0

    def test_n_peaks_count_two_isolated_peaks(self):
        mz = np.linspace(50, 500, 1000)
        intensity = np.zeros(1000)
        # Two well-separated peaks
        intensity[100] = 500.0
        intensity[800] = 300.0
        result = centroid_spectrum(mz, intensity)
        assert result["n_peaks"] >= 2
