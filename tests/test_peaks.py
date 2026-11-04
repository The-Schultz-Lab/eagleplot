"""Unit tests for plottle/peaks.py — Peak Analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from plottle.peaks import (
    compute_fwhm,
    find_peaks,
    fit_multipeak,
    fit_peak,
    integrate_peaks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _gaussian_signal(x, center, amplitude, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


@pytest.fixture
def single_peak():
    """Clean single Gaussian peak on a flat zero baseline."""
    x = np.linspace(0.0, 10.0, 500)
    y = _gaussian_signal(x, center=5.0, amplitude=2.0, sigma=0.5)
    return x, y


@pytest.fixture
def two_peaks():
    """Two Gaussian peaks, well separated, zero baseline."""
    x = np.linspace(0.0, 20.0, 1000)
    y = _gaussian_signal(x, 5.0, 3.0, 0.5) + _gaussian_signal(x, 15.0, 2.0, 0.7)
    return x, y


@pytest.fixture
def noisy_single_peak(single_peak):
    x, y = single_peak
    return x, y + RNG.normal(0, 0.02, len(y))


@pytest.fixture
def peak_on_linear_bg():
    """Single Gaussian on a rising linear baseline."""
    x = np.linspace(0.0, 10.0, 500)
    bg = 0.5 + 0.1 * x
    y = _gaussian_signal(x, 5.0, 3.0, 0.6) + bg
    return x, y


# ---------------------------------------------------------------------------
# find_peaks
# ---------------------------------------------------------------------------


class TestFindPeaks:
    def test_detects_single_peak(self, single_peak):
        x, y = single_peak
        result = find_peaks(y, x=x)
        assert result["n_peaks"] == 1

    def test_peak_position_close_to_true(self, single_peak):
        x, y = single_peak
        result = find_peaks(y, x=x)
        assert abs(result["positions"][0] - 5.0) < 0.1

    def test_peak_height_close_to_true(self, single_peak):
        x, y = single_peak
        result = find_peaks(y, x=x)
        assert abs(result["heights"][0] - 2.0) < 0.05

    def test_detects_two_peaks(self, two_peaks):
        x, y = two_peaks
        result = find_peaks(y, x=x)
        assert result["n_peaks"] == 2

    def test_returns_correct_keys(self, single_peak):
        x, y = single_peak
        result = find_peaks(y, x=x)
        expected = {
            "indices",
            "positions",
            "heights",
            "prominences",
            "widths_samples",
            "widths",
            "left_ips",
            "right_ips",
            "n_peaks",
        }
        assert set(result.keys()) == expected

    def test_no_x_positions_are_indices(self):
        y = np.array([0.0, 1.0, 0.0, 0.0, 1.5, 0.0])
        result = find_peaks(y)
        # positions should equal indices cast to float when no x supplied
        np.testing.assert_array_equal(result["positions"], result["indices"].astype(float))

    def test_empty_result_when_no_peaks(self):
        y = np.zeros(50)
        result = find_peaks(y)
        assert result["n_peaks"] == 0
        assert len(result["indices"]) == 0

    def test_height_filter(self, two_peaks):
        x, y = two_peaks
        # Only the taller peak (amplitude 3) should pass height=2.5
        result = find_peaks(y, x=x, height=2.5)
        assert result["n_peaks"] == 1
        assert abs(result["positions"][0] - 5.0) < 0.2

    def test_prominence_filter(self, two_peaks):
        x, y = two_peaks
        result = find_peaks(y, x=x, prominence=2.5)
        assert result["n_peaks"] == 1

    def test_distance_filter(self, two_peaks):
        x, y = two_peaks
        # Both peaks are ~500 samples apart; distance=600 should keep only one
        result = find_peaks(y, distance=600)
        assert result["n_peaks"] == 1

    def test_widths_positive(self, two_peaks):
        x, y = two_peaks
        result = find_peaks(y, x=x)
        assert all(w > 0 for w in result["widths"])

    def test_width_units_match_x_scale(self, single_peak):
        x, y = single_peak
        result = find_peaks(y, x=x)
        # Gaussian sigma=0.5 → FWHM ≈ 1.18; widths should be in x units (0–10)
        assert result["widths"][0] < (x.max() - x.min())

    def test_n_peaks_matches_arrays(self, two_peaks):
        x, y = two_peaks
        result = find_peaks(y, x=x)
        n = result["n_peaks"]
        assert len(result["indices"]) == n
        assert len(result["positions"]) == n
        assert len(result["heights"]) == n


# ---------------------------------------------------------------------------
# integrate_peaks
# ---------------------------------------------------------------------------


class TestIntegratePeaks:
    def test_area_positive(self, single_peak):
        x, y = single_peak
        peaks = find_peaks(y, x=x)
        result = integrate_peaks(y, x, peaks)
        assert all(a >= 0 for a in result["areas"])

    def test_area_reasonable_magnitude(self, single_peak):
        """Gaussian area ≈ A·σ·√(2π) ≈ 2.0·0.5·2.507 ≈ 2.51."""
        x, y = single_peak
        peaks = find_peaks(y, x=x)
        result = integrate_peaks(y, x, peaks)
        # Allow generous tolerance because prominence-based bounds clip the tails
        assert 1.5 < result["areas"][0] < 4.0

    def test_returns_correct_keys(self, single_peak):
        x, y = single_peak
        peaks = find_peaks(y, x=x)
        result = integrate_peaks(y, x, peaks)
        assert set(result.keys()) == {"areas", "left_bases", "right_bases"}

    def test_two_peaks_two_areas(self, two_peaks):
        x, y = two_peaks
        peaks = find_peaks(y, x=x)
        result = integrate_peaks(y, x, peaks)
        assert len(result["areas"]) == 2

    def test_empty_peaks_empty_areas(self):
        y = np.zeros(50)
        x = np.linspace(0, 5, 50)
        peaks = find_peaks(y)
        result = integrate_peaks(y, x, peaks)
        assert len(result["areas"]) == 0

    def test_left_base_less_than_right_base(self, single_peak):
        x, y = single_peak
        peaks = find_peaks(y, x=x)
        result = integrate_peaks(y, x, peaks)
        for lb, rb in zip(result["left_bases"], result["right_bases"]):
            assert lb < rb


# ---------------------------------------------------------------------------
# compute_fwhm
# ---------------------------------------------------------------------------


class TestComputeFwhm:
    def test_fwhm_close_to_analytical(self, single_peak):
        """Gaussian FWHM = 2·√(2·ln2)·σ ≈ 1.177 for σ=0.5."""
        x, y = single_peak
        peaks = find_peaks(y, x=x)
        result = compute_fwhm(y, x, peaks)
        expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * 0.5
        assert abs(result["fwhm"][0] - expected_fwhm) < 0.1

    def test_returns_correct_keys(self, single_peak):
        x, y = single_peak
        peaks = find_peaks(y, x=x)
        result = compute_fwhm(y, x, peaks)
        assert set(result.keys()) == {"fwhm", "fwhm_samples", "centers"}

    def test_fwhm_positive(self, two_peaks):
        x, y = two_peaks
        peaks = find_peaks(y, x=x)
        result = compute_fwhm(y, x, peaks)
        assert all(f > 0 for f in result["fwhm"])

    def test_empty_peaks_empty_fwhm(self):
        y = np.zeros(50)
        x = np.linspace(0, 5, 50)
        peaks = find_peaks(y)
        result = compute_fwhm(y, x, peaks)
        assert len(result["fwhm"]) == 0

    def test_centers_close_to_peak_positions(self, two_peaks):
        x, y = two_peaks
        peaks = find_peaks(y, x=x)
        result = compute_fwhm(y, x, peaks)
        np.testing.assert_allclose(result["centers"], peaks["positions"], atol=0.05)


# ---------------------------------------------------------------------------
# fit_peak
# ---------------------------------------------------------------------------


class TestFitPeak:
    def test_gaussian_fit_recovers_center(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0, model="gaussian", background="none")
        idx = result["param_names"].index("center")
        assert abs(result["params"][idx] - 5.0) < 0.05

    def test_gaussian_fit_recovers_amplitude(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0, model="gaussian", background="none")
        idx = result["param_names"].index("amplitude")
        assert abs(result["params"][idx] - 2.0) < 0.1

    def test_gaussian_fit_r_squared_high(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0, model="gaussian", background="none")
        assert result["r_squared"] > 0.99

    def test_lorentzian_fit_r_squared_high(self):
        x = np.linspace(0, 10, 500)
        y = 2.0 / (((x - 5.0) / 0.5) ** 2 + 1.0)  # Lorentzian
        result = fit_peak(y, x, center_guess=5.0, model="lorentzian", background="none")
        assert result["r_squared"] > 0.99

    def test_linear_background_fit(self, peak_on_linear_bg):
        x, y = peak_on_linear_bg
        result = fit_peak(y, x, center_guess=5.0, model="gaussian", background="linear")
        assert result["r_squared"] > 0.98
        assert "bg_offset" in result["param_names"]
        assert "bg_slope" in result["param_names"]

    def test_constant_background_fit(self):
        x = np.linspace(0, 10, 500)
        y = _gaussian_signal(x, 5.0, 2.0, 0.5) + 1.0  # flat offset
        result = fit_peak(y, x, center_guess=5.0, model="gaussian", background="constant")
        assert result["r_squared"] > 0.99
        assert "bg_offset" in result["param_names"]

    def test_returns_correct_keys(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0)
        expected = {
            "params",
            "param_names",
            "std_errors",
            "fitted_y",
            "residuals",
            "r_squared",
            "model",
            "background",
        }
        assert set(result.keys()) == expected

    def test_fitted_y_shape(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0)
        assert result["fitted_y"].shape == y.shape

    def test_residuals_small_for_good_fit(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0, model="gaussian", background="none")
        assert np.max(np.abs(result["residuals"])) < 0.05

    def test_pseudo_voigt_runs(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0, model="pseudo_voigt", background="none")
        assert result["r_squared"] > 0.95

    def test_voigt_runs(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0, model="voigt", background="none")
        assert result["r_squared"] > 0.95

    def test_invalid_model_raises(self, single_peak):
        x, y = single_peak
        with pytest.raises(ValueError, match="Unknown model"):
            fit_peak(y, x, center_guess=5.0, model="invalid_model")

    def test_std_errors_positive(self, single_peak):
        x, y = single_peak
        result = fit_peak(y, x, center_guess=5.0, model="gaussian", background="none")
        assert all(e >= 0 for e in result["std_errors"])


# ---------------------------------------------------------------------------
# fit_multipeak
# ---------------------------------------------------------------------------


class TestFitMultipeak:
    def test_single_peak_as_multipeak(self, single_peak):
        x, y = single_peak
        result = fit_multipeak(y, x, n_peaks=1, model="gaussian", background="none")
        assert result["n_peaks"] == 1
        assert result["r_squared"] > 0.99

    def test_two_peaks_fit(self, two_peaks):
        x, y = two_peaks
        result = fit_multipeak(y, x, n_peaks=2, model="gaussian", background="none")
        assert result["n_peaks"] == 2
        assert result["r_squared"] > 0.99

    def test_two_peaks_centers_recovered(self, two_peaks):
        x, y = two_peaks
        result = fit_multipeak(y, x, n_peaks=2, model="gaussian", background="none")
        centers = [s["center"] for s in result["peak_summaries"]]
        centers.sort()
        assert abs(centers[0] - 5.0) < 0.3
        assert abs(centers[1] - 15.0) < 0.3

    def test_individual_y_count(self, two_peaks):
        x, y = two_peaks
        result = fit_multipeak(y, x, n_peaks=2, model="gaussian", background="none")
        assert len(result["individual_y"]) == 2

    def test_individual_y_shapes(self, two_peaks):
        x, y = two_peaks
        result = fit_multipeak(y, x, n_peaks=2, model="gaussian", background="none")
        for iy in result["individual_y"]:
            assert iy.shape == y.shape

    def test_returns_correct_keys(self, single_peak):
        x, y = single_peak
        result = fit_multipeak(y, x, n_peaks=1)
        expected = {
            "params",
            "param_names",
            "std_errors",
            "fitted_y",
            "individual_y",
            "residuals",
            "r_squared",
            "model",
            "background",
            "n_peaks",
            "peak_summaries",
        }
        assert set(result.keys()) == expected

    def test_peak_summaries_have_fwhm(self, single_peak):
        x, y = single_peak
        result = fit_multipeak(y, x, n_peaks=1, model="gaussian", background="none")
        summary = result["peak_summaries"][0]
        assert "fwhm" in summary
        assert summary["fwhm"] > 0

    def test_with_initial_guesses(self, two_peaks):
        x, y = two_peaks
        guesses = [(5.0, 3.0, 0.5), (15.0, 2.0, 0.7)]
        result = fit_multipeak(
            y,
            x,
            n_peaks=2,
            model="gaussian",
            background="none",
            initial_guesses=guesses,
        )
        assert result["r_squared"] > 0.99

    def test_linear_background(self, peak_on_linear_bg):
        x, y = peak_on_linear_bg
        result = fit_multipeak(y, x, n_peaks=1, model="gaussian", background="linear")
        assert result["r_squared"] > 0.98
        assert "bg_offset" in result["param_names"]

    def test_constant_background(self):
        x = np.linspace(0, 10, 500)
        y = _gaussian_signal(x, 5.0, 2.0, 0.5) + 1.0
        result = fit_multipeak(y, x, n_peaks=1, model="gaussian", background="constant")
        assert result["r_squared"] > 0.99

    def test_lorentzian_model(self):
        x = np.linspace(0, 10, 500)
        y = 2.0 / (((x - 5.0) / 0.5) ** 2 + 1.0)
        result = fit_multipeak(y, x, n_peaks=1, model="lorentzian", background="none")
        assert result["r_squared"] > 0.99

    def test_pseudo_voigt_model(self, single_peak):
        x, y = single_peak
        result = fit_multipeak(y, x, n_peaks=1, model="pseudo_voigt", background="none")
        assert result["r_squared"] > 0.95

    def test_voigt_model(self, single_peak):
        x, y = single_peak
        result = fit_multipeak(y, x, n_peaks=1, model="voigt", background="none")
        assert result["r_squared"] > 0.95

    def test_invalid_model_raises(self, single_peak):
        x, y = single_peak
        with pytest.raises(ValueError, match="Unknown model"):
            fit_multipeak(y, x, n_peaks=1, model="bad_model")

    def test_param_names_count(self, two_peaks):
        x, y = two_peaks
        # 2 Gaussian peaks × 3 params + 0 bg = 6 param names
        result = fit_multipeak(y, x, n_peaks=2, model="gaussian", background="none")
        assert len(result["param_names"]) == 6

    def test_param_names_count_with_bg(self, two_peaks):
        x, y = two_peaks
        # 2 × 3 + 2 bg (linear) = 8
        result = fit_multipeak(y, x, n_peaks=2, model="gaussian", background="linear")
        assert len(result["param_names"]) == 8

    def test_auto_guess_works_when_fewer_detected(self):
        """Auto-guessing falls back to evenly-spaced guesses when auto-detection
        finds fewer peaks than requested."""
        # Signal has only one clear peak; request n_peaks=2 to trigger fallback
        x = np.linspace(0, 10, 500)
        y = _gaussian_signal(x, 5.0, 2.0, 0.5)
        # With n_peaks=2 but only 1 detected, should use evenly-spaced fallback
        result = fit_multipeak(y, x, n_peaks=2, model="gaussian", background="none")
        assert "fitted_y" in result
        assert result["n_peaks"] == 2
