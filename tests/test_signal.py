"""Unit tests for eagleplot/signal.py — Signal Processing module."""

from __future__ import annotations

import numpy as np
import pytest

from eagleplot.signal import (
    baseline_als,
    baseline_polynomial,
    baseline_rolling_ball,
    derivative,
    fft,
    filter_bandpass,
    filter_bandstop,
    filter_highpass,
    filter_lowpass,
    ifft,
    interpolate,
    power_spectrum,
    smooth_gaussian,
    smooth_moving_average,
    smooth_savitzky_golay,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


@pytest.fixture
def sine_wave():
    """Clean 1-Hz sine wave sampled at 100 Hz, 2 seconds."""
    fs = 100
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    return t, np.sin(2 * np.pi * 1.0 * t), fs


@pytest.fixture
def noisy_sine(sine_wave):
    t, y_clean, fs = sine_wave
    noise = RNG.normal(0, 0.1, len(y_clean))
    return t, y_clean + noise, fs


@pytest.fixture
def linear_signal():
    x = np.linspace(0, 10, 500)
    y = 3 * x + 5
    return x, y


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


class TestSmoothMovingAverage:
    def test_output_shape(self, noisy_sine):
        _, y, _ = noisy_sine
        out = smooth_moving_average(y, 11)
        assert out.shape == y.shape

    def test_reduces_noise(self, noisy_sine, sine_wave):
        _, y_noisy, _ = noisy_sine
        _, y_clean, _ = sine_wave
        out = smooth_moving_average(y_noisy, 21)
        # Smoothed should be closer to clean than raw noisy
        assert np.std(out - y_clean) < np.std(y_noisy - y_clean)

    def test_window_1_is_identity(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(smooth_moving_average(y, 1), y)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            smooth_moving_average(np.ones(10), 0)


class TestSmoothSavitzkyGolay:
    def test_output_shape(self, noisy_sine):
        _, y, _ = noisy_sine
        out = smooth_savitzky_golay(y, 11, 3)
        assert out.shape == y.shape

    def test_preserves_polynomial(self):
        """SG filter is exact on polynomials up to poly_order."""
        x = np.linspace(0, 10, 200)
        y = 2 * x**2 - x + 1
        out = smooth_savitzky_golay(y, 11, 3)
        np.testing.assert_allclose(out, y, atol=1e-6)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            smooth_savitzky_golay(np.ones(20), 10, 3)

    def test_poly_ge_window_raises(self):
        with pytest.raises(ValueError):
            smooth_savitzky_golay(np.ones(20), 11, 11)


class TestSmoothGaussian:
    def test_output_shape(self, noisy_sine):
        _, y, _ = noisy_sine
        out = smooth_gaussian(y, sigma=3)
        assert out.shape == y.shape

    def test_reduces_noise(self, noisy_sine, sine_wave):
        _, y_noisy, _ = noisy_sine
        _, y_clean, _ = sine_wave
        out = smooth_gaussian(y_noisy, sigma=5)
        assert np.std(out - y_clean) < np.std(y_noisy - y_clean)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            smooth_gaussian(np.ones(10), 0)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFilters:
    """Butterworth filter tests (lowpass, highpass, bandpass, bandstop)."""

    def test_lowpass_attenuates_high_freq(self):
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        low = np.sin(2 * np.pi * 5 * t)  # 5 Hz — below cutoff
        high = np.sin(2 * np.pi * 200 * t)  # 200 Hz — above cutoff
        mixed = low + high
        out = filter_lowpass(mixed, cutoff=50, fs=fs)
        # High-frequency component should be strongly attenuated
        assert np.std(out - low) < 0.1

    def test_highpass_attenuates_low_freq(self):
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        low = np.sin(2 * np.pi * 2 * t)  # 2 Hz — below cutoff
        high = np.sin(2 * np.pi * 200 * t)  # 200 Hz — above cutoff
        mixed = low + high
        out = filter_highpass(mixed, cutoff=50, fs=fs)
        assert np.std(out - high) < 0.1

    def test_bandpass_output_shape(self, sine_wave):
        t, y, fs = sine_wave
        out = filter_bandpass(y, low=0.5, high=2.0, fs=fs)
        assert out.shape == y.shape

    def test_bandstop_output_shape(self, sine_wave):
        t, y, fs = sine_wave
        out = filter_bandstop(y, low=0.5, high=2.0, fs=fs)
        assert out.shape == y.shape

    def test_output_dtype_float(self, sine_wave):
        _, y, fs = sine_wave
        out = filter_lowpass(y, cutoff=10, fs=fs)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# FFT / power spectrum
# ---------------------------------------------------------------------------


class TestFFT:
    def test_keys_present(self, sine_wave):
        _, y, _ = sine_wave
        result = fft(y)
        for key in ("frequencies", "amplitudes", "power", "phases", "spectrum", "dt", "n"):
            assert key in result

    def test_dominant_frequency(self, sine_wave):
        t, y, fs = sine_wave
        result = fft(y, dt=1.0 / fs)
        dom_idx = np.argmax(result["amplitudes"][1:]) + 1
        dom_freq = result["frequencies"][dom_idx]
        assert abs(dom_freq - 1.0) < 0.1  # dominant ≈ 1 Hz

    def test_positive_frequencies_only(self, sine_wave):
        _, y, _ = sine_wave
        result = fft(y)
        assert np.all(result["frequencies"] >= 0)

    def test_one_sided_length(self, sine_wave):
        _, y, fs = sine_wave
        result = fft(y, dt=1.0 / fs)
        n = len(y)
        # fftfreq gives n//2 non-negative bins for even n
        assert len(result["frequencies"]) == n // 2

    def test_power_equals_amplitude_squared(self, sine_wave):
        _, y, _ = sine_wave
        result = fft(y)
        np.testing.assert_allclose(result["power"], result["amplitudes"] ** 2)

    def test_default_dt_one(self, sine_wave):
        _, y, _ = sine_wave
        result = fft(y)
        assert result["dt"] == 1.0


class TestIFFT:
    def test_roundtrip(self, sine_wave):
        _, y, _ = sine_wave
        spectrum = fft(y)["spectrum"]
        y_back = ifft(spectrum)
        np.testing.assert_allclose(y_back, y, atol=1e-10)


class TestPowerSpectrum:
    def test_keys(self, sine_wave):
        _, y, _ = sine_wave
        result = power_spectrum(y)
        assert set(result.keys()) == {"frequencies", "power"}

    def test_matches_fft_power(self, sine_wave):
        _, y, fs = sine_wave
        dt = 1.0 / fs
        ps = power_spectrum(y, dt=dt)
        full = fft(y, dt=dt)
        np.testing.assert_allclose(ps["power"], full["power"])


# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------


class TestDerivative:
    def test_first_derivative_of_linear(self, linear_signal):
        x, y = linear_signal  # y = 3x + 5, dy/dx = 3
        dy = derivative(y, x=x, order=1)
        # Central differences should recover slope ≈ 3 in the interior
        np.testing.assert_allclose(dy[5:-5], 3.0, atol=1e-4)

    def test_second_derivative_of_quadratic(self):
        x = np.linspace(0, 10, 1000)
        y = x**2  # d²y/dx² = 2
        dy2 = derivative(y, x=x, order=2)
        np.testing.assert_allclose(dy2[10:-10], 2.0, atol=0.01)

    def test_output_shape(self, linear_signal):
        x, y = linear_signal
        assert derivative(y, x=x).shape == y.shape

    def test_unit_spacing(self):
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2 with dx=1
        dy = derivative(y)
        # gradient at interior points: dy[2] ≈ (9-1)/2 = 4
        assert abs(dy[2] - 4.0) < 0.5

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            derivative(np.ones(10), order=3)


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------


class TestBaselinePolynomial:
    def test_flat_baseline_subtracted(self):
        x = np.linspace(0, 10, 300)
        baseline_true = 2 * x + 1
        peak = np.exp(-((x - 5) ** 2) / 0.5)
        y = peak + baseline_true
        bl, corr = baseline_polynomial(y, x, degree=1)
        # Corrected peak should be close to the true peak
        np.testing.assert_allclose(corr, peak, atol=0.5)

    def test_returns_two_arrays(self):
        x = np.linspace(0, 5, 100)
        y = x + np.ones(100)
        result = baseline_polynomial(y, x, degree=1)
        assert len(result) == 2
        assert result[0].shape == y.shape
        assert result[1].shape == y.shape

    def test_corrected_equals_y_minus_baseline(self):
        x = np.linspace(0, 5, 100)
        y = np.ones(100) * 3
        bl, corr = baseline_polynomial(y, x, degree=2)
        np.testing.assert_allclose(corr, y - bl)


class TestBaselineRollingBall:
    def test_returns_two_arrays(self):
        y = np.ones(200) + RNG.normal(0, 0.01, 200)
        bl, corr = baseline_rolling_ball(y, radius=20)
        assert bl.shape == y.shape
        assert corr.shape == y.shape

    def test_corrected_equals_y_minus_baseline(self):
        y = np.abs(np.sin(np.linspace(0, 4 * np.pi, 300))) + 1
        bl, corr = baseline_rolling_ball(y, radius=30)
        np.testing.assert_allclose(corr, y - bl)

    def test_baseline_below_signal(self):
        y = np.abs(np.sin(np.linspace(0, 4 * np.pi, 200))) + 1
        bl, _ = baseline_rolling_ball(y, radius=20)
        assert np.all(bl <= y + 1e-10)


class TestBaselineALS:
    def test_returns_two_arrays(self):
        y = np.ones(200)
        bl, corr = baseline_als(y, lam=1e4, p=0.05)
        assert bl.shape == y.shape
        assert corr.shape == y.shape

    def test_corrected_equals_y_minus_baseline(self):
        y = np.sin(np.linspace(0, 4 * np.pi, 200)) + 3
        bl, corr = baseline_als(y, lam=1e4, p=0.1)
        np.testing.assert_allclose(corr, y - bl, atol=1e-10)

    def test_baseline_smoothness(self):
        """ALS baseline should be smoother than the original signal."""
        x = np.linspace(0, 4 * np.pi, 300)
        y = np.sin(x) + np.sin(10 * x) * 0.3 + 2
        bl, _ = baseline_als(y, lam=1e6, p=0.05)
        # Baseline roughness (std of second differences) should be small
        roughness_y = np.std(np.diff(np.diff(y)))
        roughness_bl = np.std(np.diff(np.diff(bl)))
        assert roughness_bl < roughness_y


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


class TestInterpolate:
    def test_cubic_recovers_smooth_function(self):
        x = np.linspace(0, 2 * np.pi, 30)
        y = np.sin(x)
        x_new = np.linspace(0, 2 * np.pi, 200)
        y_new = interpolate(x, y, x_new, method="cubic")
        np.testing.assert_allclose(y_new, np.sin(x_new), atol=1e-3)

    def test_linear_interpolation(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        x_new = np.array([0.5, 1.5, 2.5])
        y_new = interpolate(x, y, x_new, method="linear")
        np.testing.assert_allclose(y_new, [1.0, 3.0, 5.0])

    def test_output_shape(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        x_new = np.linspace(0, 10, 200)
        out = interpolate(x, y, x_new)
        assert out.shape == (200,)

    def test_nearest_method(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        y_new = interpolate(x, y, np.array([0.4, 0.6]), method="nearest")
        # 0.4 → nearest is 0 → 10; 0.6 → nearest is 1 → 20
        assert y_new[0] == pytest.approx(10.0, abs=1e-6)
        assert y_new[1] == pytest.approx(20.0, abs=1e-6)
