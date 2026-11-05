"""UV-Vis spectroscopy analysis workflow.

Demonstrates loading multiple spectra, normalizing, computing peak statistics,
and producing a comparison figure.

Run from the repo root:
    python examples/spectroscopy_workflow.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.math import calculate_statistics
from eagleplot.plotting import line_plot, apply_publication_style, save_figure, get_color_palette

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"


def simulate_spectrum(center, width, amplitude, noise_std=0.005):
    """Return (wavelength, absorbance) for a Gaussian absorption band."""
    np.random.seed(int(center))
    wavelength = np.linspace(350, 800, 451)
    absorbance = amplitude * np.exp(-((wavelength - center) ** 2) / (2 * width**2))
    absorbance += np.random.normal(0, noise_std, len(wavelength))
    return wavelength, np.clip(absorbance, 0, None)


def normalize_spectrum(absorbance):
    """Min-max normalize to [0, 1]."""
    mn, mx = absorbance.min(), absorbance.max()
    return (absorbance - mn) / (mx - mn) if mx > mn else absorbance


def find_lambda_max(wavelength, absorbance):
    """Return the wavelength of maximum absorbance."""
    return wavelength[np.argmax(absorbance)]


def run_spectroscopy_workflow():
    """Full UV-Vis comparison and analysis workflow."""
    print("\n" + "=" * 70)
    print("UV-Vis Spectroscopy Workflow")
    print("=" * 70)

    samples = {
        "Rhodamine B": (550, 30, 0.95),
        "Fluorescein": (490, 25, 0.80),
        "Methylene Blue": (665, 35, 0.70),
    }

    colors = get_color_palette("Color-Blind Safe (Wong)", n_colors=len(samples))

    # 1. Load (simulate) spectra
    spectra = {}
    for name, (center, width, amp) in samples.items():
        wl, ab = simulate_spectrum(center, width, amp)
        spectra[name] = (wl, ab)

    print(f"\nStep 1 — Loaded {len(spectra)} spectra")
    print(f"  {CHECK} Spectra loaded")

    # 2. Peak analysis
    print("\nStep 2 — Peak Analysis:")
    for name, (wl, ab) in spectra.items():
        lmax = find_lambda_max(wl, ab)
        stats = calculate_statistics(ab)
        print(f"  {name}:")
        print(f"    lambda_max = {lmax:.0f} nm")
        print(f"    A_max      = {ab.max():.4f}")
        print(f"    A_mean     = {stats['mean']:.4f}")
    print(f"  {CHECK} Peak analysis complete")

    # 3. Raw spectra comparison
    wl_ref = spectra["Rhodamine B"][0]
    raw_ys = [spectra[n][1] for n in samples]
    raw_labels = list(samples.keys())

    fig, ax = line_plot(
        wl_ref,
        raw_ys,
        labels=raw_labels,
        colors=colors,
        xlabel="Wavelength (nm)",
        ylabel="Absorbance",
        title="UV-Vis Absorption Spectra",
    )
    apply_publication_style(fig, ax)

    out = OUTPUT_DIR / "spectra_comparison.png"
    save_figure(fig, out, dpi=300)
    print(f"\nStep 3 — Raw spectra saved: {out}")

    # 4. Normalized comparison
    norm_ys = [normalize_spectrum(spectra[n][1]) for n in samples]

    fig2, ax2 = line_plot(
        wl_ref,
        norm_ys,
        labels=raw_labels,
        colors=colors,
        xlabel="Wavelength (nm)",
        ylabel="Normalized Absorbance",
        title="Normalized UV-Vis Spectra",
    )
    apply_publication_style(fig2, ax2)

    out2 = OUTPUT_DIR / "spectra_normalized.png"
    save_figure(fig2, out2, dpi=300)
    print(f"Step 4 — Normalized spectra saved: {out2}")
    print(f"  {CHECK} Workflow complete")


if __name__ == "__main__":
    print("EaglePlot — UV-Vis Spectroscopy Workflow")
    run_spectroscopy_workflow()
