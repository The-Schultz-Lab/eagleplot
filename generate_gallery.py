"""Gallery Generator — EaglePlot.

Run this script once from the repo root to produce pre-rendered PNG thumbnails
and a ``eagleplot/gallery/manifest.json`` file that the Gallery page reads.

Usage
-----
    python generate_gallery.py

Output
------
eagleplot/gallery/
    manifest.json           — metadata for all generated figures
    histogram.png
    line_plot.png
    ... (one file per gallery entry)

The script can be re-run at any time to regenerate all figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_EXAMPLE_DIR = _ROOT / "eagleplot" / "example-data" / "Artificial"
_GALLERY_DIR = _ROOT / "eagleplot" / "gallery"
_GALLERY_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))

from eagleplot.plotting import (  # noqa: E402
    histogram,
    line_plot,
    scatter_plot,
    heatmap,
    contour_plot,
    distribution_plot,
    box_plot,
    regression_plot,
    interactive_histogram,
    interactive_scatter,
    interactive_line,
    interactive_heatmap,
    interactive_3d_surface,
)

# ── Shared style ─────────────────────────────────────────────────────────────
# dpi is applied at save time via _save_mpl, not passed to plot functions.
_STYLE = dict(figsize=(5, 3.5))

# ── Gallery entries ──────────────────────────────────────────────────────────
# Each entry: (key, title, description, best_for, render_fn)
# render_fn() must save a PNG to _GALLERY_DIR/<key>.png and return the config
# dict that was used (so it can be stored in manifest.json).


def _save_mpl(fig, key: str) -> None:
    fig.savefig(_GALLERY_DIR / f"{key}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_plotly(fig, key: str) -> None:
    try:
        fig.write_image(str(_GALLERY_DIR / f"{key}.png"), width=550, height=385)
    except Exception:
        # kaleido not installed — write a placeholder
        _placeholder(key)


def _placeholder(key: str) -> None:
    """Write a simple placeholder PNG when a renderer is unavailable."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.text(
        0.5,
        0.5,
        f"[{key}]\n(interactive — open in browser)",
        ha="center",
        va="center",
        fontsize=10,
        color="#888",
        transform=ax.transAxes,
    )
    ax.axis("off")
    fig.savefig(_GALLERY_DIR / f"{key}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── Loaders ───────────────────────────────────────────────────────────────────


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_EXAMPLE_DIR / name)


def _npy(name: str) -> np.ndarray:
    return np.load(_EXAMPLE_DIR / name)


# ── Render functions ─────────────────────────────────────────────────────────


def _render_histogram() -> dict:
    df = _csv("normal_distribution.csv")
    cfg = dict(
        title="Temperature Distribution",
        xlabel="Temperature (K)",
        ylabel="Count",
        bins=30,
        color="#1f77b4",
        **_STYLE,
    )
    fig, ax, _ = histogram(df["temperature_K"].values, **cfg)
    _save_mpl(fig, "histogram")
    return cfg


def _render_line_plot() -> dict:
    df = _csv("sine_cosine_waves.csv")
    cfg = dict(title="Sine & Cosine Waves", xlabel="x (rad)", ylabel="Amplitude", **_STYLE)
    x = df["angle_rad"].values
    y = [df["sin_wave"].values, df["cos_wave"].values]
    fig, ax = line_plot(x, y, labels=["sin(x)", "cos(x)"], **cfg)
    _save_mpl(fig, "line_plot")
    return cfg


def _render_scatter_plot() -> dict:
    df = _csv("scatter_correlation.csv")
    cfg = dict(
        title="Reaction Time vs Yield",
        xlabel="Reaction Time (s)",
        ylabel="Yield (%)",
        alpha=0.7,
        **_STYLE,
    )
    fig, ax = scatter_plot(df["reaction_time_s"].values, df["product_yield"].values, **cfg)
    _save_mpl(fig, "scatter_plot")
    return cfg


def _render_heatmap() -> dict:
    mat = _npy("correlation_matrix.npy")
    cfg = dict(title="Correlation Matrix", cmap="coolwarm", vmin=-1, vmax=1, **_STYLE)
    fig, ax = heatmap(mat, **cfg)
    _save_mpl(fig, "heatmap")
    return cfg


def _render_contour_plot() -> dict:
    Z = _npy("gaussian_surface.npy")
    nrows, ncols = Z.shape
    X, Y = np.meshgrid(np.linspace(-3, 3, ncols), np.linspace(-3, 3, nrows))
    cfg = dict(
        title="Gaussian Potential Energy Surface",
        xlabel="x",
        ylabel="y",
        levels=20,
        cmap="viridis",
        filled=True,
        colorbar=True,
        **_STYLE,
    )
    fig, ax = contour_plot(X, Y, Z, **cfg)
    _save_mpl(fig, "contour_plot")
    return cfg


def _render_distribution_plot() -> dict:
    df = _csv("bimodal_distribution.csv")
    cfg = dict(
        title="Bimodal Absorbance Distribution",
        xlabel="Absorbance (a.u.)",
        kind="kde",
        fill=True,
        **_STYLE,
    )
    fig, ax = distribution_plot(df.iloc[:, 0].values, **cfg)
    _save_mpl(fig, "distribution_plot")
    return cfg


def _render_box_plot() -> dict:
    df = _csv("grouped_categorical.csv")
    cfg = dict(
        title="Yield by Catalyst",
        xlabel="Catalyst",
        ylabel="Yield (%)",
        x="catalyst",
        y="yield_pct",
        kind="box",
        **_STYLE,
    )
    fig, ax = box_plot(df, **cfg)
    _save_mpl(fig, "box_plot")
    return cfg


def _render_regression_plot() -> dict:
    df = _csv("scatter_correlation.csv")
    cfg = dict(
        title="Yield vs Temperature (Regression)", xlabel="Temperature (°C)", ylabel="Yield (%)"
    )
    fig, ax = regression_plot(x="temperature_C", y="product_yield", data=df, order=1, ci=95, **cfg)
    _save_mpl(fig, "regression_plot")
    return cfg


def _render_interactive_histogram() -> dict:
    df = _csv("normal_distribution.csv")
    cfg = dict(title="Interactive Temperature Histogram", xlabel="Temperature (K)", ylabel="Count")
    fig = interactive_histogram(df["temperature_K"].values, bins=30, **cfg)
    _save_plotly(fig, "interactive_histogram")
    return cfg


def _render_interactive_scatter() -> dict:
    df = _csv("scatter_correlation.csv")
    cfg = dict(title="Interactive Scatter", xlabel="Reaction Time (s)", ylabel="Yield (%)")
    fig = interactive_scatter(df["reaction_time_s"].values, df["product_yield"].values, **cfg)
    _save_plotly(fig, "interactive_scatter")
    return cfg


def _render_interactive_line() -> dict:
    df = _csv("sine_cosine_waves.csv")
    cfg = dict(title="Interactive Sine & Cosine", xlabel="x (rad)", ylabel="Amplitude")
    x = df["angle_rad"].values
    y = [df["sin_wave"].values, df["cos_wave"].values]
    fig = interactive_line(x, y, labels=["sin(x)", "cos(x)"], **cfg)
    _save_plotly(fig, "interactive_line")
    return cfg


def _render_interactive_heatmap() -> dict:
    mat = _npy("correlation_matrix.npy")
    cfg = dict(title="Interactive Correlation Matrix", colorscale="RdBu", zmin=-1, zmax=1)
    fig = interactive_heatmap(mat, **cfg)
    _save_plotly(fig, "interactive_heatmap")
    return cfg


def _render_interactive_3d_surface() -> dict:
    Z = _npy("gaussian_surface.npy")
    nrows, ncols = Z.shape
    X, Y = np.meshgrid(np.linspace(-3, 3, ncols), np.linspace(-3, 3, nrows))
    cfg = dict(title="3D Gaussian Surface", colorscale="Viridis")
    fig = interactive_3d_surface(X, Y, Z, **cfg)
    _save_plotly(fig, "interactive_3d_surface")
    return cfg


# ── Master table ─────────────────────────────────────────────────────────────

_ENTRIES = [
    {
        "key": "histogram",
        "title": "Histogram",
        "description": "Distribution of temperature measurements with 30 bins.",
        "best_for": "Single-variable distributions",
        "dataset": "normal_distribution.csv",
        "library": "Matplotlib",
        "render": _render_histogram,
    },
    {
        "key": "line_plot",
        "title": "Line Plot",
        "description": "Sine and cosine waves over 0–4π with per-series colour.",
        "best_for": "Time series, ordered data, multi-trace",
        "dataset": "sine_cosine_waves.csv",
        "library": "Matplotlib",
        "render": _render_line_plot,
    },
    {
        "key": "scatter_plot",
        "title": "Scatter Plot",
        "description": "Reaction time vs product yield showing a positive correlation.",
        "best_for": "Correlation, two-variable comparison",
        "dataset": "scatter_correlation.csv",
        "library": "Matplotlib",
        "render": _render_scatter_plot,
    },
    {
        "key": "heatmap",
        "title": "Heatmap",
        "description": "10×10 correlation matrix with colour encoding and cell annotations.",
        "best_for": "2-D matrices, correlation tables",
        "dataset": "correlation_matrix.npy",
        "library": "Matplotlib",
        "render": _render_heatmap,
    },
    {
        "key": "contour_plot",
        "title": "Contour Plot",
        "description": "Filled contours of a double-Gaussian potential energy surface.",
        "best_for": "Scalar fields, PES, 2-D functions",
        "dataset": "gaussian_surface.npy",
        "library": "Matplotlib",
        "render": _render_contour_plot,
    },
    {
        "key": "distribution_plot",
        "title": "Distribution (KDE)",
        "description": "Kernel density estimate of a bimodal absorbance distribution.",
        "best_for": "Overlaid distributions, KDE, ECDF",
        "dataset": "bimodal_distribution.csv",
        "library": "Seaborn",
        "render": _render_distribution_plot,
    },
    {
        "key": "box_plot",
        "title": "Box / Violin Plot",
        "description": "Yield distributions grouped by catalyst type.",
        "best_for": "Group comparisons, outlier detection",
        "dataset": "grouped_categorical.csv",
        "library": "Seaborn",
        "render": _render_box_plot,
    },
    {
        "key": "regression_plot",
        "title": "Regression Plot",
        "description": "Linear regression of yield vs temperature with 95% confidence band.",
        "best_for": "Trend lines, linear modelling",
        "dataset": "scatter_correlation.csv",
        "library": "Seaborn",
        "render": _render_regression_plot,
    },
    {
        "key": "interactive_histogram",
        "title": "Interactive Histogram",
        "description": "Zoomable histogram with hover counts.",
        "best_for": "Exploratory data analysis",
        "dataset": "normal_distribution.csv",
        "library": "Plotly",
        "render": _render_interactive_histogram,
    },
    {
        "key": "interactive_scatter",
        "title": "Interactive Scatter",
        "description": "Click-to-annotate scatter with hover coordinates.",
        "best_for": "Interactive correlation exploration",
        "dataset": "scatter_correlation.csv",
        "library": "Plotly",
        "render": _render_interactive_scatter,
    },
    {
        "key": "interactive_line",
        "title": "Interactive Line",
        "description": "Multi-trace interactive line chart with legend toggle.",
        "best_for": "Interactive time-series comparison",
        "dataset": "sine_cosine_waves.csv",
        "library": "Plotly",
        "render": _render_interactive_line,
    },
    {
        "key": "interactive_heatmap",
        "title": "Interactive Heatmap",
        "description": "Hover-enabled correlation matrix.",
        "best_for": "Interactive matrix inspection",
        "dataset": "correlation_matrix.npy",
        "library": "Plotly",
        "render": _render_interactive_heatmap,
    },
    {
        "key": "interactive_3d_surface",
        "title": "3D Surface",
        "description": "Rotatable 3-D potential energy surface.",
        "best_for": "3-D data exploration",
        "dataset": "gaussian_surface.npy",
        "library": "Plotly",
        "render": _render_interactive_3d_surface,
    },
]

# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    manifest = []
    total = len(_ENTRIES)
    for i, entry in enumerate(_ENTRIES, 1):
        key = entry["key"]
        print(f"[{i}/{total}] Rendering {key}...", end=" ", flush=True)
        try:
            cfg = entry["render"]()
            record = {
                "key": key,
                "title": entry["title"],
                "description": entry["description"],
                "best_for": entry["best_for"],
                "dataset": entry["dataset"],
                "library": entry["library"],
                "filename": f"{key}.png",
                "config": {k: str(v) for k, v in cfg.items() if not k.startswith("fig")},
            }
            manifest.append(record)
            print("OK")
        except Exception as exc:
            print(f"FAILED — {exc}")

    manifest_path = _GALLERY_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nDone. {len(manifest)}/{total} figures written to {_GALLERY_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
