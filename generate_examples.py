"""Generate example datasets for every plot type in the Plottle app.

Run once from the example-data/Artificial/ directory (or anywhere — paths are
resolved relative to this file):

    python generate_examples.py

All output files are written to the same directory as this script.
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent / "plottle" / "example-data" / "Artificial"
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Normal distribution (Histogram, Distribution, Interactive Histogram) ──
n = 400
normal = pd.DataFrame(
    {
        "temperature_K": rng.normal(loc=298.15, scale=12.0, size=n),
    }
)
normal.to_csv(OUT / "normal_distribution.csv", index=False)
print("Wrote normal_distribution.csv")


# ── 2. Bimodal distribution (Distribution / overlaid distributions) ───────────
bimodal = pd.DataFrame(
    {
        "absorbance": np.concatenate(
            [
                rng.normal(loc=0.35, scale=0.07, size=250),
                rng.normal(loc=0.75, scale=0.06, size=250),
            ]
        )
    }
)
bimodal.to_csv(OUT / "bimodal_distribution.csv", index=False)
print("Wrote bimodal_distribution.csv")


# ── 3. Sine / cosine waves (Line Plot, Interactive Line) ──────────────────────
x = np.linspace(0, 4 * np.pi, 300)
waves = pd.DataFrame(
    {
        "angle_rad": x,
        "sin_wave": np.sin(x),
        "cos_wave": np.cos(x),
        "damped_sin": np.exp(-x / 8) * np.sin(x),
    }
)
waves.to_csv(OUT / "sine_cosine_waves.csv", index=False)
print("Wrote sine_cosine_waves.csv")


# ── 4. Correlated scatter (Scatter Plot, Regression, Interactive Scatter) ─────
x_scat = rng.uniform(0, 10, 200)
scatter = pd.DataFrame(
    {
        "reaction_time_s": x_scat,
        "product_yield": 0.72 * x_scat + rng.normal(0, 1.0, 200),
        "temperature_C": 25 + 3 * x_scat + rng.normal(0, 2.0, 200),
    }
)
scatter.to_csv(OUT / "scatter_correlation.csv", index=False)
print("Wrote scatter_correlation.csv")


# ── 5. Grouped categorical (Box / Violin / Swarm) ────────────────────────────
groups = []
for catalyst in ["Pd", "Pt", "Rh", "Ru"]:
    for solvent in ["MeOH", "EtOH", "DCM"]:
        yields = rng.normal(
            loc={"Pd": 82, "Pt": 75, "Rh": 88, "Ru": 71}[catalyst],
            scale=5.0,
            size=20,
        )
        for y in yields:
            groups.append({"catalyst": catalyst, "solvent": solvent, "yield_pct": round(y, 2)})
groups_df = pd.DataFrame(groups)
groups_df.to_csv(OUT / "grouped_categorical.csv", index=False)
print("Wrote grouped_categorical.csv")


# ── 6. Multi-column chemistry (Advanced / Correlation Heatmap) ───────────────
n_mol = 120
mol_props = pd.DataFrame(
    {
        "MW": rng.uniform(50, 500, n_mol),
        "logP": rng.uniform(-2, 6, n_mol),
        "TPSA": rng.uniform(0, 140, n_mol),
        "HBD": rng.integers(0, 6, n_mol).astype(float),
        "HBA": rng.integers(0, 11, n_mol).astype(float),
        "RotBonds": rng.integers(0, 15, n_mol).astype(float),
    }
)
# Add correlated activity column
mol_props["pIC50"] = (
    0.012 * mol_props["MW"]
    - 0.3 * mol_props["TPSA"]
    + 0.5 * mol_props["logP"]
    + rng.normal(0, 0.8, n_mol)
)
mol_props = mol_props.round(3)
mol_props.to_csv(OUT / "molecular_properties.csv", index=False)
print("Wrote molecular_properties.csv")


# ── 7. IR spectrum (Line Plot / Scatter) ─────────────────────────────────────
wavenumbers = np.arange(4000, 399, -2)
ir_spec = pd.DataFrame(
    {
        "wavenumber_cm-1": wavenumbers,
        "transmittance": (
            0.85
            - 0.6 * np.exp(-(((wavenumbers - 2920) / 30) ** 2))  # C-H stretch
            - 0.55 * np.exp(-(((wavenumbers - 1715) / 20) ** 2))  # C=O stretch
            - 0.35 * np.exp(-(((wavenumbers - 3350) / 80) ** 2))  # O-H stretch
            - 0.25 * np.exp(-(((wavenumbers - 1250) / 30) ** 2))  # C-O stretch
            + rng.normal(0, 0.015, len(wavenumbers))
        ).clip(0, 1),
    }
)
ir_spec = ir_spec.round(4)
ir_spec.to_csv(OUT / "ir_spectrum.csv", index=False)
print("Wrote ir_spectrum.csv")


# ── 8. 2D Gaussian surface (Contour Plot, 3D Surface, Heatmap, Interactive) ──
grid_size = 60
xi = np.linspace(-3, 3, grid_size)
yi = np.linspace(-3, 3, grid_size)
Xi, Yi = np.meshgrid(xi, yi)
Z_gauss = np.exp(-0.5 * ((Xi - 0.5) ** 2 + (Yi - 0.5) ** 2)) + 0.6 * np.exp(
    -0.5 * ((Xi + 1.0) ** 2 + (Yi + 1.0) ** 2)
)
np.save(OUT / "gaussian_surface.npy", Z_gauss)
print("Wrote gaussian_surface.npy  (shape:", Z_gauss.shape, ")")


# ── 9. Correlation matrix-like heatmap (10 × 10) ─────────────────────────────
A = rng.standard_normal((50, 10))
corr_mat = np.corrcoef(A.T)  # exact correlation matrix [-1, 1]
np.save(OUT / "correlation_matrix.npy", corr_mat)
print("Wrote correlation_matrix.npy  (shape:", corr_mat.shape, ")")


# ── 10. Time-series kinetics (Line Plot / Interactive Line) ──────────────────
t = np.linspace(0, 60, 120)  # minutes
k1, k2, k3 = 0.08, 0.04, 0.02
kinetics = pd.DataFrame(
    {
        "time_min": t,
        "reactant_A": np.exp(-k1 * t),
        "intermediate_B": (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)),
        "product_C": 1 - np.exp(-k3 * t),
    }
)
kinetics = kinetics.round(5)
kinetics.to_csv(OUT / "reaction_kinetics.csv", index=False)
print("Wrote reaction_kinetics.csv")


print("\nAll example datasets generated successfully.")
