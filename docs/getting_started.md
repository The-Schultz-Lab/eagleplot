# Getting Started — Plottle

NCCU Department of Chemistry and Biochemistry

---

## What Is This Tool?

Plottle is a Python toolkit plus Streamlit GUI that makes scientific data visualization straightforward for computational chemists and physical scientists. It wraps Matplotlib, Seaborn, and Plotly behind a clean API and an interactive point-and-click interface.

You can use it two ways:

| Mode | Best for |
| --- | --- |
| **Streamlit GUI** (point-and-click) | Exploring data quickly; no coding required |
| **Python API** (scripts / notebooks) | Reproducible analysis; programmatic control |

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/The-Schultz-Lab/plottle.git
cd plottle
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
```

### Step 3 — Activate the environment

#### Windows

```bat
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### Step 4 — Install Plottle

```bash
pip install -e ".[formats,nist]"
```

The `formats` extra adds HDF5 and NetCDF support; `nist` adds the NIST WebBook
lookup. The editable install is also what puts the `plottle` command on your PATH.

### Step 5 — Verify

```bash
python -c "import streamlit, numpy, pandas, matplotlib, seaborn, plotly; print('Ready!')"
```

---

## Launching the GUI

### Windows — double-click `launch.bat` or run from a terminal

```bat
launch.bat
```

### Any platform

```bash
streamlit run plottle/Home.py
```

Open `http://localhost:8501` in your browser. Use the sidebar to navigate between pages.

---

## Your First Plot (GUI)

1. **Data Upload** — upload a CSV or drag-and-drop a `.npy` file.
2. **Quick Plot** — select your dataset, choose "Line Plot", pick X and Y columns.
3. Click **Generate Plot** — your figure appears instantly.
4. Expand **Export plot & data** to download PNG, SVG, PDF, or CSV.

---

## Your First Plot (Python API)

```python
import numpy as np
import sys

sys.path.insert(0, "path/to/plottle")

from plottle.plotting import line_plot, save_figure

# Simulate absorbance vs. wavelength
wavelength = np.linspace(400, 800, 200)
absorbance = 0.8 * np.exp(-((wavelength - 520) ** 2) / (2 * 30**2))

fig, ax = line_plot(
    wavelength,
    [absorbance],
    xlabel="Wavelength (nm)",
    ylabel="Absorbance",
    title="UV-Vis Spectrum",
    labels=["Sample A"],
)

save_figure(fig, "spectrum.png", dpi=300)
```

---

## Loading Data

The `plottle.io` module supports eight common formats.

```python
from plottle.io import load_data, save_data

# Auto-detect format from extension
df = load_data("experiment.csv")  # → pandas DataFrame
arr = load_data("matrix.npy")  # → numpy array
obj = load_data("session.pkl")  # → any Python object
```

Supported extensions:

| Extension | Format | Returns |
| --- | --- | --- |
| `.csv` | Comma-separated values | `pandas.DataFrame` |
| `.xlsx` | Excel workbook | `pandas.DataFrame` |
| `.tsv` | Tab-separated values | `pandas.DataFrame` |
| `.json` | JSON | `pandas.DataFrame` |
| `.npy` | NumPy binary | `numpy.ndarray` |
| `.npz` | NumPy compressed | `numpy.ndarray` |
| `.pkl` | Python pickle | `Any` |
| `.parquet` | Parquet | `pandas.DataFrame` |

---

## Computing Statistics

```python
from plottle.math import calculate_statistics

stats = calculate_statistics(arr)
print(f"Mean: {stats['mean']:.4f}")
print(f"Std:  {stats['std']:.4f}")
print(f"R:    [{stats['min']:.3f}, {stats['max']:.3f}]")
```

Returns: `mean`, `median`, `std`, `var`, `min`, `max`, `q1`, `q3`, `iqr`, `range`.

---

## Curve Fitting

```python
from plottle.math import fit_linear, fit_polynomial

# Beer-Lambert: A = ε·c·l  →  linear fit
result = fit_linear(concentration, absorbance)
print(f"Slope (ε·l):  {result['slope']:.4f}")
print(f"R²:           {result['r_squared']:.4f}")

# Polynomial fit (degree 2)
poly = fit_polynomial(x, y, degree=2)
y_fit = poly["predict"](x)  # call the returned predict function
```

---

## All Plot Types

| Function | Library | Returns |
| --- | --- | --- |
| `histogram(data)` | Matplotlib | `fig, ax, info` |
| `line_plot(x, y)` | Matplotlib | `fig, ax` |
| `scatter_plot(x, y)` | Matplotlib | `fig, ax` |
| `heatmap(matrix)` | Matplotlib | `fig, ax` |
| `contour_plot(X, Y, Z)` | Matplotlib | `fig, ax` |
| `distribution_plot(data)` | Seaborn | `fig, ax` |
| `box_plot(data)` | Seaborn | `fig, ax` |
| `regression_plot(x, y)` | Seaborn | `fig, ax` |
| `interactive_histogram(data)` | Plotly | `plotly.Figure` |
| `interactive_scatter(x, y)` | Plotly | `plotly.Figure` |
| `interactive_line(x, y)` | Plotly | `plotly.Figure` |
| `interactive_heatmap(matrix)` | Plotly | `plotly.Figure` |
| `interactive_3d_surface(X, Y, Z)` | Plotly | `plotly.Figure` |

---

## Saving Figures

```python
from plottle.plotting import save_figure

save_figure(fig, "plot.png", dpi=300)  # high-res raster
save_figure(fig, "plot.svg")  # vector (no dpi needed)
save_figure(fig, "plot.pdf")  # vector, publication-ready
```

---

## GUI Pages at a Glance

| Page | Purpose |
| --- | --- |
| **Home** | Dashboard overview and help tabs |
| **Data Upload** | Load files in 18 formats; preview shape, types, and statistics; batch folder import |
| **Plot → Basic** | 27 plot types with live style controls and an annotation panel |
| **Plot → Multiplot** | Up to 4×4 grid layouts with axis sharing and combined export |
| **Plot → Advanced Plotting** | Seaborn statistical plots and Plotly interactive charts |
| **Plot → Spectroscopy** | IR/Raman, NMR, UV-Vis, Mass Spec; NIST WebBook lookup by CAS |
| **Plot → Molecular Viz** | Gaussian/ORCA/Molden output; 3D structure and vibrational modes |
| **Analyze → Single** | Statistics, distributions, curve fitting, signal processing, peaks |
| **Analyze → Batch** | Batch statistics, curve fitting, and peak analysis with presets |
| **Analyze → Data Tools** | 12 non-destructive DataFrame operations |
| **Export** | Export plots, data, and analyses; save/load sessions; PDF reports |
| **Gallery** | Pre-rendered examples with "Use this config" buttons |
| **Help** | Getting started, plot types, analysis tools, formats, tips |
| **Settings** | Theme, plot defaults, and named preset management |

---

## Common Workflows

### Beer-Lambert calibration curve

```python
from plottle.io import load_dataframe
from plottle.math import fit_linear
from plottle.plotting import scatter_plot, save_figure
import numpy as np

df = load_dataframe("calibration.csv")  # columns: concentration, absorbance
result = fit_linear(df["concentration"].values, df["absorbance"].values)

x_fit = np.linspace(df["concentration"].min(), df["concentration"].max(), 100)
y_fit = result["slope"] * x_fit + result["intercept"]

fig, ax = scatter_plot(
    df["concentration"].values,
    df["absorbance"].values,
    xlabel="Concentration (mM)",
    ylabel="Absorbance (a.u.)",
    title=f"Beer-Lambert: R² = {result['r_squared']:.4f}",
)
ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Linear fit")
ax.legend()
save_figure(fig, "calibration.png", dpi=300)
```

### Comparing multiple spectra

```python
from plottle.plotting import line_plot

fig, ax = line_plot(
    wavelength,
    [spectrum_A, spectrum_B, spectrum_C],
    labels=["Sample A", "Sample B", "Sample C"],
    xlabel="Wavelength (nm)",
    ylabel="Absorbance",
    title="UV-Vis Comparison",
)
```

---

## Next Steps

- Read the [GUI Guide](tutorials/gui_guide.md) for a tour of every page.
- Read the [CLI Guide](tutorials/cli_guide.md) for scripted and batch workflows.
- Keep the [Cheatsheet](cheatsheet.md) handy as a quick reference to the Python API.
- Found a problem? See [Reporting Bugs](bug-reports.md).
- Want something added? See [Feature Requests](feature_requests.md).
