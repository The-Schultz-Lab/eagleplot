# EaglePlot

[![Tests](https://github.com/The-Schultz-Lab/eagleplot/actions/workflows/tests.yml/badge.svg)](https://github.com/The-Schultz-Lab/eagleplot/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/eagleplot)](https://pypi.org/project/eagleplot/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/The-Schultz-Lab/eagleplot/main)

A general-purpose Python toolkit for scientific data visualization and analysis, developed for research and teaching in computational science at North Carolina Central University.

> **EaglePlot was called Plottle until version 3.0.** If you used Plottle, change
> `import plottle` / `from plottle.… import …` to `eagleplot`, and run the `eagleplot`
> command instead of `plottle`. Session files saved by Plottle still load.

## Overview

EaglePlot provides a unified interface for scientific data work:

- **Data I/O** — 18 file formats (CSV, Excel, TSV, JSON, Parquet, NumPy, Pickle, JCAMP-DX, HDF5, NetCDF, SPC, ASC, mzML/mzXML, and more)
- **Mathematical analysis** — statistics, curve fitting, signal processing, peak analysis, hypothesis testing, optimization, linear algebra
- **Multi-library plotting** — 27 plot types across Matplotlib (static), Seaborn (statistical), and Plotly (interactive)
- **14-page Streamlit GUI** — exploratory data analysis without writing code
- **CLI** — batch processing and scripted workflows
- **Plugin system** — drop `plugin_*.py` into `plugins/` for custom plot types and tools

## Quick Start

### Clone the repository

#### 1 — Clone and enter the repository

```bash
git clone https://github.com/The-Schultz-Lab/eagleplot.git
cd eagleplot
```

#### 2 — Set up and launch (double-click, no terminal needed)

**Windows:**
1. Double-click **`setup.bat`** — creates the virtual environment and installs all dependencies.
2. Double-click **`launch.bat`** — starts EaglePlot. Use this every time you want to open the app.

**macOS:**
1. Double-click **`setup.command`** — creates the virtual environment and installs all dependencies.
   *(If macOS asks "Are you sure you want to open it?", click Open.)*
2. Double-click **`launch.command`** — starts EaglePlot. Use this every time you want to open the app.

**Linux:**
1. Run **`./setup.sh`** in a terminal — creates the virtual environment and installs all dependencies.
   *(On Debian/Ubuntu you may first need `sudo apt install python3-venv`.)*
2. Run **`./launch.sh`** — starts EaglePlot.

After the app starts, open **`http://localhost:8501`** in your browser (Streamlit usually opens it automatically).

#### 2 (alternative) — Set up and launch from the terminal

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install EaglePlot and its dependencies. The editable install is what puts the
# `eagleplot` command on your PATH.
pip install -e ".[formats,nist]"

streamlit run eagleplot/Home.py
```

#### 3 — Or use the CLI

```bash
eagleplot --help
eagleplot plot data.csv --plot scatter --x-column x --y-column y
eagleplot stats data.csv
```

The `eagleplot` command comes from the `pip install` above. From a source checkout
without installing, use `python -m eagleplot.cli` instead.

## GUI Pages

| Page | Description |
| --- | --- |
| **Home** | Dashboard overview and help tabs |
| **1 — Data Upload** | Upload files in 18 formats; preview shape, column types, and summary statistics; batch folder import |
| **2 — Quick Plot** | 27 plot types with live style controls, annotation panel, and Convert to Plotly toggle |
| **3 — Analysis Tools** | 8 tabs: Statistics, Distribution, Curve Fit, Optimization, Linear Algebra, Signal Processing, Peak Analysis, Statistical Tests |
| **4 — Multi-Plot Dashboard** | Up to 4×4 grid layouts with axis sharing and combined PNG/PDF export |
| **5 — Advanced Plotting** | Seaborn statistical plots and Plotly interactive charts with HTML export |
| **7 — Export Results** | Export plots, data, and analyses; save/load full sessions; generate PDF reports |
| **8 — Gallery** | Pre-generated figure gallery with Load this config buttons |
| **9 — Data Tools** | 10 operation tabs: formula columns, normalization, transpose, pivot/melt, filter, sort, merge, fill/drop NaN, resample, rolling transforms |
| **10 — Spectroscopy** | IR/Raman, NMR, UV-Vis, Mass Spec tools; NIST WebBook spectral lookup by CAS number |
| **11 — Molecular Viz** | Upload Gaussian/ORCA/Molden output files; 3D molecular structure and vibrational mode display |
| **12 — Batch Analysis** | Batch statistics, curve fitting, and peak analysis with workflow presets |
| **13 — Help** | Getting Started, Plot Types, Analysis Tools, Supported Formats, Tips |
| **14 — Settings** | Theme, DPI, plot defaults, named preset CRUD, plugin status |

## Supported File Formats (18)

| Category | Formats |
| --- | --- |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`, `.parquet` |
| Array | `.npy`, `.npz`, `.pkl` |
| Instrument / Spectral | `.jdx` / `.dx` (JCAMP-DX), `.h5` / `.hdf5` (HDF5), `.nc` / `.cdf` (NetCDF), `.spc` (Thermo Fisher), `.asc` (generic text), `.mzml` / `.mzxml` (mass spectrometry, optional) |

## CLI

EaglePlot includes a command-line interface with 5 subcommands:

```bash
eagleplot plot    <file> --plot <type> --x-column X --y-column Y [--output out.png]
eagleplot stats  <file>
eagleplot batch  <config.json> [--verbose]
eagleplot compare <file1> <file2> --plot line [--output comparison.png]
eagleplot convert <input> <output>
```

See the [CLI Tutorial](docs/tutorials/cli_guide.md) for full usage and examples.

## Python API

EaglePlot's modules can be used directly in scripts or Jupyter notebooks:

```python
from eagleplot.io import load_data
from eagleplot.plotting import line_plot, save_figure
import numpy as np

wavelength = np.linspace(400, 800, 200)
absorbance = 0.8 * np.exp(-((wavelength - 520) ** 2) / (2 * 30**2))

fig, ax, info = line_plot(
    wavelength,
    [absorbance],
    xlabel="Wavelength (nm)",
    ylabel="Absorbance",
    title="UV-Vis Spectrum",
    labels=["Sample A"],
)
save_figure(fig, "spectrum.png", dpi=300)
```

### Key modules

| Module | Description |
| --- | --- |
| `eagleplot.io` | `load_data()` / `save_data()` — auto-detects format from extension |
| `eagleplot.math` | 25 functions — statistics, curve fitting, hypothesis tests, optimization, linear algebra |
| `eagleplot.plotting` | 27 plot types; Matplotlib → `(fig, ax, info)`, Plotly → `(fig, info)` |
| `eagleplot.signal` | 16 functions — smoothing, filtering, FFT, derivatives, baseline correction, interpolation |
| `eagleplot.peaks` | 5 functions — find, integrate, FWHM, fit (Gaussian/Lorentzian/Voigt/pseudo-Voigt) |
| `eagleplot.data_tools` | 12 non-destructive DataFrame operations |
| `eagleplot.spectroscopy` | 18 functions — IR/Raman, UV-Vis, NMR, MS |
| `eagleplot.nist` | NIST WebBook integration — fetch IR spectra by CAS number |
| `eagleplot.batch` | Batch load, statistics, curve fit, peak analysis |
| `eagleplot.annotations` | 7 overlay types (hline, vline, hspan, vspan, text, rectangle, ellipse) |
| `eagleplot.report` | PDF report generation via `matplotlib.PdfPages` |
| `eagleplot.molecular` | CPK atom data, Gaussian/ORCA/Molden parsers, Plotly 3D molecule builder |

## Plugin System

Drop a `plugin_*.py` file into `plugins/` to add custom plot types or analysis tools.
See [`plugins/plugin_example.py`](plugins/plugin_example.py) for the starter template.

## Dependencies

| Package | Purpose |
| --- | --- |
| `numpy` | Numerical computing |
| `scipy` | Scientific algorithms, curve fitting, signal processing |
| `pandas` | Tabular data |
| `matplotlib` | Static plotting |
| `seaborn` | Statistical visualizations |
| `plotly` | Interactive plots |
| `streamlit` | GUI framework |
| `openpyxl` | Excel file support |
| `requests` | NIST WebBook HTTP integration |
| `h5py` | HDF5 file support |
| `xarray` / `netcdf4` | NetCDF file support |

**Optional:**

| Package | Purpose |
| --- | --- |
| `statsmodels` | Two-way ANOVA (`anova_twoway` in `eagleplot.math`) |
| `pymzml` | mzML/mzXML mass spectrometry files |
| `jcamp` | Alternative JCAMP-DX parser |

See [`requirements.txt`](requirements.txt) for pinned version ranges.

## Project Structure

```text
eagleplot/                            ← repository root
├── eagleplot/                        ← the installable package
│   ├── Home.py                     ← Streamlit entry point
│   ├── cli.py                      ← CLI entry point (5 subcommands)
│   ├── io.py                       ← 18-format data loader/saver
│   ├── math.py                     ← 25 analysis functions
│   ├── plotting.py                 ← 27 plot types (Matplotlib, Seaborn, Plotly)
│   ├── signal.py                   ← 16 signal processing functions
│   ├── peaks.py                    ← 5 peak analysis functions
│   ├── data_tools.py               ← 12 non-destructive DataFrame operations
│   ├── annotations.py              ← 7 Matplotlib overlay types
│   ├── spectroscopy.py             ← 18 spectroscopy functions
│   ├── nist.py                     ← NIST WebBook integration
│   ├── batch.py                    ← 5 batch processing functions
│   ├── report.py                   ← PDF report generation
│   ├── plugin_loader.py            ← plugin discovery and loading
│   ├── molecular/                  ← CPK atom data + vibrational parsers
│   ├── pages/                      ← 13 Streamlit pages (1_*.py – 13_*.py)
│   ├── utils/                      ← session state, plot config, user settings
│   ├── assets/                     ← logo + NCCU branding (shipped in the wheel)
│   ├── gallery/                    ← pre-rendered gallery PNGs + manifest.json
│   └── example-data/Artificial/    ← built-in sample datasets
├── tests/                          ← pytest suite, 1,000+ tests
├── plugins/                        ← plugin_example.py starter template
├── examples/                       ← 15+ standalone scripts + batch_config.json
├── docs/                           ← tutorials, cheatsheet, site built by build_docs.py
├── generate_gallery.py             ← regenerates eagleplot/gallery/
├── requirements.txt
├── pyproject.toml
├── setup.bat / .command / .sh      ← First-run setup (Windows / macOS / Linux)
└── launch.bat / .command / .sh     ← App launcher (Windows / macOS / Linux)
```

Runtime assets (`assets/`, `gallery/`, `example-data/`) live **inside** the package
rather than at the repository root, so they are present after `pip install`.

## Documentation

- [Getting Started](docs/getting_started.md) — installation walkthrough and API examples
- [Cheatsheet](docs/cheatsheet.md) — quick reference for all plot functions
- [CLI Tutorial](docs/tutorials/cli_guide.md) — step-by-step guide to all 5 CLI commands
- [GUI Tutorial](docs/tutorials/gui_guide.md) — page-by-page walkthrough of the Streamlit app
- [Examples](examples/) — 15+ standalone runnable scripts covering all major workflows

## About

**Author:** Jonathan D. Schultz, PhD — NCCU Department of Chemistry and Biochemistry
**Institution:** North Carolina Central University
**Use:** Research and teaching in computational science
**License:** See [LICENSE](LICENSE)
