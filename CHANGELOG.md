# Changelog

All notable changes to Plottle are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed

- Quick Plot (Plotly mode): clicking a data point no longer crashes the page with
  `AttributeError: 'dict' object has no attribute 'x'`. Streamlit returns selected points
  as plain dicts; they are now read through `selected_points()` in
  `plottle/utils/plot_config.py`. Marker labels on non-numeric axes (dates, categories)
  also no longer raise on the `:.4g` format.

---

## [2.0.1] - 2026-04-08

### Fixed

- Session save/load: wrap DataFrame JSON string in `io.StringIO` before passing to
  `pd.read_json()`, fixing a `FileNotFoundError` on pandas 2.x where a bare string
  argument is now treated as a file path rather than inline content
  (`modules/utils/session_state.py`)

### Changed

- Added macOS one-click setup and launch scripts (`setup.command`, `launch.command`)
- Updated `launch.bat` to check for the virtual environment and print a helpful error
  if it is missing
- README: added platform-specific setup/launch instructions, PyPI badge, and updated
  programmatic import examples to use `plottle.*` namespace
- `pyproject.toml`: bumped minimum `setuptools` to 78.0, normalized `license` field
  to PEP 639 format, corrected author email, added `build`/`twine` dev dependencies
  and changelog URL
- `modules/Home.py`: use `.resolve()` for path construction (improves symlink and
  working-directory robustness); moved NCCU wings logo to sidebar

---

## [2.0.0] - 2026-03-17

### Added

**Statistical Testing Suite (`modules/math.py`)**
- 12 new hypothesis-testing functions: one-sample/two-sample/paired t-tests, Mann-Whitney U,
  Wilcoxon signed-rank, Kruskal-Wallis, one-way ANOVA, Tukey HSD, Bonferroni correction,
  Pearson/Spearman correlation, chi-square independence
- Two-way ANOVA (`anova_twoway`) via optional `statsmodels` dependency
- Effect size display (Cohen's d with Small/Medium/Large interpretation) in the GUI

**Signal Processing (`modules/signal.py` — new module)**
- 16 functions: moving-average/Savitzky-Golay/Gaussian smoothing; Butterworth low/high/band/band-stop
  filters (zero-phase `sosfiltfilt`); FFT + inverse FFT + power spectrum; 1st/2nd derivative;
  polynomial/rolling-ball/asymmetric-least-squares baseline correction; cubic/linear/nearest interpolation
- "Signal Processing" tab (6 sub-tabs) added to Analysis Tools page

**Peak Analysis (`modules/peaks.py` — new module)**
- 5 functions: `find_peaks`, `integrate_peaks`, `compute_fwhm`, `fit_peak`, `fit_multipeak`
- Gaussian/Lorentzian/Voigt/pseudo-Voigt peak models; multi-peak simultaneous fitting
- "Peak Analysis" tab (2 sub-tabs: Find & Measure, Fit Peaks) in Analysis Tools page

**Extended Plot Types (`modules/plotting.py`)**
- 13 new plot types: bar chart (simple/grouped/stacked), waterfall plot, dual y-axis, z-colored
  scatter, bubble chart, polar plot, pair plot, 2D histogram/hexbin, interactive 3D scatter,
  scatter with regression, residual plot, interactive ternary, broken axis, inset plot
- Error bar support (`yerr`/`xerr`) added to `scatter_plot` and `line_plot`
- `COLORBLIND_PALETTE` (Okabe-Ito 8 colors) added; `get_color_palette('colorblind', n)` works
- Total plot types: **26** (was 13 at v1.0.0)

**Data Manipulation Layer (`modules/data_tools.py` — new module)**
- 12 non-destructive functions: formula columns (safe eval), normalization, transpose,
  pivot/melt, row filter, sort, merge/join, fill/drop NaN, resample, rolling transforms
- Page 9 — Data Tools: dedicated Streamlit page with 10 operation tabs

**Figure Layout & Annotation (`modules/annotations.py` — new module)**
- 7 overlay types: hline, vline, hspan, vspan, text, rectangle, ellipse
- Annotation panel added to Quick Plot (add/delete/clear overlays; persists across re-renders)
- Multi-Plot Dashboard expanded: 10 grid layouts up to 4×4; axis sharing; combined PNG/PDF export

**Instrument File Import (`modules/io.py`)**
- 5 new loaders: JCAMP-DX (`.jdx`/`.dx`), HDF5 (`.h5`/`.hdf5`), NetCDF (`.nc`/`.cdf`),
  SPC (`.spc`), generic ASC/text (`.asc`)
- mzML/mzXML loader (`load_mzml`) via optional `pymzml` dependency

**Spectroscopy & Molecular Analysis (`modules/spectroscopy.py`, `modules/molecular/` — new)**
- 18 spectroscopy functions covering IR/Raman, UV-Vis/Beer-Lambert/FRET, NMR, and mass spec
- `modules/molecular/` package: CPK atom data, Gaussian/ORCA/Molden vibrational parsers,
  Plotly 3D molecule + displacement-arrow figure builder (no RDKit dependency)
- Page 10 — Spectroscopy; Page 11 — Molecular Viz (vibrational mode animation)
- NIST WebBook integration (`modules/nist.py`): fetch IR JCAMP spectra by CAS number

**Batch GUI & Automation (`modules/batch.py` — new module)**
- 5 functions: `scan_directory`, `batch_load_files`, `batch_statistics`, `batch_curve_fit`,
  `batch_peak_analysis`
- Page 12 — Batch Analysis: Statistics/Curve Fit/Peak Analysis across multiple datasets + CSV export
- Workflow presets: save/load/delete named analysis pipeline configs persisted to `config.json`
- Batch folder import expander added to Data Upload page

**PDF Report Generation (`modules/report.py` — new module)**
- `generate_pdf_report()` using `matplotlib.backends.backend_pdf.PdfPages` (no new dependencies)
- Report section added to Export Results page (title, author, DPI slider, one-click download)

**Plugin System (`modules/plugin_loader.py` — new module)**
- Auto-discovers `plugin_*.py` files from the `plugins/` directory via `importlib.util`
- `discover_plugins`, `get_plugin_plot_types`, `get_plugin_analysis_tools`, `list_plugins`
- `plugins/plugin_example.py` starter template; plugin status shown in Settings page

**Performance Utilities (`modules/io.py`)**
- `downsample_for_preview(df, max_rows=10_000, method='systematic')` — fast preview for large data
- `load_large_csv(filepath, max_rows=500_000, chunksize=50_000)` — chunked CSV loading
- Large-file warning (> 50 MB or > 10k rows) with auto-downsampled preview in Data Upload

**New GUI Pages (7 total added)**
- Page 7 — Export Results (plots, datasets, analysis JSON, session save/load, PDF report)
- Page 8 — Gallery (pre-generated example figures for all 26 plot types; "Load this config" button)
- Page 9 — Data Tools
- Page 10 — Spectroscopy
- Page 11 — Molecular Viz
- Page 12 — Batch Analysis
- Page 13 — Help (Getting Started, Plot Types, Analysis Tools, File Formats, Tips & Tricks)
- Page 14 — Settings (moved to bottom of sidebar)
- Home page redesigned: Dashboard tab (metric cards, smart CTA) + Help tab (Quick Start, formats, tips)

**UX & Visual Polish**
- Nunito Google Font applied globally via CSS injection
- Style presets: save/load/delete named Quick Plot configurations persisted to `config.json`
- "Convert to Plotly (zoom/pan)" toggle on every matplotlib figure in Quick Plot
- PNG bytes stored in session state (fixes `MediaFileHandler: Missing file` Streamlit bug)
- Equal-height CSS for Gallery cards; compact Data Upload (3-tab layout)

**Testing**
- Test suite expanded from 258 to ~900+ tests across 17 test files
- 12 new test modules: `test_signal`, `test_peaks`, `test_data_tools`, `test_annotations`,
  `test_spectroscopy`, `test_molecular_parsers`, `test_batch`, `test_report`,
  `test_plugin_loader`, `test_nist`, plus expansions to `test_plotting`, `test_math`, `test_io`

**Code Quality**
- Switched formatter from `black` to `ruff-format`
- Line length bumped to 100; E501 enforced
- `.pre-commit-config.yaml`: ruff, ruff-format, trailing-whitespace, end-of-file-fixer,
  check-merge-conflict

### Changed

- GUI entry point renamed from `modules/gui.py` to `modules/Home.py`
- Settings page renumbered from `6_Settings.py` to `14_Settings.py` (sidebar ordering)
- `use_container_width=` fully replaced with `width='stretch'` / `width='content'` (all pages)
- `black` replaced by `ruff-format` in CI lint job and `pyproject.toml` dev dependencies
- `ruff check` now covers all new modules in CI

### Fixed

- NIST WebBook IR fetch: corrected URL format to `?JCAMP=C{cas}&Index={idx}&Type=IR-SPEC`
  (was returning HTTP 404 with old `?ID=cas&JCAMP=1` format)
- `mpl_to_plotly` conversion now re-applies style config after conversion so font sizes
  and colors persist when toggling "Convert to Plotly"
- `generate_gallery.py`: fixed 7 bugs (column name mismatches, `dpi` leaking into plot kwargs,
  duplicate kwargs, unsupported `annotate` kwarg for heatmap, broken figsize handling)

---

## [1.0.0] - 2026-03-06

### Added

**I/O Module (`modules/io.py`)**
- Universal `load_data()` / `save_data()` with auto-format detection from file extension
- Support for: `.pkl`, `.npy`, `.npz`, `.csv`, `.xlsx`, `.tsv`, `.json`, `.parquet`
- Format-specific helpers: `load_pickle`, `load_numpy`, `load_dataframe`, `save_pickle`, `save_numpy`, `save_dataframe`

**Math Module (`modules/math.py`)**
- `calculate_statistics()` — mean, median, std, var, min, max, Q1, Q3, IQR, range
- `check_normality()` — Shapiro-Wilk test returning (statistic, p_value)
- Curve fitting: `fit_linear`, `fit_polynomial`, `fit_exponential`, `fit_custom`, `fit_distribution`
- Optimization: `minimize_function`, `find_roots`
- Linear algebra: `compute_eigenvalues`, `solve_linear_system`, `matrix_decomposition` (QR, SVD, Cholesky)

**Plotting Module (`modules/plotting.py`)**
- Matplotlib (static): `histogram`, `line_plot`, `scatter_plot`, `heatmap`, `contour_plot`
- Seaborn (statistical): `distribution_plot`, `box_plot`, `regression_plot`
- Plotly (interactive): `interactive_histogram`, `interactive_scatter`, `interactive_line`, `interactive_heatmap`, `interactive_3d_surface`
- Utilities: `save_figure`, `create_figure`, `set_style`, `apply_publication_style`, `get_color_palette`, `export_interactive`
- Consistent return signatures: Matplotlib → `(fig, ax, info)`, Plotly → `(fig, info)`

**CLI (`cli.py`)**
- Five subcommands: `plot`, `stats`, `batch`, `compare`, `convert`
- Batch processing via JSON config (`batch_config.json`)
- `--version` flag

**Streamlit GUI (`modules/gui.py` + `modules/pages/`)**
- Page 1 — Data Upload: file upload for all supported formats; 10 built-in example datasets
- Page 2 — Quick Plot: all 13 plot types with dynamic widget panel; interactive/static toggle
- Page 3 — Analysis Tools: 5 tabs — Statistics, Distribution Tests, Curve Fitting, Optimization, Linear Algebra
- Page 4 — Multi-Plot Dashboard: grid layouts up to 2×3 (6 cells), mixed plot types per cell
- Page 5 — Advanced Plotting: annotated correlation heatmap, overlaid distributions, grouped categorical, 3D scatter, HTML export
- Page 6 — Settings: theme, DPI, default plot library, CSV separator, preset CRUD
- Page 7 — Export Results: PNG/SVG/PDF/HTML plot download, CSV/JSON/NPY/PKL dataset export, analysis results JSON, session save/load

**Testing**
- 258 passing tests, 9 expected skips (import-guard tests when seaborn/plotly absent)
- Coverage: `io.py` ≥ 90%, `math.py` 100%, `plotting.py` 94%, `utils/` ≈ 99%
- Test files: `test_io.py`, `test_math.py`, `test_plotting.py`, `test_plotting_advanced.py`, `test_utils.py`, `test_cli.py`, `test_integration.py`
- CI: GitHub Actions workflow on push/PR to `main` (`.github/workflows/tests.yml`)
- Lint CI: ruff + black + mypy checks (`.github/workflows/tests.yml` `lint` job)

**Documentation**
- Sphinx API docs with autodoc + Napoleon + RTD theme (`docs/conf.py`, `docs/api/`)
- GitHub Pages deployment workflow (`.github/workflows/docs.yml`)
- Tutorials: `docs/tutorials/cli_guide.md`, `docs/tutorials/gui_guide.md`
- Quick-reference cheatsheet: `docs/cheatsheet.md`
- Getting started guide: `docs/getting_started.md`
- 15 standalone example scripts in `examples/`
- 4 Jupyter tutorial notebooks in `notebooks/` (Binder-ready via `binder/requirements.txt`)

**Packaging**
- `pyproject.toml` (PEP 517/518, setuptools backend)
- `pip install .` and `pip install -e .` supported
- `plottle` CLI entry point registered

### Changed

- GUI pages migrated from deprecated `use_container_width=` parameter to `width='stretch'` / `width='content'` (Streamlit ≥ 2026-01 compatibility)
- `plotly.express` import removed from `plotting.py` (was unused)
- `typing.List`, `typing.Dict` replaced with built-in generics where unused; `TYPE_CHECKING` guard added for `pd.DataFrame` type hint
- Code formatted with `black` and linted with `ruff` across all source files

### Fixed

- `modules/utils/user_settings.py` — `get_default_settings()` returned a shallow copy sharing inner dicts across calls, causing state bleed between invocations; fixed by returning a fresh literal dict
- `tests/test_io.py` — DataFrame CSV/Excel/JSON/TSV roundtrip tests failed on Windows because `np.arange()` produces `int32` but CSV loading yields `int64`; fixed with `check_dtype=False`
- Bare `except:` in `data_preview.py` replaced with `except Exception:`

---

## [0.1.0] - 2026-01

Initial development release. Stages 0–6 complete (foundation, I/O, math, basic plotting, advanced plotting, CLI, GUI). Not publicly distributed.

---

[1.0.0]: https://github.com/The-Schultz-Lab/plottle/releases/tag/v1.0.0
[0.1.0]: https://github.com/The-Schultz-Lab/plottle/releases/tag/v0.1.0
