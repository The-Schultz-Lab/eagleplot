# Changelog

All notable changes to EaglePlot (called Plottle until 3.0) are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- **The EaglePlot logo is now in NCCU maroon and gray** (it was blue): the same disc, bars
  and line, on the Home page, in the sidebar, as the browser-tab icon and in `logo.ico`.
  `eagleplot/assets/logo.svg` is the master.
- **A new loading screen for the web app.** The logo's pieces (the rings, the axes, the
  bars, the line and the points) fly in from all sides of the screen and lock together,
  then a highlight sweeps across while Python loads. Below it, four real steps
  (downloading Python, installing the science libraries, starting EaglePlot, opening your
  workspace) replace the endlessly sliding bar. The note about the 45 MB download appears
  only on a first visit. If loading fails, the screen says why and offers **Reload**
  instead of sliding forever. With reduced motion turned on in the system settings, the
  logo appears complete and nothing moves.

---

## [3.1.0] - 2026-11-06

**EaglePlot now also runs in your browser**, with nothing to install:
<https://the-schultz-lab.github.io/eagleplot/app/>. This release also makes the GUI much
faster, gives it a simpler first screen, and fixes JCAMP-DX loading, which was broken in
3.0.0 for anyone with the current `jcamp` package.

### Added

- **EaglePlot runs in your browser:
  [the-schultz-lab.github.io/eagleplot/app/](https://the-schultz-lab.github.io/eagleplot/app/)**.
  Nothing to install and nothing uploaded: your data stays on your computer, which makes it
  usable on laptops where you can't install software. It takes 15–30 seconds to start.
  Features that can't work in a browser are turned off with an explanation instead of
  failing: loading a whole folder, downloading from NIST (the page links to NIST instead),
  and changing the theme. Settings, presets and workflows are kept in the browser. The
  documentation site and README link to it. Every change to the app is tested in a real
  browser before it is published (`packaging/web/`, `scripts/web/smoke.js`).
- **`EAGLEPLOT_CONFIG_DIR`** sets the folder that holds `config.json` (saved defaults,
  presets and workflows). Unset, it stays where it was.

### Changed

- **A "Get started" Home page.** With no data loaded, Home offers three example datasets,
  each a single click away from a plot in Quick Plot, plus buttons to upload a file or
  browse all examples, instead of "Datasets 0 · Plots 0 · Analyses 0".
- **Quick Plot shows the plot on the first screen.** Dataset, library and plot type sit in
  one row with a **Style presets** button, and the dataset size and plot description share
  one line. On a 1366×768 laptop the plot used to start below the bottom of the screen.
  All pages have less empty space at the top.
- **Clearer menu names:** *Basic* is now **Quick Plot**, *Multiplot* **Dashboard**, *Single*
  **Analysis Tools** and *Batch* **Batch Analysis**. Page addresses are unchanged, so
  bookmarks still work.
- The sidebar's **About** box starts closed, and on phones the sidebar starts closed.
- **Quick Plot is much faster and easier to use.**
  - Settings sit on the left and the plot on the right, so you see each change as you
    make it. The plot stays in view while the settings scroll.
  - **Live preview:** the plot redraws as you change settings (toggle it off if you prefer).
    **Generate Plot** still saves the plot to the plot history.
  - Changing a setting now updates only the settings and plot area, not the whole page.
    In a test with a line plot, the time from a change to seeing it on the plot fell from
    about 1.3 s to about 0.6 s.
  - Downloads are created only when you ask for one. Before, every click re-created the
    plot as PNG at 150, 300 and 600 dpi, SVG and PDF, whether or not you were downloading.
    The export panel now has one format menu and one **Download** button.
  - A marker appears as soon as you click a point on an interactive plot.
- **Export Results is much faster.** It used to re-create every saved plot as PNG (150 and
  300 dpi), SVG and PDF, and re-save the dataset as CSV and JSON, on every click: 1.2 s per
  click with three saved plots, now 0.03 s. Files are now created only when you download them.
- **PDF report:** on current Streamlit it is one click, and the report is built when you
  press **Download PDF Report**. On older Streamlit the download button no longer
  disappears after the next click.
- **Batch Analysis:** results stay on screen after you run a tool. Before, they disappeared
  at the next click, including when you clicked **Download CSV**.
- **Molecular Viz:** the HTML and frequency-table exports are one click (Streamlit 1.52+),
  instead of "Export", then "Download", with the button disappearing afterwards.
- **Analysis Tools is faster.** Each tool now updates on its own, so changing a setting or
  running a tool no longer reloads the whole page. A setting change takes about 0.23 s
  instead of about 0.38 s. A result from one tool stays on screen while you use another, and
  downloading the peak report no longer clears it.
- **Multi-Plot Dashboard:**
  - Each panel has its own settings, shown above its plot. Editing a panel redraws only that
    panel, and the dashboard updates live after **Generate Dashboard**: a title edit shows up
    in about 0.4 s, where before you had to press Generate again.
  - The dashboard stays on screen when you change something or download it.
  - Turning shared axes off restores each panel's own axis limits.
- **Advanced Plotting:** each tab updates on its own, so a change in one tab no longer
  clears the plot in another. The interactive HTML file (about 5 MB) is created only when
  you download it, and the plot stays on screen afterwards.
- **Plots are drawn at 150 dpi**, as Quick Plot already was (other pages used 200), at
  about half the cost to draw. Unchanged Multi-Plot Dashboard panels are no longer
  redrawn when the page refreshes: a refresh of a 2×2 dashboard takes about 0.4 s instead
  of 1 s.
- A missing optional package (for example h5py for HDF5 files) is reported with what to
  install, and where installing isn't possible, says so.

### Fixed

- **JCAMP-DX files (`.jdx`, `.dx`) load again.** With jcamp 1.3 installed, every such file
  failed with "the jcamp package is required". jcamp is now part of the `formats` extra
  (`pip install "eagleplot[formats]"`).
- Quick Plot's dataset CSV download was named `name.csv.csv`.
- **Multi-Plot Dashboard:** the combined PNG and PDF downloads work. Before, pressing either
  button made the dashboard disappear without downloading anything.

---

## [3.0.0] - 2026-11-05

**Plottle is now EaglePlot.** PyPI does not accept the name `plottle` (it counts as too
similar to an existing project), so 3.0.0 is the first release on PyPI, under the new name.
Nothing else about the toolkit changes in this release.

### ⚠️ Changed: name, import path and command

- **Install:** `pip install eagleplot` (with extras: `pip install "eagleplot[formats,nist]"`).
- **Import:** `import eagleplot` / `from eagleplot.io import load_data`. Code that imports
  `plottle` must be updated; there is no `plottle` compatibility package.
- **Command:** `eagleplot`, replacing `plottle`. From a source checkout:
  `python -m eagleplot.cli`, and the GUI is `streamlit run eagleplot/Home.py`.
- **Setting:** `EAGLEPLOT_ALLOW_FOLDER_LOAD` replaces `PLOTTLE_ALLOW_FOLDER_LOAD`. The old
  name still works for now and shows a deprecation warning.
- **Links:** the repository is `The-Schultz-Lab/eagleplot` and the documentation is at
  https://the-schultz-lab.github.io/eagleplot/.

### Still works

- Session files saved by Plottle 2.x load unchanged.
- Saved defaults and presets (`config.json`) stay where they were.

---

## [2.1.0] - 2026-11-04

The first release published to PyPI. It closes the security, packaging and CI findings of
the July 2026 code audit.

### ⚠️ Changed: import paths and the CLI command

- **The package is now imported as `plottle`.** Code that did `from modules.io import
  load_data` must change to `from plottle.io import load_data`. The README for 2.0.x already
  documented `plottle.*`, but the code still shipped as `modules`. The old name also meant
  installing the wheel claimed the generic top-level names `modules` and `cli`, which could
  collide with other projects.
- **The CLI is the `plottle` command**, installed by `pip install`. From a source checkout
  without installing, use `python -m plottle.cli`. The repo-root `cli.py` is gone.
- **Setup scripts install the package** (`pip install -e ".[formats,nist]"`) rather than
  `requirements.txt`, so the `plottle` command exists after setup.

### Security

- **Session files no longer execute code.** Loading a `.json` session ran `pickle.loads` on
  data inside it, so opening a session someone else shared ran their code. Arrays are now
  stored as dtype, shape and raw bytes. Sessions saved by 2.0.1 or earlier that contain
  pickled entries are refused, and the Export page lists what was skipped.
- **Formula evaluation is sandboxed properly.** Data Tools formulas used `eval()` with
  empty builtins, which can be escaped. They are now parsed and checked against a whitelist
  of operators and functions. `SECURITY.md` now describes what is and isn't guaranteed.
- `.pkl` files are no longer accepted by the GUI uploader or batch import. Use
  `plottle.io.load_pickle()` from Python for files you trust.
- Batch import from a folder reads a path on the machine running Plottle, so it is now
  available only to local connections. Set `PLOTTLE_ALLOW_FOLDER_LOAD=1` to allow it
  everywhere, or `0` to turn it off.
- Uploaded files go into a private temporary directory that is deleted after loading,
  instead of a shared folder where two users uploading `data.csv` overwrote each other.

### Fixed

- **Quick Plot failed to load on Python 3.9** (`TypeError: unsupported operand type(s) for
  |`). The Molecular Viz and Batch Analysis pages had the same problem in code paths reached
  later.
- **The Settings page failed on Python 3.9 and 3.10** (`ModuleNotFoundError: tomllib`).
  `tomli` is now used as a fallback.
- **Quick Plot (Plotly mode): clicking a data point crashed the page** with
  `AttributeError: 'dict' object has no attribute 'x'`. Marker labels on dates and
  categories no longer raise either.
- **A pip-installed GUI was missing its assets.** The logo, NCCU branding, example datasets
  and gallery images now ship inside the package.
- **Theme settings:** "Theme saved" was shown even when nothing was written, and saving
  discarded every comment in `config.toml` and could write invalid TOML. The theme is now
  written to `.streamlit/config.toml` in the working directory, where Streamlit reads it,
  with comments and other sections preserved. Page styling follows the active theme
  instead of a hard-coded dark palette, which left white text on a light background.
- **Export → Save session:** the download button disappeared after any other click, so it
  took two tries to download. It now stays available until the session changes.
- **Settings:** "Reset defaults", "Delete preset" and "Delete config.json" each need a
  confirmation tick first, and "Delete config.json" offers a backup download.
- The macOS `setup.command` and `launch.command` files are executable again, so Finder will
  run them from a fresh clone.
- The Data Tools page is reachable from the navigation. The sidebar can be collapsed again,
  and images have alt text.
- `streamlit>=1.49` is now the declared minimum. The code needed it, but the old `>=1.25`
  floor allowed versions that fail.

### Added

- `setup.sh` and `launch.sh` for Linux. All setup scripts now check the Python version
  first.
- `formats` (HDF5/NetCDF), `nist` (NIST WebBook lookups), `test` and `all` install extras,
  so a normal install no longer pulls in the developer tools.
- A documentation site built from the guides by `build_docs.py`, replacing the unbuilt
  Sphinx setup. Every guide is linked from the landing page.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue forms and a pull request template.

### Development

- CI tests Python 3.9 to 3.12 on Linux, plus macOS and Windows, and runs the whole test
  suite. It used to run a hand-written list of files that skipped the CLI and integration
  tests. Python 3.9 is pinned to `ubuntu-24.04` and macOS to `macos-15`.
- mypy runs with the project installed, so it checks real types instead of treating
  numpy, pandas and matplotlib as `Any`. ruff and mypy are pinned to exact versions.
- `ruff check` and `ruff format --check` cover the whole repository, and ruff flags
  `X | None` annotations that would fail on Python 3.9 (FA102).
- New tests:
  - every GUI page renders on a fresh session,
  - the "Seaborn/Plotly not installed" errors (9 of these tests were always skipped),
  - the Settings, Export and upload behaviour above.

  The suite has 1,100+ tests.
- The version number has a single source, `plottle/__init__.py`.

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

[Unreleased]: https://github.com/The-Schultz-Lab/eagleplot/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/The-Schultz-Lab/eagleplot/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/The-Schultz-Lab/eagleplot/compare/v2.1.0...v3.0.0
[2.1.0]: https://github.com/The-Schultz-Lab/eagleplot/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/The-Schultz-Lab/eagleplot/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/The-Schultz-Lab/eagleplot/releases/tag/v2.0.0
[1.0.0]: https://github.com/The-Schultz-Lab/eagleplot/releases/tag/v1.0.0
[0.1.0]: https://github.com/The-Schultz-Lab/eagleplot/releases/tag/v0.1.0
