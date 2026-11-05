# GUI Tutorial — EaglePlot

This guide walks through the Streamlit app page by page, then shows a complete end-to-end
workflow: upload data → analyze → plot → export.

---

## Launching the App

```bash
# Activate your virtual environment first, then:
streamlit run eagleplot/Home.py
```

Open `http://localhost:8501` in your browser. The sidebar on the left lists all pages.

---

## Data Upload

**Purpose:** Load one or more data files into the session. All subsequent pages work with
the data loaded here.

### Supported formats

| Extension | Returns |
| --- | --- |
| `.csv`, `.xlsx`, `.tsv`, `.json`, `.parquet` | pandas DataFrame |
| `.npy`, `.npz` | NumPy array |
| `.pkl` | Any Python object |

### Steps

1. Click **Browse files** (or drag-and-drop) to upload your file.
2. The page shows a preview: shape, column types, and summary statistics.
3. Upload additional files — each becomes a named dataset in the sidebar selector.
4. Select the **active dataset** from the dropdown at the top of every page.

### Tips

- Large files (> 50 MB) may be slow to preview but will work.
- `.npz` files load all arrays; select which array to use in subsequent pages.

---

## Quick Plot

**Purpose:** Generate any of the 13 supported plot types with point-and-click controls.

### Plot types available

| Category | Types |
| --- | --- |
| Matplotlib (static) | Line, Scatter, Histogram, Heatmap, Contour |
| Seaborn (statistical) | Distribution, Box / Violin, Regression |
| Plotly (interactive) | Histogram, Scatter, Line, Heatmap, 3D Surface |

### Steps

1. Select a **plot type** from the dropdown.
2. Choose your **X column** and **Y column** (or the data array for histograms).
3. Adjust style options: title, axis labels, color palette, line width, etc.
4. Click **Generate Plot**.
5. Plotly charts are interactive — zoom, pan, and hover for values.
6. Expand **Export plot & data** to download the figure as PNG, SVG, PDF, or HTML.

### Tips

- Switching plot type resets the column selectors — pick the type first.
- For multi-series line plots, select multiple Y columns with Ctrl+click.
- Plotly HTML exports are self-contained and can be shared as a single file.

---

## Analysis Tools

**Purpose:** Run quantitative analysis on the active dataset. Five tabs cover the most
common scientific workflows.

### Tab A — Descriptive Statistics

Displays mean, median, std, variance, min, max, Q1, Q3, IQR, and range for every numeric
column. Also runs a Shapiro-Wilk normality test with pass/fail indicator.

### Tab B — Distribution Fitting

Fit a theoretical distribution (Normal, Exponential, Log-normal, etc.) to a numeric column.
Shows fitted parameters, KS-test statistic, and an overlay plot.

### Tab C — Curve Fitting

Fit a model to X/Y data:

| Model | Parameters |
| ----- | ---------- |
| Linear | slope, intercept, R² |
| Polynomial | degree 1–10, coefficients, R² |
| Exponential | a, b (y = a·exp(b·x)), R² |
| Custom | Enter any Python expression using `x`, e.g., `a * np.sin(b * x) + c` |

Results show fitted parameters and a residuals plot. Click **Copy Parameters** to paste
values into a script.

### Tab D — Optimization

Find a function minimum or root:

- **Minimize:** Enter a Python expression in `x` (e.g., `(x - 3)**2 + 1`); set bounds and
  initial guess.
- **Find root:** Enter an expression; set a bracket `[a, b]` that contains a sign change.

### Tab E — Linear Algebra

Upload or enter a matrix, then compute:

- Eigenvalues and eigenvectors
- Solve a linear system A·x = b
- Matrix decomposition (QR, SVD, Cholesky)

---

## Multi-Plot Dashboard

**Purpose:** Display up to six independent plots side-by-side in a configurable grid.

### Steps

1. Choose a **grid layout** (1×1 up to 2×3).
2. For each cell, select a dataset, plot type, and columns independently.
3. Click **Generate Dashboard** — all plots render simultaneously.
4. Expand any cell to enlarge it; click its **Download** button to save.

### Tips

- Each cell is independent — you can mix Matplotlib and Plotly in the same dashboard.
- Use the 2×3 layout for a six-panel publication figure.

---

## Advanced Plotting

**Purpose:** Specialized visualizations not available on the Quick Plot page.

### Options

| Feature | Description |
| --- | --- |
| **Correlation Heatmap** | Pearson correlation matrix for all numeric columns |
| **Overlaid Distributions** | KDE / ECDF overlaid across groups (color-coded by a categorical column) |
| **Grouped Categorical** | Box, violin, or strip plot grouped by a categorical column |
| **3D Scatter** | Interactive Plotly 3D scatter with color mapping |
| **HTML Export** | Save any Plotly figure as a self-contained HTML file |

---

## Settings

**Purpose:** Persist plot defaults and create named style presets.

### Configurable defaults

- **Color palette** — choose from six palettes including two color-blind-safe options
- **Figure size** — width × height in inches
- **DPI** — default resolution for saved figures
- **Grid** — on/off
- **Font size** — base font for all text
- **Line width** — default line thickness
- **Legend** — show/hide

### Presets

Click **Save as preset** to name and store the current defaults. Switch between presets
from the dropdown. Settings are written to `config.json` in the project root and persist
across app restarts.

---

## Export Results

**Purpose:** Download figures, datasets, and analysis results generated during the session.

### Plot history

Every plot generated on Quick Plot or Advanced Plotting is recorded in the session history. For each:

- Choose a format: **PNG 150 dpi**, **PNG 300 dpi**, **SVG**, **PDF**, or **HTML** (Plotly)
- Click **Download** to save the file

### Dataset export

Export the currently loaded dataset as **CSV**, **JSON**, **NumPy (.npy)**, or **pickle (.pkl)**.

### Analysis results

Export any curve-fit or statistics result from Analysis Tools as a structured **JSON** file.

### Session save / load

- **Save session** — writes all loaded datasets, plot history, and analysis results to a
  single `.json` file you can download.
- **Load session** — drag-and-drop a previously saved session file to restore the full state.

---

## Complete Workflow Example

Goal: Upload a calibration dataset, fit a Beer-Lambert curve, export the fit result.

1. **Data Upload:** Upload `calibration.csv` (columns: `concentration`, `absorbance`).
2. **Analysis Tools → Curve Fitting:** Select `concentration` (X) and `absorbance` (Y); choose
   **Linear**; click **Fit**. Note the slope (molar absorptivity · path length) and R².
3. **Quick Plot:** Select **Scatter** plot; X = `concentration`, Y = `absorbance`; title =
   "Beer-Lambert Calibration"; click **Generate Plot**.
4. **Export Results:** Download the scatter plot as PNG 300 dpi. Download the curve-fit result as JSON.

Total time: under two minutes with no code written.
