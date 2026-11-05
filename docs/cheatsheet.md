# EaglePlot — Quick Reference Cheat Sheet

NCCU Department of Chemistry and Biochemistry

---

## Setup

```python
import sys

sys.path.insert(0, "path/to/eagleplot")

from eagleplot.io import load_data, save_data
from eagleplot.math import calculate_statistics, fit_linear, fit_polynomial
from eagleplot.plotting import line_plot, scatter_plot, histogram, save_figure
```

---

## I/O — Loading Data

```python
from eagleplot.io import load_data, load_dataframe, load_numpy, load_pickle

df = load_data("data.csv")  # auto-detect by extension
df = load_dataframe("data.csv")  # → DataFrame  (.csv .xlsx .tsv .json .parquet)
arr = load_numpy("array.npy")  # → ndarray    (.npy .npz)
obj = load_pickle("data.pkl")  # → any object (.pkl)
```

## I/O — Saving Data

```python
from eagleplot.io import save_data, save_dataframe, save_numpy, save_pickle

save_data(df, "output.csv")
save_dataframe(df, "output.xlsx")
save_numpy(arr, "matrix.npy")
save_pickle(obj, "session.pkl")
```

---

## Statistics

```python
from eagleplot.math import (
    calculate_mean,
    calculate_median,
    calculate_std,
    calculate_statistics,
    check_normality,
)

mean = calculate_mean(arr)  # scalar or array
med = calculate_median(arr)
std = calculate_std(arr)  # sample std (ddof=1)

s = calculate_statistics(arr)
# Keys: mean, median, std, var, min, max, q1, q3, iqr, range

norm = check_normality(arr)
# Keys: statistic, p_value, is_normal (bool, α=0.05)
```

---

## Curve Fitting

```python
from eagleplot.math import fit_linear, fit_polynomial, fit_exponential, fit_custom

# Linear  y = m·x + b
r = fit_linear(x, y)
# r: slope, intercept, r_value, r_squared, p_value, std_err

# Polynomial  y = aₙxⁿ + … + a₀
r = fit_polynomial(x, y, degree=2)
# r: coefficients, degree, r_squared, residuals, predict(x_new)

# Exponential  y = a·exp(b·x)
r = fit_exponential(x, y)
# r: a, b, r_squared, predict(x_new)


# Custom function
def my_func(x, A, k, c):
    return A * np.exp(-k * x) + c


r = fit_custom(x, y, func=my_func, p0=[1.0, 0.1, 0.0])
# r: parameters (dict of fitted values), r_squared, predict(x_new)
```

---

## Matplotlib Plots

```python
from eagleplot.plotting import histogram, line_plot, scatter_plot, heatmap, contour_plot

# Histogram
fig, ax, info = histogram(data, bins=20, xlabel="Value", ylabel="Count", title="Distribution")
# info: n, bins, patches

# Line plot (single or multi-series)
fig, ax = line_plot(
    x, [y1, y2], labels=["A", "B"], xlabel="Time (s)", ylabel="Signal", title="Time Series"
)

# Scatter plot
fig, ax = scatter_plot(x, y, xlabel="Conc (mM)", ylabel="Abs", title="Calibration")

# Heatmap (2-D matrix)
fig, ax = heatmap(matrix, xlabel="Col", ylabel="Row", title="Correlation")

# Contour plot (requires 2-D meshgrid)
X, Y = np.meshgrid(x_vec, y_vec)
fig, ax = contour_plot(X, Y, Z, title="Potential Energy Surface")
```

## Seaborn Plots

```python
from eagleplot.plotting import distribution_plot, box_plot, regression_plot

fig, ax = distribution_plot(data, kind="kde")  # kind: hist | kde | ecdf
fig, ax = box_plot(df)  # or box_plot(df, kind='violin')
fig, ax = regression_plot(x, y)  # scatter + regression line + CI
```

## Interactive (Plotly) Plots

```python
from eagleplot.plotting import (
    interactive_histogram,
    interactive_scatter,
    interactive_line,
    interactive_heatmap,
    interactive_3d_surface,
)

fig = interactive_histogram(data, bins=20, title="Interactive Histogram")
fig = interactive_scatter(x, y, title="Click to Zoom")
fig = interactive_line(x, [y1, y2], labels=["A", "B"])
fig = interactive_heatmap(matrix)
fig = interactive_3d_surface(X, Y, Z, title="3D Surface")

fig.show()  # opens in browser
fig.write_html("plot.html")  # save as self-contained HTML
```

---

## Saving Figures

```python
from eagleplot.plotting import save_figure

save_figure(fig, "plot.png", dpi=150)  # screen quality
save_figure(fig, "plot.png", dpi=300)  # print / publication quality
save_figure(fig, "plot.svg")  # vector (editable in Illustrator/Inkscape)
save_figure(fig, "plot.pdf")  # vector, LaTeX-ready
```

---

## Figure Customization

```python
import matplotlib.pyplot as plt

fig, ax = scatter_plot(x, y)

ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Signal", fontsize=12)
ax.set_title("My Plot", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")

# Add a second y-axis
ax2 = ax.twinx()
ax2.plot(x, y2, color="red", linestyle="--", label="Secondary")
ax2.set_ylabel("Secondary axis")

# Annotate a point
ax.annotate(
    "Peak",
    xy=(x_peak, y_peak),
    xytext=(x_peak + 0.5, y_peak + 0.1),
    arrowprops=dict(arrowstyle="->"),
)

fig.tight_layout()
```

---

## Using the CLI

```bash
# Quick plot from CSV
eagleplot plot data.csv --type line --xcol time --ycol signal --output fig.png

# Compute statistics
eagleplot stats data.csv --column signal

# Batch processing from config
eagleplot batch config.json

# Convert file formats
eagleplot convert data.csv output.pkl

# Show examples
eagleplot --examples
```

---

## GUI Keyboard Shortcuts

| Action | How |
|---|---|
| Rerun page | Press `R` or click **Rerun** in banner |
| Generate plot | Click **Generate Plot** button |
| Clear annotations | Click **Clear markers** button |
| Download figure | Expand **Export plot & data** |

---

## Common Patterns

```python
# ── Beer-Lambert calibration ──────────────────────────────────────────────
result = fit_linear(concentration, absorbance)
molar_absorptivity = result["slope"] / path_length  # ε = slope / l

# ── Normalize a spectrum ──────────────────────────────────────────────────
spectrum_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

# ── Rolling average smoothing ─────────────────────────────────────────────
window = 5
smoothed = np.convolve(signal, np.ones(window) / window, mode="same")

# ── Multi-panel figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x, y1)
axes[1].scatter(x, y2)
fig.tight_layout()
save_figure(fig, "multipanel.png", dpi=300)

# ── Correlation matrix ────────────────────────────────────────────────────
corr = df.select_dtypes("number").corr().values
fig, ax = heatmap(corr, title="Correlation Matrix")
```

---

## Color-Blind-Safe Palettes

```python
# Available in the GUI Settings page and plot_config module
from eagleplot.utils.plot_config import COLOR_PALETTES

palettes = list(COLOR_PALETTES.keys())
# 'Default', 'Color-Blind Safe (Wong)', 'Color-Blind Safe (Okabe-Ito)',
# 'Muted (Tol)', 'Pastel', 'Vibrant'

colors = COLOR_PALETTES["Color-Blind Safe (Wong)"]
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: seaborn` | `pip install seaborn` |
| `streamlit: command not found` | `python -m streamlit run gui.py` |
| Port 8501 in use | `streamlit run gui.py --server.port 8502` |
| Click-to-annotate not working | `pip install --upgrade streamlit` (need ≥ 1.33) |
| Blank plot after widget change | Click **Generate Plot** again |
| Figure looks blurry | Use `dpi=300` in `save_figure()` |
