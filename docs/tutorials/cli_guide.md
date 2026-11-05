# CLI Tutorial — EaglePlot

This guide walks through all five CLI commands with copy-pasteable examples. No GUI or
Jupyter required — everything runs from a terminal.

---

## Prerequisites

```bash
# Activate your virtual environment first
# Windows:
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# Verify the CLI is available
eagleplot --help
```

---

## Command 1 — `plot`: Create a Plot from a Data File

The `plot` command reads a CSV (or any supported format) and produces a figure.

### Quickest example — scatter from CSV

```bash
eagleplot plot examples/data/experimental_data.csv \
    --plot scatter \
    --x-column x \
    --y-column y \
    --output output/scatter.png
```

| Option | Description |
| ------ | ----------- |
| `--plot` | Plot type: `line`, `scatter`, `histogram`, `bar`, `heatmap` |
| `--x-column` | Column name for the X axis |
| `--y-column` | Column name for the Y axis |
| `--output` | Output file path (`.png`, `.svg`, `.pdf`) |
| `--title` | Optional figure title |
| `--xlabel` / `--ylabel` | Optional axis labels |
| `--style` | Matplotlib style (e.g., `seaborn-v0_8`, `ggplot`) |
| `--dpi` | Resolution in dots per inch (default 100; use 300 for publication) |

### Publication-quality line plot

```bash
eagleplot plot examples/data/experimental_data.csv \
    --plot line \
    --x-column wavelength \
    --y-column absorbance \
    --title "UV-Vis Spectrum" \
    --xlabel "Wavelength (nm)" \
    --ylabel "Absorbance (a.u.)" \
    --dpi 300 \
    --output output/spectrum_300dpi.png
```

### Histogram of a single column

```bash
eagleplot plot examples/data/experimental_data.csv \
    --plot histogram \
    --y-column signal \
    --title "Signal Distribution" \
    --output output/histogram.png
```

---

## Command 2 — `stats`: Compute Summary Statistics

```bash
eagleplot stats examples/data/experimental_data.csv
```

Output example:
```
Statistics for experimental_data.csv
=====================================
  mean    : 2.4731
  median  : 2.3800
  std     : 0.8922
  min     : 0.8100
  max     : 4.9200
  q1      : 1.7800
  q3      : 3.1200
  iqr     : 1.3400
  range   : 4.1100
```

### Stats for a specific column

```bash
eagleplot stats examples/data/experimental_data.csv --column absorbance
```

### Normality test

```bash
eagleplot stats examples/data/experimental_data.csv --normality
```

---

## Command 3 — `convert`: Change File Formats

Convert between any of the eight supported formats:

```bash
# CSV  →  pickle (preserves dtypes)
eagleplot convert examples/data/experimental_data.csv output/data.pkl

# CSV  →  NumPy binary
eagleplot convert examples/data/experimental_data.csv output/data.npy

# NumPy  →  CSV
eagleplot convert examples/data/auto_test.npy output/array.csv

# Excel  →  Parquet (smaller, faster to load)
eagleplot convert examples/data/experimental_data.xlsx output/data.parquet
```

Supported extensions: `.csv`, `.xlsx`, `.tsv`, `.json`, `.npy`, `.npz`, `.pkl`, `.parquet`

---

## Command 4 — `compare`: Overlay Multiple Datasets

Compare two or more files on the same axes:

```bash
eagleplot compare \
    examples/data/experimental_data.csv \
    examples/data/md_analysis.csv \
    --plot line \
    --x-column time \
    --y-column energy \
    --output output/comparison.png
```

Each file becomes a separate series. Labels default to the filenames; use `--labels` to
override:

```bash
eagleplot compare file1.csv file2.csv \
    --plot scatter \
    --labels "Experiment A" "Experiment B" \
    --output output/comparison.png
```

---

## Command 5 — `batch`: Process Multiple Plots from a Config File

For reproducible multi-plot workflows, define all jobs in a JSON config file.

### Config file format (`batch_config.json`)

```json
{
    "output_directory": "output/batch_plots",
    "default_dpi": 150,
    "plots": [
        {
            "input_file": "examples/data/experimental_data.csv",
            "plot_type": "scatter",
            "x_column": "x",
            "y_column": "y",
            "title": "Scatter Plot",
            "output_file": "scatter.png"
        },
        {
            "input_file": "examples/data/experimental_data.csv",
            "plot_type": "histogram",
            "y_column": "signal",
            "title": "Signal Distribution",
            "output_file": "histogram.png",
            "dpi": 300
        }
    ]
}
```

### Running a batch job

```bash
eagleplot batch examples/batch_config.json
# With verbose output:
eagleplot batch examples/batch_config.json --verbose
```

A full example config is at [examples/batch_config.json](../../examples/batch_config.json).

---

## Global Flags

| Flag | Effect |
| ---- | ------ |
| `--verbose` / `-v` | Print detailed progress messages |
| `--quiet` / `-q` | Suppress all output except errors |
| `--examples` | Print usage examples and exit |
| `--version` | Print the current version and exit |

---

## Tips

- Paths can be relative (from the repo root) or absolute.
- All output directories are created automatically if they do not exist.
- Use `--dpi 300` for any figure destined for a publication or presentation.
- Combine `batch` + `compare` in a CI pipeline to auto-generate comparison plots on each run.
