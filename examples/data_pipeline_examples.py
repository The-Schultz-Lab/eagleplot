"""Examples for data I/O pipeline and format conversion.

Demonstrates the full load → process → save pipeline using all 8 formats
supported by eagleplot.io.

Run from the repo root:
    python examples/data_pipeline_examples.py
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleplot.io import load_data, save_data
from eagleplot.math import calculate_statistics
from eagleplot.plotting import histogram, save_figure

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECK = "[OK]"
FORMATS = [".csv", ".xlsx", ".tsv", ".json", ".npy", ".pkl", ".parquet"]


def make_sample_dataframe():
    """Return a small sample DataFrame for I/O testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "time": np.linspace(0, 10, 20),
            "signal": np.sin(np.linspace(0, 4 * np.pi, 20)) + np.random.randn(20) * 0.05,
            "temperature": 298 + np.random.randn(20) * 0.5,
            "label": ["run1"] * 10 + ["run2"] * 10,
        }
    )


def example_load_existing_data():
    """Example 1: Load the bundled sample datasets."""
    print("\n" + "=" * 70)
    print("Example 1: Loading Built-In Sample Data")
    print("=" * 70)

    files = {
        "CSV DataFrame": Path("examples/data/experimental_data.csv"),
        "NumPy array  ": Path("examples/data/auto_test.npy"),
    }

    for label, path in files.items():
        if path.exists():
            data = load_data(str(path))
            if isinstance(data, pd.DataFrame):
                print(f"  {label}: shape={data.shape}, cols={list(data.columns)}")
            elif isinstance(data, np.ndarray):
                print(f"  {label}: shape={data.shape}, dtype={data.dtype}")
            print(f"  {CHECK} {path.name} loaded")
        else:
            print(f"  (skipped — {path} not found)")


def example_format_roundtrips():
    """Example 2: Save and reload a DataFrame in all 7 formats."""
    print("\n" + "=" * 70)
    print("Example 2: Format Roundtrip Test")
    print("=" * 70)

    df = make_sample_dataframe()

    with tempfile.TemporaryDirectory() as tmp:
        for ext in FORMATS:
            filepath = Path(tmp) / f"test{ext}"
            try:
                if ext == ".npy":
                    save_data(df[["time", "signal"]].values, str(filepath))
                else:
                    save_data(df, str(filepath))
                load_data(str(filepath))
                size_kb = filepath.stat().st_size / 1024
                print(f"  {ext:<10} -> {size_kb:.1f} KB  {CHECK}")
            except Exception as e:
                print(f"  {ext:<10} -> ERROR: {e}")


def example_process_and_save():
    """Example 3: Full pipeline — load, process, save, plot."""
    print("\n" + "=" * 70)
    print("Example 3: Full Data Pipeline")
    print("=" * 70)

    # 1. Load
    src = Path("examples/data/experimental_data.csv")
    if not src.exists():
        print(f"  (skipped — {src} not found)")
        return

    df = load_data(str(src))
    print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # 2. Process — compute statistics on numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  Numeric columns: {list(numeric_cols)}")

    summary = {}
    for col in numeric_cols:
        s = calculate_statistics(df[col].dropna().values)
        summary[col] = {"mean": s["mean"], "std": s["std"], "range": s["range"]}
        print(f"  {col}: mean={s['mean']:.4f}, std={s['std']:.4f}")

    # 3. Plot first numeric column
    col = numeric_cols[0]
    fig, ax, info = histogram(
        df[col].dropna().values,
        bins=20,
        title=f"Distribution of {col}",
        xlabel=col,
        ylabel="Count",
    )

    out = OUTPUT_DIR / f"pipeline_histogram_{col}.png"
    save_figure(fig, out, dpi=150)
    print(f"  Figure saved: {out}")
    print(f"  {CHECK} Full pipeline complete")


if __name__ == "__main__":
    print("EaglePlot — Data Pipeline Examples")
    print("=" * 70)
    example_load_existing_data()
    example_format_roundtrips()
    example_process_and_save()
    print("\nAll data pipeline examples complete.")
