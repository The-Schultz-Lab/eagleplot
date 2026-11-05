"""Batch processing utilities for EaglePlot.

Provides helpers for scanning directories, loading multiple files in bulk,
and running common analyses (statistics, curve fitting, peak detection) across
entire datasets in one call.

No Streamlit dependency — safe to import from CLI or notebooks.

Public API
----------
scan_directory(folder_path, extensions=None, pattern=None) -> list[Path]
    Return sorted paths matching the given criteria.
batch_load_files(file_paths, on_error='skip') -> dict
    Load many files; returns datasets, errors, and metadata.
batch_statistics(datasets, columns=None) -> pd.DataFrame
    Descriptive statistics across all DataFrame datasets.
batch_curve_fit(datasets, x_col, y_col, fit_type='linear', degree=2) -> pd.DataFrame
    Curve fit one x/y column pair across all DataFrame datasets.
batch_peak_analysis(datasets, y_col, x_col=None, ...) -> pd.DataFrame
    Find peaks in y_col across all DataFrame datasets.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eagleplot.io import load_data
from eagleplot.math import (
    calculate_statistics,
    fit_exponential,
    fit_linear,
    fit_polynomial,
)
from eagleplot.peaks import compute_fwhm, find_peaks

# ── Constants ─────────────────────────────────────────────────────────────────

_SUPPORTED_EXTENSIONS: set[str] = {
    ".pkl",
    ".npy",
    ".npz",
    ".csv",
    ".xlsx",
    ".xls",
    ".tsv",
    ".json",
    ".parquet",
    ".jdx",
    ".dx",
    ".h5",
    ".hdf5",
    ".nc",
    ".cdf",
    ".spc",
    ".asc",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalise_extensions(extensions: list[str] | None) -> set[str] | None:
    """Return a set of lower-case dot-prefixed extensions, or None for 'all'."""
    if extensions is None:
        return None
    normalised = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext.startswith("."):
            ext = "." + ext
        normalised.add(ext)
    return normalised


def _file_metadata(path: Path, data: Any) -> dict:
    """Return a metadata dict for a single loaded file."""
    meta: dict[str, Any] = {
        "source": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "suffix": path.suffix.lower(),
    }
    if isinstance(data, pd.DataFrame):
        meta["shape"] = data.shape
        meta["columns"] = list(data.columns)
    elif isinstance(data, np.ndarray):
        meta["shape"] = data.shape
        meta["dtype"] = str(data.dtype)
    return meta


# ── Public API ────────────────────────────────────────────────────────────────


def scan_directory(
    folder_path: str | Path,
    extensions: list[str] | None = None,
    pattern: str | None = None,
) -> list[Path]:
    """Scan a directory and return sorted matching file paths.

    Args:
        folder_path: Path to the directory to scan.
        extensions: List of file extensions to include, e.g. ``['csv', 'xlsx']``.
            Leading dots are optional. Pass ``None`` (default) to accept all
            EaglePlot-supported formats.
        pattern: Optional ``fnmatch``-style filename filter, e.g.
            ``'sample_*.csv'``.  Applied after the extension filter.

    Returns:
        Sorted list of ``Path`` objects for every matching file.

    Raises:
        NotADirectoryError: If ``folder_path`` is not a directory.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    allowed_exts = _normalise_extensions(extensions)
    if allowed_exts is None:
        allowed_exts = _SUPPORTED_EXTENSIONS

    results: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed_exts:
            continue
        if pattern is not None and not fnmatch.fnmatch(p.name, pattern):
            continue
        results.append(p)

    return sorted(results)


def batch_load_files(
    file_paths: list[str | Path],
    on_error: str = "skip",
) -> dict:
    """Load multiple files and return a structured result dict.

    Args:
        file_paths: Iterable of file paths to load.
        on_error: What to do when a file fails to load.
            ``'skip'`` silently records the error and continues;
            ``'raise'`` re-raises the exception immediately.

    Returns:
        A dict with three keys:

        - ``'datasets'``: ``{name: data}`` — successfully loaded objects.
        - ``'errors'``: ``{name: error_message}`` — files that failed.
        - ``'metadata'``: ``{name: {...}}`` — per-file metadata (source path,
          size_bytes, suffix; plus shape/columns for DataFrames and
          shape/dtype for ndarrays).

        The *name* key is the bare filename (e.g. ``'data.csv'``).
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    datasets: dict[str, Any] = {}
    errors: dict[str, str] = {}
    metadata: dict[str, dict] = {}

    for raw_path in file_paths:
        path = Path(raw_path)
        name = path.name
        try:
            data = load_data(str(path))
            datasets[name] = data
            metadata[name] = _file_metadata(path, data)
        except Exception as exc:  # noqa: BLE001
            if on_error == "raise":
                raise
            errors[name] = str(exc)

    return {"datasets": datasets, "errors": errors, "metadata": metadata}


def batch_statistics(
    datasets: dict[str, Any],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute descriptive statistics across multiple datasets.

    Only ``pd.DataFrame`` datasets are processed; others are silently skipped.

    Args:
        datasets: Mapping of dataset name to loaded data object.
        columns: Column names to analyse. Pass ``None`` (default) to use all
            numeric columns in each DataFrame.

    Returns:
        A ``pd.DataFrame`` with one row per (dataset, column) combination and
        columns: ``dataset, column, n, mean, median, std, min, max, q1, q3,
        iqr, range``.  Returns an empty DataFrame with those columns if there
        is nothing to compute.
    """
    _COLS = [
        "dataset",
        "column",
        "n",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "q1",
        "q3",
        "iqr",
        "range",
    ]
    rows: list[dict] = []

    for name, data in datasets.items():
        if not isinstance(data, pd.DataFrame):
            continue

        df: pd.DataFrame = data
        if columns is not None:
            target_cols = [c for c in columns if c in df.columns]
        else:
            target_cols = list(df.select_dtypes(include="number").columns)

        for col in target_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            arr = series.to_numpy(dtype=float)
            stats = calculate_statistics(arr)
            rows.append(
                {
                    "dataset": name,
                    "column": col,
                    "n": int(len(arr)),
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "q1": stats["q1"],
                    "q3": stats["q3"],
                    "iqr": stats["iqr"],
                    "range": stats["range"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows, columns=_COLS)


def batch_curve_fit(
    datasets: dict[str, Any],
    x_col: str,
    y_col: str,
    fit_type: str = "linear",
    degree: int = 2,
) -> pd.DataFrame:
    """Fit a curve to x_col vs y_col across multiple datasets.

    Only ``pd.DataFrame`` datasets that contain both ``x_col`` and ``y_col``
    are processed; others are silently skipped.

    Args:
        datasets: Mapping of dataset name to loaded data object.
        x_col: Name of the independent-variable column.
        y_col: Name of the dependent-variable column.
        fit_type: One of ``'linear'``, ``'polynomial'``, or
            ``'exponential'``.
        degree: Polynomial degree; only used when ``fit_type='polynomial'``.

    Returns:
        A ``pd.DataFrame`` with one row per dataset.  Columns depend on
        ``fit_type``:

        - *linear*: ``dataset, slope, intercept, r_squared, p_value, error``
        - *polynomial*: ``dataset, degree, r_squared, coeff_0 … coeff_N, error``
        - *exponential*: ``dataset, a, b, c, r_squared, error``

        The ``error`` column is ``None`` on success, or an error message string
        on failure.  Returns an empty ``pd.DataFrame`` if nothing could be
        computed.
    """
    rows: list[dict] = []

    for name, data in datasets.items():
        if not isinstance(data, pd.DataFrame):
            continue
        df: pd.DataFrame = data
        if x_col not in df.columns or y_col not in df.columns:
            continue

        sub = df[[x_col, y_col]].dropna()
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        row: dict[str, Any] = {"dataset": name, "error": None}

        try:
            if fit_type == "linear":
                result = fit_linear(x, y)
                row.update(
                    {
                        "slope": result["slope"],
                        "intercept": result["intercept"],
                        "r_squared": result["r_squared"],
                        "p_value": result["p_value"],
                    }
                )
            elif fit_type == "polynomial":
                result = fit_polynomial(x, y, degree)
                row["degree"] = degree
                row["r_squared"] = result["r_squared"]
                for i, coeff in enumerate(result["coefficients"]):  # type: ignore[var-annotated,arg-type]
                    row[f"coeff_{i}"] = coeff
            elif fit_type == "exponential":
                result = fit_exponential(x, y)
                row.update(
                    {
                        "a": result["a"],
                        "b": result["b"],
                        "c": result["c"],
                        "r_squared": result["r_squared"],
                    }
                )
            else:
                row["error"] = f"Unknown fit_type: {fit_type!r}"
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def batch_peak_analysis(
    datasets: dict[str, Any],
    y_col: str,
    x_col: str | None = None,
    height: float | None = None,
    prominence: float | None = None,
    distance: int | None = None,
) -> pd.DataFrame:
    """Find peaks in y_col across multiple datasets.

    Only ``pd.DataFrame`` datasets that contain ``y_col`` are processed;
    others are silently skipped.  If ``x_col`` is not given, integer row
    indices are used as the x axis.

    Args:
        datasets: Mapping of dataset name to loaded data object.
        y_col: Name of the column containing the signal.
        x_col: Name of the x-axis column.  Defaults to integer indices.
        height: Minimum absolute peak height passed to ``find_peaks``.
        prominence: Minimum peak prominence passed to ``find_peaks``.
        distance: Minimum sample distance between peaks passed to
            ``find_peaks``.

    Returns:
        A ``pd.DataFrame`` with columns: ``dataset, n_peaks,
        peak_positions, mean_height, max_height, mean_fwhm, error``.

        - ``peak_positions``: comma-separated string of x values.
        - ``mean_fwhm``: ``NaN`` if FWHM could not be computed.
        - ``error``: ``None`` on success, or an error message string.

        Returns an empty ``pd.DataFrame`` with those columns if nothing could
        be computed.
    """
    _COLS = [
        "dataset",
        "n_peaks",
        "peak_positions",
        "mean_height",
        "max_height",
        "mean_fwhm",
        "error",
    ]
    rows: list[dict] = []

    for name, data in datasets.items():
        if not isinstance(data, pd.DataFrame):
            continue
        df: pd.DataFrame = data
        if y_col not in df.columns:
            continue

        row: dict[str, Any] = {
            "dataset": name,
            "n_peaks": 0,
            "peak_positions": "",
            "mean_height": float("nan"),
            "max_height": float("nan"),
            "mean_fwhm": float("nan"),
            "error": None,
        }

        try:
            y = df[y_col].to_numpy(dtype=float)
            if x_col is not None and x_col in df.columns:
                x = df[x_col].to_numpy(dtype=float)
            else:
                x = np.arange(len(y), dtype=float)

            peaks_dict = find_peaks(
                y,
                x=x,
                height=height,
                prominence=prominence,
                distance=distance,
            )

            n = peaks_dict["n_peaks"]
            row["n_peaks"] = n

            if n > 0:
                positions = peaks_dict["positions"]
                heights = peaks_dict["heights"]
                row["peak_positions"] = ", ".join(f"{p:.6g}" for p in positions)
                row["mean_height"] = float(np.mean(heights))
                row["max_height"] = float(np.max(heights))

                try:
                    fwhm_dict = compute_fwhm(y, x, peaks_dict)
                    fwhm_vals = fwhm_dict["fwhm"]
                    if len(fwhm_vals) > 0:
                        row["mean_fwhm"] = float(np.mean(fwhm_vals))
                except Exception:  # noqa: BLE001
                    pass  # mean_fwhm stays NaN

        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows, columns=_COLS)
