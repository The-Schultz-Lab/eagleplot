"""Tests for plottle/batch.py.

Covers scan_directory, batch_load_files, batch_statistics, batch_curve_fit,
and batch_peak_analysis.  No mocking — all tests operate on real data
(temporary files and inline NumPy/pandas objects).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plottle.batch import (
    scan_directory,
    batch_load_files,
    batch_statistics,
    batch_curve_fit,
    batch_peak_analysis,
)

# ── shared fixtures ────────────────────────────────────────────────────────────

_STAT_COLS = ["dataset", "column", "n", "mean", "median", "std",
              "min", "max", "q1", "q3", "iqr", "range"]


def _make_linear_df(n: int = 50, slope: float = 2.0, intercept: float = 1.0) -> pd.DataFrame:
    """Return a DataFrame with clean linear relationship: y = slope*x + intercept."""
    x = np.linspace(0, 10, n)
    y = slope * x + intercept
    return pd.DataFrame({"x": x, "y": y})


def _make_gaussian_signal(n: int = 200) -> pd.DataFrame:
    """Return a DataFrame with a clean Gaussian peak centred at x=50."""
    x = np.arange(n, dtype=float)
    y = np.exp(-0.5 * ((x - 50) / 5) ** 2)
    return pd.DataFrame({"x": x, "y": y})


def _make_multi_peak_signal(n: int = 500) -> pd.DataFrame:
    """Three Gaussian peaks at x = 100, 250, 400."""
    x = np.arange(n, dtype=float)
    y = (
        np.exp(-0.5 * ((x - 100) / 8) ** 2)
        + np.exp(-0.5 * ((x - 250) / 8) ** 2)
        + np.exp(-0.5 * ((x - 400) / 8) ** 2)
    )
    return pd.DataFrame({"x": x, "y": y})


# ══════════════════════════════════════════════════════════════════════════════
# scan_directory
# ══════════════════════════════════════════════════════════════════════════════

class TestScanDirectory:
    """Tests for scan_directory()."""

    def test_returns_paths_in_directory(self, tmp_path):
        (tmp_path / "a.csv").write_text("x,y\n1,2\n")
        (tmp_path / "b.csv").write_text("x,y\n3,4\n")
        results = scan_directory(str(tmp_path))
        assert len(results) == 2
        assert all(isinstance(p, Path) for p in results)

    def test_extension_filter_with_dot(self, tmp_path):
        (tmp_path / "data.csv").write_text("x\n1\n")
        (tmp_path / "data.npy").write_bytes(b"\x93NUMPY")
        (tmp_path / "readme.txt").write_text("hello")
        results = scan_directory(str(tmp_path), extensions=[".csv"])
        assert all(p.suffix == ".csv" for p in results)
        assert len(results) == 1

    def test_extension_filter_without_leading_dot(self, tmp_path):
        (tmp_path / "data.csv").write_text("x\n1\n")
        (tmp_path / "data.tsv").write_text("x\n1\n")
        results = scan_directory(str(tmp_path), extensions=["csv"])
        assert all(p.suffix == ".csv" for p in results)
        assert len(results) == 1

    def test_pattern_filter_fnmatch(self, tmp_path):
        (tmp_path / "exp_001.csv").write_text("x\n1\n")
        (tmp_path / "exp_002.csv").write_text("x\n1\n")
        (tmp_path / "control.csv").write_text("x\n1\n")
        results = scan_directory(str(tmp_path), pattern="exp_*")
        names = [p.name for p in results]
        assert "exp_001.csv" in names
        assert "exp_002.csv" in names
        assert "control.csv" not in names

    def test_raises_not_a_directory(self, tmp_path):
        fake_path = tmp_path / "nonexistent_dir"
        with pytest.raises(NotADirectoryError):
            scan_directory(str(fake_path))

    def test_empty_directory_returns_empty_list(self, tmp_path):
        results = scan_directory(str(tmp_path))
        assert results == []

    def test_results_are_sorted(self, tmp_path):
        for name in ["c.csv", "a.csv", "b.csv"]:
            (tmp_path / name).write_text("x\n1\n")
        results = scan_directory(str(tmp_path))
        names = [p.name for p in results]
        assert names == sorted(names)

    def test_extension_and_pattern_combined(self, tmp_path):
        (tmp_path / "exp_001.csv").write_text("x\n1\n")
        (tmp_path / "exp_001.xlsx").write_bytes(b"")
        (tmp_path / "ctrl.csv").write_text("x\n1\n")
        results = scan_directory(str(tmp_path), extensions=[".csv"], pattern="exp_*")
        assert len(results) == 1
        assert results[0].name == "exp_001.csv"


# ══════════════════════════════════════════════════════════════════════════════
# batch_load_files
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchLoadFiles:
    """Tests for batch_load_files()."""

    def _write_csv(self, path: Path, content: str) -> Path:
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_csv_files(self, tmp_path):
        p = self._write_csv(tmp_path / "data.csv", "x,y\n1,2\n3,4\n5,6\n")
        result = batch_load_files([p])
        assert len(result["datasets"]) == 1
        assert len(result["errors"]) == 0
        key = list(result["datasets"].keys())[0]
        loaded = result["datasets"][key]
        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == ["x", "y"]

    def test_on_error_skip_skips_bad_files(self, tmp_path):
        good = self._write_csv(tmp_path / "good.csv", "x,y\n1,2\n")
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\xff\xfe garbage \x00\x00")
        result = batch_load_files([good, bad], on_error="skip")
        # Good file loaded; bad file skipped, not raised
        assert len(result["datasets"]) >= 1
        assert len(result["errors"]) >= 1
        assert "bad.csv" in " ".join(result["errors"].keys())

    def test_on_error_raise_raises_on_bad_file(self, tmp_path):
        bad = tmp_path / "bad_format.xyz_unknown"
        bad.write_text("not parseable as anything useful")
        with pytest.raises(Exception):
            batch_load_files([bad], on_error="raise")

    def test_metadata_keys_present(self, tmp_path):
        p = self._write_csv(tmp_path / "meta.csv", "a,b\n1,2\n3,4\n")
        result = batch_load_files([p])
        assert len(result["metadata"]) == 1
        meta = list(result["metadata"].values())[0]
        assert "source" in meta
        assert "size_bytes" in meta
        assert "suffix" in meta
        assert "shape" in meta
        assert "columns" in meta

    def test_metadata_source_matches_path(self, tmp_path):
        p = self._write_csv(tmp_path / "check.csv", "a\n1\n2\n")
        result = batch_load_files([p])
        meta = list(result["metadata"].values())[0]
        assert str(p) in str(meta["source"]) or meta["source"] == str(p)

    def test_metadata_shape_correct(self, tmp_path):
        p = self._write_csv(tmp_path / "shape.csv", "a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        result = batch_load_files([p])
        meta = list(result["metadata"].values())[0]
        assert meta["shape"] == (3, 3)

    def test_metadata_suffix_correct(self, tmp_path):
        p = self._write_csv(tmp_path / "sfx.csv", "a\n1\n")
        result = batch_load_files([p])
        meta = list(result["metadata"].values())[0]
        assert meta["suffix"] == ".csv"

    def test_empty_list_returns_empty_dicts(self):
        result = batch_load_files([])
        assert result["datasets"] == {}
        assert result["errors"] == {}
        assert result["metadata"] == {}

    def test_multiple_csvs_all_loaded(self, tmp_path):
        for i in range(3):
            self._write_csv(tmp_path / f"file_{i}.csv", f"x\n{i}\n")
        paths = list(tmp_path.glob("*.csv"))
        result = batch_load_files(paths)
        assert len(result["datasets"]) == 3


# ══════════════════════════════════════════════════════════════════════════════
# batch_statistics
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchStatistics:
    """Tests for batch_statistics()."""

    def test_output_columns_present(self):
        datasets = {"ds1": _make_linear_df()}
        result = batch_statistics(datasets)
        for col in _STAT_COLS:
            assert col in result.columns, f"Missing output column: {col}"

    def test_two_dataframes_produces_rows(self):
        datasets = {
            "ds1": _make_linear_df(slope=1.0),
            "ds2": _make_linear_df(slope=3.0),
        }
        result = batch_statistics(datasets)
        assert len(result) > 0
        ds_names = result["dataset"].unique().tolist()
        assert "ds1" in ds_names
        assert "ds2" in ds_names

    def test_column_filter(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = batch_statistics({"ds": df}, columns=["a"])
        assert set(result["column"].unique()) == {"a"}

    def test_all_columns_when_no_filter(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = batch_statistics({"ds": df})
        cols = set(result["column"].unique())
        assert "a" in cols
        assert "b" in cols

    def test_non_dataframe_datasets_skipped(self):
        datasets = {
            "df": pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            "arr": np.array([1.0, 2.0, 3.0]),
        }
        result = batch_statistics(datasets)
        ds_names = result["dataset"].unique().tolist()
        assert "df" in ds_names
        assert "arr" not in ds_names

    def test_empty_datasets_returns_empty_dataframe(self):
        result = batch_statistics({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        for col in _STAT_COLS:
            assert col in result.columns

    def test_n_equals_non_nan_count(self):
        df = pd.DataFrame({"x": [1.0, 2.0, float("nan"), 4.0]})
        result = batch_statistics({"ds": df}, columns=["x"])
        row = result[result["column"] == "x"].iloc[0]
        assert row["n"] == 3

    def test_mean_value_correct(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = batch_statistics({"ds": df}, columns=["x"])
        row = result[result["column"] == "x"].iloc[0]
        assert row["mean"] == pytest.approx(3.0)

    def test_median_value_correct(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = batch_statistics({"ds": df}, columns=["x"])
        row = result[result["column"] == "x"].iloc[0]
        assert row["median"] == pytest.approx(3.0)

    def test_std_value_correct(self):
        arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        df = pd.DataFrame({"x": arr})
        result = batch_statistics({"ds": df}, columns=["x"])
        row = result[result["column"] == "x"].iloc[0]
        assert row["std"] == pytest.approx(float(arr.std(ddof=1)), rel=1e-4)

    def test_iqr_equals_q3_minus_q1(self):
        df = _make_linear_df()
        result = batch_statistics({"ds": df}, columns=["y"])
        row = result[result["column"] == "y"].iloc[0]
        assert row["iqr"] == pytest.approx(row["q3"] - row["q1"], abs=1e-9)

    def test_range_equals_max_minus_min(self):
        df = _make_linear_df()
        result = batch_statistics({"ds": df}, columns=["y"])
        row = result[result["column"] == "y"].iloc[0]
        assert row["range"] == pytest.approx(row["max"] - row["min"], abs=1e-9)

    def test_non_numeric_columns_skipped(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
        result = batch_statistics({"ds": df})
        cols = set(result["column"].unique())
        assert "label" not in cols


# ══════════════════════════════════════════════════════════════════════════════
# batch_curve_fit
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchCurveFit:
    """Tests for batch_curve_fit()."""

    def test_linear_fit_returns_required_columns(self):
        datasets = {"ds1": _make_linear_df()}
        result = batch_curve_fit(datasets, x_col="x", y_col="y", fit_type="linear")
        assert "dataset" in result.columns
        assert "slope" in result.columns
        assert "intercept" in result.columns
        assert "r_squared" in result.columns
        assert "p_value" in result.columns

    def test_linear_r_squared_close_to_1_for_clean_data(self):
        datasets = {"ds1": _make_linear_df(slope=3.0, intercept=0.5)}
        result = batch_curve_fit(datasets, x_col="x", y_col="y", fit_type="linear")
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert row["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_linear_slope_and_intercept_correct(self):
        datasets = {"ds1": _make_linear_df(slope=4.0, intercept=-2.0)}
        result = batch_curve_fit(datasets, x_col="x", y_col="y", fit_type="linear")
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert row["slope"] == pytest.approx(4.0, rel=1e-4)
        assert row["intercept"] == pytest.approx(-2.0, rel=1e-3)

    def test_polynomial_fit_returns_required_columns(self):
        datasets = {"ds1": _make_linear_df()}
        result = batch_curve_fit(datasets, x_col="x", y_col="y",
                                 fit_type="polynomial", degree=3)
        assert "degree" in result.columns
        assert "r_squared" in result.columns
        assert "coeff_0" in result.columns

    def test_polynomial_degree_recorded(self):
        datasets = {"ds1": _make_linear_df()}
        result = batch_curve_fit(datasets, x_col="x", y_col="y",
                                 fit_type="polynomial", degree=4)
        row = result.iloc[0]
        assert int(row["degree"]) == 4

    def test_polynomial_coefficient_columns_present(self):
        datasets = {"ds1": _make_linear_df()}
        degree = 3
        result = batch_curve_fit(datasets, x_col="x", y_col="y",
                                 fit_type="polynomial", degree=degree)
        for i in range(degree + 1):
            assert f"coeff_{i}" in result.columns

    def test_exponential_fit_returns_required_columns(self):
        x = np.linspace(0, 3, 50)
        y = 2.0 * np.exp(0.5 * x) + 0.1
        df = pd.DataFrame({"x": x, "y": y})
        datasets = {"ds1": df}
        result = batch_curve_fit(datasets, x_col="x", y_col="y", fit_type="exponential")
        assert "r_squared" in result.columns
        # a, b, c or similar amplitude/rate/offset columns
        assert len(result) == 1

    def test_missing_x_column_produces_error_row_or_skips(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        datasets = {"ds1": df}
        result = batch_curve_fit(datasets, x_col="nonexistent", y_col="y", fit_type="linear")
        # Either skipped (no row for ds1) or an error row with error info
        if len(result) > 0:
            row = result.iloc[0]
            assert (
                "error" in row.index
                or "r_squared" not in result.columns
                or pd.isna(row.get("r_squared"))
            )

    def test_missing_y_column_produces_error_row_or_skips(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        datasets = {"ds1": df}
        result = batch_curve_fit(datasets, x_col="x", y_col="nonexistent", fit_type="linear")
        if len(result) > 0:
            row = result.iloc[0]
            assert "error" in row.index or pd.isna(row.get("r_squared"))

    def test_non_dataframe_datasets_skipped(self):
        datasets = {
            "df": _make_linear_df(),
            "arr": np.array([1.0, 2.0, 3.0]),
        }
        result = batch_curve_fit(datasets, x_col="x", y_col="y", fit_type="linear")
        ds_names = result["dataset"].tolist() if "dataset" in result.columns else []
        assert "arr" not in ds_names

    def test_invalid_fit_type_produces_error_row(self):
        datasets = {"ds1": _make_linear_df()}
        result = batch_curve_fit(datasets, x_col="x", y_col="y", fit_type="quadratic_invalid")
        # Should produce an error row, not raise
        assert len(result) == 1
        row = result.iloc[0]
        assert "error" in row.index

    def test_multiple_datasets_all_present_in_output(self):
        datasets = {
            "ds1": _make_linear_df(slope=1.0),
            "ds2": _make_linear_df(slope=2.0),
            "ds3": _make_linear_df(slope=3.0),
        }
        result = batch_curve_fit(datasets, x_col="x", y_col="y", fit_type="linear")
        ds_in_output = set(result["dataset"].tolist())
        assert {"ds1", "ds2", "ds3"} == ds_in_output


# ══════════════════════════════════════════════════════════════════════════════
# batch_peak_analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchPeakAnalysis:
    """Tests for batch_peak_analysis()."""

    def test_finds_peaks_in_clean_gaussian_signal(self):
        datasets = {"ds1": _make_gaussian_signal()}
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.1)
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert row["n_peaks"] >= 1

    def test_n_peaks_correct_single_peak(self):
        datasets = {"ds1": _make_gaussian_signal()}
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.5)
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert row["n_peaks"] == 1

    def test_n_peaks_correct_multi_peak(self):
        datasets = {"ds1": _make_multi_peak_signal()}
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.5, distance=50)
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert row["n_peaks"] == 3

    def test_peak_positions_non_empty_string_for_found_peaks(self):
        datasets = {"ds1": _make_gaussian_signal()}
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.5)
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert isinstance(row["peak_positions"], str)
        assert len(row["peak_positions"]) > 0

    def test_peak_positions_empty_string_when_no_peaks(self):
        flat = pd.DataFrame({"y": np.ones(100)})
        datasets = {"ds1": flat}
        result = batch_peak_analysis(datasets, y_col="y", height=2.0)
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert row["n_peaks"] == 0
        assert row["peak_positions"] == ""

    def test_with_x_col_provided(self):
        datasets = {"ds1": _make_gaussian_signal()}
        result = batch_peak_analysis(datasets, y_col="y", x_col="x", prominence=0.5)
        row = result[result["dataset"] == "ds1"].iloc[0]
        assert row["n_peaks"] >= 1
        # peak_positions should reflect x values (floats near 50)
        positions_str = row["peak_positions"]
        assert len(positions_str) > 0

    def test_without_x_col_uses_indices(self):
        datasets = {"ds1": _make_gaussian_signal()}
        result_no_x = batch_peak_analysis(datasets, y_col="y", x_col=None, prominence=0.5)
        result_with_x = batch_peak_analysis(datasets, y_col="y", x_col="x", prominence=0.5)
        # Both should find at least one peak
        assert result_no_x.iloc[0]["n_peaks"] >= 1
        assert result_with_x.iloc[0]["n_peaks"] >= 1

    def test_output_columns_present(self):
        datasets = {"ds1": _make_gaussian_signal()}
        result = batch_peak_analysis(datasets, y_col="y")
        expected = [
            "dataset",
            "n_peaks",
            "peak_positions",
            "mean_height",
            "max_height",
            "mean_fwhm",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_non_dataframe_datasets_skipped(self):
        datasets = {
            "df": _make_gaussian_signal(),
            "arr": np.array([1.0, 2.0, 3.0]),
        }
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.5)
        ds_names = result["dataset"].tolist()
        assert "arr" not in ds_names

    def test_missing_y_col_produces_error_row_or_skips(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        datasets = {"ds1": df}
        result = batch_peak_analysis(datasets, y_col="nonexistent")
        # Should not raise; either skipped or error row
        assert isinstance(result, pd.DataFrame)

    def test_max_height_gte_mean_height_when_peaks_found(self):
        datasets = {"ds1": _make_multi_peak_signal()}
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.5, distance=50)
        row = result[result["dataset"] == "ds1"].iloc[0]
        if row["n_peaks"] > 0:
            assert row["max_height"] >= row["mean_height"]

    def test_mean_height_approximately_correct(self):
        """Single Gaussian peak height should be ~1.0."""
        datasets = {"ds1": _make_gaussian_signal()}
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.5)
        row = result[result["dataset"] == "ds1"].iloc[0]
        if row["n_peaks"] == 1:
            assert row["max_height"] == pytest.approx(1.0, abs=0.01)

    def test_multiple_datasets_all_in_output(self):
        datasets = {
            "g1": _make_gaussian_signal(),
            "g2": _make_multi_peak_signal(),
        }
        result = batch_peak_analysis(datasets, y_col="y", prominence=0.5, distance=50)
        ds_names = set(result["dataset"].tolist())
        assert {"g1", "g2"} == ds_names

    def test_error_column_present_in_output(self):
        """The 'error' column should always be present (empty string for success)."""
        datasets = {"ds1": _make_gaussian_signal()}
        result = batch_peak_analysis(datasets, y_col="y")
        assert "error" in result.columns
