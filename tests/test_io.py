"""Unit tests for the io module.

This module tests all data input/output functions including:
- Pickle operations
- NumPy array operations
- Pandas DataFrame operations
- Universal data loader/saver
"""

import pytest
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Import functions to test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from plottle.io import (
    load_pickle,
    save_pickle,
    load_numpy,
    save_numpy,
    load_dataframe,
    save_dataframe,
    load_data,
    save_data,
    load_jcamp,
    load_hdf5,
    load_netcdf,
    load_spc,
    load_asc,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_dict():
    """Sample dictionary for testing."""
    return {"name": "experiment_1", "temperature": 300.0, "measurements": [1.0, 2.0, 3.0, 4.0, 5.0]}


@pytest.fixture
def sample_array():
    """Sample NumPy array for testing."""
    return np.random.rand(10, 3)


@pytest.fixture
def sample_dataframe():
    """Sample Pandas DataFrame for testing."""
    return pd.DataFrame(
        {"x": np.arange(10), "y": np.arange(10) ** 2, "label": ["A"] * 5 + ["B"] * 5}
    )


# ============================================================================
# Pickle Operations Tests
# ============================================================================


class TestPickleOperations:
    """Tests for pickle load/save functions."""

    def test_save_and_load_dict(self, temp_dir, sample_dict):
        """Test saving and loading a dictionary."""
        filepath = temp_dir / "test.pkl"
        save_pickle(sample_dict, filepath)
        loaded = load_pickle(filepath)
        assert loaded == sample_dict

    def test_save_and_load_list(self, temp_dir):
        """Test saving and loading a list."""
        data = [1, 2, 3, "test", {"nested": True}]
        filepath = temp_dir / "list.pkl"
        save_pickle(data, filepath)
        loaded = load_pickle(filepath)
        assert loaded == data

    def test_save_and_load_numpy_array(self, temp_dir, sample_array):
        """Test pickling NumPy arrays."""
        filepath = temp_dir / "array.pkl"
        save_pickle(sample_array, filepath)
        loaded = load_pickle(filepath)
        np.testing.assert_array_equal(loaded, sample_array)

    def test_save_creates_directories(self, temp_dir):
        """Test that save_pickle creates parent directories."""
        filepath = temp_dir / "subdir" / "nested" / "test.pkl"
        data = {"test": "data"}
        save_pickle(data, filepath)
        assert filepath.exists()
        loaded = load_pickle(filepath)
        assert loaded == data

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a file that doesn't exist."""
        filepath = temp_dir / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError):
            load_pickle(filepath)

    def test_load_corrupted_pickle(self, temp_dir):
        """Test loading a corrupted pickle file."""
        filepath = temp_dir / "corrupted.pkl"
        # Create a file with invalid pickle data
        with open(filepath, "wb") as f:
            f.write(b"not a pickle file")
        with pytest.raises(pickle.UnpicklingError):
            load_pickle(filepath)


# ============================================================================
# NumPy Operations Tests
# ============================================================================


class TestNumpyOperations:
    """Tests for NumPy array load/save functions."""

    def test_save_and_load_npy(self, temp_dir, sample_array):
        """Test saving and loading .npy files."""
        filepath = temp_dir / "array.npy"
        save_numpy(sample_array, filepath)
        loaded = load_numpy(filepath)
        np.testing.assert_array_equal(loaded, sample_array)

    def test_save_and_load_npz_single_array(self, temp_dir, sample_array):
        """Test saving single array as .npz."""
        filepath = temp_dir / "array.npz"
        save_numpy(sample_array, filepath)
        loaded = load_numpy(filepath)
        np.testing.assert_array_equal(loaded["arr_0"], sample_array)

    def test_save_and_load_npz_multiple_arrays(self, temp_dir):
        """Test saving multiple arrays as .npz."""
        arrays = {"x": np.arange(10), "y": np.arange(10) ** 2, "z": np.random.rand(5, 3)}
        filepath = temp_dir / "arrays.npz"
        save_numpy(arrays, filepath)
        loaded = load_numpy(filepath)

        for key in arrays:
            np.testing.assert_array_equal(loaded[key], arrays[key])

    def test_save_npz_compressed(self, temp_dir, sample_array):
        """Test compressed .npz saving."""
        filepath = temp_dir / "compressed.npz"
        save_numpy(sample_array, filepath, compressed=True)
        loaded = load_numpy(filepath)
        np.testing.assert_array_equal(loaded["arr_0"], sample_array)

    def test_save_npy_with_wrong_type(self, temp_dir):
        """Test that saving non-array to .npy raises error."""
        filepath = temp_dir / "wrong.npy"
        with pytest.raises(TypeError):
            save_numpy({"not": "array"}, filepath)

    def test_load_nonexistent_numpy_file(self, temp_dir):
        """Test loading nonexistent NumPy file."""
        filepath = temp_dir / "nonexistent.npy"
        with pytest.raises(FileNotFoundError):
            load_numpy(filepath)

    def test_unsupported_numpy_format(self, temp_dir, sample_array):
        """Test that unsupported formats raise errors."""
        filepath = temp_dir / "array.txt"
        with pytest.raises(ValueError):
            save_numpy(sample_array, filepath)


# ============================================================================
# Pandas DataFrame Tests
# ============================================================================


class TestDataFrameOperations:
    """Tests for Pandas DataFrame load/save functions."""

    def test_save_and_load_csv(self, temp_dir, sample_dataframe):
        """Test saving and loading CSV files."""
        filepath = temp_dir / "data.csv"
        save_dataframe(sample_dataframe, filepath)
        loaded = load_dataframe(filepath)
        pd.testing.assert_frame_equal(loaded, sample_dataframe, check_dtype=False)

    def test_save_csv_with_index(self, temp_dir, sample_dataframe):
        """Test saving CSV with index."""
        filepath = temp_dir / "data_with_index.csv"
        save_dataframe(sample_dataframe, filepath, index=True)
        loaded = load_dataframe(filepath, index_col=0)
        pd.testing.assert_frame_equal(loaded, sample_dataframe, check_dtype=False)

    def test_save_and_load_excel(self, temp_dir, sample_dataframe):
        """Test saving and loading Excel files."""
        filepath = temp_dir / "data.xlsx"
        save_dataframe(sample_dataframe, filepath)
        loaded = load_dataframe(filepath)
        pd.testing.assert_frame_equal(loaded, sample_dataframe, check_dtype=False)

    def test_save_and_load_json(self, temp_dir, sample_dataframe):
        """Test saving and loading JSON files."""
        filepath = temp_dir / "data.json"
        save_dataframe(sample_dataframe, filepath)
        loaded = load_dataframe(filepath)
        pd.testing.assert_frame_equal(loaded, sample_dataframe, check_dtype=False)

    def test_save_and_load_tsv(self, temp_dir, sample_dataframe):
        """Test saving and loading TSV files."""
        filepath = temp_dir / "data.tsv"
        save_dataframe(sample_dataframe, filepath)
        loaded = load_dataframe(filepath)
        pd.testing.assert_frame_equal(loaded, sample_dataframe, check_dtype=False)

    def test_save_and_load_parquet(self, temp_dir, sample_dataframe):
        """Test saving and loading Parquet files."""
        filepath = temp_dir / "data.parquet"
        save_dataframe(sample_dataframe, filepath)
        loaded = load_dataframe(filepath)
        pd.testing.assert_frame_equal(loaded, sample_dataframe)

    def test_save_non_dataframe(self, temp_dir):
        """Test that saving non-DataFrame raises error."""
        filepath = temp_dir / "wrong.csv"
        with pytest.raises(TypeError):
            save_dataframe([1, 2, 3], filepath)

    def test_load_nonexistent_csv(self, temp_dir):
        """Test loading nonexistent CSV."""
        filepath = temp_dir / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_dataframe(filepath)

    def test_unsupported_dataframe_format(self, temp_dir, sample_dataframe):
        """Test that unsupported formats raise errors."""
        filepath = temp_dir / "data.xyz"
        with pytest.raises(ValueError):
            save_dataframe(sample_dataframe, filepath)


# ============================================================================
# Universal Loader/Saver Tests
# ============================================================================


class TestUniversalDataOperations:
    """Tests for universal load_data and save_data functions."""

    def test_load_data_pickle(self, temp_dir, sample_dict):
        """Test universal loader with pickle file."""
        filepath = temp_dir / "test.pkl"
        save_data(sample_dict, filepath)
        loaded = load_data(filepath)
        assert loaded == sample_dict

    def test_load_data_numpy(self, temp_dir, sample_array):
        """Test universal loader with NumPy file."""
        filepath = temp_dir / "test.npy"
        save_data(sample_array, filepath)
        loaded = load_data(filepath)
        np.testing.assert_array_equal(loaded, sample_array)

    def test_load_data_csv(self, temp_dir, sample_dataframe):
        """Test universal loader with CSV file."""
        filepath = temp_dir / "test.csv"
        save_data(sample_dataframe, filepath)
        loaded = load_data(filepath)
        pd.testing.assert_frame_equal(loaded, sample_dataframe, check_dtype=False)

    def test_save_data_type_mismatch(self, temp_dir, sample_dict):
        """Test that type mismatches raise appropriate errors."""
        filepath = temp_dir / "wrong.npy"
        with pytest.raises(TypeError):
            save_data(sample_dict, filepath)

    def test_load_unsupported_format(self, temp_dir):
        """Test loading unsupported format."""
        filepath = temp_dir / "data.unknown"
        filepath.touch()  # Create empty file
        with pytest.raises(ValueError):
            load_data(filepath)

    def test_save_unsupported_format(self, temp_dir, sample_dict):
        """Test saving to unsupported format."""
        filepath = temp_dir / "data.unknown"
        with pytest.raises(ValueError):
            save_data(sample_dict, filepath)

    def test_roundtrip_all_formats(self, temp_dir, sample_dict, sample_array, sample_dataframe):
        """Test save/load roundtrip for all supported formats."""
        # Pickle
        pkl_path = temp_dir / "data.pkl"
        save_data(sample_dict, pkl_path)
        assert load_data(pkl_path) == sample_dict

        # NumPy
        npy_path = temp_dir / "data.npy"
        save_data(sample_array, npy_path)
        np.testing.assert_array_equal(load_data(npy_path), sample_array)

        # CSV
        csv_path = temp_dir / "data.csv"
        save_data(sample_dataframe, csv_path)
        pd.testing.assert_frame_equal(load_data(csv_path), sample_dataframe, check_dtype=False)

        # TSV
        tsv_path = temp_dir / "data.tsv"
        save_data(sample_dataframe, tsv_path)
        pd.testing.assert_frame_equal(load_data(tsv_path), sample_dataframe, check_dtype=False)

        # Parquet
        parquet_path = temp_dir / "data.parquet"
        save_data(sample_dataframe, parquet_path)
        pd.testing.assert_frame_equal(load_data(parquet_path), sample_dataframe)


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_empty_dataframe(self, temp_dir):
        """Test saving and loading empty DataFrame."""
        # Create a DataFrame with columns but no rows
        df = pd.DataFrame(columns=["A", "B", "C"])
        filepath = temp_dir / "empty.csv"
        save_dataframe(df, filepath)
        loaded = load_dataframe(filepath)
        assert loaded.empty
        assert list(loaded.columns) == ["A", "B", "C"]

    def test_very_large_array(self, temp_dir):
        """Test with large NumPy array."""
        large_array = np.random.rand(1000, 1000)
        filepath = temp_dir / "large.npy"
        save_numpy(large_array, filepath)
        loaded = load_numpy(filepath)
        np.testing.assert_array_equal(loaded, large_array)

    def test_special_characters_in_path(self, temp_dir):
        """Test with special characters in filename."""
        data = {"test": "data"}
        filepath = temp_dir / "file with spaces & special-chars.pkl"
        save_pickle(data, filepath)
        loaded = load_pickle(filepath)
        assert loaded == data

    def test_unicode_data(self, temp_dir):
        """Test with unicode characters in data."""
        data = {"name": "Ångström", "symbol": "Å", "greek": "αβγδε"}
        filepath = temp_dir / "unicode.pkl"
        save_pickle(data, filepath)
        loaded = load_pickle(filepath)
        assert loaded == data


# ============================================================================
# Instrument File Format Tests (M17)
# ============================================================================

# Minimal valid JCAMP-DX string for synthetic test files
_JCAMP_CONTENT = """\
##TITLE= Test IR Spectrum
##JCAMP-DX= 4.24
##DATA TYPE= INFRARED SPECTRUM
##XUNITS= 1/CM
##YUNITS= ABSORBANCE
##FIRSTX= 400.0
##LASTX= 1300.0
##NPOINTS= 4
##DELTAX= 300.0
##XFACTOR= 1.0
##YFACTOR= 1.0
##XYDATA= (X++(Y..Y))
400.0 0.05 0.35 0.20 0.10
##END=
"""


class TestJCAMPOperations:
    """Tests for JCAMP-DX loader. Skipped if jcamp is not installed."""

    @pytest.fixture(autouse=True)
    def _require_jcamp(self):
        pytest.importorskip("jcamp")

    def test_load_jcamp_returns_dataframe(self, temp_dir):
        """load_jcamp returns a two-column DataFrame."""
        p = temp_dir / "spectrum.jdx"
        p.write_text(_JCAMP_CONTENT, encoding="utf-8")
        df = load_jcamp(p)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]

    def test_load_jcamp_correct_shape(self, temp_dir):
        """DataFrame has one row per data point."""
        p = temp_dir / "spectrum.jdx"
        p.write_text(_JCAMP_CONTENT, encoding="utf-8")
        df = load_jcamp(p)
        assert len(df) == 4

    def test_load_jcamp_attrs(self, temp_dir):
        """Metadata is stored in df.attrs."""
        p = temp_dir / "spectrum.jdx"
        p.write_text(_JCAMP_CONTENT, encoding="utf-8")
        df = load_jcamp(p)
        assert "xunits" in df.attrs or "title" in df.attrs

    def test_load_jcamp_via_load_data(self, temp_dir):
        """load_data dispatches .jdx to load_jcamp."""
        p = temp_dir / "spectrum.jdx"
        p.write_text(_JCAMP_CONTENT, encoding="utf-8")
        df = load_data(p)
        assert isinstance(df, pd.DataFrame)

    def test_load_jcamp_dx_extension(self, temp_dir):
        """load_data also dispatches .dx extension."""
        p = temp_dir / "spectrum.dx"
        p.write_text(_JCAMP_CONTENT, encoding="utf-8")
        df = load_data(p)
        assert isinstance(df, pd.DataFrame)

    def test_load_jcamp_file_not_found(self, temp_dir):
        """FileNotFoundError raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_jcamp(temp_dir / "missing.jdx")


class TestHDF5Operations:
    """Tests for HDF5 loader. Skipped if h5py is not installed."""

    @pytest.fixture(autouse=True)
    def _require_h5py(self):
        pytest.importorskip("h5py")

    def _write_hdf5(self, path):
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("x", data=np.arange(10, dtype=float))
            f.create_dataset("y", data=np.arange(10, dtype=float) ** 2)
            grp = f.create_group("meta")
            grp.create_dataset("label", data=np.array([1.0, 2.0]))
        return path

    def test_load_hdf5_returns_dict(self, temp_dir):
        """load_hdf5 returns a dict of arrays."""
        p = self._write_hdf5(temp_dir / "data.h5")
        result = load_hdf5(p)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result

    def test_load_hdf5_array_values(self, temp_dir):
        """Loaded arrays match what was written."""
        p = self._write_hdf5(temp_dir / "data.h5")
        result = load_hdf5(p)
        np.testing.assert_array_equal(result["x"], np.arange(10, dtype=float))

    def test_load_hdf5_nested_group(self, temp_dir):
        """Nested group datasets are flattened with slash paths."""
        p = self._write_hdf5(temp_dir / "data.h5")
        result = load_hdf5(p)
        assert "meta/label" in result

    def test_load_hdf5_single_dataset(self, temp_dir):
        """dataset= kwarg returns a single ndarray."""
        p = self._write_hdf5(temp_dir / "data.h5")
        arr = load_hdf5(p, dataset="x")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 10

    def test_load_hdf5_missing_dataset_key(self, temp_dir):
        """KeyError raised for unknown dataset path."""
        p = self._write_hdf5(temp_dir / "data.h5")
        with pytest.raises(KeyError):
            load_hdf5(p, dataset="nonexistent")

    def test_load_hdf5_via_load_data(self, temp_dir):
        """load_data dispatches .h5 and .hdf5."""
        p = self._write_hdf5(temp_dir / "data.h5")
        result = load_data(p)
        assert isinstance(result, dict)

    def test_load_hdf5_file_not_found(self, temp_dir):
        """FileNotFoundError raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_hdf5(temp_dir / "missing.h5")


class TestNetCDFOperations:
    """Tests for NetCDF loader. Skipped if xarray is not installed."""

    @pytest.fixture(autouse=True)
    def _require_xarray(self):
        pytest.importorskip("xarray")

    def _write_netcdf(self, path):
        import xarray as xr

        ds = xr.Dataset(
            {
                "temperature": (["time"], np.linspace(20.0, 25.0, 5)),
                "pressure": (["time"], np.linspace(1.0, 1.05, 5)),
            },
            coords={"time": np.arange(5, dtype=float)},
        )
        ds.to_netcdf(str(path))
        return path

    def test_load_netcdf_returns_dict(self, temp_dir):
        """load_netcdf returns a dict."""
        p = self._write_netcdf(temp_dir / "data.nc")
        result = load_netcdf(p)
        assert isinstance(result, dict)

    def test_load_netcdf_variables_present(self, temp_dir):
        """Data variables appear in the returned dict."""
        p = self._write_netcdf(temp_dir / "data.nc")
        result = load_netcdf(p)
        assert "temperature" in result
        assert "pressure" in result

    def test_load_netcdf_coordinate_present(self, temp_dir):
        """Coordinate arrays are included in the dict."""
        p = self._write_netcdf(temp_dir / "data.nc")
        result = load_netcdf(p)
        assert "time" in result

    def test_load_netcdf_array_values(self, temp_dir):
        """Array values match what was written."""
        p = self._write_netcdf(temp_dir / "data.nc")
        result = load_netcdf(p)
        np.testing.assert_allclose(result["temperature"], np.linspace(20.0, 25.0, 5))

    def test_load_netcdf_via_load_data(self, temp_dir):
        """load_data dispatches .nc extension."""
        p = self._write_netcdf(temp_dir / "data.nc")
        result = load_data(p)
        assert isinstance(result, dict)

    def test_load_netcdf_file_not_found(self, temp_dir):
        """FileNotFoundError raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_netcdf(temp_dir / "missing.nc")


class TestSPCOperations:
    """Tests for SPC loader. Skipped if spc is not installed."""

    @pytest.fixture(autouse=True)
    def _require_spc(self):
        pytest.importorskip("spc")

    def test_load_spc_import_error_without_package(self):
        """ImportError raised (and caught) when spc absent — mocked."""
        import unittest.mock as mock
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spc":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="spc"):
                load_spc("any.spc")

    def test_load_spc_file_not_found(self, temp_dir):
        """FileNotFoundError raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_spc(temp_dir / "missing.spc")


class TestASCOperations:
    """Tests for ASC / plain-text instrument export loader."""

    def test_load_asc_whitespace_delimited(self, temp_dir):
        """Whitespace-delimited two-column file loads correctly."""
        p = temp_dir / "spectrum.asc"
        p.write_text("100.0 0.12\n200.0 0.25\n300.0 0.08\n")
        df = load_asc(p)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)

    def test_load_asc_comma_delimited(self, temp_dir):
        """Comma-delimited file loads correctly."""
        p = temp_dir / "spectrum.asc"
        p.write_text("100.0,0.12\n200.0,0.25\n300.0,0.08\n")
        df = load_asc(p)
        assert df.shape == (3, 2)

    def test_load_asc_comment_lines_skipped(self, temp_dir):
        """Lines starting with # are skipped."""
        p = temp_dir / "spectrum.asc"
        p.write_text(
            "# Raman spectrum of sample A\n# Wavenumber  Intensity\n100.0 0.12\n200.0 0.25\n"
        )
        df = load_asc(p)
        assert len(df) == 2

    def test_load_asc_column_names_without_header(self, temp_dir):
        """Columns named col_0, col_1 when no header present."""
        p = temp_dir / "spectrum.asc"
        p.write_text("100.0 0.12\n200.0 0.25\n")
        df = load_asc(p)
        assert list(df.columns) == ["col_0", "col_1"]

    def test_load_asc_with_header_row(self, temp_dir):
        """Header row preserved when first line is non-numeric."""
        p = temp_dir / "spectrum.asc"
        p.write_text("wavenumber intensity\n100.0 0.12\n200.0 0.25\n")
        df = load_asc(p)
        assert "wavenumber" in df.columns
        assert "intensity" in df.columns

    def test_load_asc_tab_delimited(self, temp_dir):
        """Tab-delimited file loads correctly."""
        p = temp_dir / "spectrum.asc"
        p.write_text("100.0\t0.12\n200.0\t0.25\n300.0\t0.08\n")
        df = load_asc(p)
        assert df.shape == (3, 2)

    def test_load_asc_via_load_data(self, temp_dir):
        """load_data dispatches .asc extension."""
        p = temp_dir / "spectrum.asc"
        p.write_text("100.0 0.12\n200.0 0.25\n")
        df = load_data(p)
        assert isinstance(df, pd.DataFrame)

    def test_load_asc_file_not_found(self, temp_dir):
        """FileNotFoundError raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_asc(temp_dir / "missing.asc")


class TestInstrumentImportErrorHandling:
    """Tests that missing optional dependencies raise clear ImportError."""

    def _mock_no_import(self, blocked_name, func, *args):
        """Call func(*args) with `blocked_name` blocked from importing."""
        import unittest.mock as mock
        import builtins

        real_import = builtins.__import__

        def _side_effect(name, *a, **kw):
            if name == blocked_name:
                raise ImportError("mocked absent")
            return real_import(name, *a, **kw)

        with mock.patch("builtins.__import__", side_effect=_side_effect):
            func(*args)

    def test_jcamp_import_error(self, temp_dir):
        p = temp_dir / "x.jdx"
        p.touch()
        with pytest.raises(ImportError, match="jcamp"):
            self._mock_no_import("jcamp", load_jcamp, p)

    def test_h5py_import_error(self, temp_dir):
        p = temp_dir / "x.h5"
        p.touch()
        with pytest.raises(ImportError, match="h5py"):
            self._mock_no_import("h5py", load_hdf5, p)

    def test_xarray_import_error(self, temp_dir):
        p = temp_dir / "x.nc"
        p.touch()
        with pytest.raises(ImportError, match="xarray"):
            self._mock_no_import("xarray", load_netcdf, p)

    def test_load_data_unknown_extension_still_fails(self, temp_dir):
        """Unrecognised extensions still raise ValueError."""
        p = temp_dir / "data.xyz"
        p.touch()
        with pytest.raises(ValueError):
            load_data(p)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
