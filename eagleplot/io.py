"""Input/Output Module.

This module provides utilities for reading and writing scientific data
in various formats commonly used in computational chemistry.

Supported Formats
-----------------
- Pickle (.pkl): Python object serialization
- NumPy arrays (.npy, .npz): Numerical array data
- Pandas DataFrames (.csv, .xlsx, .tsv, .json, .parquet): Tabular data
- JCAMP-DX (.jdx, .dx): Spectroscopy data (IR, Raman, NMR, MS) — requires ``jcamp``
- HDF5 (.h5, .hdf5): Hierarchical data arrays — requires ``h5py``
- NetCDF (.nc, .cdf): Array/climate data — requires ``xarray``
- SPC (.spc): Thermo Fisher spectroscopy — requires ``spc``
- ASC (.asc): Generic instrument text export (no extra dependency)

Examples
--------
>>> from eagleplot.io import load_data, save_data
>>> data = load_data('experiment.pkl')
>>> save_data(processed_data, 'results.pkl')
>>> spectrum = load_data('spectrum.jdx')   # returns pd.DataFrame
>>> arrays  = load_data('simulation.h5')   # returns dict of np.ndarray
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Union

# ============================================================================
# Pickle Operations
# ============================================================================


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from a pickle file.

    Parameters
    ----------
    filepath : str or Path
        Path to the pickle file (.pkl)

    Returns
    -------
    data : Any
        The deserialized Python object

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    pickle.UnpicklingError
        If the file is corrupted or not a valid pickle file

    Examples
    --------
    >>> data = load_pickle('experiment.pkl')
    >>> print(type(data))
    <class 'dict'>
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(
            f"Failed to unpickle file {filepath}. "
            f"The file may be corrupted or not a valid pickle file."
        ) from e


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to a pickle file.

    Parameters
    ----------
    data : Any
        Python object to serialize
    filepath : str or Path
        Output file path (.pkl)

    Raises
    ------
    PermissionError
        If there are insufficient permissions to write to the file
    pickle.PicklingError
        If the object cannot be pickled

    Examples
    --------
    >>> data = {'temperature': [300, 310, 320], 'pressure': [1.0, 1.1, 1.2]}
    >>> save_pickle(data, 'results.pkl')
    """
    filepath = Path(filepath)

    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    except pickle.PicklingError as e:
        raise pickle.PicklingError(
            f"Failed to pickle object of type {type(data).__name__}. "
            f"Ensure the object is serializable."
        ) from e


# ============================================================================
# NumPy Array Operations
# ============================================================================


def load_numpy(filepath: Union[str, Path]) -> Union[np.ndarray, dict]:
    """Load NumPy array(s) from file.

    Supports both .npy (single array) and .npz (multiple arrays) formats.

    Parameters
    ----------
    filepath : str or Path
        Path to NumPy file (.npy or .npz)

    Returns
    -------
    data : np.ndarray or dict
        For .npy files: returns a single NumPy array
        For .npz files: returns a dict-like object with array names as keys

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported

    Examples
    --------
    >>> array = load_numpy('data.npy')
    >>> print(array.shape)
    (100, 3)

    >>> arrays = load_numpy('multiple.npz')
    >>> print(arrays['x'].shape)
    (100,)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".npy":
        return np.load(filepath)
    elif suffix == ".npz":
        return np.load(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .npy or .npz")


def save_numpy(
    data: Union[np.ndarray, dict], filepath: Union[str, Path], compressed: bool = False
) -> None:
    """Save NumPy array(s) to file.

    Parameters
    ----------
    data : np.ndarray or dict
        For .npy: single NumPy array
        For .npz: dict of arrays with names as keys
    filepath : str or Path
        Output file path (.npy or .npz)
    compressed : bool, default False
        If True and saving .npz, use compression

    Raises
    ------
    ValueError
        If data type doesn't match file format
    TypeError
        If data is not a NumPy array or dict of arrays

    Examples
    --------
    >>> array = np.random.rand(100, 3)
    >>> save_numpy(array, 'data.npy')

    >>> arrays = {'x': np.arange(10), 'y': np.arange(10)**2}
    >>> save_numpy(arrays, 'multiple.npz')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    suffix = filepath.suffix.lower()

    if suffix == ".npy":
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"For .npy format, data must be a NumPy array, got {type(data).__name__}"
            )
        np.save(filepath, data)

    elif suffix == ".npz":
        if isinstance(data, dict):
            if compressed:
                np.savez_compressed(filepath, **data)
            else:
                np.savez(filepath, **data)
        elif isinstance(data, np.ndarray):
            # Save single array with default name
            if compressed:
                np.savez_compressed(filepath, arr_0=data)
            else:
                np.savez(filepath, arr_0=data)
        else:
            raise TypeError(
                f"For .npz format, data must be a NumPy array or dict of arrays, "
                f"got {type(data).__name__}"
            )

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .npy or .npz")


# ============================================================================
# Pandas DataFrame Operations
# ============================================================================


def load_dataframe(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load tabular data into a Pandas DataFrame.

    Automatically detects format from file extension.
    Supports CSV, Excel, and other Pandas-compatible formats.

    Parameters
    ----------
    filepath : str or Path
        Path to data file (.csv, .xlsx, .xls, etc.)
    kwargs : dict
        Additional keyword arguments passed to pandas read function

    Returns
    -------
    df : pd.DataFrame
        Loaded DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported

    Examples
    --------
    >>> df = load_dataframe('data.csv')
    >>> print(df.head())

    >>> df = load_dataframe('data.xlsx', sheet_name='Sheet1')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(filepath, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(filepath, **kwargs)
    elif suffix == ".tsv":
        return pd.read_csv(filepath, sep="\t", **kwargs)
    elif suffix == ".json":
        return pd.read_json(filepath, **kwargs)
    elif suffix == ".parquet":
        return pd.read_parquet(filepath, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .csv, .xlsx, .xls, .tsv, .json, .parquet"
        )


def save_dataframe(
    df: pd.DataFrame, filepath: Union[str, Path], index: bool = False, **kwargs
) -> None:
    """Save Pandas DataFrame to file.

    Automatically selects format based on file extension.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output file path
    index : bool, default False
        Whether to include the index in the output
    kwargs : dict
        Additional keyword arguments passed to pandas write function

    Raises
    ------
    ValueError
        If the file format is not supported
    TypeError
        If df is not a Pandas DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    >>> save_dataframe(df, 'output.csv')

    >>> save_dataframe(df, 'output.xlsx', sheet_name='Results')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        df.to_csv(filepath, index=index, **kwargs)
    elif suffix in [".xlsx", ".xls"]:
        df.to_excel(filepath, index=index, **kwargs)
    elif suffix == ".tsv":
        df.to_csv(filepath, sep="\t", index=index, **kwargs)
    elif suffix == ".json":
        df.to_json(filepath, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(filepath, index=index, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .csv, .xlsx, .xls, .tsv, .json, .parquet"
        )


# ============================================================================
# Instrument File Formats (M17)
# ============================================================================


def load_jcamp(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a JCAMP-DX spectroscopy file (.jdx or .dx).

    Requires the ``jcamp`` package: ``pip install jcamp``

    Returns a :class:`pandas.DataFrame` with columns ``x`` and ``y``.
    Spectral metadata (title, data type, units) is stored in ``df.attrs``.

    Parameters
    ----------
    filepath : str or Path
        Path to the JCAMP-DX file (.jdx or .dx)

    Returns
    -------
    df : pd.DataFrame
        Two-column DataFrame: ``x`` (frequency/wavenumber axis) and
        ``y`` (intensity/absorbance/transmittance). Metadata available
        via ``df.attrs``.

    Raises
    ------
    ImportError
        If the ``jcamp`` package is not installed
    FileNotFoundError
        If the file does not exist

    Examples
    --------
    >>> df = load_jcamp('spectrum.jdx')
    >>> print(df.columns.tolist())
    ['x', 'y']
    >>> print(df.attrs.get('xunits'))
    1/CM
    """
    try:
        from jcamp import JCAMP_reader as _jcamp_read
    except ImportError:
        try:
            from jcamp import jcamp_read as _jcamp_read  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "The `jcamp` package is required to load JCAMP-DX files. "
                "Install it with: pip install jcamp"
            )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = _jcamp_read(str(filepath))

    x = np.asarray(data.get("x", []), dtype=float)
    y = np.asarray(data.get("y", []), dtype=float)

    df = pd.DataFrame({"x": x, "y": y})

    for key in ("title", "data type", "xunits", "yunits", "origin", "owner"):
        if key in data:
            df.attrs[key] = data[key]

    return df


def load_hdf5(
    filepath: Union[str, Path], dataset: Union[str, None] = None
) -> Union[dict, np.ndarray]:
    """Load an HDF5 file (.h5 or .hdf5).

    Requires the ``h5py`` package: ``pip install h5py``

    By default returns a flat ``dict`` mapping every dataset path to a
    :class:`numpy.ndarray`.  If *dataset* is given, returns just that
    single array.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file
    dataset : str, optional
        Slash-separated path to a specific dataset inside the file
        (e.g. ``"group/signal"``).  When omitted all datasets are
        returned.

    Returns
    -------
    data : dict[str, np.ndarray] or np.ndarray
        All datasets as a flat dict, or the requested single array.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed
    FileNotFoundError
        If the file does not exist
    KeyError
        If the requested *dataset* path does not exist in the file

    Examples
    --------
    >>> arrays = load_hdf5('simulation.h5')
    >>> print(list(arrays.keys()))
    ['time', 'temperature', 'pressure']

    >>> arr = load_hdf5('simulation.h5', dataset='temperature')
    >>> print(arr.shape)
    (1000,)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "The `h5py` package is required to load HDF5 files. Install it with: pip install h5py"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    def _flatten(group: "h5py.Group", prefix: str = "") -> dict:
        result: dict = {}
        for key, item in group.items():
            full = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                result[full] = np.array(item)
            elif isinstance(item, h5py.Group):
                result.update(_flatten(item, full))
        return result

    with h5py.File(filepath, "r") as f:
        if dataset is not None:
            if dataset not in f:
                raise KeyError(f"Dataset '{dataset}' not found in {filepath}")
            return np.array(f[dataset])
        return _flatten(f)


def load_netcdf(filepath: Union[str, Path]) -> dict:
    """Load a NetCDF file (.nc or .cdf).

    Requires the ``xarray`` package: ``pip install xarray netcdf4``

    Returns a ``dict`` mapping variable names to :class:`numpy.ndarray`
    values.  Coordinate arrays are included alongside data variables.

    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF file

    Returns
    -------
    data : dict[str, np.ndarray]
        All data variables and coordinates as numpy arrays.

    Raises
    ------
    ImportError
        If ``xarray`` (or its NetCDF engine) is not installed
    FileNotFoundError
        If the file does not exist

    Examples
    --------
    >>> data = load_netcdf('climate.nc')
    >>> print(list(data.keys()))
    ['time', 'lat', 'lon', 'temperature']
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "The `xarray` package is required to load NetCDF files. "
            "Install it with: pip install xarray netcdf4"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ds = xr.open_dataset(str(filepath))
    result: dict = {}
    for name, var in ds.data_vars.items():
        result[str(name)] = np.array(var.values)
    for name, coord in ds.coords.items():
        if str(name) not in result:
            result[str(name)] = np.array(coord.values)
    ds.close()
    return result


def load_spc(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a Thermo Fisher SPC spectroscopy file (.spc).

    Requires the ``spc`` package: ``pip install spc``

    Returns a :class:`pandas.DataFrame` with a ``wavenumber`` column
    and one ``intensity`` column per trace (``intensity_0``,
    ``intensity_1``, … for multi-trace files).

    Parameters
    ----------
    filepath : str or Path
        Path to the SPC file

    Returns
    -------
    df : pd.DataFrame
        Spectral data. Single-trace files have columns
        ``['wavenumber', 'intensity']``; multi-trace files add a
        numeric suffix to each intensity column.

    Raises
    ------
    ImportError
        If the ``spc`` package is not installed
    FileNotFoundError
        If the file does not exist

    Examples
    --------
    >>> df = load_spc('raman.spc')
    >>> print(df.columns.tolist())
    ['wavenumber', 'intensity']
    """
    try:
        import spc  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The `spc` package is required to load SPC files. Install it with: pip install spc"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    f = spc.File(str(filepath))

    # Shared x-axis (may live on the file or on each sub-file)
    shared_x = np.asarray(f.x) if hasattr(f, "x") and f.x is not None else None

    columns: dict = {}
    for i, sub in enumerate(f.sub):
        x_data = np.asarray(sub.x) if (hasattr(sub, "x") and sub.x is not None) else shared_x
        y_data = np.asarray(sub.y)

        if i == 0 and x_data is not None:
            columns["wavenumber"] = x_data

        suffix = f"_{i}" if len(f.sub) > 1 else ""
        columns[f"intensity{suffix}"] = y_data

    return pd.DataFrame(columns)


def load_asc(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a generic instrument ASC / plain-text export file (.asc).

    Handles whitespace-, comma-, and tab-delimited files.  Lines
    starting with ``#`` are treated as comments.  If the first
    non-comment line contains only numeric tokens the file is assumed
    to have no header row and columns are named ``col_0``, ``col_1``,
    etc.  Otherwise the first line is used as a header.

    No extra package is required beyond Pandas.

    Parameters
    ----------
    filepath : str or Path
        Path to the ASC or plain-text file

    Returns
    -------
    df : pd.DataFrame
        Parsed tabular data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file cannot be parsed as delimited text

    Examples
    --------
    >>> df = load_asc('spectrum.asc')
    >>> print(df.shape)
    (512, 2)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect whether the first non-comment line is a header
    first_data_line = ""
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_data_line = stripped
                break

    parts = first_data_line.replace(",", " ").replace("\t", " ").split()
    try:
        [float(p) for p in parts if p]
        header_row: Union[int, None] = None  # all numeric → no header
    except ValueError:
        header_row = 0  # first line is a header

    try:
        df = pd.read_csv(
            filepath,
            sep=None,
            comment="#",
            header=header_row,
            engine="python",
            skip_blank_lines=True,
        )
    except Exception as exc:
        raise ValueError(f"Could not parse {filepath} as delimited text: {exc}") from exc

    if header_row is None:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


def load_mzml(filepath: Union[str, Path]) -> "pd.DataFrame":
    """Load a mass spectrometry mzML or mzXML file.

    Requires the ``pymzml`` package (``pip install pymzml``).
    Returns a summary DataFrame with one row per scan.

    Parameters
    ----------
    filepath : str
        Path to a ``.mzML`` or ``.mzXML`` file.

    Returns
    -------
    pd.DataFrame
        Columns: ``scan``, ``rt`` (s), ``ms_level``, ``tic``,
        ``base_peak_mz``, ``base_peak_intensity``.
        Raw (mz, intensity) arrays are available via
        ``df.attrs["spectra"]`` — a list of dicts with keys
        ``scan``, ``mz``, ``intensity``.
    """
    try:
        import pymzml
    except ImportError as exc:
        raise ImportError(
            "pymzml is required for mzML/mzXML files. Install it with: pip install pymzml"
        ) from exc

    rows = []
    spectra = []
    run = pymzml.run.Reader(str(filepath))
    for scan_idx, spectrum in enumerate(run, start=1):
        mz_arr = np.asarray(spectrum.mz, dtype=float)
        int_arr = np.asarray(spectrum.i, dtype=float)
        if hasattr(spectrum, "scan_time_in_minutes"):
            rt = float(spectrum.scan_time_in_minutes() * 60.0)
        else:
            rt = float(spectrum.get("MS:1000016", 0.0))
        ms_level = int(spectrum.ms_level) if spectrum.ms_level else 1
        tic = float(int_arr.sum()) if len(int_arr) else 0.0
        if len(int_arr) > 0:
            bp_idx = int(np.argmax(int_arr))
            bp_mz = float(mz_arr[bp_idx])
            bp_int = float(int_arr[bp_idx])
        else:
            bp_mz, bp_int = float("nan"), 0.0
        rows.append(
            {
                "scan": scan_idx,
                "rt": rt,
                "ms_level": ms_level,
                "tic": tic,
                "base_peak_mz": bp_mz,
                "base_peak_intensity": bp_int,
            }
        )
        spectra.append({"scan": scan_idx, "mz": mz_arr, "intensity": int_arr})

    df = pd.DataFrame(rows)
    df.attrs["spectra"] = spectra
    df.attrs["source_file"] = str(filepath)
    return df


# ============================================================================
# Universal Data Loader/Saver
# ============================================================================


def load_data(filepath: Union[str, Path], **kwargs) -> Any:
    """Universal data loader with automatic format detection.

    Automatically detects the file format from extension and uses
    the appropriate loading function.

    Supported formats:
    - .pkl: Pickle files
    - .npy, .npz: NumPy arrays
    - .csv, .xlsx, .xls, .tsv, .json, .parquet: Tabular data (Pandas)
    - .jdx, .dx: JCAMP-DX spectroscopy (requires ``jcamp``)
    - .h5, .hdf5: HDF5 arrays (requires ``h5py``)
    - .nc, .cdf: NetCDF (requires ``xarray``)
    - .spc: Thermo Fisher SPC (requires ``spc``)
    - .asc: Generic instrument text export

    Parameters
    ----------
    filepath : str or Path
        Path to data file
    kwargs : dict
        Additional arguments passed to format-specific loader

    Returns
    -------
    data : Any
        Loaded data in appropriate format

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported

    Examples
    --------
    >>> data = load_data('experiment.pkl')
    >>> array = load_data('data.npy')
    >>> df = load_data('results.csv')
    >>> df = load_data('spectrum.jdx')
    >>> arrays = load_data('simulation.h5')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    # Pickle files
    if suffix == ".pkl":
        return load_pickle(filepath)

    # NumPy files
    elif suffix in [".npy", ".npz"]:
        return load_numpy(filepath)

    # Tabular data files
    elif suffix in [".csv", ".xlsx", ".xls", ".tsv", ".json", ".parquet"]:
        return load_dataframe(filepath, **kwargs)

    # JCAMP-DX spectroscopy
    elif suffix in [".jdx", ".dx"]:
        return load_jcamp(filepath)

    # HDF5
    elif suffix in [".h5", ".hdf5"]:
        return load_hdf5(filepath, **kwargs)

    # NetCDF
    elif suffix in [".nc", ".cdf"]:
        return load_netcdf(filepath)

    # Thermo SPC
    elif suffix == ".spc":
        return load_spc(filepath)

    # Generic instrument text
    elif suffix == ".asc":
        return load_asc(filepath)

    # mzML / mzXML mass spectrometry
    elif suffix in [".mzml", ".mzxml"]:
        return load_mzml(filepath)

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .pkl, .npy, .npz, .csv, .xlsx, .xls, "
            f".tsv, .json, .parquet, .jdx, .dx, .h5, .hdf5, .nc, .cdf, "
            f".spc, .asc, .mzml, .mzxml"
        )


def save_data(data: Any, filepath: Union[str, Path], **kwargs) -> None:
    """Universal data saver with automatic format detection.

    Automatically detects the target format from file extension and uses
    the appropriate saving function.

    Parameters
    ----------
    data : Any
        Data to save (Python object, NumPy array, or Pandas DataFrame)
    filepath : str or Path
        Output file path
    kwargs : dict
        Additional arguments passed to format-specific saver

    Raises
    ------
    ValueError
        If the file format is not supported or data type incompatible

    Examples
    --------
    >>> save_data({'temp': 300, 'pressure': 1.0}, 'params.pkl')
    >>> save_data(np.array([1, 2, 3]), 'data.npy')
    >>> save_data(df, 'results.csv')
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    # Pickle files - accept any object
    if suffix == ".pkl":
        save_pickle(data, filepath)

    # NumPy files - require NumPy array or dict of arrays
    elif suffix in [".npy", ".npz"]:
        if isinstance(data, (np.ndarray, dict)):
            save_numpy(data, filepath, **kwargs)
        else:
            raise TypeError(
                f"For {suffix} format, data must be NumPy array or dict of arrays, "
                f"got {type(data).__name__}"
            )

    # Tabular data - require Pandas DataFrame
    elif suffix in [".csv", ".xlsx", ".xls", ".tsv", ".json", ".parquet"]:
        if isinstance(data, pd.DataFrame):
            save_dataframe(data, filepath, **kwargs)
        else:
            raise TypeError(
                f"For {suffix} format, data must be pandas DataFrame, got {type(data).__name__}"
            )

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .pkl, .npy, .npz, .csv, .xlsx, .xls, "
            f".tsv, .json, .parquet"
        )


# ============================================================================
# Large Dataset Utilities
# ============================================================================


def downsample_for_preview(
    data: pd.DataFrame,
    max_rows: int = 10_000,
    method: str = "systematic",
) -> pd.DataFrame:
    """Downsample a DataFrame to at most *max_rows* rows for GUI performance.

    The original DataFrame is returned unchanged when it already fits within
    the limit, so callers can use the result unconditionally.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    max_rows : int
        Row threshold above which downsampling is applied. Default 10 000.
    method : {'systematic', 'random'}
        ``'systematic'`` selects every N-th row (preserves data shape).
        ``'random'`` draws a random sample (uniform coverage).

    Returns
    -------
    pd.DataFrame
        Downsampled (or original) DataFrame.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> large = pd.DataFrame({'x': np.arange(50_000)})
    >>> small = downsample_for_preview(large, max_rows=1000)
    >>> len(small) <= 1000
    True
    """
    if not isinstance(data, pd.DataFrame) or len(data) <= max_rows:
        return data

    if method == "random":
        return data.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # systematic: every step-th row
    step = max(1, len(data) // max_rows)
    return data.iloc[::step].reset_index(drop=True)


def load_large_csv(
    filepath: Union[str, Path],
    max_rows: int = 500_000,
    chunksize: int = 50_000,
    **kwargs,
) -> pd.DataFrame:
    """Load a large CSV in chunks, stopping once *max_rows* rows are read.

    Useful for very large files where ``pd.read_csv`` would exhaust memory.
    For files where all rows are needed, use :func:`load_data` directly.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    max_rows : int
        Maximum rows to load. Default 500 000.
    chunksize : int
        Rows per chunk. Default 50 000.
    kwargs
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of at most *max_rows* rows.

    Examples
    --------
    >>> df = load_large_csv('huge.csv', max_rows=100_000)
    >>> len(df) <= 100_000
    True
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    chunks = []
    rows_read = 0
    reader = pd.read_csv(filepath, chunksize=chunksize, **kwargs)
    for chunk in reader:
        remaining = max_rows - rows_read
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]
        chunks.append(chunk)
        rows_read += len(chunk)
        if rows_read >= max_rows:
            break

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)
