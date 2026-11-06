"""Session State Management for Streamlit GUI.

This module provides centralized session state management to persist data,
plots, and analysis results across page navigation in the Streamlit multi-page app.

Functions
---------
initialize_session_state()
    Initialize all session state variables with defaults
add_dataset(name, data, metadata=None)
    Add a dataset to session state
get_current_dataset()
    Retrieve the currently selected dataset
get_dataset(name)
    Retrieve a specific dataset by name
delete_dataset(name)
    Remove a dataset from session state
add_plot_to_history(plot_data)
    Add a generated plot to history
clear_plot_history()
    Remove all plots from history
add_analysis_result(result_data)
    Add analysis result to session state
save_session_to_file(filepath)
    Persist session state to JSON file
load_session_from_file(filepath)
    Restore session state from JSON file
clear_session()
    Reset all session state to defaults

Examples
--------
>>> import streamlit as st
>>> from eagleplot.utils.session_state import initialize_session_state, add_dataset
>>> initialize_session_state()
>>> data = np.array([1, 2, 3, 4, 5])
>>> add_dataset('test_data.npy', data)
>>> dataset = get_current_dataset()
"""

from __future__ import annotations

import io
import streamlit as st
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# numpy and pandas are imported inside the functions that use them: Home.py imports this
# module, and pandas alone takes ~2.3 s to import in the browser build (roadmap D4.1).
if TYPE_CHECKING:
    import numpy as np


def initialize_session_state():
    """Initialize all session state variables with default values.

    This should be called once at the start of the app (in gui.py).
    Creates the following state variables:
    - datasets: Dict[str, Any] - Loaded datasets keyed by filename
    - current_dataset: str | None - Name of currently selected dataset
    - plot_history: List[Dict] - History of generated plots
    - analysis_results: List[Dict] - Results from mathematical operations
    - plot_config: Dict - Current plot configuration settings
    - export_queue: List[Dict] - Items queued for export
    """
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}

    if "dataset_metadata" not in st.session_state:
        st.session_state.dataset_metadata = {}

    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None

    if "plot_history" not in st.session_state:
        st.session_state.plot_history = []

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []

    if "plot_config" not in st.session_state:
        st.session_state.plot_config = {
            "default_figsize": (8, 6),
            "default_dpi": 100,
            "default_style": "default",
        }

    if "export_queue" not in st.session_state:
        st.session_state.export_queue = []

    if "mol_vib_data" not in st.session_state:
        st.session_state.mol_vib_data = None


def add_dataset(name: str, data: Any, metadata: Optional[Dict] = None):
    """Add a dataset to session state.

    Parameters
    ----------
    name : str
        Dataset name (usually filename)
    data : Any
        Dataset (DataFrame, ndarray, dict, etc.)
    metadata : dict, optional
        Additional metadata (file size, upload time, etc.)
    """
    import numpy as np
    import pandas as pd

    st.session_state.datasets[name] = data

    # Store metadata
    if metadata is None:
        metadata = {}

    metadata["added_time"] = datetime.now().isoformat()
    metadata["data_type"] = type(data).__name__

    # Add shape/size info
    if isinstance(data, pd.DataFrame):
        metadata["shape"] = data.shape
        metadata["columns"] = list(data.columns)
        metadata["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
    elif isinstance(data, np.ndarray):
        metadata["shape"] = data.shape
        metadata["dtype"] = str(data.dtype)
    elif isinstance(data, (list, tuple)):
        metadata["length"] = len(data)

    st.session_state.dataset_metadata[name] = metadata

    # Set as current dataset if it's the first one
    if st.session_state.current_dataset is None:
        st.session_state.current_dataset = name


def get_current_dataset() -> Optional[Any]:
    """Retrieve the currently selected dataset.

    Returns
    -------
    data : Any or None
        Current dataset, or None if no dataset is selected
    """
    if st.session_state.current_dataset is None:
        return None

    return st.session_state.datasets.get(st.session_state.current_dataset)


def get_dataset(name: str) -> Optional[Any]:
    """Retrieve a specific dataset by name.

    Parameters
    ----------
    name : str
        Dataset name

    Returns
    -------
    data : Any or None
        Dataset if found, None otherwise
    """
    return st.session_state.datasets.get(name)


def delete_dataset(name: str) -> bool:
    """Remove a dataset from session state.

    Parameters
    ----------
    name : str
        Dataset name to delete

    Returns
    -------
    success : bool
        True if dataset was deleted, False if not found
    """
    if name in st.session_state.datasets:
        del st.session_state.datasets[name]
        del st.session_state.dataset_metadata[name]

        # Update current dataset if we deleted it
        if st.session_state.current_dataset == name:
            if st.session_state.datasets:
                st.session_state.current_dataset = list(st.session_state.datasets.keys())[0]
            else:
                st.session_state.current_dataset = None

        return True

    return False


_MAX_PLOT_HISTORY = 50


def add_plot_to_history(plot_data: Dict):
    """Add a generated plot to history (capped at ``_MAX_PLOT_HISTORY`` entries).

    Parameters
    ----------
    plot_data : dict
        Plot information with keys:
        - type: Plot type (e.g., 'histogram', 'scatter')
        - dataset: Dataset name used
        - config: Plot configuration dict
        - timestamp: When plot was created
        - figure: Matplotlib/Plotly figure object (optional)

    Notes
    -----
    Figure objects are stripped from older entries once the history exceeds
    five entries to prevent unbounded memory growth.
    """
    if "timestamp" not in plot_data:
        plot_data["timestamp"] = datetime.now().isoformat()

    st.session_state.plot_history.append(plot_data)

    # Drop figure objects from all but the 5 most recent entries.
    history = st.session_state.plot_history
    for entry in history[:-5]:
        entry.pop("figure", None)

    # Rolling cap — evict oldest entries when limit is exceeded.
    if len(history) > _MAX_PLOT_HISTORY:
        st.session_state.plot_history = history[-_MAX_PLOT_HISTORY:]


def clear_plot_history():
    """Remove all plots from history."""
    st.session_state.plot_history = []


def add_analysis_result(result_data: Dict):
    """Add analysis result to session state.

    Parameters
    ----------
    result_data : dict
        Analysis result with keys:
        - type: Analysis type (e.g., 'statistics', 'curve_fit')
        - dataset: Dataset name used
        - results: Analysis output dict
        - timestamp: When analysis was performed
    """
    if "timestamp" not in result_data:
        result_data["timestamp"] = datetime.now().isoformat()

    st.session_state.analysis_results.append(result_data)


def _serialize_data(data: Any) -> Any:
    """Serialize data for JSON storage.

    Only types with a safe, self-describing encoding are serialized:
    DataFrames (via ``to_json``), numeric/boolean ndarrays (via an explicit
    dtype + shape + raw-buffer triple), JSON scalars, and lists/dicts of the
    above.  Anything else is replaced by an ``unsupported`` placeholder rather
    than being pickled — see Notes.

    Parameters
    ----------
    data : Any
        Data to serialize

    Returns
    -------
    serialized : Any
        Serialized representation — a dict for tagged types, otherwise the
        value itself.

    Notes
    -----
    Session files are written and re-read through the GUI's Export page, which
    accepts uploads.  Any ``pickle``-based encoding would therefore make
    "open a session file" equivalent to "execute arbitrary code", so pickle is
    deliberately not used here.  Object-dtype arrays are also refused, since
    their contents cannot be represented without pickling.
    """
    import numpy as np
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return {"__type__": "DataFrame", "__data__": data.to_json(orient="split")}
    elif isinstance(data, np.ndarray):
        if data.dtype.hasobject:
            return {
                "__type__": "unsupported",
                "__class__": f"numpy.ndarray[{data.dtype}]",
                "__reason__": "object-dtype arrays cannot be serialized safely",
            }
        return {
            "__type__": "ndarray",
            "__dtype__": data.dtype.str,
            "__shape__": list(data.shape),
            "__data__": base64.b64encode(np.ascontiguousarray(data).tobytes()).decode("utf-8"),
        }
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, (list, tuple)):
        return [_serialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: _serialize_data(value) for key, value in data.items()}
    else:
        return {
            "__type__": "unsupported",
            "__class__": type(data).__name__,
            "__reason__": "no safe JSON encoding for this type",
        }


def _decode_ndarray(data: Dict) -> np.ndarray:
    """Rebuild an ndarray from a serialized dtype/shape/buffer triple.

    Parameters
    ----------
    data : dict
        Mapping with ``__dtype__``, ``__shape__``, and ``__data__`` keys as
        written by :func:`_serialize_data`.

    Returns
    -------
    numpy.ndarray
        A writable array with the recorded dtype and shape.

    Raises
    ------
    ValueError
        If the dtype is absent, is an object dtype, or if the decoded buffer
        length does not match ``dtype.itemsize * prod(shape)``.
    """
    import numpy as np

    dtype_str = data.get("__dtype__")
    if not dtype_str:
        raise ValueError("ndarray entry is missing '__dtype__'")

    dtype = np.dtype(dtype_str)
    if dtype.hasobject:
        raise ValueError(f"refusing to decode object dtype {dtype!r}")

    shape = tuple(int(n) for n in data.get("__shape__", ()))
    if any(n < 0 for n in shape):
        raise ValueError(f"invalid shape {shape}")

    buf = base64.b64decode(data["__data__"])
    # np.frombuffer validates that the buffer is a whole number of items, but
    # not that it matches `shape`; check explicitly so a truncated or padded
    # file fails here rather than producing a silently wrong array.
    expected = dtype.itemsize * int(np.prod(shape, dtype=np.int64)) if shape else dtype.itemsize
    if len(buf) != expected:
        raise ValueError(
            f"ndarray buffer is {len(buf)} bytes; expected {expected} "
            f"for dtype {dtype.str} and shape {shape}"
        )

    # frombuffer returns a read-only view onto `buf`; copy so callers can write.
    return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()


def _deserialize_data(data: Any, skipped: Optional[List[str]] = None) -> Any:
    """Deserialize data from JSON storage.

    Parameters
    ----------
    data : Any
        Serialized data
    skipped : list of str, optional
        If given, a human-readable note is appended for every entry that could
        not be restored.  Callers use this to tell the user what was dropped.

    Returns
    -------
    deserialized : Any
        Deserialized data, or ``None`` for entries that could not be restored.

    Notes
    -----
    Entries written by Plottle 2.0.1 and earlier used a pickle-based encoding
    (``__type__`` of ``"pickled"``, and ``"ndarray"`` without a ``__dtype__``).
    Those are refused rather than unpickled — see :func:`_serialize_data`.
    """
    import pandas as pd

    if isinstance(data, dict):
        kind = data.get("__type__")
        if kind is None:
            return {key: _deserialize_data(value, skipped) for key, value in data.items()}

        if kind == "DataFrame":
            return pd.read_json(io.StringIO(data["__data__"]), orient="split")

        if kind == "ndarray":
            if "__dtype__" not in data:
                if skipped is not None:
                    skipped.append(
                        "an array saved by Plottle 2.0.1 or earlier "
                        "(pickle-encoded; refused for safety)"
                    )
                return None
            try:
                return _decode_ndarray(data)
            except (ValueError, TypeError) as exc:
                if skipped is not None:
                    skipped.append(f"a corrupt array entry ({exc})")
                return None

        if kind == "pickled":
            if skipped is not None:
                skipped.append(
                    f"a pickled {data.get('__class__', 'object')} saved by Plottle 2.0.1 "
                    "or earlier (refused for safety)"
                )
            return None

        if kind == "unsupported":
            if skipped is not None:
                skipped.append(
                    f"{data.get('__class__', 'an object')} "
                    f"({data.get('__reason__', 'unsupported type')})"
                )
            return None

        if skipped is not None:
            skipped.append(f"an entry of unrecognized type {kind!r}")
        return None

    elif isinstance(data, list):
        return [_deserialize_data(item, skipped) for item in data]
    else:
        return data


def save_session_to_file(filepath: str):
    """Persist session state to JSON file.

    Parameters
    ----------
    filepath : str
        Path to save session file

    Notes
    -----
    This saves datasets, plot history, and analysis results.
    Matplotlib figure objects are NOT saved (too large).
    """
    session_data = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "datasets": {},
        "dataset_metadata": st.session_state.dataset_metadata,
        "current_dataset": st.session_state.current_dataset,
        "plot_history": [
            {k: v for k, v in plot.items() if k != "figure"}
            for plot in st.session_state.plot_history
        ],
        "analysis_results": st.session_state.analysis_results,
        "plot_config": st.session_state.plot_config,
    }

    # Serialize datasets
    for name, data in st.session_state.datasets.items():
        session_data["datasets"][name] = _serialize_data(data)

    filepath = Path(filepath)
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)


def load_session_from_file(filepath: str) -> List[str]:
    """Restore session state from JSON file.

    Parameters
    ----------
    filepath : str
        Path to session file

    Returns
    -------
    skipped : list of str
        One human-readable note per dataset that could not be restored — for
        example entries saved by Plottle 2.0.1 or earlier, which used a
        pickle-based encoding that is refused for safety.  Empty when the whole
        session was restored.  Callers should surface this to the user.
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        session_data = json.load(f)

    # Deserialize datasets
    skipped: List[str] = []
    st.session_state.datasets = {}
    for name, data in session_data["datasets"].items():
        notes: List[str] = []
        restored = _deserialize_data(data, notes)
        if restored is None and notes:
            skipped.extend(f"{name}: {note}" for note in notes)
            continue
        skipped.extend(f"{name}: {note}" for note in notes)
        st.session_state.datasets[name] = restored

    st.session_state.dataset_metadata = session_data["dataset_metadata"]
    st.session_state.current_dataset = session_data["current_dataset"]
    st.session_state.plot_history = session_data["plot_history"]
    st.session_state.analysis_results = session_data["analysis_results"]
    st.session_state.plot_config = session_data["plot_config"]

    # A dataset that was dropped must not stay selected.
    if st.session_state.current_dataset not in st.session_state.datasets:
        st.session_state.current_dataset = next(iter(st.session_state.datasets), None)

    return skipped


def clear_session():
    """Reset all session state to defaults."""
    st.session_state.datasets = {}
    st.session_state.dataset_metadata = {}
    st.session_state.current_dataset = None
    st.session_state.plot_history = []
    st.session_state.analysis_results = []
    st.session_state.plot_config = {
        "default_figsize": (8, 6),
        "default_dpi": 100,
        "default_style": "default",
    }
    st.session_state.export_queue = []


def get_session_summary() -> Dict:
    """Get a summary of current session state.

    Returns
    -------
    summary : dict
        Summary with counts and current selections
    """
    return {
        "num_datasets": len(st.session_state.datasets),
        "current_dataset": st.session_state.current_dataset,
        "num_plots": len(st.session_state.plot_history),
        "num_analyses": len(st.session_state.analysis_results),
        "dataset_names": list(st.session_state.datasets.keys()),
    }
