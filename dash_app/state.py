"""Server-side session state for the Plottle Dash app.

Replaces Streamlit's st.session_state with a thread-safe, server-side
dictionary. Designed for single-user research-tool deployments where a
shared in-process store is the simplest correct choice.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_LOCK = threading.Lock()
_MAX_PLOT_HISTORY = 50

_STATE: Dict[str, Any] = {
    "datasets": {},
    "dataset_metadata": {},
    "current_dataset": None,
    "plot_history": [],
    "analysis_results": [],
    "mol_vib_data": None,
}

# ── Public accessors ──────────────────────────────────────────────────────────


def add_dataset(name: str, data: Any, metadata: Optional[Dict] = None) -> None:
    """Store a dataset and its metadata; set as current if first."""
    with _LOCK:
        _STATE["datasets"][name] = data
        meta: Dict[str, Any] = metadata or {}
        meta["added_time"] = datetime.now().isoformat()
        meta["data_type"] = type(data).__name__
        if isinstance(data, pd.DataFrame):
            meta["shape"] = list(data.shape)
            meta["columns"] = list(data.columns)
            meta["dtypes"] = {c: str(d) for c, d in data.dtypes.items()}
        elif isinstance(data, np.ndarray):
            meta["shape"] = list(data.shape)
            meta["dtype"] = str(data.dtype)
        elif isinstance(data, (list, tuple)):
            meta["length"] = len(data)
        _STATE["dataset_metadata"][name] = meta
        if _STATE["current_dataset"] is None:
            _STATE["current_dataset"] = name


def get_dataset(name: str) -> Optional[Any]:
    return _STATE["datasets"].get(name)


def get_current_dataset() -> Optional[Any]:
    name = _STATE["current_dataset"]
    return _STATE["datasets"].get(name) if name else None


def get_dataset_names() -> List[str]:
    return list(_STATE["datasets"].keys())


def get_dataset_metadata(name: str) -> Dict:
    return dict(_STATE["dataset_metadata"].get(name, {}))


def set_current_dataset(name: str) -> None:
    if name in _STATE["datasets"]:
        with _LOCK:
            _STATE["current_dataset"] = name


def delete_dataset(name: str) -> bool:
    with _LOCK:
        if name not in _STATE["datasets"]:
            return False
        del _STATE["datasets"][name]
        del _STATE["dataset_metadata"][name]
        if _STATE["current_dataset"] == name:
            remaining = list(_STATE["datasets"].keys())
            _STATE["current_dataset"] = remaining[0] if remaining else None
        return True


def add_plot_to_history(plot_data: Dict) -> None:
    with _LOCK:
        plot_data.setdefault("timestamp", datetime.now().isoformat())
        _STATE["plot_history"].append(plot_data)
        history = _STATE["plot_history"]
        for entry in history[:-5]:
            entry.pop("figure", None)
        if len(history) > _MAX_PLOT_HISTORY:
            _STATE["plot_history"] = history[-_MAX_PLOT_HISTORY:]


def clear_plot_history() -> None:
    with _LOCK:
        _STATE["plot_history"] = []


def get_plot_history() -> List[Dict]:
    return list(_STATE["plot_history"])


def add_analysis_result(result_data: Dict) -> None:
    with _LOCK:
        result_data.setdefault("timestamp", datetime.now().isoformat())
        _STATE["analysis_results"].append(result_data)


def get_analysis_results() -> List[Dict]:
    return list(_STATE["analysis_results"])


def clear_analysis_results() -> None:
    with _LOCK:
        _STATE["analysis_results"] = []


def get_session_summary() -> Dict:
    return {
        "num_datasets": len(_STATE["datasets"]),
        "current_dataset": _STATE["current_dataset"],
        "num_plots": len(_STATE["plot_history"]),
        "num_analyses": len(_STATE["analysis_results"]),
        "dataset_names": list(_STATE["datasets"].keys()),
    }


def set_mol_vib_data(data: Any) -> None:
    with _LOCK:
        _STATE["mol_vib_data"] = data


def get_mol_vib_data() -> Any:
    return _STATE["mol_vib_data"]


def clear_session() -> None:
    with _LOCK:
        _STATE["datasets"] = {}
        _STATE["dataset_metadata"] = {}
        _STATE["current_dataset"] = None
        _STATE["plot_history"] = []
        _STATE["analysis_results"] = []
        _STATE["mol_vib_data"] = None
