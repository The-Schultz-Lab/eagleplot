"""Data Preview Utilities for Streamlit GUI.

This module provides helper functions for displaying data previews,
summaries, and metadata in the Streamlit interface.

Functions
---------
preview_dataframe(df, n_rows=10)
    Display DataFrame preview with first and last rows
get_dataframe_info(df)
    Get DataFrame information (shape, types, memory)
get_array_info(array)
    Get NumPy array information (shape, dtype, stats)
format_data_size(size_bytes)
    Format data size in human-readable format
display_dataset_card(name, data, metadata)
    Display a dataset summary card in Streamlit
get_column_suggestions(df)
    Suggest numeric/categorical columns for plotting

Examples
--------
>>> import streamlit as st
>>> import pandas as pd
>>> from eagleplot.utils.data_preview import preview_dataframe, get_dataframe_info
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
>>> info = get_dataframe_info(df)
>>> st.write(info)
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional


def preview_dataframe(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """Display DataFrame preview with first and last rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to preview
    n_rows : int, default 10
        Number of rows to show from head and tail

    Returns
    -------
    preview : pd.DataFrame
        Preview DataFrame (first n_rows + last n_rows if applicable)
    """
    if len(df) <= 2 * n_rows:
        return df
    else:
        # Show first n_rows and last n_rows with separator
        head = df.head(n_rows)
        tail = df.tail(n_rows)
        return pd.concat([head, tail])


@st.cache_data(show_spinner=False)
def get_dataframe_info(df: pd.DataFrame) -> Dict:
    """Get DataFrame information (shape, types, memory).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    info : dict
        Dictionary with DataFrame metadata:
        - shape: (rows, cols)
        - columns: list of column names
        - dtypes: dict of column types
        - memory_usage: total memory in bytes
        - missing_values: dict of missing counts per column
        - numeric_columns: list of numeric column names
        - categorical_columns: list of object/category column names
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage": df.memory_usage(deep=True).sum(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
    }

    return info


@st.cache_data(show_spinner=False)
def get_array_info(array: np.ndarray) -> Dict:
    """Get NumPy array information (shape, dtype, stats).

    Parameters
    ----------
    array : np.ndarray
        Array to analyze

    Returns
    -------
    info : dict
        Dictionary with array metadata:
        - shape: array shape
        - dtype: data type
        - size: total number of elements
        - memory_usage: total memory in bytes
        - ndim: number of dimensions
        - min: minimum value (if numeric)
        - max: maximum value (if numeric)
        - mean: mean value (if numeric)
        - std: standard deviation (if numeric)
    """
    info = {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "size": array.size,
        "memory_usage": array.nbytes,
        "ndim": array.ndim,
    }

    # Add statistics for numeric arrays
    if np.issubdtype(array.dtype, np.number):
        try:
            info["min"] = float(np.min(array))
            info["max"] = float(np.max(array))
            info["mean"] = float(np.mean(array))
            info["std"] = float(np.std(array))
        except Exception:
            pass

    return info


def format_data_size(size_bytes: int) -> str:
    """Format data size in human-readable format.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    formatted : str
        Human-readable size (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def display_dataset_card(name: str, data: Any, metadata: Optional[Dict] = None):
    """Display a dataset summary card in Streamlit.

    Parameters
    ----------
    name : str
        Dataset name
    data : Any
        Dataset (DataFrame, ndarray, etc.)
    metadata : dict, optional
        Additional metadata to display
    """
    with st.container():
        st.subheader(f"📊 {name}")

        if isinstance(data, pd.DataFrame):
            info = get_dataframe_info(data)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", info["shape"][0])
            col2.metric("Columns", info["shape"][1])
            col3.metric("Memory", format_data_size(info["memory_usage"]))

            st.write("**Column Types:**")
            dtypes_df = pd.DataFrame(
                [{"Column": col, "Type": dtype} for col, dtype in info["dtypes"].items()]
            )
            st.dataframe(dtypes_df, hide_index=True, width="stretch")

            # Show missing values if any
            missing_total = sum(info["missing_values"].values())
            if missing_total > 0:
                st.warning(f"⚠️ {missing_total} missing values found")

        elif isinstance(data, np.ndarray):
            info = get_array_info(data)

            col1, col2, col3 = st.columns(3)
            col1.metric("Shape", str(info["shape"]))
            col2.metric("Type", info["dtype"])
            col3.metric("Memory", format_data_size(info["memory_usage"]))

            if "mean" in info:
                st.write("**Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Min", f"{info['min']:.4g}")
                col2.metric("Max", f"{info['max']:.4g}")
                col3.metric("Mean", f"{info['mean']:.4g}")
                col4.metric("Std Dev", f"{info['std']:.4g}")

        else:
            st.write(f"**Type:** {type(data).__name__}")
            if hasattr(data, "__len__"):
                st.write(f"**Length:** {len(data)}")

        # Display additional metadata if provided
        if metadata:
            with st.expander("Additional Metadata"):
                st.json(metadata)


def get_column_suggestions(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Suggest numeric/categorical columns for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    suggestions : dict
        Dictionary with suggestions:
        - x_candidates: columns suitable for X-axis
        - y_candidates: columns suitable for Y-axis
        - hue_candidates: columns suitable for grouping/color
    """
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    # For X-axis: prefer numeric or datetime
    x_candidates = numeric_cols.copy()

    # For Y-axis: numeric only
    y_candidates = numeric_cols.copy()

    # For hue/grouping: categorical or low-cardinality numeric
    hue_candidates = categorical_cols.copy()
    for col in numeric_cols:
        if df[col].nunique() <= 10:  # Low cardinality
            hue_candidates.append(col)

    return {
        "x_candidates": x_candidates,
        "y_candidates": y_candidates,
        "hue_candidates": hue_candidates,
        "numeric": numeric_cols,
        "categorical": categorical_cols,
    }


def display_data_preview(data: Any, name: str = "Data"):
    """Display an intelligent preview of any data type.

    Parameters
    ----------
    data : Any
        Data to preview
    name : str, default "Data"
        Name to display in preview
    """
    st.write(f"### {name} Preview")

    if isinstance(data, pd.DataFrame):
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")

        # Show preview
        preview = preview_dataframe(data, n_rows=5)
        st.dataframe(preview, width="stretch")

        # Quick stats for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            with st.expander("Quick Statistics"):
                st.dataframe(data[numeric_cols].describe(), width="stretch")

    elif isinstance(data, np.ndarray):
        info = get_array_info(data)
        st.write(f"**Shape:** {info['shape']}, **Type:** {info['dtype']}")

        # For 1D or 2D arrays, show a sample
        if data.ndim == 1:
            st.write("**First 10 elements:**")
            st.write(data[:10])
        elif data.ndim == 2:
            st.write("**First 5 rows:**")
            st.write(data[:5])
        else:
            st.write(f"**{data.ndim}D array** - too complex to preview directly")

    elif isinstance(data, dict):
        st.write(f"**Dictionary with {len(data)} keys:**")
        st.json({k: str(v)[:100] for k, v in list(data.items())[:10]})

    elif isinstance(data, (list, tuple)):
        st.write(f"**{type(data).__name__} with {len(data)} elements:**")
        st.write(data[:10] if len(data) > 10 else data)

    else:
        st.write(f"**Type:** {type(data).__name__}")
        st.write(data)


def get_plottable_arrays(data: Any) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Extract plottable arrays from various data types.

    Parameters
    ----------
    data : Any
        Input data (DataFrame, ndarray, dict, etc.)

    Returns
    -------
    arrays : np.ndarray or None
        Extracted numeric arrays, or None if not applicable
    labels : list of str or None
        Column/array labels, or None
    """
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return data[numeric_cols].values, list(numeric_cols)
        return None, None

    elif isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            return data, None
        return None, None

    elif isinstance(data, dict):
        # Try to extract numeric arrays from dict
        numeric_items = {
            k: v
            for k, v in data.items()
            if isinstance(v, (np.ndarray, list))
            and len(v) > 0
            and isinstance(v[0] if isinstance(v, list) else v.flat[0], (int, float, np.number))
        }
        if numeric_items:
            arrays = np.column_stack(list(numeric_items.values()))
            return arrays, list(numeric_items.keys())
        return None, None

    return None, None
