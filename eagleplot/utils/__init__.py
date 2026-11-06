"""Utility modules for Streamlit GUI."""

# The names below are loaded from their submodules on first use (PEP 562), not when the
# package is imported. Importing session_state or data_preview pulls in pandas, which takes
# ~2.3 s in the browser build; Home and small helpers (``eagleplot.utils.runtime`` …) must
# not pay that at start-up (roadmap D4.1). ``from eagleplot.utils import X`` works as before.
_EXPORTS = {
    "session_state": (
        "initialize_session_state",
        "add_dataset",
        "get_current_dataset",
        "get_dataset",
        "delete_dataset",
        "add_plot_to_history",
        "clear_plot_history",
        "add_analysis_result",
        "save_session_to_file",
        "load_session_from_file",
        "clear_session",
        "get_session_summary",
    ),
    "data_preview": (
        "preview_dataframe",
        "get_dataframe_info",
        "get_array_info",
        "format_data_size",
        "display_dataset_card",
        "get_column_suggestions",
        "display_data_preview",
        "get_plottable_arrays",
    ),
    "user_settings": (
        "get_config_path",
        "load_config",
        "save_config",
        "get_defaults",
        "save_defaults",
        "list_presets",
        "save_preset",
        "load_preset",
        "delete_preset",
        "SAVEABLE_KEYS",
    ),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}


def __getattr__(name):
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_MODULE_OF))


__all__ = [
    # Session state
    "initialize_session_state",
    "add_dataset",
    "get_current_dataset",
    "get_dataset",
    "delete_dataset",
    "add_plot_to_history",
    "clear_plot_history",
    "add_analysis_result",
    "save_session_to_file",
    "load_session_from_file",
    "clear_session",
    "get_session_summary",
    # Data preview
    "preview_dataframe",
    "get_dataframe_info",
    "get_array_info",
    "format_data_size",
    "display_dataset_card",
    "get_column_suggestions",
    "display_data_preview",
    "get_plottable_arrays",
    # User settings
    "get_config_path",
    "load_config",
    "save_config",
    "get_defaults",
    "save_defaults",
    "list_presets",
    "save_preset",
    "load_preset",
    "delete_preset",
    "SAVEABLE_KEYS",
]
