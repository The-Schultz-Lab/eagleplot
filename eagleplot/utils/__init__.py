"""Utility modules for Streamlit GUI."""

from .session_state import (
    initialize_session_state,
    add_dataset,
    get_current_dataset,
    get_dataset,
    delete_dataset,
    add_plot_to_history,
    clear_plot_history,
    add_analysis_result,
    save_session_to_file,
    load_session_from_file,
    clear_session,
    get_session_summary,
)

from .data_preview import (
    preview_dataframe,
    get_dataframe_info,
    get_array_info,
    format_data_size,
    display_dataset_card,
    get_column_suggestions,
    display_data_preview,
    get_plottable_arrays,
)

from .user_settings import (
    get_config_path,
    load_config,
    save_config,
    get_defaults,
    save_defaults,
    list_presets,
    save_preset,
    load_preset,
    delete_preset,
    SAVEABLE_KEYS,
)

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
