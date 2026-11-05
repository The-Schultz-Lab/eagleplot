"""Example EaglePlot plugin.

This file demonstrates the EaglePlot plugin interface. Drop a plugin_*.py file
into the plugins/ directory to have it auto-discovered by the app.

To create a custom plot type:
1. Define a plotting function with the standard EaglePlot signature:
   - Matplotlib: returns (fig, ax, info_dict)
   - Plotly: returns (fig, info_dict)
2. Add it to the list returned by get_plot_types().
"""

PLUGIN_NAME = "Example Plugin"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Demonstrates the EaglePlot plugin interface (no custom plots added)."


def get_plot_types() -> list[dict]:
    """Return custom plot types provided by this plugin.

    Returns
    -------
    list[dict]
        Each dict has keys: name, label, function, description.
        Return an empty list if this plugin adds no plot types.
    """
    return []


def get_analysis_tools() -> list[dict]:
    """Return custom analysis tools provided by this plugin.

    Returns
    -------
    list[dict]
        Each dict has keys: name, label, function, description.
        Return an empty list if this plugin adds no analysis tools.
    """
    return []
