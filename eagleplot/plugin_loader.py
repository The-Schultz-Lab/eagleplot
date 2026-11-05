"""Plugin loader for EaglePlot.

Discovers and loads plugin_*.py files from the plugins/ directory.
Each plugin is a standard Python module that optionally exposes:

    PLUGIN_NAME: str          — human-readable name
    PLUGIN_VERSION: str       — semver string
    PLUGIN_DESCRIPTION: str   — one-line description

    get_plot_types() -> list[dict]
        Returns a list of plot-type registration dicts, each with keys:
        - 'name': str  (unique identifier)
        - 'label': str (display label)
        - 'function': callable  (plotting function)
        - 'description': str (optional)

    get_analysis_tools() -> list[dict]
        Returns a list of analysis-tool dicts with keys:
        - 'name': str
        - 'label': str
        - 'function': callable
        - 'description': str (optional)

Public API
----------
discover_plugins(plugins_dir=None) -> dict[str, types.ModuleType]
    Scan plugins_dir (default: repo-root/plugins/) for plugin_*.py files.
    Import each valid plugin and return {filename_stem: module}.
    Silently skips files that raise ImportError or other exceptions during load.

get_plugin_plot_types(plugins=None) -> list[dict]
    Return all plot-type dicts from all discovered plugins.
    plugins: pass result of discover_plugins() to avoid re-scanning.

get_plugin_analysis_tools(plugins=None) -> list[dict]
    Return all analysis-tool dicts from all discovered plugins.

list_plugins(plugins=None) -> list[dict]
    Return a summary list of loaded plugins:
    [{'name': ..., 'version': ..., 'description': ..., 'file': ...}, ...]
"""

import importlib.util
import types
from pathlib import Path
from typing import Optional

_PLUGINS_DIR = Path(__file__).parent.parent / "plugins"


def discover_plugins(
    plugins_dir: Optional[Path] = None,
) -> dict[str, types.ModuleType]:
    """Scan plugins_dir for plugin_*.py files and import each one.

    Parameters
    ----------
    plugins_dir : Path, optional
        Directory to scan. Defaults to ``<repo-root>/plugins/``.

    Returns
    -------
    dict[str, types.ModuleType]
        Mapping of filename stem (e.g. ``"plugin_example"``) to the loaded
        module object. Files that fail to import are silently skipped.
    """
    if plugins_dir is None:
        plugins_dir = _PLUGINS_DIR

    plugins_dir = Path(plugins_dir)

    if not plugins_dir.is_dir():
        return {}

    loaded: dict[str, types.ModuleType] = {}

    for path in sorted(plugins_dir.glob("plugin_*.py")):
        stem = path.stem
        try:
            spec = importlib.util.spec_from_file_location(stem, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            loaded[stem] = module
        except Exception:
            # Bad plugins must never crash the app
            pass

    return loaded


def get_plugin_plot_types(
    plugins: Optional[dict[str, types.ModuleType]] = None,
) -> list[dict]:
    """Return all plot-type dicts contributed by discovered plugins.

    Parameters
    ----------
    plugins : dict, optional
        Pre-loaded plugins dict from :func:`discover_plugins`. If *None*,
        :func:`discover_plugins` is called automatically.

    Returns
    -------
    list[dict]
        Concatenated list of plot-type dicts from every plugin that exposes
        ``get_plot_types()``. Each dict has at minimum ``name``, ``label``,
        and ``function`` keys.
    """
    if plugins is None:
        plugins = discover_plugins()

    result: list[dict] = []

    for module in plugins.values():
        if not hasattr(module, "get_plot_types"):
            continue
        try:
            items = module.get_plot_types()
            if isinstance(items, list):
                result.extend(items)
        except Exception:
            pass

    return result


def get_plugin_analysis_tools(
    plugins: Optional[dict[str, types.ModuleType]] = None,
) -> list[dict]:
    """Return all analysis-tool dicts contributed by discovered plugins.

    Parameters
    ----------
    plugins : dict, optional
        Pre-loaded plugins dict from :func:`discover_plugins`. If *None*,
        :func:`discover_plugins` is called automatically.

    Returns
    -------
    list[dict]
        Concatenated list of analysis-tool dicts from every plugin that
        exposes ``get_analysis_tools()``. Each dict has at minimum ``name``,
        ``label``, and ``function`` keys.
    """
    if plugins is None:
        plugins = discover_plugins()

    result: list[dict] = []

    for module in plugins.values():
        if not hasattr(module, "get_analysis_tools"):
            continue
        try:
            items = module.get_analysis_tools()
            if isinstance(items, list):
                result.extend(items)
        except Exception:
            pass

    return result


def list_plugins(
    plugins: Optional[dict[str, types.ModuleType]] = None,
) -> list[dict]:
    """Return a human-readable summary of all loaded plugins.

    Parameters
    ----------
    plugins : dict, optional
        Pre-loaded plugins dict from :func:`discover_plugins`. If *None*,
        :func:`discover_plugins` is called automatically.

    Returns
    -------
    list[dict]
        One dict per plugin with keys:

        - ``name`` — ``PLUGIN_NAME`` attribute, or the file stem if absent
        - ``version`` — ``PLUGIN_VERSION`` attribute, or ``"unknown"``
        - ``description`` — ``PLUGIN_DESCRIPTION`` attribute, or ``""``
        - ``file`` — absolute :class:`~pathlib.Path` to the plugin file
    """
    if plugins is None:
        plugins = discover_plugins()

    summary: list[dict] = []

    for stem, module in plugins.items():
        name = getattr(module, "PLUGIN_NAME", stem)
        version = getattr(module, "PLUGIN_VERSION", "unknown")
        description = getattr(module, "PLUGIN_DESCRIPTION", "")

        # Retrieve the file path from the module's __file__ attribute
        file_path = Path(getattr(module, "__file__", ""))

        summary.append(
            {
                "name": name,
                "version": version,
                "description": description,
                "file": file_path,
            }
        )

    return summary
