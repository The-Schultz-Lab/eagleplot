"""User settings persistence for the EaglePlot GUI.

Saves user defaults and named configuration presets to a local
``config.json`` file alongside the app.  The file is intentionally
kept outside version control (add it to .gitignore).

Storage layout
--------------
config.json ::

    {
        "version": "1.0",
        "defaults": {
            "grid": true,
            "fontsize": 12,
            ...
        },
        "presets": {
            "My Dark Theme": { "grid": false, ... },
            "Publication": { "fontsize": 14, ... }
        },
        "workflows": {
            "My Pipeline": [
                {"operation": "statistics", "params": {}},
                {"operation": "peak_analysis",
                 "params": {"y_col": "intensity"}}
            ]
        }
    }

Public API
----------
get_config_path() -> Path
    Return the path to config.json (next to gui.py).
load_config() -> dict
    Read config.json; returns empty structure if missing.
save_config(config: dict) -> None
    Write the full config dict to disk.
get_defaults() -> dict
    Return the "defaults" sub-dict.
save_defaults(defaults: dict) -> None
    Overwrite the stored defaults.
list_presets() -> list[str]
    Return sorted preset names.
save_preset(name: str, settings: dict) -> None
    Save (or overwrite) a named preset.
load_preset(name: str) -> dict
    Return a preset dict, or {} if not found.
delete_preset(name: str) -> bool
    Remove a preset; returns True if it existed.
list_workflows() -> list[str]
    Return sorted workflow names.
save_workflow(name: str, steps: list[dict]) -> None
    Save (or overwrite) a named workflow.
load_workflow(name: str) -> list[dict]
    Return a workflow's step list, or [] if not found.
delete_workflow(name: str) -> bool
    Remove a workflow; returns True if it existed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# ── Location ──────────────────────────────────────────────────────────────
# Stored next to gui.py (two directories up from this file).
_APP_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _APP_ROOT / "config.json"

_EMPTY: Dict = {
    "version": "1.0",
    "defaults": {},
    "presets": {},
    "workflows": {},
}

# Keys exposed in the Settings UI (subset of _STYLE_KEYS that make sense as
# persistent defaults; data-column selectors are omitted).
SAVEABLE_KEYS: List[str] = [
    "grid",
    "grid_linestyle",
    "grid_which",
    "fontsize",
    "fontfamily",
    "fontcolor",
    "linewidth",
    "legend_frameon",
    "legend_framealpha",
    "legend_position",
    "y_scale",
    "y_notation",
    "y_transform",
    "color_palette",
]

# ── Read / write ──────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """Return the absolute path to config.json."""
    return _CONFIG_PATH


def load_config() -> Dict:
    """Read config.json from disk.

    Returns
    -------
    config : dict
        Full config dict, or an empty structure if the file is missing or
        corrupt.
    """
    if not _CONFIG_PATH.exists():
        return {
            "version": "1.0",
            "defaults": {},
            "presets": {},
            "workflows": {},
        }
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Ensure required top-level keys exist
        data.setdefault("version", "1.0")
        data.setdefault("defaults", {})
        data.setdefault("presets", {})
        data.setdefault("workflows", {})
        return data
    except (json.JSONDecodeError, OSError):
        return dict(_EMPTY)


def save_config(config: Dict) -> None:
    """Write the full config dict to disk.

    Parameters
    ----------
    config : dict
        Full config structure (version, defaults, presets).
    """
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


# ── Defaults ──────────────────────────────────────────────────────────────


def get_defaults() -> Dict:
    """Return the stored defaults dict (may be empty).

    Returns
    -------
    defaults : dict
        Persisted default settings, e.g. ``{"grid": True, "fontsize": 12}``.
    """
    return load_config().get("defaults", {})


def save_defaults(defaults: Dict) -> None:
    """Overwrite the stored defaults.

    Parameters
    ----------
    defaults : dict
        Mapping of setting keys to values.
    """
    config = load_config()
    config["defaults"] = defaults
    save_config(config)


# ── Presets ───────────────────────────────────────────────────────────────


def list_presets() -> List[str]:
    """Return a sorted list of saved preset names.

    Returns
    -------
    names : list[str]
    """
    return sorted(load_config().get("presets", {}).keys())


def save_preset(name: str, settings: Dict) -> None:
    """Save (or overwrite) a named preset.

    Parameters
    ----------
    name : str
        Human-readable preset name.
    settings : dict
        Configuration dict to save.
    """
    config = load_config()
    config["presets"][name] = settings
    save_config(config)


def load_preset(name: str) -> Dict:
    """Return a preset dict by name.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    settings : dict
        Preset dict, or ``{}`` if the preset does not exist.
    """
    return load_config().get("presets", {}).get(name, {})


def delete_preset(name: str) -> bool:
    """Remove a named preset.

    Parameters
    ----------
    name : str
        Preset name to remove.

    Returns
    -------
    removed : bool
        True if the preset existed and was removed.
    """
    config = load_config()
    if name in config.get("presets", {}):
        del config["presets"][name]
        save_config(config)
        return True
    return False


# ── Workflows ─────────────────────────────────────────────────────────────


def list_workflows() -> List[str]:
    """Return a sorted list of saved workflow names.

    Returns
    -------
    names : list[str]
        Alphabetically sorted workflow names stored in config.json.
    """
    return sorted(load_config().get("workflows", {}).keys())


def save_workflow(name: str, steps: List[Dict]) -> None:
    """Save (or overwrite) a named workflow.

    Args:
        name: Human-readable workflow name.
        steps: List of operation dicts, e.g.
            ``[{"operation": "statistics", "params": {}}]``.
            Stored under ``config["workflows"][name]``.
    """
    config = load_config()
    config["workflows"][name] = steps
    save_config(config)


def load_workflow(name: str) -> List[Dict]:
    """Return a workflow's step list by name.

    Args:
        name: Workflow name.

    Returns:
        List of step dicts, or ``[]`` if the workflow does not exist.
    """
    return load_config().get("workflows", {}).get(name, [])


def delete_workflow(name: str) -> bool:
    """Remove a named workflow.

    Args:
        name: Workflow name to remove.

    Returns:
        ``True`` if the workflow existed and was removed, ``False`` otherwise.
    """
    config = load_config()
    if name in config.get("workflows", {}):
        del config["workflows"][name]
        save_config(config)
        return True
    return False
