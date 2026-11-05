"""Settings Page.

Manage persistent user defaults and named configuration presets that are
saved to a local ``config.json`` file next to the app.  The file is
untracked by version control — add ``config.json`` to .gitignore.

Sections
--------
1. Default Settings  — global style defaults applied on every new plot
2. Presets           — save / load / delete named configurations
3. Config file info  — path and raw JSON viewer
"""

import json
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.utils import initialize_session_state
from eagleplot.utils.user_settings import (
    get_config_path,
    get_defaults,
    save_defaults,
    list_presets,
    save_preset,
    load_preset,
    delete_preset,
)
from eagleplot.utils.plot_config import COLOR_PALETTE_NAMES, _FONT_OPTIONS
from eagleplot.utils.theming import (
    STREAMLIT_DEFAULTS,
    read_theme,
    write_theme,
)

# Aliased: this page shows two different config files, and an unaliased import
# would shadow user_settings.get_config_path() (config.json) with the theming
# one (config.toml), silently repointing the "Config File" section below.
from eagleplot.utils.theming import get_config_path as get_theme_config_path

initialize_session_state()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("Settings")
st.caption(
    "Persistent defaults and named presets are saved to **`config.json`**. "
    "Changes take effect the next time you open Quick Plot."
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 0 — Theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## Theme")
st.markdown(
    "Choose a base theme and optionally customize colours. "
    "Click **Save theme** then **restart the app** for changes to take effect."
)
st.caption(
    f"Theme is saved to `{get_theme_config_path()}` — the location Streamlit reads when "
    "launched from this directory."
)

_cur = read_theme()

with st.form("theme_form"):
    _tc1, _tc2 = st.columns(2)
    with _tc1:
        _base = st.selectbox(
            "Base theme",
            ["light", "dark"],
            index=0 if _cur.get("base", "light") == "light" else 1,
        )
    with _tc2:
        _primary = st.color_picker(
            "Primary colour",
            value=_cur.get("primaryColor", "#1f77b4"),
        )

    _tc3, _tc4, _tc5 = st.columns(3)
    with _tc3:
        _bg = st.color_picker(
            "Background",
            value=_cur.get("backgroundColor", "#ffffff"),
        )
    with _tc4:
        _sbg = st.color_picker(
            "Sidebar / widget background",
            value=_cur.get("secondaryBackgroundColor", "#f0f2f6"),
        )
    with _tc5:
        _text = st.color_picker(
            "Text colour",
            value=_cur.get("textColor", "#262730"),
        )

    _theme_submitted = st.form_submit_button("Save theme", type="primary")


def _save_theme(theme: dict, success_message: str) -> None:
    """Write a theme and report what actually happened.

    The previous implementation returned silently when config.toml was missing --
    the default for every pip-installed user -- while still printing a success
    message. See audit A-18.
    """
    try:
        written = write_theme(theme)
    except OSError as exc:
        st.error(
            f"Could not write the theme: {exc}\n\n"
            f"EaglePlot tried to write `{get_theme_config_path()}`. If that directory is "
            "read-only, launch EaglePlot from a directory you can write to, or create "
            "`.streamlit/config.toml` there yourself."
        )
        return
    st.success(f"{success_message} Saved to `{written}` — restart the app to apply.")


if _theme_submitted:
    _save_theme(
        {
            "base": _base,
            "primaryColor": _primary,
            "backgroundColor": _bg,
            "secondaryBackgroundColor": _sbg,
            "textColor": _text,
        },
        "Theme saved.",
    )

if st.button("Reset theme to Streamlit default"):
    _save_theme(dict(STREAMLIT_DEFAULTS), "Theme reset to Streamlit defaults.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Default settings
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Default Settings")
st.markdown(
    "These values are pre-filled in the Quick Plot appearance controls "
    "whenever a new session starts."
)

current = get_defaults()

with st.form("defaults_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        grid = st.checkbox(
            "Show grid by default",
            value=current.get("grid", True),
        )
        grid_linestyle = st.selectbox(
            "Default grid style",
            ["--", "-", ":", "-."],
            index=["--", "-", ":", "-."].index(current.get("grid_linestyle", "--")),
        )

    with c2:
        fontsize = st.slider(
            "Default font size",
            8,
            24,
            value=int(current.get("fontsize", 11)),
        )
        fontfamily = st.selectbox(
            "Default font family",
            _FONT_OPTIONS,
            index=(
                _FONT_OPTIONS.index(current.get("fontfamily", "sans-serif"))
                if current.get("fontfamily", "sans-serif") in _FONT_OPTIONS
                else 0
            ),
        )

    with c3:
        color_palette = st.selectbox(
            "Default color palette",
            COLOR_PALETTE_NAMES,
            index=(
                COLOR_PALETTE_NAMES.index(current.get("color_palette", "Default"))
                if current.get("color_palette", "Default") in COLOR_PALETTE_NAMES
                else 0
            ),
        )
        linewidth = st.slider(
            "Default line width",
            0.5,
            6.0,
            value=float(current.get("linewidth", 1.5)),
            step=0.5,
        )

    c4, c5 = st.columns(2)
    with c4:
        legend_position = st.selectbox(
            "Default legend position",
            [
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
                "outside right",
            ],
            index=[
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
                "outside right",
            ].index(current.get("legend_position", "best")),
        )
    with c5:
        y_notation = st.selectbox(
            "Default Y notation",
            ["default", "scientific", "engineering"],
            index=["default", "scientific", "engineering"].index(
                current.get("y_notation", "default")
            ),
        )

    submitted = st.form_submit_button("Save defaults", type="primary")

if submitted:
    save_defaults(
        {
            "grid": grid,
            "grid_linestyle": grid_linestyle,
            "fontsize": fontsize,
            "fontfamily": fontfamily,
            "color_palette": color_palette,
            "linewidth": linewidth,
            "legend_position": legend_position,
            "y_notation": y_notation,
        }
    )
    st.session_state.pop("user_defaults_loaded", None)
    st.success("Defaults saved to config.json.")

# ── Reset button (outside the form) ──────────────────────────────────────────
# Each destructive action on this page is disabled until its own confirmation
# box is ticked. They used to fire on a single click. See audit A-33.
_confirm_reset = st.checkbox(
    "I understand resetting clears my saved defaults",
    key="confirm_reset_defaults",
)
if st.button("Reset defaults to factory settings", disabled=not _confirm_reset):
    save_defaults({})
    st.session_state.pop("user_defaults_loaded", None)
    st.session_state.pop("confirm_reset_defaults", None)  # re-arm for next time
    st.success("Defaults cleared.")
    st.rerun()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Presets
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Presets")
st.markdown(
    "A preset captures the current default settings under a name so you can "
    "switch between configurations quickly (e.g., *Publication*, *Poster*, "
    "*Dark Background*)."
)

tab_save, tab_load, tab_delete = st.tabs(["Save preset", "Load preset", "Delete preset"])

# ── Save ──────────────────────────────────────────────────────────────────────
with tab_save:
    st.markdown(
        "Enter a name and click **Save** to snapshot the current defaults "
        "as a named preset.  An existing preset with the same name will be "
        "overwritten."
    )
    preset_name_save = st.text_input(
        "Preset name",
        placeholder="e.g. Publication",
        key="preset_save_name",
    )
    if st.button("Save as preset", type="primary", key="preset_save_btn"):
        if not preset_name_save.strip():
            st.warning("Please enter a preset name.")
        else:
            save_preset(preset_name_save.strip(), get_defaults())
            st.success(f"Preset **{preset_name_save.strip()}** saved.")

# ── Load ──────────────────────────────────────────────────────────────────────
with tab_load:
    presets = list_presets()
    if not presets:
        st.info("No presets saved yet.")
    else:
        preset_name_load = st.selectbox(
            "Choose preset",
            presets,
            key="preset_load_sel",
        )
        if st.button("Load preset", type="primary", key="preset_load_btn"):
            settings = load_preset(preset_name_load)
            save_defaults(settings)
            st.session_state.pop("user_defaults_loaded", None)
            st.success(f"Preset **{preset_name_load}** loaded as defaults.")
            st.rerun()

        # Preview
        with st.expander("Preview preset values"):
            st.json(load_preset(preset_name_load))

# ── Delete ────────────────────────────────────────────────────────────────────
with tab_delete:
    presets = list_presets()
    if not presets:
        st.info("No presets saved yet.")
    else:
        preset_name_del = st.selectbox(
            "Choose preset to delete",
            presets,
            key="preset_del_sel",
        )
        _confirm_del = st.checkbox(
            f"I understand **{preset_name_del}** will be deleted permanently",
            key="confirm_preset_del",
        )
        if st.button(
            "Delete preset",
            type="primary",
            key="preset_del_btn",
            disabled=not _confirm_del,
        ):
            removed = delete_preset(preset_name_del)
            if removed:
                st.session_state.pop("confirm_preset_del", None)
                st.success(f"Preset **{preset_name_del}** deleted.")
                st.rerun()
            else:
                st.error("Preset not found.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Config file info
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Config File")

cfg_path = get_config_path()
st.markdown(f"**Location:** `{cfg_path}`")

if cfg_path.exists():
    st.markdown(
        f"**Size:** {cfg_path.stat().st_size:,} bytes  | **Presets saved:** {len(list_presets())}"
    )
    with st.expander("View raw config.json"):
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            st.json(raw)
        except Exception as exc:
            st.error(f"Could not read config.json: {exc}")

    st.download_button(
        "Download a backup of config.json",
        data=cfg_path.read_bytes(),
        file_name="eagleplot_config_backup.json",
        mime="application/json",
        key="dl_config_backup",
    )
    _confirm_wipe = st.checkbox(
        "I understand this deletes all my defaults **and** presets",
        key="confirm_config_delete",
    )
    if st.button("Delete config.json (reset everything)", disabled=not _confirm_wipe):
        # missing_ok: the file can vanish between the exists() check above and
        # here, which used to surface a raw FileNotFoundError.
        cfg_path.unlink(missing_ok=True)
        st.session_state.pop("confirm_config_delete", None)
        st.success("config.json deleted.  All defaults and presets cleared.")
        st.rerun()
else:
    st.info("No config.json file found — it will be created when you save defaults or a preset.")
