"""Gallery Page.

Displays pre-rendered example figures grouped by library (Matplotlib,
Seaborn, Plotly).  Each card shows a thumbnail, title, description, and a
"Use this config" button that pre-fills Quick Plot with the figure's settings.

To populate the gallery run once from the repo root::

    python generate_gallery.py

The script writes PNGs and ``docs/gallery/manifest.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_APP_ROOT = Path(__file__).resolve().parent.parent.parent
# Packaged alongside the code so the Gallery is populated after `pip install`
# rather than only in a source checkout -- see audit A-09.
_GALLERY_DIR = Path(__file__).resolve().parent.parent / "gallery"
_MANIFEST_PATH = _GALLERY_DIR / "manifest.json"

sys.path.insert(0, str(_APP_ROOT))

from plottle.utils import initialize_session_state

initialize_session_state()

# ── Equal-height cards ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]
        > [data-testid="stVerticalBlock"] {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]
        > [data-testid="stVerticalBlock"]
        > [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("Gallery")
st.caption(
    "Browse example figures. Click **Use this config** to pre-fill Quick Plot "
    "with that figure's settings."
)

# ── Guard: manifest missing ───────────────────────────────────────────────────
if not _MANIFEST_PATH.exists():
    st.warning(
        "No gallery figures found.  "
        "Run **`python generate_gallery.py`** from the repo root to generate them."
    )
    st.code("python generate_gallery.py", language="bash")
    st.stop()

# ── Load manifest ─────────────────────────────────────────────────────────────
with open(_MANIFEST_PATH, encoding="utf-8") as fh:
    _manifest: list[dict] = json.load(fh)

# ── Group by library ──────────────────────────────────────────────────────────
_LIBRARY_ORDER = ["Matplotlib", "Seaborn", "Plotly"]

_by_library: dict[str, list[dict]] = {}
for entry in _manifest:
    _by_library.setdefault(entry.get("library", "Other"), []).append(entry)

# Add any user-saved entries that live in the gallery dir but not in manifest
_user_entries: list[dict] = []
_manifest_keys = {e["key"] for e in _manifest}
for png in sorted(_GALLERY_DIR.glob("user_*.png")):
    key = png.stem
    meta_path = _GALLERY_DIR / f"{key}.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if key not in _manifest_keys:
                _user_entries.append(meta)
        except Exception:
            pass

if _user_entries:
    _by_library.setdefault("My Figures", _user_entries)

# ── Render sections ───────────────────────────────────────────────────────────
_lib_order = [lib for lib in _LIBRARY_ORDER if lib in _by_library]
_lib_order += [lib for lib in _by_library if lib not in _LIBRARY_ORDER]

# _DEFAULTS_KEY_MAP inverse — needed to restore settings into widget keys.
# Mirror of the map in 2_Quick_Plot.py.
_CFG_TO_WIDGET: dict[str, str] = {
    "fontfamily": "_shared_fontfamily",
    "fontcolor": "_shared_fontcolor",
    "linewidth": "_shared_lw",
    "grid": "_shared_grid",
    "grid_linestyle": "_shared_grid_ls",
    "grid_which": "_shared_grid_which",
    "legend_frameon": "_shared_leg_frame",
    "legend_framealpha": "_shared_leg_bg",
    "legend_position": "_shared_leg_pos",
    "color_palette": "_shared_palette",
    "fontsize": "_shared_ply_fontsize",
    "fontsize_label": "_shared_fs_label",
    "fontsize_tick": "_shared_fs_tick",
    "fontsize_title": "_shared_fs_title",
    "fontsize_legend": "_shared_fs_legend",
}


def _apply_gallery_config(cfg: dict) -> None:
    """Write a gallery config dict into the shared Quick Plot widget keys."""
    for cfg_key, widget_key in _CFG_TO_WIDGET.items():
        if cfg_key in cfg:
            # Attempt type coercion — manifest stores everything as strings.
            val = cfg[cfg_key]
            current = st.session_state.get(widget_key)
            if isinstance(current, bool):
                st.session_state[widget_key] = str(val).lower() in ("true", "1")
            elif isinstance(current, (int, float)) or current is None:
                try:
                    st.session_state[widget_key] = (
                        type(current)(val) if current is not None else val
                    )
                except (ValueError, TypeError):
                    st.session_state[widget_key] = val
            else:
                st.session_state[widget_key] = val


for lib in _lib_order:
    entries = _by_library[lib]
    st.markdown(f"## {lib}")

    # 3-column grid — new st.columns() per row so cards align across columns
    for row_start in range(0, len(entries), 3):
        row_entries = entries[row_start : row_start + 3]
        cols = st.columns(3)
        for col, entry in zip(cols, row_entries):
            with col:
                with st.container(border=True):
                    img_path = _GALLERY_DIR / entry.get("filename", f"{entry['key']}.png")
                    if img_path.exists():
                        st.image(str(img_path), width="stretch")
                    else:
                        st.caption("_(image not generated yet)_")

                    st.markdown(
                        f"<div style='min-height:5.5em'>"
                        f"<strong>{entry['title']}</strong><br>"
                        f"<span style='font-size:0.82rem;color:inherit'>"
                        f"{entry.get('description', '')}</span><br><br>"
                        f"<span style='font-size:0.8rem'>"
                        f"<em>Best for:</em> {entry.get('best_for', '—')}</span>"
                        f"</div>"
                        f"<p style='font-size:0.78rem;margin-top:0.3rem'>"
                        f"Dataset: <code>{entry.get('dataset', '—')}</code></p>",
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "Use this config",
                        key=f"gallery_apply_{entry['key']}",
                        width="stretch",
                    ):
                        _apply_gallery_config(entry.get("config", {}))
                        st.session_state["qp_category"] = lib
                        st.success(
                            f"Config loaded.  Go to **Quick Plot** and choose "
                            f"**{entry['title']}** to generate the figure."
                        )

    st.markdown("---")
