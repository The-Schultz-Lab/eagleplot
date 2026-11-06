"""Theme reading, writing, and CSS generation for the EaglePlot GUI.

Why this module exists
----------------------
Three related defects motivated it (audit A-17, A-18, A-19):

- **A-17** — ``Home.py`` injected a stylesheet with the NCCU dark palette baked in
  as literals, while the Settings page let the user change the theme. Nothing
  connected the two, so on a light theme the hero subtitle was white-on-white and
  the scrollbar stayed black. Worse, two of those literals disagreed with each
  other, so one was always low-contrast *whatever* theme was active.
  :func:`app_css` now derives every colour from the live theme.

- **A-18** — the Settings page wrote to ``.streamlit/config.toml`` resolved relative
  to the package, returned silently when the file did not exist, and reported
  "Theme saved" regardless. That silent path was the default for every
  pip-installed user, since the wheel does not ship ``.streamlit/``.
  :func:`write_theme` targets the location Streamlit actually reads, creates it
  when missing, and raises on failure so the caller can tell the truth.

- **A-19** — the old writer re-serialised the whole file from parsed TOML, which
  destroyed every comment, dropped non-dict top-level values, and emitted Python
  ``repr`` for lists (invalid TOML). :func:`write_theme` rewrites only the
  ``[theme]`` keys and leaves the rest of the file byte-identical, writing
  atomically so an interrupted save cannot truncate the config.

Config file location
--------------------
Streamlit reads ``$CWD/.streamlit/config.toml`` and ``~/.streamlit/config.toml``.
It does *not* look inside the installed package, so the project-level path
relative to the working directory is the only target that works both from a source
checkout and from a ``pip install``.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# tomllib is stdlib from Python 3.11; EaglePlot supports 3.9+ (G-013).
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 / 3.10
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = [
    "THEME_KEYS",
    "STREAMLIT_DEFAULTS",
    "get_config_path",
    "read_theme",
    "write_theme",
    "active_theme",
    "app_css",
]

THEME_KEYS: List[str] = [
    "base",
    "primaryColor",
    "backgroundColor",
    "secondaryBackgroundColor",
    "textColor",
]

#: Streamlit's own light-theme defaults, used when nothing else is available.
STREAMLIT_DEFAULTS: Dict[str, str] = {
    "base": "light",
    "primaryColor": "#1f77b4",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f0f2f6",
    "textColor": "#262730",
}

_KEY_LINE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")
_SECTION_LINE = re.compile(r"^\s*\[")


def get_config_path() -> Path:
    """Return the ``config.toml`` path Streamlit will actually read.

    Returns
    -------
    Path
        ``<cwd>/.streamlit/config.toml``. Resolved against the working directory
        rather than the package, because Streamlit does not read config from
        inside an installed package -- see the module docstring.
    """
    return Path.cwd() / ".streamlit" / "config.toml"


def read_theme() -> Dict[str, str]:
    """Read the ``[theme]`` table, falling back to the live theme then defaults.

    Returns
    -------
    dict
        All five :data:`THEME_KEYS`, always populated.
    """
    theme = dict(STREAMLIT_DEFAULTS)
    theme.update(active_theme())

    path = get_config_path()
    if path.is_file():
        try:
            with open(path, "rb") as fh:
                stored = tomllib.load(fh).get("theme", {})
            theme.update({k: str(v) for k, v in stored.items() if k in THEME_KEYS})
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return theme


def active_theme() -> Dict[str, str]:
    """Return the theme Streamlit is currently running with.

    Reads ``st.get_option("theme.*")``, which reflects the merged config actually
    in effect -- so injected CSS matches what the user sees even when the config
    file is absent, unreadable, or overridden by a CLI flag or environment
    variable.

    Returns
    -------
    dict
        Subset of :data:`THEME_KEYS` that Streamlit reports a value for. Empty if
        Streamlit is unavailable (so this module stays importable in tests).
    """
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover - streamlit is a hard dependency
        return {}

    resolved: Dict[str, str] = {}
    for key in THEME_KEYS:
        try:
            value = st.get_option(f"theme.{key}")
        except Exception:
            continue
        if value:
            resolved[key] = str(value)
    return resolved


def _render_theme_body(theme: Dict[str, str]) -> List[str]:
    """Render the ``[theme]`` key lines, in :data:`THEME_KEYS` order."""
    return [f'{key} = "{theme[key]}"\n' for key in THEME_KEYS if theme.get(key)]


def write_theme(theme: Dict[str, str]) -> Path:
    """Rewrite only the ``[theme]`` keys in ``config.toml``.

    Everything outside the ``[theme]`` table -- other sections, and every comment
    including the file header -- is preserved byte-for-byte. Comments *inside* the
    table are kept too; only ``key = value`` lines are replaced. The write is
    atomic, so an interrupted save cannot leave a truncated config.

    Parameters
    ----------
    theme : dict
        Values for :data:`THEME_KEYS`. Keys outside that set are ignored; keys
        with a falsy value are omitted from the output.

    Returns
    -------
    Path
        The file that was written, for display to the user.

    Raises
    ------
    OSError
        If the file or its parent directory cannot be created or written. Callers
        must surface this rather than reporting success -- see A-18.
    """
    path = get_config_path()
    existing: List[str] = []
    if path.is_file():
        existing = path.read_text(encoding="utf-8").splitlines(keepends=True)

    out: List[str] = []
    index = 0
    replaced = False

    while index < len(existing):
        line = existing[index]
        if line.strip() != "[theme]":
            out.append(line)
            index += 1
            continue

        # Found the table: emit the header, then the new keys, then walk the old
        # body dropping only `key = value` lines so comments and blanks survive.
        out.append(line)
        out.extend(_render_theme_body(theme))
        replaced = True
        index += 1
        while index < len(existing) and not _SECTION_LINE.match(existing[index]):
            if not _KEY_LINE.match(existing[index]):
                out.append(existing[index])
            index += 1

    if not replaced:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append("\n[theme]\n")
        out.extend(_render_theme_body(theme))

    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic replace: write a sibling temp file, flush, then rename over the target.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".config-", suffix=".toml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("".join(out))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def _hex_to_rgba(colour: str, alpha: float) -> str:
    """Convert ``#rrggbb`` (or ``#rgb``) to a CSS ``rgba()`` string.

    Falls back to a mid grey for anything unparseable, so a malformed colour in
    config.toml degrades to readable rather than breaking the stylesheet.
    """
    text = colour.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        red, green, blue = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        red, green, blue = (128, 128, 128)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def app_css(theme: Optional[Dict[str, str]] = None) -> str:
    """Build the app-wide stylesheet from the active theme.

    Every colour is derived from *theme*, so the injected CSS follows whatever the
    user selected in Settings instead of assuming the NCCU dark palette (A-17).

    Parameters
    ----------
    theme : dict, optional
        Theme mapping; defaults to :func:`read_theme`.

    Returns
    -------
    str
        A ``<style>`` block, ready for ``st.markdown(..., unsafe_allow_html=True)``.

    Notes
    -----
    The font is declared as a stack of locally-available families. An earlier
    version pulled Nunito from ``fonts.googleapis.com``, which fails on an
    air-gapped lab machine and discloses a third-party request on every load
    (A-35); Streamlit also does not reliably preserve ``<link>`` elements passed
    through ``st.markdown``, so it may never have loaded at all.
    """
    resolved = dict(theme or read_theme())
    primary = resolved.get("primaryColor", STREAMLIT_DEFAULTS["primaryColor"])
    text = resolved.get("textColor", STREAMLIT_DEFAULTS["textColor"])
    background = resolved.get("backgroundColor", STREAMLIT_DEFAULTS["backgroundColor"])
    secondary = resolved.get(
        "secondaryBackgroundColor", STREAMLIT_DEFAULTS["secondaryBackgroundColor"]
    )

    muted = _hex_to_rgba(text, 0.55)
    faint = _hex_to_rgba(text, 0.45)

    return f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Nunito', 'Avenir Next', 'Avenir', 'Segoe UI',
                     'Helvetica Neue', Arial, sans-serif !important;
    }}
    /* Hide Streamlit chrome. The sidebar collapse control is deliberately NOT
       hidden -- doing so locked the sidebar open, which cost plot canvas width on
       small screens and at high browser zoom (A-29). */
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stDecoration"]   {{ display: none !important; }}
    /* Streamlit's default 6rem of space above each page pushed plots below the fold
       on a 768 px laptop screen; 3rem still clears the sidebar toggle. */
    [data-testid="stMainBlockContainer"] {{ padding-top: 3rem !important; }}
    [data-testid="stToolbar"]      {{ display: none !important; }}
    #MainMenu                      {{ display: none !important; }}
    footer                         {{ display: none !important; }}
    /* Active page indicator */
    [data-testid="stSidebar"] a[aria-current="page"] {{
        border-left: 3px solid {primary} !important;
        padding-left: 0.4rem !important;
        color: {primary} !important;
        font-weight: 600 !important;
    }}
    /* Muted text helpers, derived from the theme's text colour */
    .eagleplot-muted {{ color: {muted}; margin-top: 0; }}
    .eagleplot-section-label {{
        margin: 0.6rem 0 0.15rem 0;
        font-size: 0.68rem;
        color: {faint};
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
    }}
    /* Scrollbar, tinted to the active theme rather than a fixed dark palette */
    ::-webkit-scrollbar             {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track       {{ background: {background}; }}
    ::-webkit-scrollbar-thumb       {{ background: {secondary}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {primary}; }}
    </style>
    """
