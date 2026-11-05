"""Render smoke tests for every Streamlit page (audit finding A-22).

Each page script is executed headlessly with ``streamlit.testing.v1.AppTest`` on a
fresh session -- no datasets loaded, no saved settings -- and must finish without an
uncaught exception. Pages that need data are expected to show an info/warning
message and stop; that is fine. Only exceptions fail.

This catches import errors, renamed helpers, and top-level code that assumes
session state a first-time visitor does not have. It does not exercise widget
interactions.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "eagleplot"
_HOME = _PACKAGE_ROOT / "Home.py"
_PAGES = sorted(p for p in (_PACKAGE_ROOT / "pages").glob("*.py") if not p.name.startswith("_"))
_ALL_SCRIPTS = [_HOME] + _PAGES

# Generous: the first page run in a process pays for importing scipy/plotly/etc.
_TIMEOUT_S = 30


def _script_id(path: Path) -> str:
    return str(path.relative_to(_PACKAGE_ROOT))


def test_page_discovery_found_the_app():
    """Guard against the glob silently matching nothing (which would pass vacuously)."""
    assert _HOME.is_file()
    assert len(_PAGES) >= 10, [p.name for p in _PAGES]


@pytest.mark.parametrize("script", _ALL_SCRIPTS, ids=_script_id)
def test_page_renders_on_fresh_session(script: Path):
    at = AppTest.from_file(str(script), default_timeout=_TIMEOUT_S).run()
    errors = [f"{e.message}\n{''.join(e.stack_trace)}" for e in at.exception]
    assert not at.exception, f"{_script_id(script)} raised on a fresh session:\n" + "\n".join(
        errors
    )
