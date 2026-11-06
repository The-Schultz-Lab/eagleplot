"""What the current runtime can do (roadmap D2.1).

EaglePlot runs in three places:

* **server**: ``streamlit run`` on CPython, local or shared (the classic GUI);
* **browser**: stlite, i.e. Streamlit on Pyodide/WebAssembly (``sys.platform ==
  "emscripten"``), served as a static web app;
* **desktop**: the same stlite build inside Electron, which sets
  ``EAGLEPLOT_DESKTOP=1``.

Pages should branch on *what is possible* (:class:`Capabilities`), never on *which
build this is*, so a capability can change in one build without touching the pages.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

#: Set to ``1`` by the Electron desktop build.
DESKTOP_ENV = "EAGLEPLOT_DESKTOP"

#: Running under Pyodide (stlite, in a browser or in Electron).
IS_PYODIDE = sys.platform == "emscripten"
#: Running in the Electron desktop build (always Pyodide as well).
IS_DESKTOP = IS_PYODIDE and os.environ.get(DESKTOP_ENV) == "1"


@dataclass(frozen=True)
class Capabilities:
    """What the current runtime allows. Computed once per process by :func:`capabilities`."""

    #: A folder path typed by the user names a real directory this process can read.
    local_folders: bool
    #: Outbound HTTP(S) requests work (e.g. NIST WebBook lookups). In the browser they
    #: are subject to CORS, and NIST is expected to block them.
    outbound_http: bool
    #: Files written now (config.json, presets, plugins) are still there next session.
    persistent_storage: bool
    #: Native CPython speed. ``False`` under WebAssembly: pages may lower display dpi
    #: or skip expensive previews.
    native_speed: bool


def _browser_storage_is_persistent(environ: Mapping[str, str]) -> bool:
    """Hook for D3: return ``True`` once the browser build mounts IDBFS for user data.

    Until then everything the browser build writes lives in the in-memory Emscripten
    file system and is gone on reload.
    """
    return False


def compute_capabilities(platform: str, environ: Mapping[str, str]) -> Capabilities:
    """The capabilities of a runtime with this ``sys.platform`` and environment (pure)."""
    pyodide = platform == "emscripten"
    desktop = pyodide and environ.get(DESKTOP_ENV) == "1"
    if not pyodide:  # server
        return Capabilities(
            local_folders=True,
            outbound_http=True,
            persistent_storage=True,
            native_speed=True,
        )
    if desktop:
        return Capabilities(
            local_folders=True,
            outbound_http=True,
            persistent_storage=True,
            native_speed=False,
        )
    return Capabilities(  # browser
        local_folders=False,
        outbound_http=False,
        persistent_storage=_browser_storage_is_persistent(environ),
        native_speed=False,
    )


@lru_cache(maxsize=1)
def capabilities() -> Capabilities:
    """The capabilities of this process (computed once)."""
    return compute_capabilities(sys.platform, os.environ)
