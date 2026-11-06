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
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

#: Set to ``1`` by the Electron desktop build.
DESKTOP_ENV = "EAGLEPLOT_DESKTOP"
#: Directory for ``config.json`` (defaults, presets, workflows). The browser build sets it
#: to a folder it mounts on browser storage (IndexedDB), so settings survive a reload.
CONFIG_DIR_ENV = "EAGLEPLOT_CONFIG_DIR"

#: Where the desktop build (packaging/desktop) mounts the user's app-data folder
#: (Electron's ``userData``): ``config.json`` goes there and survives restarts.
DESKTOP_CONFIG_DIR = "/home/pyodide/.eagleplot"


def _running_in_electron() -> bool:
    """Under Pyodide in the desktop build: Python runs in a Node.js worker inside Electron.

    @stlite/desktop can't pass environment variables to Python, so the desktop build is
    recognised from Node's ``process.versions.electron``.
    """
    if sys.platform != "emscripten":
        return False
    try:
        import js  # type: ignore[import-not-found]  # only exists under Pyodide

        return bool(getattr(js.process.versions, "electron", None))
    except Exception:  # noqa: BLE001 - a browser has no ``process``
        return False


#: Running under Pyodide (stlite, in a browser or in Electron).
IS_PYODIDE = sys.platform == "emscripten"
#: Running in the Electron desktop build (always Pyodide as well).
IS_DESKTOP = IS_PYODIDE and (os.environ.get(DESKTOP_ENV) == "1" or _running_in_electron())


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
    #: Streamlit reads a ``.streamlit/config.toml`` the app writes, at its next start
    #: (Settings → Theme). Under stlite the theme comes from the site's configuration.
    app_config_file: bool = True
    #: The user can add a missing optional package with ``pip install``. Not in the
    #: browser or desktop builds, whose packages are fixed when the site is built.
    pip_install: bool = True
    #: ``st.download_button(data=callable)`` works (the file is built on click). Under
    #: stlite these requests go to the placeholder origin https://stlite.invalid; the
    #: browser build routes them to Python (packaging/web/index.html), but the desktop
    #: build uses stlite's own page, where they fail, so files are prepared in advance.
    on_click_downloads: bool = True


def _browser_storage_is_persistent(environ: Mapping[str, str]) -> bool:
    """Whether the browser build keeps user settings across a reload.

    The site build (``packaging/web``) mounts a folder on browser storage (IndexedDB) and
    points :data:`CONFIG_DIR_ENV` at it. Without that, everything the browser build
    writes lives in the in-memory Emscripten file system and is gone on reload.
    """
    return bool(environ.get(CONFIG_DIR_ENV))


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
            # Folder import needs the user's typed paths (C:\Users\…) mapped onto the
            # folders the desktop build mounts: roadmap D5.2. Off until then.
            local_folders=False,
            # Python runs in a Node.js worker, where Pyodide's requests support (built on
            # the browser's XMLHttpRequest) doesn't work; NIST links out instead.
            outbound_http=False,
            persistent_storage=True,
            native_speed=False,
            app_config_file=False,
            pip_install=False,
            on_click_downloads=False,
        )
    return Capabilities(  # browser
        local_folders=False,
        outbound_http=False,
        persistent_storage=_browser_storage_is_persistent(environ),
        native_speed=False,
        app_config_file=False,
        pip_install=False,
    )


@lru_cache(maxsize=1)
def capabilities() -> Capabilities:
    """The capabilities of this process (computed once)."""
    environ: Mapping[str, str] = os.environ
    if IS_DESKTOP:
        environ = {**os.environ, DESKTOP_ENV: "1"}
    return compute_capabilities(sys.platform, environ)


_PIP_HINT = re.compile(
    r"\s*(?:Install it with|Install with)?:?\s*`*pip install ([^`\n]*?)`*\.?$", re.I
)


def missing_package_message(message: str, caps: Capabilities | None = None) -> str:
    """*message* (an ``ImportError`` text ending in a ``pip install`` hint) for this runtime.

    Where packages can't be installed (the browser and desktop builds ship a fixed set),
    the hint is replaced by one the user can act on.
    """
    caps = capabilities() if caps is None else caps
    if caps.pip_install:
        return message
    match = _PIP_HINT.search(message)
    if match is None:
        return message
    packages = match.group(1).strip()
    reason = message[: match.start()].rstrip(" .:")
    reason = f"{reason}. " if reason else f"This needs {packages}. "
    return (
        f"{reason}It isn't available in this version of EaglePlot. To use this feature, "
        f"install EaglePlot on your computer (`pip install eagleplot`), then `pip install "
        f"{packages}`."
    )


# ── Packages loaded after start-up (browser build, roadmap D4.1) ───────────────
#: Not loaded when the browser build starts: scipy alone added ~2 s to every start, and
#: the Home page doesn't need it. The build (packaging/web/build_site.py) leaves these
#: out; Home.py starts loading them in the background as soon as the app is up, and other
#: pages wait for them (see ``missing_deferred_packages``).
DEFERRED_PACKAGES = ("scipy",)

_deferred_load_started = False
#: Set when Pyodide's loadPackage promise settles. Until then a deferred package can be
#: half there: its files unpacked (so ``find_spec`` finds it) but its compiled modules not
#: loaded yet, and importing it fails with "dynamic module does not define module export
#: function" (seen with scipy.spatial).
_deferred_load_settled = False


def _mark_deferred_load_settled(*_args) -> None:
    global _deferred_load_settled
    _deferred_load_settled = True


def start_deferred_loads(packages: tuple = DEFERRED_PACKAGES) -> bool:
    """Under Pyodide, start loading *packages* in the background (once per process).

    Returns ``True`` if this call started the load. Elsewhere it does nothing: on a
    server every package is installed already.
    """
    global _deferred_load_started
    # The desktop build bundles every package (nothing to fetch, and no network needed).
    if not IS_PYODIDE or IS_DESKTOP or _deferred_load_started:
        return False
    import pyodide_js  # type: ignore[import-not-found]  # only exists under Pyodide
    from pyodide.ffi import create_once_callable  # type: ignore[import-not-found]

    # A JS promise, not awaited, so the page carries on while the packages load. Both
    # outcomes settle it: after a failed load, pages report the import error instead of
    # waiting for ever.
    promise = pyodide_js.loadPackage(list(packages))
    promise.then(
        create_once_callable(_mark_deferred_load_settled),
        create_once_callable(_mark_deferred_load_settled),
    )
    _deferred_load_started = True
    return True


def missing_deferred_packages(packages: tuple = DEFERRED_PACKAGES) -> list:
    """The *packages* that pages can't import yet (always empty on a server).

    Under Pyodide that is all of them until their load has finished.
    """
    if IS_PYODIDE and _deferred_load_started and not _deferred_load_settled:
        return list(packages)
    import importlib
    import importlib.util

    importlib.invalidate_caches()
    return [p for p in packages if importlib.util.find_spec(p) is None]


#: Imported in the background after Home is drawn (browser build), in this order, so the
#: first visit to a data page doesn't pay for them: pandas alone takes ~2.3 s to import
#: under Pyodide, and the plotting stack (matplotlib, seaborn, plotly) several more.
#: The ``eagleplot.*`` entries need the deferred packages, so they wait for those.
WARM_IMPORTS = ("pandas", "matplotlib.pyplot", "eagleplot.plotting", "eagleplot.math")

_warm_imports_started = False


def warm_imports_in_background(modules: tuple = WARM_IMPORTS, delay: float = 0.5) -> bool:
    """Under Pyodide, import *modules* one by one once the event loop is idle.

    Python runs on one thread in the browser, so each import still blocks while it runs;
    doing them one per loop turn lets a click be handled between two imports. A module
    that fails to import is skipped (the page that needs it reports the error).
    Returns ``True`` if this call scheduled the imports; elsewhere it does nothing.
    """
    global _warm_imports_started
    if not IS_PYODIDE or _warm_imports_started:
        return False
    import asyncio
    import importlib

    loop = asyncio.get_event_loop()
    queue = list(modules)

    def step() -> None:
        if not queue:
            return
        if queue[0].startswith("eagleplot.") and missing_deferred_packages():
            loop.call_later(0.25, step)  # wait for scipy & co.
            return
        name = queue.pop(0)
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - reported by the page that imports it
            pass
        loop.call_later(0.05, step)

    loop.call_later(delay, step)
    _warm_imports_started = True
    return True
