"""Folders on the user's computer, read from the desktop build (roadmap D5.2).

In the desktop app (packaging/desktop) Python runs on Pyodide in a Node.js worker. Its file
system is Pyodide's own, not the computer's, so a path the user types
(``C:\\Users\\you\\data``, ``/Users/you/data``, ``~/data``) means nothing to it until that
folder is mounted. :func:`local_folder` mounts the typed folder with Emscripten's NODEFS
(the same mechanism @stlite/desktop uses for its ``nodefsMountpoints``) and returns the
path to read. Everywhere else it returns the path unchanged.

Only folders the user names are mounted, one at a time, under :data:`MOUNT_ROOT`.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
from pathlib import Path
from typing import Dict, Optional

from eagleplot.utils import runtime

#: Where the desktop build mounts the folders users type.
MOUNT_ROOT = "/mnt/host"

#: Host folder (as normalised by :func:`host_path`) → its mount point. Mounts last until
#: the app closes; the same folder typed again reuses its mount.
_mounts: Dict[str, str] = {}


def host_path(text: str, platform: str, home: Optional[str]) -> str:
    """The absolute, normalised host path the user typed (pure).

    Parameters
    ----------
    text : str
        What the user typed. Surrounding whitespace and quotes (from "Copy as path" on
        Windows) are ignored; a leading ``~`` is the user's home folder.
    platform : str
        Node's ``process.platform`` on the host: ``win32``, ``darwin`` or ``linux``.
    home : str or None
        The user's home folder on the host.

    Raises
    ------
    NotADirectoryError
        If the path is empty or relative: there is no working directory to resolve it
        against that the user would recognise.
    """
    flavour = ntpath if platform == "win32" else posixpath
    path = text.strip().strip("\"'").strip()
    if path == "~" or path.startswith(("~/", "~\\")):
        if not home:
            raise NotADirectoryError("Can't tell where your home folder is: type the full path.")
        path = home + path[1:]
    if not path or not flavour.isabs(path):
        example = "C:\\Users\\you\\data" if platform == "win32" else "/Users/you/data"
        raise NotADirectoryError(
            f"Type the folder's full path, for example {example} (got {text.strip()!r})."
        )
    return flavour.normpath(path)


def _host() -> tuple:
    """``(platform, home)`` of the computer the desktop app runs on."""
    import js  # type: ignore[import-not-found]  # only exists under Pyodide

    env = js.process.env
    home = getattr(env, "HOME", None) or getattr(env, "USERPROFILE", None)
    return str(js.process.platform), (str(home) if home else None)


def _mount(host: str, mountpoint: str) -> None:
    """Mount host folder *host* at *mountpoint* (NODEFS), or raise ``NotADirectoryError``."""
    import pyodide_js  # type: ignore[import-not-found]  # only exists under Pyodide
    from js import Object  # type: ignore[import-not-found]
    from pyodide.ffi import to_js  # type: ignore[import-not-found]

    fs = pyodide_js.FS
    os.makedirs(mountpoint, exist_ok=True)
    mounted = False
    try:
        options = to_js({"root": host}, dict_converter=Object.fromEntries)
        fs.mount(fs.filesystems.NODEFS, options, mountpoint)
        mounted = True
        if not os.path.isdir(mountpoint):
            raise OSError("not a folder")
        os.listdir(mountpoint)  # a file, a missing folder or no permission fails here
    except Exception:  # noqa: BLE001 - FS errors arrive as JS objects, not OSError
        if mounted:
            try:
                fs.unmount(mountpoint)
            except Exception:  # noqa: BLE001
                pass
        try:
            os.rmdir(mountpoint)
        except OSError:
            pass
        raise NotADirectoryError(f"Not a folder EaglePlot can open: {host}") from None


def local_folder(text: str) -> Path:
    """The folder named by *text*, as a path this process can read.

    On a server this is just ``Path(text)``. In the desktop build the host folder is
    mounted first (see the module docstring). The browser build has no local folders
    (:attr:`~eagleplot.utils.runtime.Capabilities.local_folders`) and never gets here.

    Raises
    ------
    NotADirectoryError
        In the desktop build, if *text* isn't the full path of a folder that can be read.
    """
    if not runtime.IS_DESKTOP:
        return Path(text)
    platform, home = _host()
    host = host_path(text, platform, home)
    key = host.lower() if platform in ("win32", "darwin") else host  # case-insensitive
    if key not in _mounts:
        mountpoint = f"{MOUNT_ROOT}/{len(_mounts) + 1}"
        _mount(host, mountpoint)
        _mounts[key] = mountpoint
    return Path(_mounts[key])
