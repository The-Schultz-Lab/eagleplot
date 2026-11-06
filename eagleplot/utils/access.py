"""Gates for GUI features that reach outside the browser sandbox.

EaglePlot is meant to be run locally (``launch.bat`` / ``launch.command``), where
reading a folder on the user's own disk is the point. On a shared deployment the
same feature lets any visitor list and read files on the *server*. See audit A-15.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Mapping, Optional

from eagleplot.utils.runtime import capabilities, compute_capabilities

#: Environment variable that overrides the folder-loading default.
#: ``1``/``true``/``yes``/``on`` always allow; ``0``/``false``/``no``/``off`` always deny.
FOLDER_LOAD_ENV = "EAGLEPLOT_ALLOW_FOLDER_LOAD"
#: Name used before the Plottle -> EaglePlot rename (3.0.0). Still honoured, with a
#: deprecation warning, when the new name is not set. Remove in 4.0.
LEGACY_FOLDER_LOAD_ENV = "PLOTTLE_ALLOW_FOLDER_LOAD"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def folder_loading_allowed(env_value: Optional[str], client_ip: Optional[str]) -> bool:
    """Decide whether the GUI may read a folder path typed by the user.

    Parameters
    ----------
    env_value : str or None
        Value of :data:`FOLDER_LOAD_ENV`, or ``None`` if unset.
    client_ip : str or None
        ``st.context.ip_address``. Streamlit reports ``None`` for a connection
        from ``localhost`` and the remote address otherwise.

    Returns
    -------
    bool
        An explicit setting wins. Unset or unrecognised, folder loading is allowed
        only for a local connection. The IP check alone is not a security control
        (a reverse proxy on the same host looks local), so shared deployments
        should set ``EAGLEPLOT_ALLOW_FOLDER_LOAD=0``.
    """
    if env_value is not None:
        value = env_value.strip().lower()
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
    return client_ip is None


def _client_ip() -> Optional[str]:
    """``st.context.ip_address`` for the current session (``None`` = localhost)."""
    import streamlit as st

    return st.context.ip_address


def folder_loading_allowed_here(environ: Optional[Mapping[str, str]] = None) -> bool:
    """:func:`folder_loading_allowed` for the current Streamlit session.

    Always ``False`` where the runtime has no local disk to read (the browser build,
    see :mod:`eagleplot.utils.runtime`), whatever :data:`FOLDER_LOAD_ENV` says.
    """
    if environ is None:
        env: Mapping[str, str] = os.environ
        caps = capabilities()
    else:
        env = environ
        caps = compute_capabilities(sys.platform, env)
    if not caps.local_folders:
        return False
    return folder_loading_allowed(folder_load_setting(env), _client_ip())


def folder_load_setting(env: Mapping[str, str]) -> Optional[str]:
    """The folder-loading setting from *env*, accepting the pre-3.0 variable name."""
    value = env.get(FOLDER_LOAD_ENV)
    if value is None and LEGACY_FOLDER_LOAD_ENV in env:
        warnings.warn(
            f"{LEGACY_FOLDER_LOAD_ENV} is deprecated; set {FOLDER_LOAD_ENV} instead "
            "(Plottle was renamed EaglePlot in 3.0).",
            DeprecationWarning,
            stacklevel=2,
        )
        value = env[LEGACY_FOLDER_LOAD_ENV]
    return value
