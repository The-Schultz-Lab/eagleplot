"""Gates for GUI features that reach outside the browser sandbox.

Plottle is meant to be run locally (``launch.bat`` / ``launch.command``), where
reading a folder on the user's own disk is the point. On a shared deployment the
same feature lets any visitor list and read files on the *server*. See audit A-15.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

#: Environment variable that overrides the folder-loading default.
#: ``1``/``true``/``yes``/``on`` always allow; ``0``/``false``/``no``/``off`` always deny.
FOLDER_LOAD_ENV = "PLOTTLE_ALLOW_FOLDER_LOAD"

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
        should set ``PLOTTLE_ALLOW_FOLDER_LOAD=0``.
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
    """:func:`folder_loading_allowed` for the current Streamlit session."""
    env = os.environ if environ is None else environ
    return folder_loading_allowed(env.get(FOLDER_LOAD_ENV), _client_ip())
