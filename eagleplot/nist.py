"""NIST Chemistry WebBook integration.

Provides functions to fetch IR spectra and compound information from the
NIST Chemistry WebBook (https://webbook.nist.gov) using CAS Registry Numbers
or compound names.

No authentication is required. Requires the ``requests`` package.

Public API
----------
fetch_ir_spectrum(cas_or_id, index=0) -> pd.DataFrame
    Download the IR JCAMP-DX spectrum for a compound and return as DataFrame.
get_compound_url(cas_or_id) -> str
    Return the NIST WebBook URL for a compound page.
get_ir_jcamp_url(cas_or_id, index=0) -> str
    Return the direct JCAMP-DX download URL for an IR spectrum.
search_url(query, by='name') -> str
    Return the NIST WebBook search URL (for 'name' or 'formula' queries).
list_ir_spectra(cas_or_id) -> list[dict]
    Return metadata for all available IR spectra for a compound.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List

import numpy as np  # noqa: F401  (kept for downstream users who import from this module)

_NIST_BASE = "https://webbook.nist.gov/cgi/cbook.cgi"


def _to_nist_id(cas_or_id: str) -> str:
    """Convert a CAS number to NIST compound ID format (e.g. '64-17-5' → 'C64175').

    If the value already looks like a NIST ID (starts with a letter other than
    a digit), it is returned unchanged.
    """
    stripped = cas_or_id.strip()
    # Already a NIST-style ID (e.g. 'C64175', 'InChI=...') — leave it alone
    if not stripped[0].isdigit():
        return stripped
    # CAS number: strip dashes and prepend 'C'
    return "C" + stripped.replace("-", "")


def get_compound_url(cas_or_id: str) -> str:
    """Return the NIST WebBook compound page URL for a CAS number or NIST ID.

    Parameters
    ----------
    cas_or_id : str
        CAS Registry Number (e.g. ``'64-17-5'``) or NIST compound ID
        (e.g. ``'C64175'``).

    Returns
    -------
    str
        Full URL to the compound's NIST WebBook page.

    Examples
    --------
    >>> get_compound_url('64-17-5')
    'https://webbook.nist.gov/cgi/cbook.cgi?ID=64-17-5&Units=SI'
    """
    return f"{_NIST_BASE}?ID={cas_or_id}&Units=SI"


def get_ir_jcamp_url(cas_or_id: str, index: int = 0) -> str:
    """Return the direct JCAMP-DX download URL for an IR spectrum.

    Parameters
    ----------
    cas_or_id : str
        CAS Registry Number (e.g. ``'64-17-5'``) or NIST compound ID
        (e.g. ``'C64175'``).
    index : int, default 0
        Spectrum index (0 = first available IR spectrum).

    Returns
    -------
    str
        Direct URL to the JCAMP-DX file.

    Notes
    -----
    NIST's JCAMP endpoint uses ``?JCAMP=<compound_id>`` as the primary
    parameter, where the compound ID is the CAS number with dashes removed
    and a ``C`` prefix (e.g. CAS ``64-17-5`` → ``C64175``).
    """
    nist_id = _to_nist_id(cas_or_id)
    return f"{_NIST_BASE}?JCAMP={nist_id}&Index={index}&Type=IR-SPEC"


def search_url(query: str, by: str = "name") -> str:
    """Return a NIST WebBook search URL.

    Parameters
    ----------
    query : str
        Search term.
    by : {'name', 'formula', 'cas'}, default 'name'
        Search type.

    Returns
    -------
    str
        NIST WebBook search URL (open in a browser or via requests).
    """
    query_enc = query.replace(" ", "+")
    if by == "formula":
        return f"{_NIST_BASE}?Formula={query_enc}&Units=SI&cST=on"
    elif by == "cas":
        return get_compound_url(query)
    else:
        return f"{_NIST_BASE}?Name={query_enc}&Units=SI&cST=on"


def fetch_ir_spectrum(cas_or_id: str, index: int = 0) -> Any:
    """Download the IR spectrum for a compound from NIST WebBook.

    Fetches the JCAMP-DX file for the given compound and returns it as a
    ``pd.DataFrame`` with columns ``x`` (wavenumber, cm⁻¹) and ``y``
    (absorbance or transmittance). Compound metadata is stored in
    ``df.attrs``.

    Parameters
    ----------
    cas_or_id : str
        CAS Registry Number (e.g. ``'64-17-5'``) or NIST compound ID.
    index : int, default 0
        Spectrum index when multiple IR spectra are available.

    Returns
    -------
    pd.DataFrame
        Columns: ``x`` (wavenumber, cm⁻¹), ``y`` (absorbance / transmittance).
        ``df.attrs`` contains JCAMP metadata (``TITLE``, ``DATA TYPE``, etc.).

    Raises
    ------
    ImportError
        If ``requests`` is not installed.
    RuntimeError
        If the NIST server returns a non-200 status or no spectrum is found.

    Examples
    --------
    >>> df = fetch_ir_spectrum('64-17-5')  # ethanol
    >>> df.columns.tolist()
    ['x', 'y']
    >>> df.attrs.get('TITLE')
    'Ethanol'
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "requests is required for NIST WebBook access. Install it with: pip install requests"
        ) from exc

    from eagleplot.io import load_jcamp

    url = get_ir_jcamp_url(cas_or_id, index)
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"NIST WebBook returned HTTP {resp.status_code} for CAS '{cas_or_id}'. "
            "Check that the CAS number is correct and the spectrum exists."
        )
    content = resp.text
    if "##TITLE=" not in content and "##title=" not in content.lower():
        raise RuntimeError(
            f"No IR spectrum found for CAS '{cas_or_id}' (index={index}). "
            "The compound may not have an IR spectrum in NIST WebBook."
        )

    # Write to a temp file so load_jcamp can parse it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jdx", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        df = load_jcamp(tmp_path)
    finally:
        os.unlink(tmp_path)

    df.attrs["nist_cas"] = cas_or_id
    df.attrs["nist_url"] = get_compound_url(cas_or_id)
    return df


def list_ir_spectra(cas_or_id: str) -> List[Dict[str, Any]]:
    """List available IR spectra for a compound on NIST WebBook.

    Attempts to fetch the first few spectrum indices (0–4) and returns
    metadata for each one that exists.

    Parameters
    ----------
    cas_or_id : str
        CAS Registry Number or NIST compound ID.

    Returns
    -------
    list of dict
        Each dict has keys: ``index`` (int), ``url`` (str),
        ``title`` (str), ``data_type`` (str).
        Empty list if no spectra are found or ``requests`` is not available.
    """
    try:
        import requests
    except ImportError:
        return []

    results = []
    for idx in range(5):
        url = get_ir_jcamp_url(cas_or_id, idx)
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                break
            content = resp.text
            if "##TITLE=" not in content and "##title=" not in content.lower():
                break
            title = ""
            data_type = ""
            for line in content.splitlines():
                if line.upper().startswith("##TITLE="):
                    title = line.split("=", 1)[1].strip()
                elif line.upper().startswith("##DATA TYPE="):
                    data_type = line.split("=", 1)[1].strip()
                if title and data_type:
                    break
            results.append({"index": idx, "url": url, "title": title, "data_type": data_type})
        except Exception:
            break
    return results
