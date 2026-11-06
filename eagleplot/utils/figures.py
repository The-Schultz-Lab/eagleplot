"""Figure display and download helpers shared by the GUI pages.

Rules these helpers enforce (roadmap D1.3):

* **Nothing is encoded for download on a plain rerun.** Quick Plot used to re-render a
  figure as PNG 150/300/600 dpi, SVG and PDF on *every* interaction (~1.2 s per click),
  because ``st.download_button`` needs its payload up front and expanders still execute
  their contents. Here a payload is either a callable that Streamlit runs only when the
  button is clicked (Streamlit >= 1.52), or computed once per figure and cached.
* **Figures are shown without media-file races in the browser build.** Under Pyodide
  (stlite) ``st.image`` PNGs are served by the single Python worker, and quick successive
  edits can lose intermediate images (404). There the PNG is embedded as a data URI.
  Pages show figures with :func:`show_figure` (roadmap D2.3), never ``st.pyplot``.
"""

from __future__ import annotations

import base64
import html
import io
import sys
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import streamlit as st

from eagleplot.utils.runtime import capabilities

#: Download formats offered for a Matplotlib figure: label -> (format, dpi, MIME, suffix).
EXPORT_FORMATS: Dict[str, Tuple[str, Optional[int], str, str]] = {
    "PNG (150 dpi)": ("png", 150, "image/png", ".png"),
    "PNG (300 dpi)": ("png", 300, "image/png", "_300dpi.png"),
    "PNG (600 dpi)": ("png", 600, "image/png", "_600dpi.png"),
    "SVG (vector)": ("svg", None, "image/svg+xml", ".svg"),
    "PDF (vector)": ("pdf", None, "application/pdf", ".pdf"),
}

#: Resolution of the PNG pages render for display; a download at this dpi can reuse it.
DISPLAY_DPI = 150
#: Display resolution where Python runs without native speed (WebAssembly): encoding a
#: PNG costs about the square of the dpi, and the plot is scaled to the column anyway.
SLOW_RUNTIME_DISPLAY_DPI = 110


def lazy_download_supported(version: Optional[str] = None) -> bool:
    """Whether ``st.download_button(data=callable)`` is available (Streamlit >= 1.52)."""
    version = st.__version__ if version is None else version
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (1, 52)


LAZY_DOWNLOAD = lazy_download_supported()


def figure_bytes(fig: Any, fmt: str, dpi: Optional[int] = None) -> bytes:
    """Render a Matplotlib figure to bytes in *fmt* (``png``, ``svg``, ``pdf`` …)."""
    buf = io.BytesIO()
    if dpi is None:
        fig.savefig(buf, format=fmt, bbox_inches="tight")
    else:
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def cached_for(cache_name: str, version: Hashable, key: Hashable, make: Callable[[], Any]) -> Any:
    """Return ``make()``, computed at most once per *version* of the thing it depends on.

    The cache lives in ``st.session_state[cache_name]`` and is cleared whenever *version*
    changes (pages bump a counter each time they display a new figure).
    """
    cache = st.session_state.setdefault(cache_name, {})
    if cache.get("_version") != version:
        cache.clear()
        cache["_version"] = version
    if key not in cache:
        cache[key] = make()
    return cache[key]


def download_payload(
    make: Callable[[], Any], *, cache_name: str, version: Hashable, key: Hashable
) -> Any:
    """Payload for ``st.download_button``: lazy where Streamlit allows, else cached."""
    if LAZY_DOWNLOAD:
        return make
    return cached_for(cache_name, version, key, make)


def dataset_csv_cached(data: Any, name: str, *, cache_name: str = "_eagleplot_csv_cache") -> str:
    """``data.to_csv(index=False)``, recomputed only when the dataset object changes."""
    return cached_for(cache_name, (name, id(data)), "csv", lambda: data.to_csv(index=False))


def download_figure(
    fig: Any,
    *,
    key_prefix: str,
    version: Hashable,
    display_png: Optional[bytes] = None,
    file_stem: str = "plot",
) -> None:
    """A format selector and one Download button for a Matplotlib figure.

    Only the selected format is ever encoded: on click (Streamlit >= 1.52) or once per
    *version*. If *display_png* is the figure already rendered at :data:`DISPLAY_DPI`, a
    PNG download at that resolution reuses it.
    """
    col_fmt, col_btn = st.columns([3, 1])
    with col_fmt:
        label = st.selectbox(
            "Format",
            list(EXPORT_FORMATS),
            key=f"{key_prefix}_export_fmt",
            label_visibility="collapsed",
        )
    fmt, dpi, mime, suffix = EXPORT_FORMATS[label]
    if fmt == "png" and dpi == DISPLAY_DPI and display_png is not None:
        payload: Any = display_png
    else:
        payload = download_payload(
            lambda: figure_bytes(fig, fmt, dpi),
            cache_name=f"_{key_prefix}_export_cache",
            version=version,
            key=label,
        )
    with col_btn:
        st.download_button(
            "Download",
            data=payload,
            file_name=f"{file_stem}{suffix}",
            mime=mime,
            type="primary",
            key=f"{key_prefix}_dl_plot",
            width="stretch",
        )


def show_png(png: bytes, *, alt: str = "Plot") -> None:
    """Display a rendered PNG; inline as a data URI in the browser build (see module doc)."""
    if sys.platform == "emscripten":
        b64 = base64.b64encode(png).decode("ascii")
        st.html(
            f'<img data-eagleplot-plot="1" alt="{html.escape(alt)}" '
            f'src="data:image/png;base64,{b64}" '
            'style="width:100%;height:auto;border-radius:0.5rem;">'
        )
    else:
        st.image(png, width="stretch")


def display_dpi() -> int:
    """Resolution :func:`show_figure` renders at by default in this runtime."""
    return DISPLAY_DPI if capabilities().native_speed else SLOW_RUNTIME_DISPLAY_DPI


def _is_plotly(fig: Any) -> bool:
    return hasattr(fig, "to_plotly_json")


def show_figure(
    fig: Any,
    *,
    alt: str = "Plot",
    dpi: Optional[int] = None,
    key: Optional[str] = None,
    cache_key: Optional[Hashable] = None,
    version: Hashable = None,
) -> Optional[bytes]:
    """Show a Matplotlib or Plotly figure. Use this instead of ``st.pyplot``.

    A Matplotlib figure is rendered to PNG (``bbox_inches="tight"``, as ``st.pyplot``
    does) at *dpi*, :func:`display_dpi` by default, and shown with :func:`show_png`.
    A Plotly figure goes to ``st.plotly_chart`` at full width, with *key*.

    With a *cache_key*, the PNG is encoded once per (*version*, dpi) and reused on later
    reruns. The caller owns *version*: it must change whenever the figure would draw
    differently (a rebuilt figure, new axis limits …). Leave *cache_key* out for a
    figure that is rebuilt on every run.

    Returns the PNG bytes (for a download at the same dpi), or ``None`` for Plotly.
    The figure is not closed; that stays with the caller.
    """
    if _is_plotly(fig):
        st.plotly_chart(fig, width="stretch", key=key)
        return None
    res = display_dpi() if dpi is None else dpi

    def make() -> bytes:
        return figure_bytes(fig, "png", res)

    if cache_key is None:
        png = make()
    else:
        png = cached_for(f"_eagleplot_png::{cache_key}", (version, res), "png", make)
    show_png(png, alt=alt)
    return png
