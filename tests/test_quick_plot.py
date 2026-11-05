"""Quick Plot behaviour (roadmap D1.1): live preview, history, and no wasted export work."""

from pathlib import Path

import matplotlib.figure
import pandas as pd
import pytest

from eagleplot.utils.figures import EXPORT_FORMATS

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "eagleplot" / "pages" / "2_Quick_Plot.py"
SPECTRUM = ROOT / "eagleplot" / "example-data" / "Artificial" / "ir_spectrum.csv"


@pytest.fixture
def savefig_calls(monkeypatch):
    """Record every Figure.savefig call made while the page runs (same process)."""
    calls = []
    original = matplotlib.figure.Figure.savefig

    def recording(self, fname, *args, **kwargs):
        calls.append((kwargs.get("format"), kwargs.get("dpi")))
        return original(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording)
    return calls


@pytest.fixture
def page():
    """Quick Plot with ir_spectrum.csv loaded and a Matplotlib line plot selected."""
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    at.session_state["datasets"] = {"ir_spectrum.csv": pd.read_csv(SPECTRUM)}
    at.session_state["current_dataset"] = "ir_spectrum.csv"
    at.run()
    at.selectbox(key="qp_category").set_value("Matplotlib").run()
    at.selectbox(key="qp_plottype").set_value("Line Plot").run()
    assert not at.exception
    return at


def _has_download_button(at, key):
    # Streamlit 1.50 (the newest on Python 3.9) exposes download buttons to AppTest as
    # UnknownElement without a .key; the key is still the suffix of the element id.
    return any(b.proto.id.endswith("-" + key) for b in at.get("download_button"))


def _set_title(at, text):
    at.text_input(key="line_plot_title").set_value(text).run()
    assert not at.exception


class TestLivePreview:
    def test_plot_appears_without_pressing_generate(self, page):
        assert page.session_state["qp_mpl_png"][:4] == b"\x89PNG"
        assert len(page.session_state["plot_history"]) == 0

    def test_setting_change_redraws_but_does_not_save(self, page):
        version, png = page.session_state["qp_fig_version"], page.session_state["qp_mpl_png"]
        _set_title(page, "IR spectrum")
        assert page.session_state["qp_fig_version"] == version + 1
        assert page.session_state["qp_mpl_png"] != png
        assert len(page.session_state["plot_history"]) == 0

    def test_generate_saves_to_history(self, page):
        page.button(key="qp_generate").click().run()
        assert not page.exception
        assert len(page.session_state["plot_history"]) == 1

    def test_turning_live_preview_off_stops_redraws(self, page):
        page.toggle(key="qp_live").set_value(False).run()
        version = page.session_state["qp_fig_version"]
        _set_title(page, "not drawn yet")
        assert page.session_state["qp_fig_version"] == version
        page.button(key="qp_generate").click().run()
        assert page.session_state["qp_fig_version"] == version + 1


class TestExportsAreNotRebuiltOnEveryRerun:
    def test_offers_every_export_format(self, page):
        assert page.selectbox(key="qp_export_fmt").options == list(EXPORT_FORMATS)
        assert _has_download_button(page, "qp_dl_plot")

    def test_a_setting_change_encodes_at_most_display_plus_one_format(self, page, savefig_calls):
        # Before 3.1 the page encoded PNG 150/300/600 dpi, SVG and PDF on *every* rerun:
        # 5 savefig calls per title edit (~1.2 s). Now: the display PNG, plus at most the
        # selected format when Streamlit is too old for on-click (callable) downloads.
        page.selectbox(key="qp_export_fmt").set_value("PDF (vector)").run()
        savefig_calls.clear()
        _set_title(page, "Edited")
        assert len(savefig_calls) <= 2, savefig_calls
        assert ("png", 600) not in savefig_calls
        assert ("svg", None) not in savefig_calls
