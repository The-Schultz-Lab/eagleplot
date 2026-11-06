"""Advanced Plotting (roadmap D1.2): tabs as fragments, HTML built only when downloaded."""

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.basedatatypes
import pytest

from eagleplot.utils import figures

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "eagleplot" / "pages" / "5_Advanced_Plotting.py"
EXAMPLES = ROOT / "eagleplot" / "example-data" / "Artificial"


@pytest.fixture
def work(monkeypatch):
    """Count savefig, to_html and to_csv calls made while the page runs."""
    counts = {"savefig": 0, "to_html": 0, "to_csv": 0}

    def counting(name, original):
        def inner(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return inner

    monkeypatch.setattr(
        matplotlib.figure.Figure, "savefig", counting("savefig", matplotlib.figure.Figure.savefig)
    )
    monkeypatch.setattr(
        plotly.basedatatypes.BaseFigure,
        "to_html",
        counting("to_html", plotly.basedatatypes.BaseFigure.to_html),
    )
    monkeypatch.setattr(pd.DataFrame, "to_csv", counting("to_csv", pd.DataFrame.to_csv))

    def reset():
        for key in ("savefig", "to_html", "to_csv"):
            counts[key] = 0

    counts["reset"] = reset
    return counts


def _page(datasets):
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    at.session_state["datasets"] = datasets
    at.session_state["current_dataset"] = next(iter(datasets))
    at.run()
    assert not at.exception
    return at


@pytest.fixture
def page():
    name = "molecular_properties.csv"
    return _page({name: pd.read_csv(EXAMPLES / name)})


@pytest.fixture
def grouped():
    name = "grouped_categorical.csv"
    return _page({name: pd.read_csv(EXAMPLES / name)})


def _images(at):
    # st.pyplot output: "imgs" on Streamlit 1.50, "image" on newer versions.
    return len(at.get("imgs")) + len(at.get("image"))


def _has_download_button(at, key):
    # Streamlit 1.50 exposes download buttons to AppTest without a .key; the key is
    # still the suffix of the element id.
    return any(b.proto.id.endswith("-" + key) for b in at.get("download_button"))


def _click(at, key):
    at.button(key=key).click().run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    return at


class TestRendering:
    def test_without_datasets_the_page_asks_for_data(self):
        at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.run()
        assert not at.exception
        assert any("No datasets loaded" in w.value for w in at.warning)

    def test_all_tabs_render(self, page):
        assert len(page.tabs) == 5
        for key in ("corr_run", "dist_run", "cat_run", "s3d_run", "exp_run"):
            assert any(b.key == key for b in page.button)

    def test_numpy_array_dataset(self):
        at = _page({"m.npy": np.load(EXAMPLES / "correlation_matrix.npy")})
        assert any("requires a **DataFrame**" in w.value for w in at.warning)
        _click(at, "dist_run")
        assert _images(at) == 1


class TestPlots:
    def test_correlation_heatmap(self, page):
        _click(page, "corr_run")
        assert _images(page) == 1
        assert plt.get_fignums() == []

    @pytest.mark.parametrize("kind", ["Histogram", "KDE"])
    def test_overlaid_distributions(self, page, kind):
        page.radio(key="dist_kind").set_value(kind).run()
        _click(page, "dist_run")
        assert _images(page) == 1

    @pytest.mark.parametrize("kind", ["Box", "Violin", "Swarm"])
    def test_grouped_categorical(self, grouped, kind):
        grouped.selectbox(key="cat_kind").set_value(kind).run()
        grouped.selectbox(key="cat_x").set_value("catalyst").run()
        grouped.selectbox(key="cat_hue").set_value("solvent").run()
        _click(grouped, "cat_run")
        assert _images(grouped) == 1

    def test_3d_scatter_offers_html(self, page):
        page.selectbox(key="s3d_color").set_value("pIC50").run()
        _click(page, "s3d_run")
        assert len(page.get("plotly_chart")) == 1
        assert _has_download_button(page, "s3d_dl_html")

    @pytest.mark.parametrize(
        "plot_type",
        [
            "interactive_histogram",
            "interactive_scatter",
            "interactive_line",
            "interactive_heatmap",
            "interactive_3d_surface",
        ],
    )
    def test_html_export_tab(self, page, plot_type):
        page.selectbox(key="exp_type").set_value(plot_type).run()
        _click(page, "exp_run")
        assert len(page.get("plotly_chart")) == 1
        assert _has_download_button(page, "exp_dl_html")


class TestNoWastedWork:
    def test_a_plain_rerun_encodes_nothing(self, page, work):
        _click(page, "s3d_run")
        work["reset"]()
        page.run()
        page.slider(key="corr_figw").set_value(10).run()
        page.text_input(key="exp_title").set_value("Renamed").run()
        assert (work["savefig"], work["to_html"], work["to_csv"]) == (0, 0, 0)

    def test_generate_does_not_build_html_until_downloaded(self, page, work):
        if not figures.LAZY_DOWNLOAD:
            pytest.skip("Streamlit < 1.52 has no on-click (callable) downloads")
        work["reset"]()
        _click(page, "s3d_run")
        _click(page, "exp_run")
        assert work["to_html"] == 0
        assert _has_download_button(page, "exp_dl_html")

    def test_older_streamlit_builds_html_once_per_plot(self, page, work, monkeypatch):
        monkeypatch.setattr(figures, "LAZY_DOWNLOAD", False)
        work["reset"]()
        _click(page, "exp_run")
        assert work["to_html"] == 1
        html = page.session_state["_adv_html_cache"]["exp_dl_html"]
        assert "<html>" in html[:200].lower() and "plotly" in html
        _click(page, "exp_run")
        assert work["to_html"] == 2  # a new figure, a new file
