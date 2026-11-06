"""Multi-Plot Dashboard (roadmap D1.2): per-cell redraws and on-demand exports."""

from pathlib import Path

import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from eagleplot.utils import figures

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "eagleplot" / "pages" / "4_Multi_Plot_Dashboard.py"
EXAMPLES = ROOT / "eagleplot" / "example-data" / "Artificial"

# (dataset, plot type) per cell of a 2 × 2 dashboard: three Matplotlib panels
# (one of them seaborn) and one Plotly panel.
CELLS = [
    ("ir_spectrum.csv", "line_plot"),
    ("normal_distribution.csv", "histogram"),
    ("scatter_correlation.csv", "interactive_scatter"),
    ("grouped_categorical.csv", "box_plot"),
]
MPL_CELLS = [0, 1, 3]


@pytest.fixture
def work(monkeypatch):
    """Count figure creations, savefig calls and PdfPages builds made by the page."""
    counts = {"figures": 0, "savefig": [], "pdfpages": 0}
    fig_init = matplotlib.figure.Figure.__init__
    savefig = matplotlib.figure.Figure.savefig
    pdf_init = backend_pdf.PdfPages.__init__

    def counting_init(self, *args, **kwargs):
        counts["figures"] += 1
        return fig_init(self, *args, **kwargs)

    def counting_savefig(self, fname, *args, **kwargs):
        counts["savefig"].append((kwargs.get("format"), kwargs.get("dpi")))
        return savefig(self, fname, *args, **kwargs)

    def counting_pdf(self, *args, **kwargs):
        counts["pdfpages"] += 1
        return pdf_init(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "__init__", counting_init)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", counting_savefig)
    monkeypatch.setattr(backend_pdf.PdfPages, "__init__", counting_pdf)

    def reset():
        counts["figures"] = 0
        counts["pdfpages"] = 0
        counts["savefig"].clear()

    counts["reset"] = reset
    return counts


def _configured(at):
    at.run()
    at.session_state["datasets"] = {name: pd.read_csv(EXAMPLES / name) for name, _ in CELLS}
    at.run()
    at.radio(key="dash_layout").set_value("2 × 2").run()
    for i, (name, plot_type) in enumerate(CELLS):
        at.selectbox(key=f"cell_{i}_ds").set_value(name).run()
        at.selectbox(key=f"cell_{i}_pt").set_value(plot_type).run()
    assert not at.exception
    return at


@pytest.fixture
def page():
    """A 2 × 2 dashboard, configured but not generated yet."""
    return _configured(AppTest.from_file(str(PAGE), default_timeout=60))


@pytest.fixture
def dashboard(page):
    page.button(key="dash_generate").click().run()
    assert not page.exception
    return page


def _images(at):
    # st.pyplot output: "imgs" on Streamlit 1.50, "image" on newer versions.
    return len(at.get("imgs")) + len(at.get("image"))


def _has_download_button(at, key):
    # Streamlit 1.50 exposes download buttons to AppTest without a .key; the key is
    # still the suffix of the element id.
    return any(b.proto.id.endswith("-" + key) for b in at.get("download_button"))


def _has_button(at, key):
    return any(b.key == key for b in at.button)


def _cell_figs(at):
    cells = at.session_state["dash_cells"]
    return {i: cells[i]["fig"] for i in range(len(CELLS))}


class TestRendering:
    def test_without_datasets_the_page_asks_for_data(self):
        at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.run()
        assert not at.exception
        assert any("No datasets loaded" in w.value for w in at.warning)

    def test_nothing_is_drawn_before_generate(self, page, work):
        work["reset"]()
        page.run()
        assert _images(page) == 0
        assert work["figures"] == 0 and not work["savefig"]

    def test_generate_draws_every_cell(self, dashboard):
        assert _images(dashboard) == len(MPL_CELLS)
        assert len(dashboard.get("plotly_chart")) == 1
        assert not dashboard.error
        assert pd.Series([c.value for c in dashboard.caption]).str.contains("Cell 4").any()

    def test_errors_are_shown_in_their_cell(self, dashboard):
        # The dashboard draws a subset of the plot catalogue; the rest get a cell error.
        dashboard.selectbox(key="cell_0_pt").set_value("bar_chart").run()
        assert not dashboard.exception
        assert any("Unhandled plot type: bar_chart" in e.value for e in dashboard.error)
        assert _images(dashboard) == len(MPL_CELLS) - 1

    def test_matplotlib_figures_do_not_pile_up_in_pyplot(self, dashboard):
        dashboard.text_input(key="cell_1_title").set_value("Replaced").run()
        assert plt.get_fignums() == []


class TestRedraws:
    def test_dashboard_stays_on_screen_after_an_unrelated_rerun(self, dashboard, work):
        # Before D1.2 the dashboard vanished on every rerun after "Generate".
        figs = _cell_figs(dashboard)
        work["reset"]()
        dashboard.run()
        assert _images(dashboard) == len(MPL_CELLS)
        assert work["figures"] == 0, "a plain rerun regenerated cells"
        assert _cell_figs(dashboard) == figs
        # The only encoding left is st.pyplot showing the three cached figures.
        assert len(work["savefig"]) == len(MPL_CELLS)
        assert work["pdfpages"] == 0

    def test_changing_one_cell_rebuilds_only_that_cell(self, dashboard, work):
        before = _cell_figs(dashboard)
        version = dashboard.session_state["dash_fig_version"]
        work["reset"]()
        dashboard.text_input(key="cell_1_title").set_value("Temperatures").run()
        after = _cell_figs(dashboard)
        assert after[1] is not before[1]
        assert after[1].axes[0].get_title() == "Temperatures"
        assert all(after[i] is before[i] for i in (0, 2, 3))
        assert work["figures"] == 1  # the histogram's one new figure
        assert dashboard.session_state["dash_fig_version"] > version

    def test_generate_redraws_every_cell(self, dashboard):
        before = _cell_figs(dashboard)
        dashboard.button(key="dash_generate").click().run()
        after = _cell_figs(dashboard)
        assert all(after[i] is not before[i] for i in range(len(CELLS)))

    def test_shared_x_axis_syncs_a_row_and_unsharing_restores_it(self, dashboard):
        cells = dashboard.session_state["dash_cells"]
        own = [cells[i]["fig"].axes[0].get_xlim() for i in (0, 1)]
        dashboard.checkbox(key="dash_sharex").check().run()
        assert not dashboard.exception
        cells = dashboard.session_state["dash_cells"]
        shared = [cells[i]["fig"].axes[0].get_xlim() for i in (0, 1)]
        assert shared[0] == shared[1]
        assert shared[0][0] == min(lo for lo, _ in own)
        assert shared[0][1] == max(hi for _, hi in own)
        dashboard.checkbox(key="dash_sharex").uncheck().run()
        cells = dashboard.session_state["dash_cells"]
        assert [cells[i]["fig"].axes[0].get_xlim() for i in (0, 1)] == own


class TestExports:
    def test_a_plain_rerun_builds_no_export(self, dashboard, work):
        work["reset"]()
        dashboard.text_input(key="dash_fig_title").set_value("Report").run()
        dashboard.select_slider(key="dash_export_dpi").set_value(300).run()
        assert work["pdfpages"] == 0
        assert work["figures"] == 0
        # Only st.pyplot's display renders (PNG at 200 dpi; two reruns × three panels):
        # no 300 dpi combined PNG, no PDF.
        assert set(work["savefig"]) == {("png", 200)}
        assert len(work["savefig"]) == 2 * len(MPL_CELLS)

    def test_on_click_downloads_are_offered_without_building(self, dashboard):
        if not figures.LAZY_DOWNLOAD:
            pytest.skip("Streamlit < 1.52 has no on-click (callable) downloads")
        assert _has_download_button(dashboard, "dash_dl_png")
        assert _has_download_button(dashboard, "dash_dl_pdf")
        assert not _has_button(dashboard, "dash_gen_png")
        assert "_dash_export_cache" not in dashboard.session_state

    def test_older_streamlit_builds_each_export_once_on_request(self, dashboard, work, monkeypatch):
        monkeypatch.setattr(figures, "LAZY_DOWNLOAD", False)
        dashboard.text_input(key="dash_fig_title").set_value("Report").run()
        assert _has_button(dashboard, "dash_gen_png") and _has_button(dashboard, "dash_gen_pdf")
        assert not _has_download_button(dashboard, "dash_dl_pdf")

        work["reset"]()
        dashboard.button(key="dash_gen_pdf").click().run()
        assert work["pdfpages"] == 1
        pdf = dashboard.session_state["_dash_export_cache"]["dash_dl_pdf"][1]
        assert pdf.startswith(b"%PDF")
        # A title page plus one page per Matplotlib panel (the Plotly one is left out).
        assert pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") == 1 + len(MPL_CELLS)
        assert _has_download_button(dashboard, "dash_dl_pdf")
        # Page titles go on copies, not on the displayed figures.
        figs = _cell_figs(dashboard)
        assert all(not figs[i].get_suptitle() for i in MPL_CELLS)

        dashboard.button(key="dash_gen_png").click().run()
        png = dashboard.session_state["_dash_export_cache"]["dash_dl_png"][1]
        assert png[:4] == b"\x89PNG"

        work["reset"]()
        dashboard.run()
        assert work["pdfpages"] == 0
        assert len(work["savefig"]) == len(MPL_CELLS)  # display only, nothing rebuilt
        assert _has_download_button(dashboard, "dash_dl_pdf")
        assert _has_download_button(dashboard, "dash_dl_png")

        # A changed cell makes the prepared files stale: they are offered for rebuild.
        dashboard.text_input(key="cell_0_title").set_value("New").run()
        assert _has_button(dashboard, "dash_gen_pdf")
        assert not _has_download_button(dashboard, "dash_dl_pdf")

    def test_combined_png_has_the_grid_layout(self, dashboard, monkeypatch):
        monkeypatch.setattr(figures, "LAZY_DOWNLOAD", False)
        dashboard.run()
        dashboard.select_slider(key="dash_export_dpi").set_value(150).run()
        dashboard.button(key="dash_gen_png").click().run()
        png = dashboard.session_state["_dash_export_cache"]["dash_dl_png"][1]
        width = int.from_bytes(png[16:20], "big")
        height = int.from_bytes(png[20:24], "big")
        assert width > height * 0.8  # two columns of 5 × 4 in panels
