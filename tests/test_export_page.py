"""Export Results page (roadmap D1.2): downloads are built on demand, not on every rerun."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from eagleplot.utils.figures import LAZY_DOWNLOAD  # noqa: E402

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

PAGE = Path(__file__).resolve().parent.parent / "eagleplot" / "pages" / "6_Export_Results.py"


def _fig(n):
    fig, ax = plt.subplots()
    ax.plot(range(n), range(n))
    return fig


@pytest.fixture
def work(monkeypatch):
    """Count export work done while the page runs (same process as AppTest)."""
    calls = []
    for owner, name in [
        (matplotlib.figure.Figure, "savefig"),
        (pd.DataFrame, "to_csv"),
        (pd.DataFrame, "to_json"),
    ]:
        original = getattr(owner, name)

        def recording(self, *a, _orig=original, _name=name, **kw):
            calls.append(_name)
            return _orig(self, *a, **kw)

        monkeypatch.setattr(owner, name, recording)
    return calls


@pytest.fixture
def page():
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    at.session_state["plot_history"] = [
        {
            "type": "line_plot",
            "dataset": "d.csv",
            "timestamp": "2026-11-05T10:00:00",
            "figure": _fig(5),
        },
        {
            "type": "histogram",
            "dataset": "d.csv",
            "timestamp": "2026-11-05T10:01:00",
            "figure": _fig(9),
        },
        {
            "type": "scatter",
            "dataset": "d.csv",
            "timestamp": "2026-11-05T10:02:00",
            "figure": _fig(3),
        },
    ]
    at.session_state["datasets"] = {"d.csv": pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})}
    at.session_state["current_dataset"] = "d.csv"
    at.session_state["analysis_results"] = [
        {"type": "statistics", "timestamp": "2026-11-05T10:03:00", "results": {"mean": 2.0}}
    ]
    at.run()
    assert not at.exception
    yield at
    plt.close("all")


def _download_keys(at):
    # Streamlit 1.50 (Python 3.9) gives AppTest download buttons without a .key; the key is
    # the suffix of the element id on every version.
    return {b.proto.id.rsplit("-", 1)[-1] for b in at.get("download_button")}


def test_every_download_is_still_offered(page):
    keys = _download_keys(page)
    for i in range(3):
        assert {f"dl_png150_{i}", f"dl_png300_{i}", f"dl_svg_{i}", f"dl_pdf_{i}"} <= keys
    assert {"dl_dataset_csv", "dl_dataset_json", "dl_analysis_json"} <= keys


def test_a_plain_rerun_builds_no_downloads(page, work):
    # Before D1.2 every rerun rendered each plot 4 times (PNG 150, PNG 300, SVG, PDF)
    # and serialised the dataset to CSV and JSON: 14 calls here.
    work.clear()
    page.run()
    assert not page.exception
    assert work == []


@pytest.mark.skipif(not LAZY_DOWNLOAD, reason="Streamlit < 1.52 has no on-click downloads")
def test_pdf_report_is_one_click_on_new_streamlit(page):
    assert "dl_pdf_report" in _download_keys(page)
    assert not any(b.key == "pdf_report_btn" for b in page.button)


@pytest.mark.skipif(LAZY_DOWNLOAD, reason="covers the Streamlit < 1.52 path")
def test_pdf_report_download_persists_on_old_streamlit(page):
    assert "dl_pdf_report" not in _download_keys(page)
    next(b for b in page.button if b.key == "pdf_report_btn").click().run()
    assert not page.exception
    assert "dl_pdf_report" in _download_keys(page)
    page.run()  # any later interaction: the A-32 bug made the button vanish here
    assert "dl_pdf_report" in _download_keys(page)
