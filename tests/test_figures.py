"""eagleplot.utils.figures: display and download helpers shared by the GUI pages."""

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import eagleplot.utils.figures as figures  # noqa: E402


@pytest.fixture
def fake_st(monkeypatch):
    """Replace ``st`` in the module with a recorder backed by a plain-dict session state."""
    calls = []
    fake = types.SimpleNamespace(
        session_state={},
        __version__="1.66.0",
        html=lambda body: calls.append(("html", body)),
        image=lambda data, **kw: calls.append(("image", data)),
    )
    monkeypatch.setattr(figures, "st", fake)
    fake.calls = calls
    return fake


@pytest.fixture
def fig():
    f, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 3, 2])
    yield f
    plt.close(f)


class TestFigureBytes:
    @pytest.mark.parametrize(
        "fmt, magic",
        [("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")],
    )
    def test_each_export_format_produces_that_format(self, fig, fmt, magic):
        assert figure_bytes_start(fig, fmt) == magic

    def test_dpi_changes_png_size(self, fig):
        small = figures.figure_bytes(fig, "png", 72)
        large = figures.figure_bytes(fig, "png", 300)
        assert len(large) > len(small)

    def test_every_offered_format_renders(self, fig):
        for label, (fmt, dpi, mime, suffix) in figures.EXPORT_FORMATS.items():
            data = figures.figure_bytes(fig, fmt, dpi)
            assert data, label
            assert suffix.endswith("." + fmt)
            assert mime.split("/")[1].startswith(fmt[:3])


def figure_bytes_start(fig, fmt):
    return figures.figure_bytes(fig, fmt)[: {"png": 4, "pdf": 4, "svg": 5}[fmt]]


class TestLazyDownloadSupport:
    @pytest.mark.parametrize(
        "version, expected",
        [("1.49.0", False), ("1.51.2", False), ("1.52.0", True), ("1.66.1", True), ("2.0", True)],
    )
    def test_version_threshold(self, version, expected):
        assert figures.lazy_download_supported(version) is expected

    def test_unparseable_version_is_treated_as_unsupported(self):
        assert figures.lazy_download_supported("nightly") is False


class TestCaching:
    def test_cached_for_computes_once_per_version(self, fake_st):
        made = []

        def make():
            made.append(1)
            return b"x"

        assert figures.cached_for("c", 1, "k", make) == b"x"
        assert figures.cached_for("c", 1, "k", make) == b"x"
        assert len(made) == 1
        figures.cached_for("c", 2, "k", make)  # new figure version invalidates
        assert len(made) == 2

    def test_download_payload_is_lazy_when_supported(self, fake_st, monkeypatch):
        monkeypatch.setattr(figures, "LAZY_DOWNLOAD", True)
        made = []
        payload = figures.download_payload(
            lambda: made.append(1) or b"x", cache_name="c", version=1, key="k"
        )
        assert callable(payload)
        assert made == []  # nothing encoded until Streamlit calls it on click

    def test_download_payload_caches_when_not_lazy(self, fake_st, monkeypatch):
        monkeypatch.setattr(figures, "LAZY_DOWNLOAD", False)
        made = []

        def make():
            made.append(1)
            return b"x"

        for _ in range(3):
            assert figures.download_payload(make, cache_name="c", version=7, key="k") == b"x"
        assert len(made) == 1

    def test_dataset_csv_recomputed_only_for_a_new_object(self, fake_st):
        class Data:
            n = 0

            def to_csv(self, index):
                Data.n += 1
                return "a,b\n1,2\n"

        d = Data()
        for _ in range(3):
            assert figures.dataset_csv_cached(d, "d.csv") == "a,b\n1,2\n"
        assert Data.n == 1
        figures.dataset_csv_cached(Data(), "d.csv")
        assert Data.n == 2


class TestShowPng:
    def test_uses_st_image_normally(self, fake_st, monkeypatch):
        monkeypatch.setattr(figures.sys, "platform", "linux")
        figures.show_png(b"\x89PNG...")
        assert fake_st.calls == [("image", b"\x89PNG...")]

    def test_inlines_a_data_uri_in_the_browser_build(self, fake_st, monkeypatch):
        monkeypatch.setattr(figures.sys, "platform", "emscripten")
        figures.show_png(b"\x89PNG...", alt="My plot")
        kind, body = fake_st.calls[0]
        assert kind == "html"
        assert 'src="data:image/png;base64,iVBORy4uLg=="' in body
        assert 'alt="My plot"' in body
