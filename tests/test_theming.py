"""Tests for eagleplot.utils.theming — audit A-17, A-18, A-19."""

from pathlib import Path

import pytest

from eagleplot.utils import theming
from eagleplot.utils.theming import (
    STREAMLIT_DEFAULTS,
    THEME_KEYS,
    _hex_to_rgba,
    app_css,
    get_config_path,
    read_theme,
    write_theme,
)


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run with cwd pointed at a temp dir, so config writes are isolated."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


_SHIPPED_CONFIG = """\
# EaglePlot — Streamlit configuration

# ── NCCU theme ────────────────────────────────────────────────────────────────
# primaryColor:           soft rose — accent, tab underlines, active states
[theme]
base = "dark"
primaryColor = "#e0a3a3"
backgroundColor = "#1b1b1b"
secondaryBackgroundColor = "#240000"
textColor = "#ffffff"

[server]
maxUploadSize = 200
enableCORS = true
enableXsrfProtection = true

[runner]
fastReruns = true
"""


class TestConfigPath:
    def test_path_is_relative_to_cwd_not_the_package(self, in_tmp_cwd):
        """Streamlit reads $CWD/.streamlit/config.toml, never the installed package.

        Resolving this against the package was why a pip-installed user's theme
        save silently did nothing (A-18).
        """
        assert get_config_path() == in_tmp_cwd / ".streamlit" / "config.toml"
        assert "site-packages" not in str(get_config_path())


class TestWriteTheme:
    def test_creates_file_and_directory_when_absent(self, in_tmp_cwd):
        written = write_theme({"base": "light", "primaryColor": "#123456"})
        assert written.is_file()
        text = written.read_text(encoding="utf-8")
        assert "[theme]" in text
        assert 'base = "light"' in text
        assert 'primaryColor = "#123456"' in text

    def test_roundtrips_through_read_theme(self, in_tmp_cwd):
        theme = {
            "base": "dark",
            "primaryColor": "#aabbcc",
            "backgroundColor": "#111111",
            "secondaryBackgroundColor": "#222222",
            "textColor": "#eeeeee",
        }
        write_theme(theme)
        restored = read_theme()
        for key, value in theme.items():
            assert restored[key] == value

    def test_preserves_comments_and_other_sections(self, in_tmp_cwd):
        """A-19: the old writer re-serialised the file and destroyed every comment."""
        cfg = in_tmp_cwd / ".streamlit" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(_SHIPPED_CONFIG, encoding="utf-8")

        write_theme({**STREAMLIT_DEFAULTS, "primaryColor": "#ff0000"})
        text = cfg.read_text(encoding="utf-8")

        # Header and section comments survive.
        assert "# EaglePlot — Streamlit configuration" in text
        assert "# ── NCCU theme" in text
        assert "soft rose" in text
        # Other sections survive verbatim, values included.
        assert "[server]" in text
        assert "maxUploadSize = 200" in text
        assert "enableXsrfProtection = true" in text
        assert "[runner]" in text
        assert "fastReruns = true" in text
        # And the theme actually changed.
        assert 'primaryColor = "#ff0000"' in text
        assert "#e0a3a3" not in text

    def test_does_not_duplicate_the_theme_table(self, in_tmp_cwd):
        cfg = in_tmp_cwd / ".streamlit" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(_SHIPPED_CONFIG, encoding="utf-8")
        write_theme(dict(STREAMLIT_DEFAULTS))
        write_theme(dict(STREAMLIT_DEFAULTS))
        text = cfg.read_text(encoding="utf-8")
        assert text.count("[theme]") == 1
        for key in THEME_KEYS:
            assert text.count(f"{key} = ") == 1, f"{key} written more than once"

    def test_appends_theme_table_when_config_has_none(self, in_tmp_cwd):
        cfg = in_tmp_cwd / ".streamlit" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("[server]\nmaxUploadSize = 50\n", encoding="utf-8")
        write_theme({"base": "dark"})
        text = cfg.read_text(encoding="utf-8")
        assert "maxUploadSize = 50" in text
        assert "[theme]" in text
        assert 'base = "dark"' in text

    def test_output_is_valid_toml(self, in_tmp_cwd):
        cfg = in_tmp_cwd / ".streamlit" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(_SHIPPED_CONFIG, encoding="utf-8")
        write_theme({**STREAMLIT_DEFAULTS, "base": "dark"})
        with open(cfg, "rb") as fh:
            parsed = theming.tomllib.load(fh)
        assert parsed["theme"]["base"] == "dark"
        assert parsed["server"]["maxUploadSize"] == 200

    def test_list_valued_option_elsewhere_survives(self, in_tmp_cwd):
        """The old hand-rolled serialiser emitted Python repr for lists, which is
        invalid TOML. Only the [theme] table is touched now, so this holds."""
        cfg = in_tmp_cwd / ".streamlit" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[server]\nfolderWatchBlacklist = ["a", "b"]\n\n[theme]\nbase = "dark"\n',
            encoding="utf-8",
        )
        write_theme({"base": "light"})
        with open(cfg, "rb") as fh:
            parsed = theming.tomllib.load(fh)
        assert parsed["server"]["folderWatchBlacklist"] == ["a", "b"]
        assert parsed["theme"]["base"] == "light"

    def test_raises_oserror_when_target_is_unwritable(self, in_tmp_cwd, monkeypatch):
        """A-18: failures must raise so the caller does not report success."""

        def _boom(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(theming.os, "replace", _boom)
        with pytest.raises(OSError):
            write_theme(dict(STREAMLIT_DEFAULTS))

    def test_no_temp_files_left_behind_on_failure(self, in_tmp_cwd, monkeypatch):
        def _boom(*args, **kwargs):
            raise OSError("nope")

        monkeypatch.setattr(theming.os, "replace", _boom)
        with pytest.raises(OSError):
            write_theme(dict(STREAMLIT_DEFAULTS))
        leftovers = list((in_tmp_cwd / ".streamlit").glob(".config-*"))
        assert not leftovers, f"temp files left behind: {leftovers}"


class TestReadTheme:
    def test_returns_all_keys_when_no_config_exists(self, in_tmp_cwd):
        theme = read_theme()
        assert set(THEME_KEYS) <= set(theme)
        assert all(theme[k] for k in THEME_KEYS)

    def test_corrupt_config_falls_back_instead_of_raising(self, in_tmp_cwd):
        cfg = in_tmp_cwd / ".streamlit" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("this is not valid toml }{", encoding="utf-8")
        theme = read_theme()
        assert all(theme[k] for k in THEME_KEYS)


class TestHexToRgba:
    @pytest.mark.parametrize(
        "colour,expected",
        [
            ("#ffffff", "rgba(255, 255, 255, 0.5)"),
            ("ffffff", "rgba(255, 255, 255, 0.5)"),
            ("#000000", "rgba(0, 0, 0, 0.5)"),
            ("#fff", "rgba(255, 255, 255, 0.5)"),
            ("#1b1b1b", "rgba(27, 27, 27, 0.5)"),
        ],
    )
    def test_parses_hex_forms(self, colour, expected):
        assert _hex_to_rgba(colour, 0.5) == expected

    @pytest.mark.parametrize("colour", ["", "not-a-colour", "#12", "#gggggg"])
    def test_degrades_to_grey_rather_than_raising(self, colour):
        # A malformed colour in config.toml should not break the whole stylesheet.
        assert _hex_to_rgba(colour, 0.4) == "rgba(128, 128, 128, 0.4)"


class TestAppCss:
    def test_uses_the_supplied_theme_colours(self):
        css = app_css(
            {
                "base": "light",
                "primaryColor": "#ff0000",
                "backgroundColor": "#00ff00",
                "secondaryBackgroundColor": "#0000ff",
                "textColor": "#123456",
            }
        )
        assert "#ff0000" in css
        assert "#00ff00" in css
        assert "#0000ff" in css
        assert "rgba(18, 52, 86" in css  # textColor, as muted rgba

    def test_no_hardcoded_dark_palette_literals(self):
        """A-17: the old CSS baked in the NCCU dark palette regardless of theme."""
        css = app_css(dict(STREAMLIT_DEFAULTS))
        for literal in ("#e0a3a3", "#1b1b1b", "#5a0010", "#240000"):
            assert literal not in css, f"{literal} is hardcoded in app_css output"
        assert "rgba(255,255,255,0.55)" not in css
        assert "rgba(49,51,63,0.45)" not in css

    def test_does_not_hide_the_sidebar_collapse_control(self):
        """A-29: hiding it locked the sidebar open, costing canvas width."""
        css = app_css(dict(STREAMLIT_DEFAULTS))
        assert "stSidebarCollapseButton" not in css
        assert "collapsedControl" not in css

    def test_fetches_no_external_resources(self):
        """A-35: the Google Fonts <link> broke offline and disclosed a request."""
        css = app_css(dict(STREAMLIT_DEFAULTS))
        for token in ("http://", "https://", "@import", "fonts.googleapis"):
            assert token not in css, f"app_css reaches out to {token}"

    def test_missing_keys_fall_back_to_defaults(self):
        css = app_css({"base": "light"})
        assert STREAMLIT_DEFAULTS["primaryColor"] in css


class TestHomePageUsesThemedCss:
    """A-17 end-to-end: Home.py must not reintroduce hardcoded colours."""

    @pytest.fixture(scope="class")
    def home_source(self) -> str:
        import eagleplot

        return (Path(eagleplot.__file__).resolve().parent / "Home.py").read_text("utf-8")

    def test_home_delegates_styling_to_app_css(self, home_source: str):
        assert "app_css()" in home_source

    def test_home_has_no_hardcoded_theme_colours(self, home_source: str):
        for literal in ("#e0a3a3", "#1b1b1b", "#5a0010", "rgba(255,255,255,0.55)"):
            assert literal not in home_source, f"{literal} is back in Home.py"

    def test_home_does_not_load_external_fonts(self, home_source: str):
        assert "fonts.googleapis.com" not in home_source

    def test_every_image_has_alt_text(self, home_source: str):
        """A-29: four base64 <img> tags carried no alt attribute."""
        img_count = home_source.count("<img src=")
        alt_count = home_source.count("alt=")
        assert img_count > 0
        assert alt_count >= img_count, f"{img_count} <img> tags but only {alt_count} alt attributes"

    def test_data_tools_is_reachable_from_the_nav(self, home_source: str):
        """A-30: the page was routable by URL but had no sidebar link, and
        st.navigation(position="hidden") makes the sidebar the only navigation."""
        assert "st.page_link(datatools_pg" in home_source
