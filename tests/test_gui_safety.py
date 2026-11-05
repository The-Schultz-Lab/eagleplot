"""GUI safety behaviour from audit Gate 3: A-15 (folder access), A-32 (session
download), A-33 (destructive actions need confirmation)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from eagleplot.utils.access import FOLDER_LOAD_ENV, folder_loading_allowed

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

PAGES = Path(__file__).resolve().parent.parent / "eagleplot" / "pages"


# ── A-15 ─────────────────────────────────────────────────────────────────────


class TestFolderLoadingAllowed:
    def test_local_connection_allowed_by_default(self):
        assert folder_loading_allowed(None, None) is True

    def test_remote_connection_denied_by_default(self):
        assert folder_loading_allowed(None, "203.0.113.7") is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_env_can_force_on_for_remote(self, value):
        assert folder_loading_allowed(value, "203.0.113.7") is True

    @pytest.mark.parametrize("value", ["0", "false", "No", "off"])
    def test_env_can_force_off_for_local(self, value):
        assert folder_loading_allowed(value, None) is False

    def test_unrecognised_env_value_falls_back_to_connection(self):
        assert folder_loading_allowed("maybe", None) is True
        assert folder_loading_allowed("maybe", "203.0.113.7") is False


class TestBatchFolderTab:
    # AppTest has no real connection (st.context.ip_address is a MagicMock), so the
    # client address is patched to what Streamlit reports for each case.
    def test_tab_works_locally(self, monkeypatch):
        monkeypatch.delenv(FOLDER_LOAD_ENV, raising=False)
        monkeypatch.setattr("eagleplot.utils.access._client_ip", lambda: None)
        at = AppTest.from_file(str(PAGES / "1_Data_Upload.py"), default_timeout=30).run()
        assert not at.exception
        assert any(t.key == "batch_folder_path" for t in at.text_input)

    def test_tab_hidden_for_remote_connection(self, monkeypatch):
        monkeypatch.delenv(FOLDER_LOAD_ENV, raising=False)
        monkeypatch.setattr("eagleplot.utils.access._client_ip", lambda: "203.0.113.7")
        at = AppTest.from_file(str(PAGES / "1_Data_Upload.py"), default_timeout=30).run()
        assert not at.exception
        assert not any(t.key == "batch_folder_path" for t in at.text_input)

    def test_tab_hidden_when_disabled(self, monkeypatch):
        monkeypatch.setenv(FOLDER_LOAD_ENV, "0")
        monkeypatch.setattr("eagleplot.utils.access._client_ip", lambda: None)
        at = AppTest.from_file(str(PAGES / "1_Data_Upload.py"), default_timeout=30).run()
        assert not at.exception
        assert not any(t.key == "batch_folder_path" for t in at.text_input)
        assert any(FOLDER_LOAD_ENV in i.value for i in at.info)


# ── A-33 ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def settings_page(tmp_path, monkeypatch):
    """Settings page with config.json redirected to a temp file holding 2 presets."""
    import eagleplot.utils.user_settings as us

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "version": "1.0",
                "defaults": {"fontsize": 14},
                "presets": {"Poster": {"fontsize": 20}, "Talk": {"fontsize": 18}},
                "workflows": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(us, "_CONFIG_PATH", cfg)
    at = AppTest.from_file(str(PAGES / "13_Settings.py"), default_timeout=30).run()
    assert not at.exception
    return at, cfg


def _button(at, label):
    return next(b for b in at.button if b.label == label)


class TestDestructiveActionsNeedConfirmation:
    @pytest.mark.parametrize(
        "label",
        [
            "Reset defaults to factory settings",
            "Delete preset",
            "Delete config.json (reset everything)",
        ],
    )
    def test_disabled_until_confirmed(self, settings_page, label):
        at, _ = settings_page
        assert _button(at, label).disabled

    def test_preset_delete_needs_confirmation_each_time(self, settings_page):
        at, cfg = settings_page
        at.checkbox(key="confirm_preset_del").check().run()
        _button(at, "Delete preset").click().run()
        assert not at.exception
        presets = json.loads(cfg.read_text(encoding="utf-8"))["presets"]
        assert len(presets) == 1
        # The tick must not carry over to the next preset.
        assert not at.checkbox(key="confirm_preset_del").value
        assert _button(at, "Delete preset").disabled

    def test_config_delete_offers_backup_and_tolerates_missing_file(self, settings_page):
        at, cfg = settings_page
        assert any(d.label == "Download a backup of config.json" for d in at.get("download_button"))
        at.checkbox(key="confirm_config_delete").check().run()
        cfg.unlink()  # file vanishes between render and click: used to raise
        _button(at, "Delete config.json (reset everything)").click().run()
        assert not at.exception
        assert not cfg.exists()


# ── A-32 ─────────────────────────────────────────────────────────────────────


def _session_downloads(at):
    return [d for d in at.get("download_button") if d.label == "Download session.json"]


class TestSessionDownloadPersists:
    @pytest.fixture
    def export_page(self):
        at = AppTest.from_file(str(PAGES / "6_Export_Results.py"), default_timeout=30)
        at.run()
        at.session_state["datasets"] = {"a.csv": pd.DataFrame({"x": [1, 2, 3]})}
        at.run()
        assert not at.exception
        return at

    def test_download_survives_later_interaction(self, export_page):
        at = export_page
        assert not _session_downloads(at)
        _button(at, "Generate session file").click().run()
        assert len(_session_downloads(at)) == 1
        at.run()  # any later interaction is a rerun
        assert len(_session_downloads(at)) == 1

    def test_download_withdrawn_when_session_changes(self, export_page):
        at = export_page
        _button(at, "Generate session file").click().run()
        datasets = dict(at.session_state["datasets"])
        datasets["b.csv"] = pd.DataFrame({"y": [4.0]})
        at.session_state["datasets"] = datasets
        at.run()
        assert not _session_downloads(at)
        assert any(b.label == "Generate session file" for b in at.button)
