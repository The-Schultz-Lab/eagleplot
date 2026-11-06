"""eagleplot.utils.runtime: capabilities of the server, browser and desktop builds (D2.1)."""

import dataclasses

import pytest

from eagleplot.utils import access, runtime

SERVER = runtime.compute_capabilities("linux", {})
BROWSER = runtime.compute_capabilities("emscripten", {})
DESKTOP = runtime.compute_capabilities("emscripten", {"EAGLEPLOT_DESKTOP": "1"})


class TestComputeCapabilities:
    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_server_can_do_everything(self, platform):
        caps = runtime.compute_capabilities(platform, {})
        assert caps == runtime.Capabilities(
            local_folders=True, outbound_http=True, persistent_storage=True, native_speed=True
        )

    def test_desktop_flag_alone_does_not_change_a_server(self):
        assert runtime.compute_capabilities("linux", {"EAGLEPLOT_DESKTOP": "1"}) == SERVER

    def test_browser(self):
        assert BROWSER == runtime.Capabilities(
            local_folders=False, outbound_http=False, persistent_storage=False, native_speed=False
        )

    def test_desktop(self):
        assert DESKTOP == runtime.Capabilities(
            local_folders=True, outbound_http=True, persistent_storage=True, native_speed=False
        )

    @pytest.mark.parametrize("value", ["0", "", "true", "yes"])
    def test_only_exactly_1_marks_the_desktop_build(self, value):
        assert runtime.compute_capabilities("emscripten", {"EAGLEPLOT_DESKTOP": value}) == BROWSER

    def test_browser_storage_hook(self, monkeypatch):
        monkeypatch.setattr(runtime, "_browser_storage_is_persistent", lambda environ: True)
        assert runtime.compute_capabilities("emscripten", {}).persistent_storage is True

    def test_capabilities_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SERVER.local_folders = False  # type: ignore[misc]


class TestThisProcess:
    def test_tests_run_on_a_server_runtime(self):
        assert runtime.IS_PYODIDE is False
        assert runtime.IS_DESKTOP is False
        assert runtime.capabilities() == SERVER

    def test_capabilities_are_computed_once(self):
        assert runtime.capabilities() is runtime.capabilities()


class TestFolderLoading:
    """The browser build has no local disk: folder loading is off whatever the env says."""

    @pytest.fixture(autouse=True)
    def local_client(self, monkeypatch):
        monkeypatch.setattr(access, "_client_ip", lambda: None)

    def _with_platform(self, monkeypatch, platform):
        monkeypatch.setattr(access.sys, "platform", platform)

    @pytest.mark.parametrize("env", [{}, {"EAGLEPLOT_ALLOW_FOLDER_LOAD": "1"}])
    def test_browser_never_loads_folders(self, monkeypatch, env):
        self._with_platform(monkeypatch, "emscripten")
        assert access.folder_loading_allowed_here(env) is False

    def test_browser_ignores_the_legacy_variable_too(self, monkeypatch):
        self._with_platform(monkeypatch, "emscripten")
        assert access.folder_loading_allowed_here({"PLOTTLE_ALLOW_FOLDER_LOAD": "1"}) is False

    def test_desktop_follows_the_usual_rules(self, monkeypatch):
        self._with_platform(monkeypatch, "emscripten")
        assert access.folder_loading_allowed_here({"EAGLEPLOT_DESKTOP": "1"}) is True
        env = {"EAGLEPLOT_DESKTOP": "1", "EAGLEPLOT_ALLOW_FOLDER_LOAD": "0"}
        assert access.folder_loading_allowed_here(env) is False

    def test_server_follows_the_usual_rules(self, monkeypatch):
        self._with_platform(monkeypatch, "linux")
        assert access.folder_loading_allowed_here({}) is True
        assert access.folder_loading_allowed_here({"EAGLEPLOT_ALLOW_FOLDER_LOAD": "0"}) is False

    def test_process_capabilities_gate_the_default_environment(self, monkeypatch):
        monkeypatch.setattr(access, "capabilities", lambda: BROWSER)
        monkeypatch.setenv("EAGLEPLOT_ALLOW_FOLDER_LOAD", "1")
        assert access.folder_loading_allowed_here() is False


def test_data_upload_explains_folder_import_in_the_browser(monkeypatch):
    from pathlib import Path

    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setenv("EAGLEPLOT_ALLOW_FOLDER_LOAD", "1")
    monkeypatch.setattr(runtime, "capabilities", lambda: BROWSER)
    monkeypatch.setattr(access, "capabilities", lambda: BROWSER)
    page = Path(__file__).resolve().parent.parent / "eagleplot" / "pages" / "1_Data_Upload.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception
    assert not any(t.key == "batch_folder_path" for t in at.text_input)
    assert any("running in your browser" in i.value for i in at.info)
