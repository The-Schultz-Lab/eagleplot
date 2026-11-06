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
            local_folders=False,
            outbound_http=False,
            persistent_storage=False,
            native_speed=False,
            app_config_file=False,
            pip_install=False,
        )

    def test_desktop(self):
        assert DESKTOP == runtime.Capabilities(
            local_folders=True,
            outbound_http=True,
            persistent_storage=True,
            native_speed=False,
            app_config_file=False,
            pip_install=False,
        )

    def test_browser_settings_persist_when_the_site_mounts_a_config_dir(self):
        caps = runtime.compute_capabilities("emscripten", {"EAGLEPLOT_CONFIG_DIR": "/x"})
        assert caps.persistent_storage is True
        assert runtime.compute_capabilities("emscripten", {"EAGLEPLOT_CONFIG_DIR": ""}) == BROWSER

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


class TestMissingPackageMessage:
    @pytest.mark.parametrize(
        "message",
        [
            "The `h5py` package is required to load HDF5 files. Install it with: pip install h5py",
            "Two-way ANOVA requires statsmodels. Install it with: `pip install statsmodels`",
        ],
    )
    def test_unchanged_where_pip_works(self, message):
        assert runtime.missing_package_message(message, SERVER) == message

    def test_browser_names_the_package_without_a_useless_pip_hint(self):
        msg = runtime.missing_package_message(
            "The `h5py` package is required to load HDF5 files. Install it with: pip install h5py",
            BROWSER,
        )
        assert msg.startswith("The `h5py` package is required to load HDF5 files. It isn't")
        assert "install EaglePlot on your computer" in msg
        assert msg.endswith("`pip install h5py`.")

    def test_a_bare_hint_still_names_the_package(self):
        msg = runtime.missing_package_message(
            "Install it with: pip install xarray netcdf4", BROWSER
        )
        assert msg.startswith("This needs xarray netcdf4.")

    def test_other_errors_are_left_alone(self):
        assert runtime.missing_package_message("File is empty.", BROWSER) == "File is empty."


def test_config_dir_env_moves_config_json(tmp_path):
    """The browser build points EAGLEPLOT_CONFIG_DIR at storage that survives a reload."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, EAGLEPLOT_CONFIG_DIR=str(tmp_path))
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from eagleplot.utils import user_settings as us; "
            "us.save_preset('p', {'grid': True}); print(us.get_config_path())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == str(tmp_path / "config.json")
    assert '"p"' in (tmp_path / "config.json").read_text(encoding="utf-8")


def test_nothing_in_eagleplot_needs_threads_processes_or_sleep():
    """Pyodide has no threads or subprocesses, and time.sleep blocks its only worker."""
    import re
    from pathlib import Path

    banned = re.compile(
        r"^\s*(import|from)\s+(threading|subprocess|multiprocessing|concurrent\.futures)\b"
        r"|\btime\.sleep\(",
        re.M,
    )
    root = Path(runtime.__file__).resolve().parents[1]
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if banned.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def _browser_page(monkeypatch, page_file, **caps):
    from pathlib import Path

    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    fake = dataclasses.replace(BROWSER, **caps)
    monkeypatch.setattr(runtime, "capabilities", lambda: fake)
    import pandas as pd

    page = Path(__file__).resolve().parent.parent / "eagleplot" / "pages" / page_file
    at = AppTest.from_file(str(page), default_timeout=30)
    # Spectroscopy and Batch Analysis show their tools only once a dataset is loaded.
    at.session_state["datasets"] = {"d.csv": pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})}
    at.session_state["current_dataset"] = "d.csv"
    at.run()
    assert not at.exception
    return at


def test_settings_page_in_the_browser(monkeypatch):
    at = _browser_page(monkeypatch, "13_Settings.py")
    assert not any(b.label == "Save theme" for b in at.get("form_submit_button"))
    assert any("theme is set by the app" in i.value for i in at.info)
    assert any("until you close or reload" in w.value for w in at.warning)


def test_settings_page_with_persistent_browser_storage(monkeypatch):
    at = _browser_page(monkeypatch, "13_Settings.py", persistent_storage=True)
    assert not any("until you close or reload" in w.value for w in at.warning)


def test_spectroscopy_links_to_nist_instead_of_fetching_in_the_browser(monkeypatch):
    at = _browser_page(monkeypatch, "9_Spectroscopy.py")
    assert not any(b.key == "nist_fetch_btn" for b in at.button)
    assert any("can't download from NIST directly" in i.value for i in at.info)


def test_batch_workflows_are_not_promised_to_persist_in_the_browser(monkeypatch):
    at = _browser_page(monkeypatch, "11_Batch_Analysis.py")
    texts = [i.value for i in at.info]
    assert not any("persist across app restarts" in t for t in texts)
    assert any("until you close or reload" in t for t in texts)


@pytest.mark.parametrize(
    "pyodide, desktop, shown", [(False, False, False), (True, False, True), (True, True, False)]
)
def test_home_says_when_it_runs_in_the_browser(monkeypatch, pyodide, desktop, shown):
    from pathlib import Path

    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setattr(runtime, "IS_PYODIDE", pyodide)
    monkeypatch.setattr(runtime, "IS_DESKTOP", desktop)
    home = Path(__file__).resolve().parent.parent / "eagleplot" / "Home.py"
    at = AppTest.from_file(str(home), default_timeout=30).run()
    assert not at.exception
    found = any("running in your browser" in c.value for c in at.sidebar.caption)
    assert found is shown
