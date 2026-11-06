"""eagleplot.utils.host_folders: typed folders in the desktop build (roadmap D5.2)."""

from pathlib import Path

import pytest

from eagleplot.utils import host_folders, runtime


class TestHostPath:
    @pytest.mark.parametrize(
        "typed, expected",
        [
            ("C:\\Users\\ana\\data", "C:\\Users\\ana\\data"),
            ("C:/Users/ana/data/", "C:\\Users\\ana\\data"),  # forward slashes work too
            ('  "C:\\Users\\ana\\my data"  ', "C:\\Users\\ana\\my data"),  # Copy as path
            ("~\\data", "C:\\Users\\ana\\data"),
            ("~/data", "C:\\Users\\ana\\data"),
            ("D:\\runs\\..\\spectra", "D:\\spectra"),
        ],
    )
    def test_windows(self, typed, expected):
        assert host_folders.host_path(typed, "win32", "C:\\Users\\ana") == expected

    @pytest.mark.parametrize(
        "typed, expected",
        [
            ("/Users/ana/data", "/Users/ana/data"),
            ("/Users/ana/data/", "/Users/ana/data"),
            ("'/Users/ana/my data'", "/Users/ana/my data"),
            ("~", "/Users/ana"),
            ("~/data", "/Users/ana/data"),
            ("/Volumes/USB/runs", "/Volumes/USB/runs"),
        ],
    )
    def test_macos_and_linux(self, typed, expected):
        assert host_folders.host_path(typed, "darwin", "/Users/ana") == expected

    @pytest.mark.parametrize("typed", ["", "   ", "data", "./data", "..\\data"])
    @pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
    def test_relative_paths_ask_for_the_full_path(self, typed, platform):
        with pytest.raises(NotADirectoryError, match="full path"):
            host_folders.host_path(typed, platform, "/home/ana")

    def test_a_posix_path_is_not_a_full_windows_path(self):
        with pytest.raises(NotADirectoryError, match=r"C:\\Users\\you\\data"):
            host_folders.host_path("Users\\ana", "win32", "C:\\Users\\ana")

    def test_home_unknown(self):
        with pytest.raises(NotADirectoryError, match="home folder"):
            host_folders.host_path("~/data", "linux", None)


class TestLocalFolder:
    def test_server_paths_are_used_as_typed(self, tmp_path):
        assert runtime.IS_DESKTOP is False
        assert host_folders.local_folder(str(tmp_path)) == Path(str(tmp_path))

    @pytest.fixture
    def desktop(self, monkeypatch):
        """A fake desktop build that records mounts instead of making them."""
        mounts = []
        monkeypatch.setattr(runtime, "IS_DESKTOP", True)
        monkeypatch.setattr(host_folders, "_mounts", {})
        monkeypatch.setattr(host_folders, "_host", lambda: ("win32", "C:\\Users\\ana"))
        monkeypatch.setattr(host_folders, "_mount", lambda host, mp: mounts.append((host, mp)))
        return mounts

    def test_desktop_mounts_the_typed_folder(self, desktop):
        assert host_folders.local_folder("~\\data") == Path("/mnt/host/1")
        assert desktop == [("C:\\Users\\ana\\data", "/mnt/host/1")]

    def test_the_same_folder_is_mounted_once(self, desktop):
        first = host_folders.local_folder("C:\\Users\\ana\\data")
        again = host_folders.local_folder('"c:/users/ana/data/"')  # Windows ignores case
        other = host_folders.local_folder("D:\\spectra")
        assert first == again == Path("/mnt/host/1")
        assert other == Path("/mnt/host/2")
        assert [host for host, _ in desktop] == ["C:\\Users\\ana\\data", "D:\\spectra"]

    def test_a_failed_mount_is_not_remembered(self, desktop, monkeypatch):
        def refuse(host, mp):
            raise NotADirectoryError(f"Not a folder EaglePlot can open: {host}")

        monkeypatch.setattr(host_folders, "_mount", refuse)
        with pytest.raises(NotADirectoryError, match="can open"):
            host_folders.local_folder("C:\\nope")
        assert host_folders._mounts == {}
