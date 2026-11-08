"""scripts/release/check_release.py: the checks the release workflow runs (roadmap D6)."""

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    path = REPO / "scripts" / "release" / "check_release.py"
    spec = importlib.util.spec_from_file_location("_check_release", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()
VERSION = check.package_version()


def test_the_repository_is_consistent_for_its_own_version():
    """__version__, the CHANGELOG section and the docs badge agree (G-035)."""
    assert check.check_sources(VERSION, f"v{VERSION}") == []


def test_a_wrong_tag_is_reported():
    problems = check.check_sources(VERSION, "v0.0.1")
    assert len(problems) == 1 and "doesn't match the package version" in problems[0]


class TestChangelogSection:
    CHANGELOG = (
        "# Changelog\n\n## [Unreleased]\n\n---\n\n"
        "## [2.0.0] - 2026-01-02\n\nBig release.\n\n### Added\n\n- A thing.\n\n---\n\n"
        "## [1.0.0] - 2025-05-06\n\n- Old.\n\n---\n\n"
        "[Unreleased]: https://x/compare/v2.0.0...HEAD\n[2.0.0]: https://x/compare/v1.0.0...v2.0.0\n"
    )

    def test_section_body(self):
        body = check.changelog_section(self.CHANGELOG, "2.0.0")
        assert body == "Big release.\n\n### Added\n\n- A thing."

    def test_last_section_stops_at_the_links(self):
        assert check.changelog_section(self.CHANGELOG, "1.0.0") == "- Old."

    @pytest.mark.parametrize("version", ["3.0.0", "Unreleased", "2.0"])
    def test_missing_or_undated(self, version):
        assert check.changelog_section(self.CHANGELOG, version) is None


def _wheel(dist: Path, version: str, files: dict, top_level: str = "eagleplot\n") -> Path:
    path = dist / f"eagleplot-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as whl:
        for name, data in files.items():
            whl.writestr(name, data)
        whl.writestr(f"eagleplot-{version}.dist-info/top_level.txt", top_level)
    return path


def _sdist(dist: Path, version: str, names=("PKG-INFO",)) -> Path:
    path = dist / f"eagleplot-{version}.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            info = tarfile.TarInfo(f"eagleplot-{version}/{name}")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    return path


GOOD_FILES = {name: b"ok\n" for name in check.WHEEL_DATA_FILES}


class TestDist:
    def test_a_good_build_passes(self, tmp_path):
        _wheel(tmp_path, "9.9.9", GOOD_FILES)
        _sdist(tmp_path, "9.9.9")
        assert check.check_dist(tmp_path, "9.9.9") == []

    def test_wrong_version_in_the_file_names(self, tmp_path):
        _wheel(tmp_path, "9.9.8", GOOD_FILES)
        _sdist(tmp_path, "9.9.8")
        problems = check.check_dist(tmp_path, "9.9.9")
        assert len(problems) == 2 and all("expected exactly" in p for p in problems)

    def test_another_top_level_package(self, tmp_path):
        _wheel(tmp_path, "9.9.9", {**GOOD_FILES, "modules/io.py": b""}, "eagleplot\nmodules\n")
        _sdist(tmp_path, "9.9.9")
        problems = "\n".join(check.check_dist(tmp_path, "9.9.9"))
        assert "top_level.txt" in problems and "outside eagleplot/" in problems

    def test_missing_gui_data_files(self, tmp_path):
        files = dict(GOOD_FILES)
        del files["eagleplot/assets/logo.png"]
        _wheel(tmp_path, "9.9.9", files)
        _sdist(tmp_path, "9.9.9")
        (problem,) = check.check_dist(tmp_path, "9.9.9")
        assert "G-012" in problem and "logo.png" in problem

    def test_windows_line_endings(self, tmp_path):
        _wheel(tmp_path, "9.9.9", {**GOOD_FILES, "eagleplot/io.py": b"import os\r\n"})
        _sdist(tmp_path, "9.9.9")
        (problem,) = check.check_dist(tmp_path, "9.9.9")
        assert "G-032" in problem and "eagleplot/io.py" in problem

    def test_desktop_build_output_in_the_sdist(self, tmp_path):
        _wheel(tmp_path, "9.9.9", GOOD_FILES)
        _sdist(tmp_path, "9.9.9", ["PKG-INFO", "packaging/desktop/node_modules/x/index.js"])
        (problem,) = check.check_dist(tmp_path, "9.9.9")
        assert "desktop build output" in problem


def test_notes_are_written_for_the_release(tmp_path):
    notes = tmp_path / "notes.md"
    assert check.main(["--tag", f"v{VERSION}", "--notes", str(notes)]) == 0
    text = notes.read_text(encoding="utf-8")
    assert text.strip() and "## [" not in text and "[Unreleased]" not in text


def test_the_release_workflow_publishes_without_a_token():
    """PyPI upload by Trusted Publishing only: OIDC permission, no password or token."""
    import yaml

    text = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    jobs = workflow["jobs"]
    assert jobs["pypi"]["permissions"] == {"id-token": "write"}
    assert jobs["pypi"]["environment"]["name"] == "pypi"
    assert jobs["pypi"]["needs"] == "build"
    publish = [s for s in jobs["pypi"]["steps"] if "gh-action-pypi-publish" in s.get("uses", "")]
    assert publish and "password" not in publish[0].get("with", {})
    assert "secrets." not in text
    build_steps = " ".join(s.get("run", "") for s in jobs["build"]["steps"])
    assert "check_release.py" in build_steps
    assert workflow.get("permissions") == {"contents": "read"}  # write only where needed
