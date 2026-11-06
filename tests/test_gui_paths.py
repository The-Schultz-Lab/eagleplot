"""Path-resolution guards for the Streamlit GUI.

These tests exist because two shipped defects came from path handling that only
broke on a platform or Python version CI did not test:

- **G-010** — ``Path(__file__)`` is *relative* when Streamlit invokes the entry
  script, so ``Path(__file__).parent / "pages"`` produced a relative path that
  ``st.Page()`` could not stat. Reported by a student on macOS / Python 3.9.
  The fix was ``.resolve()``; these tests assert it stays.

- **A-09** — runtime assets were resolved relative to the *repository root*, so a
  pip-installed GUI came up with no logo, no branding, no example data and an
  empty gallery. Every ``.exists()`` guard in ``Home.py`` meant this failed
  silently. These tests assert the assets are inside the package and present.

Both checks are OS- and version-independent, which is the point: they hold on
every matrix cell rather than only on the one where the bug happened to show.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def package_root() -> Path:
    import eagleplot

    return Path(eagleplot.__file__).resolve().parent


# ── G-010: every GUI path must be absolute ────────────────────────────────────


class TestGuiPathsAreAbsolute:
    """Regression guard for G-010.

    ``Home.py`` imports streamlit at module scope, so rather than importing it we
    read the source and assert the path expressions are anchored with
    ``.resolve()``. That keeps the test dependency-free and lets it run on every
    matrix cell.
    """

    @pytest.fixture(scope="class")
    def home_source(self, package_root: Path) -> str:
        return (package_root / "Home.py").read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", ["_PAGES_DIR", "_ASSETS_DIR"])
    def test_path_constants_are_resolved(self, home_source: str, name: str) -> None:
        line = next(
            (ln for ln in home_source.splitlines() if ln.strip().startswith(f"{name} =")),
            None,
        )
        assert line is not None, f"{name} not found in Home.py"
        assert ".resolve()" in line, (
            f"{name} must be anchored with Path(__file__).resolve() -- a relative "
            f"__file__ under `streamlit run` is what caused G-010. Got: {line.strip()}"
        )

    def test_no_unresolved_file_parent_paths(self, home_source: str) -> None:
        offenders = [
            ln.strip()
            for ln in home_source.splitlines()
            if "Path(__file__).parent" in ln and ".resolve()" not in ln
        ]
        assert not offenders, (
            f"Path(__file__).parent without .resolve() reintroduces G-010: {offenders}"
        )

    def test_page_files_referenced_by_home_all_exist(self, package_root: Path) -> None:
        import re

        source = (package_root / "Home.py").read_text(encoding="utf-8")
        referenced = set(re.findall(r'_PAGES_DIR / "([^"]+)"', source))
        assert referenced, "expected Home.py to reference page files via _PAGES_DIR"

        pages_dir = package_root / "pages"
        missing = sorted(name for name in referenced if not (pages_dir / name).exists())
        assert not missing, f"Home.py routes to page files that do not exist: {missing}"


# ── A-09: assets must live inside the package and be present ──────────────────


class TestPackagedAssetsArePresent:
    """Regression guard for A-09.

    Asserted against the installed package directory, so this fails if an asset is
    moved back outside the package or dropped from ``package-data``.
    """

    @pytest.mark.parametrize(
        "relative_path",
        [
            "assets/logo.png",
            "assets/logo.ico",
            "assets/icon.png",
            "assets/nccu-horiz-logo.png",
            "assets/nccu-wings.png",
            "gallery/manifest.json",
            "example-data/Artificial/normal_distribution.csv",
            "example-data/Artificial/gaussian_surface.npy",
        ],
    )
    def test_asset_ships_with_the_package(self, package_root: Path, relative_path: str) -> None:
        asset = package_root / relative_path
        assert asset.is_file(), (
            f"{relative_path} is missing from the package. Assets resolved relative to "
            "the repo root are absent from a pip install -- see audit A-09."
        )
        assert asset.stat().st_size > 0, f"{relative_path} is empty"

    def test_every_gallery_manifest_entry_has_its_png(self, package_root: Path) -> None:
        manifest = json.loads((package_root / "gallery" / "manifest.json").read_text("utf-8"))
        missing = [
            entry["filename"]
            for entry in manifest
            if not (package_root / "gallery" / entry["filename"]).is_file()
        ]
        assert not missing, f"gallery manifest references missing PNGs: {missing}"

    def test_example_metadata_matches_shipped_files(self, package_root: Path) -> None:
        """Every dataset the Data Upload page advertises must actually ship."""
        import re

        source = (package_root / "pages" / "1_Data_Upload.py").read_text(encoding="utf-8")

        # Slice from `_EXAMPLES = {` to its closing brace at column 0, so this does not
        # depend on whatever happens to follow the dict.
        lines = source.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("_EXAMPLES = {"))
        end = next(i for i, ln in enumerate(lines[start:], start) if ln == "}")
        block = "\n".join(lines[start : end + 1])
        advertised = set(re.findall(r'"([\w-]+\.(?:csv|npy))":', block))
        assert advertised, "expected _EXAMPLES to list example filenames"

        example_dir = package_root / "example-data" / "Artificial"
        missing = sorted(name for name in advertised if not (example_dir / name).exists())
        assert not missing, f"_EXAMPLES advertises datasets that do not ship: {missing}"
