"""Tests for the documentation site build — audit A-13.

The published site used to be a single landing page that linked to none of the
seven markdown guides, while a full Sphinx setup sat in ``docs/`` that CI never
built. Sphinx was dropped (see the mediator repo's decision log) and
``build_docs.py`` now renders the guides into a navigable site.

These tests guard the two ways that can rot:

1. A guide is added to ``build_docs.PAGES`` but never linked from the landing
   page, so it ships unreachable — the original A-13 failure, exactly.
2. The build itself breaks, or starts emitting dead links.

The static checks run everywhere. The full build needs the ``markdown`` package
from the ``docs`` extra, which the test matrix does not install, so it skips when
absent — the docs workflow runs the real build on every docs change.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_docs():
    """Import build_docs.py by path — it lives at the repo root, not in the package."""
    spec = importlib.util.spec_from_file_location(
        "_build_docs_under_test", _REPO_ROOT / "build_docs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_docs = _load_build_docs()

_HAS_MARKDOWN = importlib.util.find_spec("markdown") is not None
_needs_markdown = pytest.mark.skipif(
    not _HAS_MARKDOWN, reason="requires the `markdown` package (pip install -e '.[docs]')"
)


class TestPageManifest:
    def test_every_source_file_exists(self):
        missing = [
            str(page.source.relative_to(_REPO_ROOT))
            for page in build_docs.PAGES
            if not page.source.is_file()
        ]
        assert not missing, f"build_docs.PAGES references missing sources: {missing}"

    def test_slugs_are_unique(self):
        slugs = [page.slug for page in build_docs.PAGES]
        assert len(slugs) == len(set(slugs)), f"duplicate slugs in PAGES: {slugs}"

    def test_every_markdown_guide_is_published_or_deliberately_excluded(self):
        """A guide added to docs/ but not to PAGES would ship unreachable.

        ``docs/archive/`` is excluded on purpose: superseded internal notes.
        """
        published = {page.source.resolve() for page in build_docs.PAGES}
        found = {
            path.resolve()
            for path in (_REPO_ROOT / "docs").rglob("*.md")
            if "archive" not in path.parts and "_site" not in path.parts
        }
        unpublished = sorted(str(p.relative_to(_REPO_ROOT)) for p in found - published)
        assert not unpublished, (
            f"markdown guides not in build_docs.PAGES, so unreachable on the site: "
            f"{unpublished}. Add them to PAGES and link them from docs/index.html, "
            "or move them under docs/archive/."
        )


class TestLandingPageLinksEveryGuide:
    """Regression guard for A-13 itself."""

    @pytest.fixture(scope="class")
    def index_html(self) -> str:
        return (_REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    def test_index_links_every_built_page(self, index_html: str):
        unlinked = [
            page.output_name
            for page in build_docs.PAGES
            if f'href="{page.output_name}"' not in index_html
        ]
        assert not unlinked, (
            f"docs/index.html does not link: {unlinked}. These pages would be "
            "published but reachable only by guessing the URL — the original A-13 bug."
        )

    def test_index_has_no_links_to_raw_markdown(self, index_html: str):
        md_links = re.findall(r'href="([^"]+\.md)"', index_html)
        assert not md_links, (
            f"docs/index.html links raw markdown {md_links}, which browsers download "
            "rather than render. Link the built .html page instead."
        )

    def test_no_links_to_removed_sphinx_output(self, index_html: str):
        for stale in ("genindex", "py-modindex", "_build/", "index.rst"):
            assert stale not in index_html, (
                f"index.html references removed Sphinx artifact: {stale}"
            )


class TestSphinxIsFullyRemoved:
    """Half-removing Sphinx would leave dead config that looks authoritative."""

    @pytest.mark.parametrize(
        "path", ["docs/conf.py", "docs/index.rst", "docs/Makefile", "docs/api"]
    )
    def test_sphinx_artifact_is_gone(self, path: str):
        assert not (_REPO_ROOT / path).exists(), f"{path} still present after dropping Sphinx"

    def test_docs_extra_does_not_pin_sphinx(self):
        pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        start = pyproject.index("docs = [")
        docs_extra = pyproject[start : pyproject.index("]", start)]
        assert "sphinx" not in docs_extra.lower()
        assert "furo" not in docs_extra.lower()
        assert "markdown" in docs_extra.lower()


@_needs_markdown
class TestFullBuild:
    @pytest.fixture(scope="class")
    def built_site(self, tmp_path_factory) -> Path:
        assert build_docs.main() == 0
        return build_docs._SITE

    def test_builds_every_page_plus_index_and_css(self, built_site: Path):
        expected = {"index.html", "docs.css"} | {p.output_name for p in build_docs.PAGES}
        actual = {p.name for p in built_site.iterdir()}
        assert expected <= actual, f"missing from build: {sorted(expected - actual)}"

    # Added to the site by .github/workflows/docs.yml after build_docs.py runs: the
    # in-browser app (packaging/web/build_site.py) is built into _site/app/.
    DEPLOY_ADDED = {"app/"}

    def test_no_broken_internal_links(self, built_site: Path):
        broken = []
        for page in sorted(built_site.glob("*.html")):
            html = page.read_text(encoding="utf-8")
            for href in re.findall(r'href="([^"#?]+)"', html):
                if href.startswith(("http://", "https://", "mailto:")):
                    continue
                if href in self.DEPLOY_ADDED:
                    continue
                if not (built_site / href).exists():
                    broken.append(f"{page.name} -> {href}")
        assert not broken, f"broken internal links in built site: {broken}"

    def test_cross_references_between_guides_are_rewritten(self, built_site: Path):
        """Guides link each other as `tutorials/cli_guide.md`, which would 404."""
        for page in built_site.glob("*.html"):
            md_links = re.findall(r'href="([^"]+\.md)"', page.read_text(encoding="utf-8"))
            assert not md_links, f"{page.name} still links raw markdown: {md_links}"

    def test_every_page_has_nav_and_stylesheet(self, built_site: Path):
        for page in built_site.glob("*.html"):
            html = page.read_text(encoding="utf-8")
            assert 'class="nav"' in html, f"{page.name} has no nav"
            if page.name != "index.html":  # index.html carries its own inline styles
                assert 'href="docs.css"' in html, f"{page.name} does not load docs.css"

    def test_wide_tables_scroll_in_their_own_container(self, built_site: Path):
        """A wide reference table must not make the whole page scroll sideways."""
        cheatsheet = (built_site / "cheatsheet.html").read_text(encoding="utf-8")
        assert cheatsheet.count("<table>") == cheatsheet.count('<div class="table-scroll">')
        assert cheatsheet.count("<table>") > 0

    def test_generated_pages_fetch_no_external_resources(self, built_site: Path):
        """Same offline/privacy concern as A-35 in the app itself."""
        for page in built_site.glob("*.html"):
            if page.name == "index.html":
                continue  # hand-written; checked separately if it ever gains assets
            html = page.read_text(encoding="utf-8")
            # Loaded by the browser: src= (scripts, images, frames) and <link href> (styles,
            # fonts). Ordinary <a href> links to other sites are only followed on a click.
            external = re.findall(r'src="(https?://[^"]+)"', html)
            external += re.findall(r'<link\b[^>]*\bhref="(https?://[^"]+)"', html)
            non_repo = [u for u in external if "github.com/The-Schultz-Lab" not in u]
            assert not non_repo, f"{page.name} loads external resources: {non_repo}"


def test_the_docs_workflow_builds_what_the_docs_link_to():
    """The landing page links app/; the Pages workflow must build it there."""
    workflow = (Path(__file__).resolve().parent.parent / ".github/workflows/docs.yml").read_text(
        encoding="utf-8"
    )
    assert "build_site.py docs/_site/app" in workflow
    assert TestFullBuild.DEPLOY_ADDED == {"app/"}
