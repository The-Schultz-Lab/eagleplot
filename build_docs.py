#!/usr/bin/env python3
"""Build the static documentation site into ``docs/_site/``.

EaglePlot's docs used to contain two disconnected systems: a full Sphinx setup that
CI never built, and a hand-written ``docs/index.html`` that linked to none of the
markdown guides. The published site was therefore a single marketing page, with
seven markdown files and an entire API reference deployed but unreachable except
by guessing URLs — and ``.md`` served raw downloads rather than renders
(audit A-13).

Sphinx was dropped in favour of this script, which renders each markdown guide
into a page styled to match ``index.html`` and wires them together with a shared
nav. Run it locally to preview, or let ``.github/workflows/docs.yml`` run it.

Usage
-----
    pip install markdown
    python build_docs.py
    # then open docs/_site/index.html

Output
------
``docs/_site/`` — ``index.html`` (copied verbatim), one HTML page per guide, and
``docs.css``. The directory is gitignored and is what the Pages workflow uploads.
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent
_DOCS = _ROOT / "docs"
_SITE = _DOCS / "_site"

REPO_URL = "https://github.com/The-Schultz-Lab/eagleplot"


@dataclass(frozen=True)
class Page:
    """One markdown guide and where it lands in the built site."""

    source: Path
    slug: str
    title: str
    blurb: str

    @property
    def output_name(self) -> str:
        return f"{self.slug}.html"


# Ordered — this is also the nav order and the order of the cards on the landing
# page. `docs/archive/` is deliberately excluded: it is superseded internal notes,
# not user documentation.
PAGES: List[Page] = [
    Page(
        _DOCS / "getting_started.md",
        "getting-started",
        "Getting Started",
        "Install EaglePlot, load your first dataset, and make your first plot.",
    ),
    Page(
        _DOCS / "download.md",
        "download",
        "Download",
        "Web app, desktop app or Python package; code signing and privacy policy.",
    ),
    Page(
        _DOCS / "tutorials" / "gui_guide.md",
        "gui-guide",
        "GUI Guide",
        "A tour of all 14 pages of the Streamlit interface.",
    ),
    Page(
        _DOCS / "tutorials" / "cli_guide.md",
        "cli-guide",
        "CLI Guide",
        "The five `eagleplot` subcommands, with worked examples.",
    ),
    Page(
        _DOCS / "cheatsheet.md",
        "cheatsheet",
        "Cheatsheet",
        "Quick reference for the Python API — every module, at a glance.",
    ),
    Page(
        _DOCS / "bug-reports.md",
        "bug-reports",
        "Reporting Bugs",
        "What to include so a problem can be reproduced and fixed.",
    ),
    Page(
        _DOCS / "feature_requests.md",
        "feature-requests",
        "Feature Requests",
        "How to propose new functionality.",
    ),
]

_CSS = """\
/* Documentation pages — palette matched to index.html. */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; scroll-behavior: smooth; }
body {
  font-family: Inter, 'Segoe UI', -apple-system, BlinkMacSystemFont,
               'Helvetica Neue', Arial, sans-serif;
  background: #ffffff; color: #334155; line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}
img, svg { display: block; max-width: 100%; }

.nav {
  position: sticky; top: 0; z-index: 100;
  background: rgba(15, 23, 42, 0.92);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.nav__inner {
  max-width: 1160px; margin: 0 auto; padding: 0.85rem 1.5rem;
  display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap;
}
.nav__logo {
  font-weight: 700; font-size: 1.0625rem; color: #f8fafc;
  text-decoration: none; letter-spacing: -0.01em;
}
.nav__link {
  color: #94a3b8; text-decoration: none; font-size: 0.9375rem;
  transition: color 0.15s ease;
}
.nav__link:hover, .nav__link[aria-current="page"] { color: #f8fafc; }
.nav__link[aria-current="page"] { font-weight: 600; }
.nav__spacer { margin-left: auto; }
.btn--ghost {
  color: #e2e8f0; text-decoration: none; font-size: 0.875rem;
  padding: 0.4rem 0.85rem; border: 1px solid rgba(203,213,225,0.22);
  border-radius: 7px; transition: background 0.15s ease;
}
.btn--ghost:hover { background: rgba(255,255,255,0.07); color: #f8fafc; }

.wrap { max-width: 820px; margin: 0 auto; padding: 3rem 1.5rem 5rem; }
.crumb { font-size: 0.875rem; color: #64748b; margin-bottom: 1.5rem; }
.crumb a { color: #8B1A2B; text-decoration: none; }
.crumb a:hover { text-decoration: underline; }

.wrap h1 {
  font-size: clamp(1.9rem, 4vw, 2.4rem); font-weight: 700; color: #0f172a;
  letter-spacing: -0.02em; line-height: 1.2; margin-bottom: 1.25rem;
}
.wrap h2 {
  font-size: 1.5rem; font-weight: 700; color: #0f172a;
  margin: 2.75rem 0 0.85rem; padding-bottom: 0.4rem;
  border-bottom: 1px solid #e2e8f0;
}
.wrap h3 { font-size: 1.15rem; font-weight: 650; color: #0f172a; margin: 2rem 0 0.6rem; }
.wrap h4 { font-size: 1rem; font-weight: 650; color: #475569; margin: 1.5rem 0 0.5rem; }
.wrap p, .wrap li { font-size: 1rem; }
.wrap p { margin: 0.85rem 0; }
.wrap ul, .wrap ol { margin: 0.85rem 0 0.85rem 1.5rem; }
.wrap li { margin: 0.35rem 0; }
.wrap a { color: #8B1A2B; text-decoration: none; }
.wrap a:hover { text-decoration: underline; }
.wrap strong { color: #0f172a; font-weight: 650; }
.wrap hr { border: 0; border-top: 1px solid #e2e8f0; margin: 2.5rem 0; }
.wrap blockquote {
  border-left: 3px solid #D94054; padding: 0.15rem 0 0.15rem 1.1rem;
  margin: 1.25rem 0; color: #475569;
}

.wrap code {
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.875em; background: #f1f5f9; color: #8B1A2B;
  padding: 0.15em 0.4em; border-radius: 4px;
}
.wrap pre {
  background: #0f172a; border-radius: 10px; padding: 1.15rem 1.35rem;
  overflow-x: auto; margin: 1.25rem 0;
}
.wrap pre code {
  background: none; color: #e2e8f0; padding: 0; font-size: 0.875rem; line-height: 1.65;
}

/* Wide tables scroll inside their own container so the page never does. */
.table-scroll { overflow-x: auto; margin: 1.25rem 0; }
.wrap table { border-collapse: collapse; width: 100%; font-size: 0.9375rem; }
.wrap th, .wrap td {
  text-align: left; padding: 0.6rem 0.85rem; border-bottom: 1px solid #e2e8f0;
  vertical-align: top;
}
.wrap th { color: #0f172a; font-weight: 650; background: #f8fafc; white-space: nowrap; }

.footer { border-top: 1px solid #e2e8f0; background: #f8fafc; }
.footer__inner {
  max-width: 1160px; margin: 0 auto; padding: 2rem 1.5rem;
  display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  font-size: 0.875rem; color: #64748b;
}
.footer a { color: #475569; text-decoration: none; }
.footer a:hover { color: #0f172a; }

@media (max-width: 640px) {
  .wrap { padding: 2rem 1.15rem 3.5rem; }
  .nav__inner { gap: 0.85rem; }
}
"""


def _nav_html(current_slug: str = "") -> str:
    # Built without backslashes inside f-string expressions: that is a syntax error
    # before Python 3.12, and this script must run on the same floor as the package.
    links = []
    for page in PAGES[:4]:
        current = ' aria-current="page"' if page.slug == current_slug else ""
        links.append(
            '<a class="nav__link"' + current + f' href="{page.output_name}">{page.title}</a>'
        )
    return f"""<nav class="nav" role="navigation">
  <div class="nav__inner">
    <a class="nav__logo" href="index.html">EaglePlot</a>
    {"".join(links)}
    <span class="nav__spacer"></span>
    <a class="btn--ghost" href="{REPO_URL}">GitHub &rarr;</a>
  </div>
</nav>"""


def _footer_html() -> str:
    return f"""<footer class="footer">
  <div class="footer__inner">
    <div>EaglePlot &middot; NCCU Department of Chemistry and Biochemistry</div>
    <div>
      <a href="{REPO_URL}">GitHub</a> &nbsp;&middot;&nbsp;
      <a href="{REPO_URL}/blob/main/LICENSE">MIT License</a> &nbsp;&middot;&nbsp;
      <a href="{REPO_URL}/issues">Issues</a>
    </div>
  </div>
</footer>"""


def _wrap_tables(html: str) -> str:
    """Put each table in a horizontally scrollable container.

    Wide reference tables would otherwise force the whole page to scroll
    sideways on a narrow screen.
    """
    return re.sub(
        r"(<table>.*?</table>)",
        r'<div class="table-scroll">\1</div>',
        html,
        flags=re.DOTALL,
    )


def _rewrite_internal_links(html: str) -> str:
    """Point cross-references at built pages instead of source markdown.

    The guides link to each other with paths like ``tutorials/cli_guide.md``,
    which would 404 in the built site.
    """
    by_source_name = {p.source.name: p.output_name for p in PAGES}
    for source_name, output_name in by_source_name.items():
        html = re.sub(
            rf'href="(?:\.\./)*(?:tutorials/)?{re.escape(source_name)}"',
            f'href="{output_name}"',
            html,
        )
    return html


def _render(page: Page, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{page.title} — EaglePlot</title>
<meta name="description" content="{page.blurb}">
<meta name="theme-color" content="#1a0c0e">
<link rel="stylesheet" href="docs.css">
</head>
<body>
{_nav_html(page.slug)}
<main class="wrap">
<p class="crumb"><a href="index.html">EaglePlot</a> &rsaquo; {page.title}</p>
{body_html}
</main>
{_footer_html()}
</body>
</html>
"""


def main() -> int:
    try:
        import markdown
    except ImportError:
        print(
            "error: the `markdown` package is required.\n"
            '       pip install markdown   (or: pip install -e ".[docs]")',
            file=sys.stderr,
        )
        return 1

    missing = [p.source for p in PAGES if not p.source.is_file()]
    if missing:
        for path in missing:
            print(f"error: missing source file {path}", file=sys.stderr)
        return 1

    if _SITE.exists():
        shutil.rmtree(_SITE)
    _SITE.mkdir(parents=True)

    (_SITE / "docs.css").write_text(_CSS, encoding="utf-8")

    # The landing page is hand-written and copied verbatim.
    shutil.copy2(_DOCS / "index.html", _SITE / "index.html")
    print("  index.html          <- docs/index.html")

    converter = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc"],
        output_format="html5",
    )
    for page in PAGES:
        converter.reset()
        body = converter.convert(page.source.read_text(encoding="utf-8"))
        body = _wrap_tables(_rewrite_internal_links(body))
        (_SITE / page.output_name).write_text(_render(page, body), encoding="utf-8")
        print(f"  {page.output_name:<20}<- {page.source.relative_to(_ROOT)}")

    print(f"\nBuilt {len(PAGES) + 1} pages into {_SITE.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
