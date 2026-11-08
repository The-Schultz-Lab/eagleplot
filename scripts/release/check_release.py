#!/usr/bin/env python3
"""Check that a release is consistent before it is published (roadmap D6).

    python scripts/release/check_release.py [--tag vX.Y.Z] [--dist DIR] [--notes FILE]

Checks, against the version in ``eagleplot/__init__.py``:

- ``--tag``: the release tag is ``v<version>``;
- ``CHANGELOG.md`` has a dated ``## [<version>] - YYYY-MM-DD`` section and a compare link;
- the documentation site's version badge (``docs/index.html``) shows the version;
- ``--dist``: the built wheel and sdist (``python -m build``) are named for the version,
  the wheel installs only the ``eagleplot`` package, ships the GUI's data files, and has
  no Windows line endings; the sdist has no desktop-build output.

``--notes FILE`` writes the version's CHANGELOG section to FILE (the GitHub Release
notes). Exits 1 and lists every problem found. ``.github/workflows/release.yml`` runs it
on each published release; run it locally before opening a release PR.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]

#: Files a pip-installed GUI needs (G-012: they once went missing from the wheel).
WHEEL_DATA_FILES = (
    "eagleplot/Home.py",
    "eagleplot/pages/2_Quick_Plot.py",
    "eagleplot/assets/logo.png",
    "eagleplot/assets/logo.ico",
    "eagleplot/assets/nccu-horiz-logo.png",
    "eagleplot/example-data/Artificial/sine_cosine_waves.csv",
)
#: Never in the sdist: the desktop build's staging and output (packaging/desktop).
SDIST_FORBIDDEN = ("node_modules/", "packaging/desktop/build/", "packaging/desktop/dist/")


def package_version(root: Path = ROOT) -> str:
    """``__version__`` from ``eagleplot/__init__.py``, read without importing it."""
    text = (root / "eagleplot" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit("eagleplot/__init__.py has no __version__")
    return match.group(1)


def changelog_section(changelog: str, version: str) -> Optional[str]:
    """The body of ``## [version] - date`` in *changelog*, or ``None``."""
    heading = re.search(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}[ \t]*$", changelog, re.M
    )
    if not heading:
        return None
    rest = changelog[heading.end() :]
    end = re.search(r"^## \[|^\[[^\]]+\]: http", rest, re.M)
    body = rest[: end.start()] if end else rest
    return body.strip().removesuffix("---").strip()


def check_sources(version: str, tag: Optional[str], root: Path = ROOT) -> List[str]:
    """Problems with the tag, CHANGELOG and docs badge for *version*."""
    problems = []
    if tag is not None and tag != f"v{version}":
        problems.append(
            f"release tag {tag!r} doesn't match the package version: expected 'v{version}' "
            "(bump eagleplot/__init__.py, or tag the right commit)"
        )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if changelog_section(changelog, version) is None:
        problems.append(
            f"CHANGELOG.md has no dated section '## [{version}] - YYYY-MM-DD' "
            "(rename [Unreleased] when preparing the release)"
        )
    if not re.search(rf"^\[{re.escape(version)}\]: https://", changelog, re.M):
        problems.append(f"CHANGELOG.md has no compare link '[{version}]: https://…' at the end")
    index = (root / "docs" / "index.html").read_text(encoding="utf-8")
    badge = re.search(r'class="nav__badge">v([^ <&]+)', index)
    if not badge or badge.group(1) != version:
        found = badge.group(1) if badge else "none"
        problems.append(f"docs/index.html version badge shows {found}, expected {version} (G-035)")
    return problems


def check_dist(dist: Path, version: str) -> List[str]:
    """Problems with the wheel and sdist in *dist*."""
    problems: List[str] = []
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    want_wheel = f"eagleplot-{version}-py3-none-any.whl"
    want_sdist = f"eagleplot-{version}.tar.gz"
    if [w.name for w in wheels] != [want_wheel]:
        problems.append(f"{dist}: expected exactly {want_wheel}, found {[w.name for w in wheels]}")
    if [s.name for s in sdists] != [want_sdist]:
        problems.append(f"{dist}: expected exactly {want_sdist}, found {[s.name for s in sdists]}")

    if (dist / want_wheel).is_file():
        with zipfile.ZipFile(dist / want_wheel) as whl:
            names = whl.namelist()
            top = [n for n in names if n.endswith(".dist-info/top_level.txt")]
            top_level = whl.read(top[0]).decode().split() if top else []
            if top_level != ["eagleplot"]:
                problems.append(
                    f"wheel top_level.txt is {top_level}, expected ['eagleplot'] (G-011)"
                )
            outside = sorted(
                {n.split("/")[0] for n in names} - {"eagleplot", f"eagleplot-{version}.dist-info"}
            )
            if outside:
                problems.append(f"wheel installs files outside eagleplot/: {outside}")
            missing = [f for f in WHEEL_DATA_FILES if f not in names]
            if missing:
                problems.append(f"wheel is missing GUI data files (G-012): {missing}")
            crlf = [n for n in names if n.endswith(".py") and b"\r\n" in whl.read(n)]
            if crlf:
                problems.append(
                    f"{len(crlf)} .py files in the wheel have Windows line endings, e.g. "
                    f"{crlf[0]} (G-032: build from a clean checkout on Linux or CI)"
                )

    if (dist / want_sdist).is_file():
        with tarfile.open(dist / want_sdist) as sdist:
            leaked = [n for n in sdist.getnames() if any(f in n for f in SDIST_FORBIDDEN)]
        if leaked:
            problems.append(f"sdist contains desktop build output, e.g. {leaked[0]}")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", help="the release tag, e.g. v3.2.0")
    parser.add_argument("--dist", type=Path, help="folder with the built wheel and sdist")
    parser.add_argument("--notes", type=Path, help="write the version's CHANGELOG section here")
    args = parser.parse_args(argv)

    version = package_version()
    problems = check_sources(version, args.tag)
    if args.dist is not None:
        problems += check_dist(args.dist, version)
    if args.notes is not None:
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        args.notes.parent.mkdir(parents=True, exist_ok=True)
        args.notes.write_text(
            (changelog_section(changelog, version) or "") + "\n", encoding="utf-8"
        )

    if problems:
        print(f"EaglePlot {version}: {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"EaglePlot {version}: release checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
