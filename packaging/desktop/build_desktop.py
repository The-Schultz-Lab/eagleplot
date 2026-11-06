"""Build the EaglePlot desktop app (roadmap D5): Electron + @stlite/desktop.

    python packaging/desktop/build_desktop.py [--pyodide-source DIR] [--dir] [-- BUILDER_ARGS]

Steps, all inside packaging/desktop/:

1. Stage the app: copy ``eagleplot/`` and ``.streamlit/config.toml`` from the repository
   (the files listed under ``stlite.desktop.files`` in package.json).
2. ``npm ci`` (Electron, electron-builder, @stlite/desktop at the versions in
   package-lock.json).
3. ``dump-stlite-desktop-artifacts``: writes ``build/`` with Pyodide, the installed Python
   packages and the app files, so the app needs no network. ``--pyodide-source`` uses a
   local Pyodide distribution instead of the jsDelivr CDN.
4. Patch the Electron main process so that https links open in the system browser
   (@stlite/desktop blocks every navigation and new window).
5. ``electron-builder`` for the current OS: an installer in ``dist/`` (``--dir``: an
   unpacked app only, faster, for testing). Arguments after ``--`` go to electron-builder.

The version is EaglePlot's own (``eagleplot/__init__.py``). Needs Node.js 22 and Python.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGED = ("eagleplot", ".streamlit")


def eagleplot_version() -> str:
    text = (REPO / "eagleplot" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if not match:
        sys.exit("eagleplot/__init__.py has no __version__")
    return match.group(1)


def stage_app() -> None:
    """Copy the app into packaging/desktop/ (git-ignored), as package.json expects."""
    for name in STAGED:
        target = HERE / name
        if target.exists():
            shutil.rmtree(target)
    shutil.copytree(
        REPO / "eagleplot",
        HERE / "eagleplot",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (HERE / ".streamlit").mkdir()
    shutil.copy(REPO / ".streamlit" / "config.toml", HERE / ".streamlit" / "config.toml")


# @stlite/desktop 0.105's Electron main process refuses every new window, so links in the
# app (Help, the NIST page, "install EaglePlot") do nothing. Open http(s) links in the
# system browser instead; everything else stays refused.
_WINDOW_OPEN = re.compile(
    r"(\w+)\.setWindowOpenHandler\(\(\{url:(\w+)\}\)=>\(console\.error\("
    r'"Opening a new window is not allowed\.",\2\),\{action:"deny"\}\)\)'
)
_ELECTRON_VAR = re.compile(r'(\w+)\.app\.on\("web-contents-created"')


def patch_external_links(main_js: Path) -> None:
    code = main_js.read_text(encoding="utf-8")
    electron = _ELECTRON_VAR.search(code)
    handler = _WINDOW_OPEN.search(code)
    if not electron or not handler:
        sys.exit(
            f"{main_js}: the window-open handler has changed (new @stlite/desktop?); "
            "update patch_external_links in build_desktop.py"
        )
    contents, url = handler.group(1), handler.group(2)
    replacement = (
        f"{contents}.setWindowOpenHandler(({{url:{url}}})=>"
        f"(/^https?:\\/\\//.test({url})&&{electron.group(1)}.shell.openExternal({url}),"
        f'{{action:"deny"}}))'
    )
    main_js.write_text(code[: handler.start()] + replacement + code[handler.end() :], "utf-8")


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True, shell=sys.platform == "win32")


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    builder_args: List[str] = []
    if "--" in argv:
        builder_args = argv[argv.index("--") + 1 :]
        argv = argv[: argv.index("--")]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pyodide-source", help="local Pyodide distribution (default: jsDelivr)")
    ap.add_argument("--dir", action="store_true", help="unpacked app only, no installer")
    ap.add_argument("--skip-install", action="store_true", help="reuse node_modules")
    args = ap.parse_args(argv)

    version = eagleplot_version()
    stage_app()
    if not args.skip_install:
        run(["npm", "ci", "--no-audit", "--no-fund"])
    dump = ["npx", "dump-stlite-desktop-artifacts"]
    if args.pyodide_source:
        dump += ["--pyodide-source", str(Path(args.pyodide_source).resolve()) + "/"]
    run(dump)
    patch_external_links(HERE / "build" / "electron" / "main.js")
    builder = ["npx", "electron-builder", f"-c.extraMetadata.version={version}"]
    if args.dir:
        builder.append("--dir")
    run(builder + builder_args)
    print(f"EaglePlot {version} desktop app in {HERE / 'dist'}")


if __name__ == "__main__":
    main()
