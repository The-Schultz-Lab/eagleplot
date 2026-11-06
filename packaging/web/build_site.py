"""Build the in-browser (stlite) EaglePlot site.

    python packaging/web/build_site.py OUT_DIR [--pyodide-dir DIR | --pyodide-url URL]

Produces a static site that runs the EaglePlot Streamlit app entirely in the browser
(Streamlit on Pyodide/WebAssembly via stlite). Serve it over HTTP, e.g. GitHub Pages or
``python -m http.server -d OUT_DIR``; ``file://`` cannot run it (module workers, fetch).

- The stlite runtime is downloaded from the npm registry and copied into the site, with
  source maps stripped (no third-party CDN for the app shell).
- The app (``eagleplot/`` with its assets, gallery and example data, plus
  ``.streamlit/config.toml``) is zipped into ``eagleplot-app.zip``.
- Pure-Python wheels that are not in the Pyodide distribution are bundled under
  ``wheels/``, so nothing is fetched from PyPI at runtime.
- Pyodide loads from the jsDelivr CDN unless ``--pyodide-dir`` points at an unpacked
  *full* Pyodide distribution of the version stlite expects, which is then served with
  the site (offline-capable, ~500 MB unpacked; browsers fetch only what they need), or
  ``--pyodide-url`` names a ``pyodide.mjs`` that is already served somewhere.

Pins that move together: @stlite/browser 1.12.0 <-> Streamlit 1.65 <-> Pyodide 314.0.3.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

STLITE_VERSION = "1.12.0"  # bundles Streamlit 1.65 and expects Pyodide 314.0.3
PYODIDE_VERSION = "314.0.3"
PYODIDE_CDN_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/pyodide.mjs"
PYODIDE_PYTHON = "3.14"  # CPython version inside that Pyodide (for `pip download`)

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
APP_PACKAGE = "eagleplot"
ENTRYPOINT = f"{APP_PACKAGE}/Home.py"

# Provided by the Pyodide distribution (compiled for WebAssembly). Covers the runtime
# dependencies in pyproject.toml that Pyodide ships: numpy, scipy, pandas, matplotlib,
# PyYAML and Pillow (Home.py's page icon). tomli is not needed: Pyodide is Python 3.14.
# Optional extras (h5py, xarray, netCDF4, statsmodels) are deliberately not preloaded.
PYODIDE_PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "pyyaml", "pillow"]

# Pure-Python packages bundled with the site as wheels: the remaining pyproject.toml
# dependencies (seaborn, plotly + narwhals, openpyxl + et-xmlfile), and two Streamlit
# dependencies that are not in the Pyodide distribution (python-multipart, itsdangerous).
# Also jcamp (13 KB) for JCAMP-DX (.jdx) uploads: the browser build can't fetch from NIST,
# so users download the .jdx from the NIST page and load it (Spectroscopy page).
# Streamlit itself comes with stlite.
PURE_WHEELS = [
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
    "narwhals",
    "openpyxl>=3.1.0",
    "et-xmlfile",
    "python-multipart>=0.0.10,<1",
    "itsdangerous>=2.1.2,<3",
    "jcamp>=1.2",
]

# Emscripten home directory in stlite; the app zip is unpacked here and it is the cwd.
PYODIDE_HOME = "/home/pyodide"
# Persistent per-browser storage (IndexedDB) for user settings. stlite mounts it before
# the app starts and syncs it after every script run. Nothing in eagleplot/ writes here
# yet: user_settings.py still writes <app root>/config.json (see README).
CONFIG_DIR = f"{PYODIDE_HOME}/.eagleplot"

# Environment variables for the Python worker. (Folder loading needs no variable: it is
# off under Pyodide, see eagleplot/utils/runtime.py.)
APP_ENV = {"EAGLEPLOT_CONFIG_DIR": CONFIG_DIR}

# Same theme as .streamlit/config.toml, passed as flags so it applies regardless of
# where Streamlit looks for that file under Pyodide.
STREAMLIT_CONFIG = {
    "theme.base": "dark",
    "theme.primaryColor": "#e0a3a3",
    "theme.backgroundColor": "#1b1b1b",
    "theme.secondaryBackgroundColor": "#240000",
    "theme.textColor": "#ffffff",
    "client.toolbarMode": "viewer",
    # As on the server (.streamlit/config.toml). In the browser the real limit is the
    # tab's memory (WebAssembly heap, ~2-4 GB), so this keeps uploads well inside it.
    "server.maxUploadSize": 200,
}


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https URLs
        return resp.read()


def fetch_stlite(dest: Path) -> None:
    """Copy the @stlite/browser build into *dest*, without source maps."""
    meta = json.loads(_read_url(f"https://registry.npmjs.org/@stlite/browser/{STLITE_VERSION}"))
    data = _read_url(meta["dist"]["tarball"])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if not member.isfile() or not name.startswith("package/build/"):
                continue
            if name.endswith(".map"):
                continue
            rel = Path(name).relative_to("package/build")
            if ".." in rel.parts or rel.is_absolute():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())


def build_app_zip(dest: Path) -> int:
    """Zip ``eagleplot/`` and ``.streamlit/config.toml``; return the number of files."""
    count = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((REPO / APP_PACKAGE).rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            zf.write(path, path.relative_to(REPO).as_posix())
            count += 1
        zf.write(REPO / ".streamlit" / "config.toml", ".streamlit/config.toml")
    return count + 1


def download_wheels(dest: Path) -> List[str]:
    """``pip download`` the pure-Python wheels; return their site-relative paths."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--quiet",
            "--disable-pip-version-check",
            "--no-deps",
            "--only-binary=:all:",
            "--python-version",
            PYODIDE_PYTHON,
            "--platform",
            "any",
            "--dest",
            str(dest),
            *PURE_WHEELS,
        ],
        check=True,
    )
    wheels = sorted(dest.glob("*.whl"))
    impure = [w.name for w in wheels if not w.name.endswith("-none-any.whl")]
    if impure:
        sys.exit(f"Expected pure-Python wheels only, got: {', '.join(impure)}")
    return [f"wheels/{w.name}" for w in wheels]


def site_size_mb(root: Path, exclude: Optional[Path] = None) -> float:
    total = 0
    for path in root.rglob("*"):
        if exclude is not None and exclude in path.parents:
            continue
        if path.is_file():
            total += path.stat().st_size
    return total / 1e6


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", type=Path, metavar="OUT_DIR", help="output directory (replaced)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--pyodide-dir",
        type=Path,
        help="unpacked full Pyodide distribution to copy into the site and self-host",
    )
    group.add_argument(
        "--pyodide-url",
        help="URL or site-relative path of a pyodide.mjs that is already served (no copy)",
    )
    args = ap.parse_args(argv)

    out: Path = args.out.resolve()
    if out == REPO or out in REPO.parents:
        sys.exit(f"Refusing to replace {out}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    fetch_stlite(out / "stlite")
    n_files = build_app_zip(out / "eagleplot-app.zip")
    wheels = download_wheels(out / "wheels")
    shutil.copy(HERE / "index.html", out / "index.html")
    # Streamlit pages have their own paths (…/settings). A reload or bookmark there asks
    # the static host for a file that doesn't exist; GitHub Pages then serves 404.html,
    # so make that the app too (it opens on Home).
    shutil.copy(HERE / "index.html", out / "404.html")
    # Favicon: the app icon (the eagle head on a maroon tile).
    shutil.copy(REPO / APP_PACKAGE / "assets" / "icon.png", out / "eagleplot-icon.png")

    pyodide_url: str = args.pyodide_url or PYODIDE_CDN_URL
    if args.pyodide_url and args.pyodide_url.endswith("pyodide.js"):
        sys.exit("--pyodide-url must name pyodide.mjs: stlite's worker is an ES module")
    if args.pyodide_dir:
        src = args.pyodide_dir.resolve()
        if not (src / "pyodide.mjs").exists() or not (src / "pyodide-lock.json").exists():
            sys.exit(f"{src} is not an unpacked Pyodide distribution (no pyodide.mjs)")
        version = json.loads((src / "pyodide-lock.json").read_text())["info"].get("version")
        if version and version != PYODIDE_VERSION:
            print(f"warning: {src} is Pyodide {version}, stlite expects {PYODIDE_VERSION}")
        shutil.copytree(src, out / "pyodide")
        # stlite runs Python in a module worker, which can only import the .mjs build.
        pyodide_url = "pyodide/pyodide.mjs"

    config: Dict[str, object] = {
        "entrypoint": ENTRYPOINT,
        "appArchive": "eagleplot-app.zip",
        "pyodidePackages": PYODIDE_PACKAGES,
        "wheels": wheels,
        "streamlitConfig": STREAMLIT_CONFIG,
        "pyodideUrl": pyodide_url,
        "idbfsMountpoints": [CONFIG_DIR],
        "env": APP_ENV,
    }
    (out / "site-config.js").write_text(
        "window.EAGLEPLOT_STLITE = " + json.dumps(config, indent=2) + ";\n", encoding="utf-8"
    )
    (out / ".nojekyll").write_text("")

    pyodide_dir = out / "pyodide"
    size = site_size_mb(out, exclude=pyodide_dir)
    where = "self-hosted" if args.pyodide_dir else pyodide_url
    msg = f"Built {out}: {size:.1f} MB without Pyodide, {n_files} app files, "
    msg += f"{len(wheels)} bundled wheels, Pyodide from {where}"
    if pyodide_dir.exists():
        msg += f" (+{site_size_mb(pyodide_dir):.0f} MB Pyodide)"
    print(msg)


if __name__ == "__main__":
    main()
