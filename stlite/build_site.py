"""Build the in-browser (stlite) Plottle site.

    python stlite/build_site.py OUT_DIR [--pyodide-dir DIR]

Produces a static site that runs the Plottle Streamlit app entirely in the browser
(Streamlit on Pyodide/WebAssembly). Serve it over HTTP, e.g. GitHub Pages or
``python -m http.server -d OUT_DIR``; ``file://`` cannot run it.

- The stlite runtime is copied into the site (no jsDelivr dependency for it).
- Pyodide loads from the jsDelivr CDN unless ``--pyodide-dir`` points at an
  unpacked *full* Pyodide distribution of the version stlite expects, which is
  then served with the site (offline-capable, but several hundred MB).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

STLITE_VERSION = "1.12.0"  # bundles Streamlit 1.65 and expects Pyodide 314.0.3
REPO = Path(__file__).resolve().parent.parent

# Provided by the Pyodide distribution (compiled for WebAssembly).
PYODIDE_PACKAGES = ["numpy", "scipy", "pandas", "matplotlib", "pyyaml", "pillow"]
# Pure-Python packages bundled with the site as wheels. The last two are Streamlit
# dependencies that are not in the Pyodide distribution; bundling them means nothing
# is fetched from PyPI at runtime.
PURE_WHEELS = [
    "seaborn>=0.12",
    "plotly>=5.14",
    "narwhals",
    "openpyxl",
    "et-xmlfile",
    "python-multipart>=0.0.10,<1",
    "itsdangerous>=2.1.2,<3",
]

STREAMLIT_CONFIG = {
    "theme.base": "dark",
    "theme.primaryColor": "#e0a3a3",
    "theme.backgroundColor": "#1b1b1b",
    "theme.secondaryBackgroundColor": "#240000",
    "theme.textColor": "#ffffff",
    "client.toolbarMode": "viewer",
}


def fetch_stlite(dest: Path) -> None:
    meta = json.load(
        urllib.request.urlopen(f"https://registry.npmjs.org/@stlite/browser/{STLITE_VERSION}")
    )
    data = urllib.request.urlopen(meta["dist"]["tarball"]).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.startswith("package/build/") or m.name.endswith(".map"):
                continue
            rel = Path(m.name).relative_to("package/build")
            if ".." in rel.parts:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())


def build_app_zip(dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((REPO / "plottle").rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            zf.write(path, path.relative_to(REPO).as_posix())
        zf.write(REPO / ".streamlit" / "config.toml", ".streamlit/config.toml")


def download_wheels(dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--quiet",
            "--no-deps",
            "--only-binary=:all:",
            "--python-version",
            "3.14",
            "--platform",
            "any",
            "--dest",
            str(dest),
            *PURE_WHEELS,
        ],
        check=True,
    )
    return sorted(f"wheels/{p.name}" for p in dest.glob("*.whl"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--pyodide-dir", type=Path, help="unpacked full Pyodide distribution to self-host"
    )
    ap.add_argument(
        "--pyodide-url",
        help="pyodide.mjs already served at this URL or site-relative path (no copy)",
    )
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    fetch_stlite(out / "stlite")
    build_app_zip(out / "plottle-app.zip")
    wheels = download_wheels(out / "wheels")
    shutil.copy(REPO / "stlite" / "index.html", out / "index.html")
    shutil.copy(REPO / "plottle" / "assets" / "logo.png", out / "plottle-icon.png")

    pyodide_url = args.pyodide_url
    if args.pyodide_dir:
        src = args.pyodide_dir.resolve()
        if not (src / "pyodide.js").exists():
            sys.exit(f"{src} is not an unpacked Pyodide distribution (no pyodide.js)")
        shutil.copytree(src, out / "pyodide")
        # stlite runs Python in a module worker, which can only import the .mjs build.
        pyodide_url = "pyodide/pyodide.mjs"

    config = {
        "pyodidePackages": PYODIDE_PACKAGES,
        "wheels": wheels,
        "streamlitConfig": STREAMLIT_CONFIG,
        "pyodideUrl": pyodide_url,
    }
    (out / "site-config.js").write_text(
        "window.PLOTTLE_STLITE = " + json.dumps(config, indent=2) + ";\n", encoding="utf-8"
    )
    (out / ".nojekyll").write_text("")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(
        f"Built {out} ({size:.0f} MB): {len(wheels)} bundled wheels, "
        f"Pyodide {'self-hosted' if pyodide_url else 'from CDN'}"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory():
        main()
