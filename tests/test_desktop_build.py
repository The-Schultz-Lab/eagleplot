"""Desktop app build (roadmap D5): packaging/desktop."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DESKTOP = REPO / "packaging" / "desktop"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_json():
    return json.loads((DESKTOP / "package.json").read_text(encoding="utf-8"))


# The relevant part of @stlite/desktop 0.105.1's (minified) Electron main process.
STLITE_MAIN = (
    'f.app.on("web-contents-created",(i,s)=>{s.on("will-attach-webview",c=>{c.preventDefault()}),'
    's.on("will-navigate",(c,u)=>{console.debug("will-navigate",u),c.preventDefault()}),'
    's.setWindowOpenHandler(({url:c})=>(console.error("Opening a new window is not allowed.",c),'
    '{action:"deny"}))});'
)


def test_links_are_patched_to_open_in_the_system_browser(tmp_path):
    build = _load(DESKTOP / "build_desktop.py", "build_desktop")
    main_js = tmp_path / "main.js"
    main_js.write_text(STLITE_MAIN, encoding="utf-8")
    build.patch_external_links(main_js)
    code = main_js.read_text(encoding="utf-8")
    assert "f.shell.openExternal(c)" in code
    assert "/^https?:\\/\\//.test(c)" in code  # only web links, nothing else
    assert '{action:"deny"}' in code  # still no new Electron window
    assert "will-navigate" in code  # in-app navigation stays blocked


def test_the_build_stops_if_stlite_changes_the_handler(tmp_path):
    import pytest

    build = _load(DESKTOP / "build_desktop.py", "build_desktop")
    main_js = tmp_path / "main.js"
    main_js.write_text('f.app.on("web-contents-created",()=>{});', encoding="utf-8")
    with pytest.raises(SystemExit, match="window-open handler has changed"):
        build.patch_external_links(main_js)


def test_desktop_bundles_everything_the_browser_build_loads():
    web = _load(REPO / "packaging" / "web" / "build_site.py", "build_site")
    from eagleplot.utils import runtime

    desktop = set(_package_json()["stlite"]["desktop"]["dependencies"])
    wheels = {w.split(">")[0].split("<")[0].split("=")[0] for w in web.PURE_WHEELS}
    needed = set(web.PYODIDE_PACKAGES) | set(runtime.DEFERRED_PACKAGES) | wheels
    # Streamlit's own dependencies come with @stlite/desktop; narwhals and et-xmlfile are
    # pulled in by plotly and openpyxl.
    needed -= {"python-multipart", "itsdangerous", "narwhals", "et-xmlfile"}
    assert needed <= desktop, sorted(needed - desktop)


def test_desktop_versions_match_the_browser_build():
    pkg = _package_json()
    lock = json.loads((DESKTOP / "package-lock.json").read_text(encoding="utf-8"))
    for name in ("@stlite/desktop", "electron", "electron-builder"):
        assert lock["packages"][f"node_modules/{name}"]["version"] == pkg["devDependencies"][name]
    # @stlite/desktop 0.105.x and @stlite/browser 1.12.x both run Pyodide 314.0.3.
    assert lock["packages"]["node_modules/@stlite/desktop"]["dependencies"]["pyodide"] == "314.0.3"
    web = _load(REPO / "packaging" / "web" / "build_site.py", "build_site")
    assert web.PYODIDE_VERSION == "314.0.3"


def test_settings_folder_is_mounted_where_runtime_expects_it():
    from eagleplot.utils import runtime

    desktop = _package_json()["stlite"]["desktop"]
    assert desktop["nodeJsWorker"] is True
    assert desktop["nodefsMountpoints"][runtime.DESKTOP_CONFIG_DIR] == "{{userData}}"
    assert desktop["entrypoint"] == "eagleplot/Home.py"
