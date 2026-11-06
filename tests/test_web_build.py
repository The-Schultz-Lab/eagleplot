"""Browser build start-up (roadmap D4.1): scipy is loaded after the app is up, not before."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from eagleplot.utils import runtime

REPO = Path(__file__).resolve().parent.parent


def _build_site():
    spec = importlib.util.spec_from_file_location(
        "build_site", REPO / "packaging" / "web" / "build_site.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deferred_packages_are_not_loaded_at_start_up():
    preloaded = set(_build_site().PYODIDE_PACKAGES)
    assert "scipy" in runtime.DEFERRED_PACKAGES
    assert not preloaded & set(runtime.DEFERRED_PACKAGES)
    # Home.py must not need them: it is what runs while they load.
    assert {"numpy", "pandas", "matplotlib"} <= preloaded


class _Promise:
    """Stands in for the JS promise loadPackage returns; settle() runs its callbacks."""

    def __init__(self):
        self.callbacks = []

    def then(self, ok, err):
        self.callbacks.append(ok)

    def settle(self):
        for cb in self.callbacks:
            cb(None)


class _Calls(list):
    promise: "_Promise"


@pytest.fixture
def fake_pyodide(monkeypatch):
    calls = _Calls()
    promise = _Promise()

    def load_package(packages):
        calls.append(packages)
        return promise

    monkeypatch.setitem(sys.modules, "pyodide_js", types.SimpleNamespace(loadPackage=load_package))
    monkeypatch.setitem(
        sys.modules, "pyodide.ffi", types.SimpleNamespace(create_once_callable=lambda f: f)
    )
    monkeypatch.setitem(sys.modules, "pyodide", types.SimpleNamespace())
    monkeypatch.setattr(runtime, "IS_PYODIDE", True)
    monkeypatch.setattr(runtime, "_deferred_load_started", False)
    monkeypatch.setattr(runtime, "_deferred_load_settled", False)
    calls.promise = promise
    return calls


def test_start_deferred_loads_does_nothing_on_a_server(monkeypatch):
    monkeypatch.setattr(runtime, "_deferred_load_started", False)
    assert runtime.IS_PYODIDE is False
    assert runtime.start_deferred_loads() is False


def test_start_deferred_loads_starts_once_in_the_browser(fake_pyodide):
    assert runtime.start_deferred_loads() is True
    assert runtime.start_deferred_loads() is False
    assert fake_pyodide == [list(runtime.DEFERRED_PACKAGES)]


def test_missing_deferred_packages():
    assert runtime.missing_deferred_packages() == []  # scipy is installed here
    assert runtime.missing_deferred_packages(("no_such_package_xyz",)) == ["no_such_package_xyz"]


def test_a_deferred_package_counts_as_missing_until_its_load_settles(fake_pyodide):
    # scipy's files can be found before its compiled modules are loaded; only the
    # settled promise means it can be imported.
    runtime.start_deferred_loads()
    assert runtime.missing_deferred_packages() == ["scipy"]
    fake_pyodide.promise.settle()
    assert runtime.missing_deferred_packages() == []


def _home_while_scipy_loads(monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setattr(runtime, "missing_deferred_packages", lambda packages=(): ["scipy"])
    warm = []
    monkeypatch.setattr(runtime, "warm_imports_in_background", lambda *a: warm.append(1))
    at = AppTest.from_file(str(REPO / "eagleplot" / "Home.py"), default_timeout=60).run()
    assert not at.exception
    assert warm  # Home schedules the background imports
    assert not any("Loading the science libraries" in i.value for i in at.info)
    return at


def test_a_page_opened_before_scipy_arrives_waits_instead_of_failing(fake_pyodide, monkeypatch):
    at = _home_while_scipy_loads(monkeypatch)
    assert fake_pyodide == [list(runtime.DEFERRED_PACKAGES)]  # Home started the load
    # Data Upload imports scipy (via eagleplot.math): it waits instead of failing.
    next(b for b in at.button if b.key == "home_upload").click().run()
    assert not at.exception
    assert any("Loading the science libraries" in i.value for i in at.info)
    assert not any(t.value == "Data Upload" for t in at.title)


def test_quick_plot_does_not_wait_for_scipy(fake_pyodide, monkeypatch):
    """Home's one-click example goes straight to a plot: Quick Plot doesn't import scipy."""
    at = _home_while_scipy_loads(monkeypatch)
    next(b for b in at.button if b.key == "home_try_sine_cosine_waves.csv").click().run()
    assert not at.exception
    assert not any("Loading the science libraries" in i.value for i in at.info)
    assert any(t.value == "Quick Plot" for t in at.title)


def test_plotting_does_not_import_seaborn_plotly_or_scipy():
    """They cost seconds to import in the browser; a Matplotlib plot needs none of them."""
    import subprocess

    code = (
        "import sys, eagleplot.plotting as p; "
        "print(sorted(m for m in ('seaborn', 'plotly', 'scipy') if m in sys.modules), "
        "p.HAS_SEABORN, p.HAS_PLOTLY)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    ).stdout.strip()
    assert out == "[] True True"


def test_home_does_not_import_pandas():
    """pandas takes ~2.3 s to import in the browser; Home runs before anything else.

    (plotly is not checked: Streamlit itself imports it when it is installed.)
    """
    import subprocess

    code = (
        "import sys; from eagleplot.utils import runtime, initialize_session_state, "
        "get_session_summary; from eagleplot.utils.theming import app_css; "
        "print(sorted(m for m in ('pandas', 'scipy', 'seaborn') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    ).stdout.strip()
    assert out.splitlines()[-1] == "[]"
