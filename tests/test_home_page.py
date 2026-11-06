"""Home page: a first screen that gets a new user to a plot in one click."""

from pathlib import Path

import pandas as pd
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

HOME = Path(__file__).resolve().parent.parent / "eagleplot" / "Home.py"


def _home():
    at = AppTest.from_file(str(HOME), default_timeout=60).run()
    assert not at.exception
    return at


def test_first_screen_offers_examples_instead_of_empty_counters():
    at = _home()
    keys = {b.key for b in at.button}
    assert {"home_try_sine_cosine_waves.csv", "home_upload", "home_examples"} <= keys
    assert not at.metric  # no "Datasets 0 · Plots 0 · Analyses 0"


def test_plot_this_example_loads_it_and_opens_quick_plot():
    at = _home()
    next(b for b in at.button if b.key == "home_try_sine_cosine_waves.csv").click().run()
    assert not at.exception
    assert isinstance(at.session_state["datasets"]["sine_cosine_waves.csv"], pd.DataFrame)
    assert at.session_state["current_dataset"] == "sine_cosine_waves.csv"
    assert any(t.value == "Quick Plot" for t in at.title)


def test_with_data_home_shows_the_session_and_a_way_on():
    at = AppTest.from_file(str(HOME), default_timeout=60)
    at.session_state["datasets"] = {"d.csv": pd.DataFrame({"x": [1, 2]})}
    at.session_state["current_dataset"] = "d.csv"
    at.run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Datasets", "Plots", "Analyses"]
    assert not any(b.key == "home_upload" for b in at.button)


def test_menu_uses_the_page_names():
    source = HOME.read_text(encoding="utf-8")
    for label in ("Quick Plot", "Dashboard", "Analysis Tools", "Batch Analysis"):
        assert f'label="{label}"' in source
    for old in ('label="Basic"', 'label="Multiplot"', 'label="Single"', 'label="Batch"'):
        assert old not in source
