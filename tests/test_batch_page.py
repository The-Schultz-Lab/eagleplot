"""Batch Analysis page (roadmap D1.2): results persist, and the CSV is not rebuilt every rerun."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

PAGE = Path(__file__).resolve().parent.parent / "eagleplot" / "pages" / "11_Batch_Analysis.py"


@pytest.fixture
def to_csv_calls(monkeypatch):
    calls = []
    original = pd.DataFrame.to_csv

    def recording(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", recording)
    return calls


@pytest.fixture
def stats_run():
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    x = np.linspace(0, 1, 20)
    at.session_state["datasets"] = {
        "a.csv": pd.DataFrame({"x": x, "y": 2 * x}),
        "b.csv": pd.DataFrame({"x": x, "y": 3 * x + 1}),
    }
    at.run()
    next(b for b in at.button if b.key == "ba_stats_run").click().run()
    assert not at.exception
    return at


def _download_keys(at):
    return {b.proto.id.rsplit("-", 1)[-1] for b in at.get("download_button")}


def test_results_stay_after_a_later_interaction(stats_run):
    at = stats_run
    assert "ba_stats_result_dl" in _download_keys(at)
    assert len(at.dataframe) >= 1
    # Any later interaction (here: changing another tool's setting). Before D1.2 the
    # table and its download existed only on the rerun right after "Run Statistics".
    at.text_input(key="ba_fit_x").set_value("x").run()
    at.text_input(key="ba_fit_x").set_value("xx").run()
    assert not at.exception
    assert "ba_stats_result_dl" in _download_keys(at)
    assert len(at.dataframe) >= 1


def test_csv_is_not_rebuilt_on_every_rerun(stats_run, to_csv_calls):
    stats_run.run()  # first rerun may fill the cache on Streamlit < 1.52
    to_csv_calls.clear()
    for _ in range(3):
        stats_run.run()
    assert to_csv_calls == []
