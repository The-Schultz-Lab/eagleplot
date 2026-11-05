"""Plottle -> EaglePlot (3.0) compatibility: what existing users already have must keep working."""

import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eagleplot.utils.session_state as _ss
import eagleplot.utils.user_settings as _us
from eagleplot.utils.access import (
    FOLDER_LOAD_ENV,
    LEGACY_FOLDER_LOAD_ENV,
    folder_load_setting,
)
from eagleplot.utils.session_state import (
    get_dataset,
    initialize_session_state,
    load_session_from_file,
)

DATA = Path(__file__).resolve().parent / "data"
REPO = Path(__file__).resolve().parent.parent


class _State:
    def __contains__(self, key):
        return key in self.__dict__


@pytest.fixture
def session(monkeypatch):
    state = _State()
    monkeypatch.setattr(_ss, "st", types.SimpleNamespace(session_state=state))
    initialize_session_state()
    return state


class TestPlottleSessionFiles:
    """tests/data/plottle_2_1_0_session.json was written by the real Plottle 2.1.0 code
    (git tag v2.1.0, save_session_to_file), not constructed by hand."""

    def test_loads_without_skipping_anything(self, session):
        skipped = load_session_from_file(str(DATA / "plottle_2_1_0_session.json"))
        assert skipped == []

    def test_dataframe_and_array_round_trip(self, session):
        load_session_from_file(str(DATA / "plottle_2_1_0_session.json"))
        df = get_dataset("spectrum.csv")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["wavenumber", "transmittance", "label"]
        # pandas JSON round trips can differ in the last bit (0.8200000000000001).
        assert df["transmittance"].tolist() == pytest.approx([0.85, 0.82, 0.31])
        arr = get_dataset("signal.npy")
        np.testing.assert_array_equal(arr, np.array([[1.5, 2.0], [3.25, -4.0]]))

    def test_history_and_results_restored(self, session):
        load_session_from_file(str(DATA / "plottle_2_1_0_session.json"))
        assert session.plot_history[0]["type"] == "line_plot"
        assert session.analysis_results[0]["results"]["mean"] == 0.66


class TestLegacyFolderLoadVariable:
    def test_new_name_is_used(self):
        assert folder_load_setting({FOLDER_LOAD_ENV: "0"}) == "0"

    def test_old_name_still_works_with_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match=FOLDER_LOAD_ENV):
            assert folder_load_setting({LEGACY_FOLDER_LOAD_ENV: "1"}) == "1"

    def test_new_name_wins_and_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env = {FOLDER_LOAD_ENV: "0", LEGACY_FOLDER_LOAD_ENV: "1"}
            assert folder_load_setting(env) == "0"

    def test_unset_is_none(self):
        assert folder_load_setting({}) is None


def test_config_json_location_is_independent_of_package_name():
    # config.json is resolved three levels above utils/user_settings.py: the repo root
    # in a source checkout. The rename must not move users' saved defaults and presets.
    assert _us.get_config_path() == REPO / "config.json"
