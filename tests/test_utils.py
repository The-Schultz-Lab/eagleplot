"""Unit tests for the utils modules.

Covers:
- plottle.utils.user_settings  — config.json CRUD operations
- plottle.utils.plot_config    — get_plot_kwargs, COLOR_PALETTES, PLOT_TYPES
- plottle.utils.session_state  — dataset management, serialization, save/load
- plottle.utils.data_preview   — pure info helpers and Streamlit display helpers
"""

import json
import types as _types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import plottle.utils.user_settings as us
import plottle.utils.session_state as _ss
import plottle.utils.data_preview as _dp
from plottle.utils.plot_config import (
    COLOR_PALETTES,
    COLOR_PALETTE_NAMES,
    PLOT_TYPES,
    format_marker_label,
    get_plot_kwargs,
    selected_points,
)
from plottle.utils.session_state import (
    _serialize_data,
    _deserialize_data,
    initialize_session_state,
    add_dataset,
    get_current_dataset,
    get_dataset,
    delete_dataset,
    add_plot_to_history,
    clear_plot_history,
    add_analysis_result,
    save_session_to_file,
    load_session_from_file,
    clear_session,
    get_session_summary,
)
from plottle.utils.data_preview import (
    preview_dataframe,
    get_dataframe_info,
    get_array_info,
    format_data_size,
    get_column_suggestions,
    get_plottable_arrays,
    display_dataset_card,
    display_data_preview,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect every test to a fresh temporary config.json path."""
    cfg = tmp_path / "config.json"
    monkeypatch.setattr(us, "_CONFIG_PATH", cfg)
    yield cfg


# ============================================================================
# user_settings — load_config / save_config
# ============================================================================


class TestLoadSaveConfig:
    def test_load_missing_returns_empty_structure(self):
        cfg = us.load_config()
        assert cfg["version"] == "1.0"
        assert cfg["defaults"] == {}
        assert cfg["presets"] == {}

    def test_save_and_load_roundtrip(self):
        data = {"version": "1.0", "defaults": {"fontsize": 14}, "presets": {}}
        us.save_config(data)
        loaded = us.load_config()
        assert loaded["defaults"]["fontsize"] == 14

    def test_load_corrupt_json_returns_empty_structure(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text("not valid json }{", encoding="utf-8")
        monkeypatch.setattr(us, "_CONFIG_PATH", cfg)
        loaded = us.load_config()
        assert loaded == {"version": "1.0", "defaults": {}, "presets": {}, "workflows": {}}

    def test_load_missing_top_keys_filled_in(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text('{"version": "1.0"}', encoding="utf-8")
        monkeypatch.setattr(us, "_CONFIG_PATH", cfg)
        loaded = us.load_config()
        assert "defaults" in loaded
        assert "presets" in loaded


# ============================================================================
# user_settings — defaults
# ============================================================================


class TestDefaults:
    def test_get_defaults_empty_when_no_file(self):
        assert us.get_defaults() == {}

    def test_save_and_get_defaults(self):
        us.save_defaults({"grid": True, "fontsize": 12})
        d = us.get_defaults()
        assert d["grid"] is True
        assert d["fontsize"] == 12

    def test_save_defaults_overwrites_previous(self):
        us.save_defaults({"fontsize": 10})
        us.save_defaults({"fontsize": 16})
        assert us.get_defaults()["fontsize"] == 16

    def test_save_empty_defaults_clears(self):
        us.save_defaults({"fontsize": 10})
        us.save_defaults({})
        assert us.get_defaults() == {}

    def test_save_defaults_preserves_presets(self):
        us.save_preset("P1", {"fontsize": 11})
        us.save_defaults({"grid": False})
        assert "P1" in us.list_presets()


# ============================================================================
# user_settings — presets
# ============================================================================


class TestPresets:
    def test_list_presets_empty_when_no_file(self):
        assert us.list_presets() == []

    def test_save_and_list_preset(self):
        us.save_preset("Publication", {"fontsize": 14, "grid": False})
        names = us.list_presets()
        assert names == ["Publication"]

    def test_list_presets_sorted(self):
        us.save_preset("Zebra", {})
        us.save_preset("Alpha", {})
        assert us.list_presets() == ["Alpha", "Zebra"]

    def test_load_existing_preset(self):
        us.save_preset("Dark", {"grid": False, "fontsize": 9})
        p = us.load_preset("Dark")
        assert p["grid"] is False
        assert p["fontsize"] == 9

    def test_load_nonexistent_preset_returns_empty(self):
        assert us.load_preset("nonexistent") == {}

    def test_overwrite_preset(self):
        us.save_preset("P", {"fontsize": 10})
        us.save_preset("P", {"fontsize": 20})
        assert us.load_preset("P")["fontsize"] == 20

    def test_delete_existing_preset_returns_true(self):
        us.save_preset("ToDelete", {})
        result = us.delete_preset("ToDelete")
        assert result is True
        assert "ToDelete" not in us.list_presets()

    def test_delete_nonexistent_preset_returns_false(self):
        result = us.delete_preset("ghost")
        assert result is False

    def test_delete_does_not_remove_other_presets(self):
        us.save_preset("Keep", {"fontsize": 12})
        us.save_preset("Remove", {})
        us.delete_preset("Remove")
        assert "Keep" in us.list_presets()

    def test_multiple_presets_coexist(self):
        us.save_preset("A", {"fontsize": 10})
        us.save_preset("B", {"fontsize": 12})
        us.save_preset("C", {"fontsize": 14})
        assert set(us.list_presets()) == {"A", "B", "C"}

    def test_preset_survives_save_defaults(self):
        us.save_preset("Stable", {"linewidth": 2.0})
        us.save_defaults({"grid": True})
        assert us.load_preset("Stable")["linewidth"] == 2.0


# ============================================================================
# user_settings — get_config_path
# ============================================================================


class TestGetConfigPath:
    def test_returns_path_object(self):
        p = us.get_config_path()
        # In tests the path is monkeypatched so just check type
        assert isinstance(p, Path)

    def test_returns_monkeypatched_path(self, isolated_config):
        assert us.get_config_path() == isolated_config


# ============================================================================
# user_settings — SAVEABLE_KEYS
# ============================================================================


class TestSaveableKeys:
    def test_saveable_keys_is_list(self):
        assert isinstance(us.SAVEABLE_KEYS, list)

    def test_known_keys_present(self):
        for key in ("grid", "fontsize", "fontfamily", "linewidth", "color_palette"):
            assert key in us.SAVEABLE_KEYS

    def test_no_private_keys(self):
        for key in us.SAVEABLE_KEYS:
            assert not key.startswith("_"), f"Private key in SAVEABLE_KEYS: {key}"


# ============================================================================
# plot_config — COLOR_PALETTES
# ============================================================================


class TestColorPalettes:
    def test_palette_names_list_matches_dict_keys(self):
        assert COLOR_PALETTE_NAMES == list(COLOR_PALETTES.keys())

    def test_default_palette_exists(self):
        assert "Default" in COLOR_PALETTES

    def test_colorblind_palettes_exist(self):
        assert "Color-Blind Safe (Wong)" in COLOR_PALETTES
        assert "Color-Blind Safe (Okabe-Ito)" in COLOR_PALETTES

    def test_all_palettes_have_8_colors(self):
        for name, colors in COLOR_PALETTES.items():
            assert len(colors) == 8, f"Palette '{name}' has {len(colors)} colors, expected 8"

    def test_all_colors_are_strings(self):
        for name, colors in COLOR_PALETTES.items():
            for c in colors:
                assert isinstance(c, str), f"Non-string color in palette '{name}': {c!r}"

    def test_hex_colors_valid_format(self):
        import re

        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, colors in COLOR_PALETTES.items():
            for c in colors:
                if c.startswith("#"):
                    assert hex_re.match(c), f"Invalid hex color '{c}' in palette '{name}'"


# ============================================================================
# plot_config — PLOT_TYPES
# ============================================================================


class TestPlotTypes:
    def test_plot_types_not_empty(self):
        assert len(PLOT_TYPES) > 0

    def test_all_have_label_and_category(self):
        for key, info in PLOT_TYPES.items():
            assert "label" in info, f"'{key}' missing 'label'"
            assert "category" in info, f"'{key}' missing 'category'"

    def test_expected_keys_present(self):
        expected = {
            "histogram",
            "line_plot",
            "scatter_plot",
            "interactive_scatter",
            "interactive_line",
            "interactive_3d_surface",
        }
        assert expected.issubset(set(PLOT_TYPES.keys()))

    def test_categories_are_known(self):
        known = {"Matplotlib", "Seaborn", "Plotly", "Specialty"}
        for key, info in PLOT_TYPES.items():
            assert info["category"] in known, f"Unknown category '{info['category']}' for '{key}'"


# ============================================================================
# plot_config — click-to-annotate helpers (Quick Plot, Plotly mode)
# ============================================================================


class _AttrDict(dict):
    """Mimics Streamlit's AttributeDictionary: the event and its ``selection``
    support attribute access, but each point is a plain dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class TestSelectedPoints:
    def test_points_are_plain_dicts(self):
        # Regression: Quick Plot crashed with
        # AttributeError: 'dict' object has no attribute 'x'
        event = _AttrDict(
            selection=_AttrDict(
                points=[
                    {"curve_number": 0, "point_index": 3, "x": 1.5, "y": 2.5},
                    {"curve_number": 0, "point_index": 7, "x": 4.0, "y": -1.0},
                ]
            )
        )
        assert selected_points(event) == [{"x": 1.5, "y": 2.5}, {"x": 4.0, "y": -1.0}]

    def test_plain_dict_event(self):
        event = {"selection": {"points": [{"x": "a", "y": 3}]}}
        assert selected_points(event) == [{"x": "a", "y": 3}]

    def test_attribute_style_points_still_work(self):
        pt = type("Pt", (), {"x": 1, "y": 2})()
        assert selected_points({"selection": {"points": [pt]}}) == [{"x": 1, "y": 2}]

    @pytest.mark.parametrize(
        "event", [None, {}, {"selection": None}, {"selection": {"points": []}}]
    )
    def test_empty_or_missing(self, event):
        assert selected_points(event) == []

    def test_skips_points_without_coordinates(self):
        event = {"selection": {"points": [{"label": "slice", "value": 4}, {"x": 0, "y": 0}]}}
        assert selected_points(event) == [{"x": 0, "y": 0}]


class TestFormatMarkerLabel:
    def test_numeric(self):
        assert format_marker_label(1.234567, 2) == "(1.235, 2)"

    def test_numpy_scalar(self):
        assert format_marker_label(np.float64(0.000123456), np.int64(5)) == "(0.0001235, 5)"

    def test_non_numeric_does_not_raise(self):
        assert format_marker_label("2024-01-05", "group A") == "(2024-01-05, group A)"


# ============================================================================
# plot_config — get_plot_kwargs
# ============================================================================


class TestGetPlotKwargs:
    def test_strips_underscore_prefixed_keys(self):
        config = {"title": "My Plot", "_x_col": "x", "_y_cols": ["y"]}
        kwargs = get_plot_kwargs(config)
        assert "_x_col" not in kwargs
        assert "_y_cols" not in kwargs
        assert "title" in kwargs

    def test_strips_none_values(self):
        config = {"title": "Test", "xlabel": None, "ylabel": "Y"}
        kwargs = get_plot_kwargs(config)
        assert "xlabel" not in kwargs
        assert "ylabel" in kwargs

    def test_strips_style_keys(self):
        config = {
            "title": "Test",
            "grid": True,
            "fontsize": 12,
            "linewidth": 2.0,
            "color_palette": "Default",
        }
        kwargs = get_plot_kwargs(config)
        assert "grid" not in kwargs
        assert "fontsize" not in kwargs
        assert "linewidth" not in kwargs
        assert "color_palette" not in kwargs
        assert "title" in kwargs

    def test_empty_config_returns_empty(self):
        assert get_plot_kwargs({}) == {}

    def test_preserves_non_style_non_private_keys(self):
        config = {
            "bins": 20,
            "title": "Histogram",
            "xlabel": "Value",
            "color": "steelblue",
        }
        kwargs = get_plot_kwargs(config)
        assert kwargs == config

    def test_zero_values_not_stripped(self):
        """0, False, and empty string are falsy but valid kwargs — must NOT be dropped."""
        config = {"bins": 0, "color": "blue"}
        kwargs = get_plot_kwargs(config)
        assert "bins" in kwargs

    def test_false_value_not_stripped(self):
        config = {"legend": False, "title": "T"}
        kwargs = get_plot_kwargs(config)
        assert "legend" in kwargs
        assert kwargs["legend"] is False


# ============================================================================
# Helpers shared by session_state and data_preview tests
# ============================================================================


class _MockState:
    """Minimal drop-in for st.session_state: attribute store + `in` operator."""

    def __contains__(self, key):
        return key in self.__dict__


class _NullCM:
    """Null context manager for mocking st.container() / st.expander()."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _MockCol:
    def metric(self, *a, **kw):
        pass


def _make_mock_st_for_dp():
    """Build a SimpleNamespace that satisfies all st calls in data_preview.py."""
    return _types.SimpleNamespace(
        container=lambda: _NullCM(),
        subheader=lambda *a, **kw: None,
        columns=lambda n: [_MockCol() for _ in range(n)],
        metric=lambda *a, **kw: None,
        write=lambda *a, **kw: None,
        dataframe=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        expander=lambda *a, **kw: _NullCM(),
        json=lambda *a, **kw: None,
    )


@pytest.fixture
def mock_session_state(monkeypatch):
    """Patch st inside session_state module; initialize state; return the state object."""
    state = _MockState()
    monkeypatch.setattr(_ss, "st", _types.SimpleNamespace(session_state=state))
    initialize_session_state()
    return state


@pytest.fixture
def mock_dp_st(monkeypatch):
    """Patch st inside data_preview module with a full display mock."""
    mock = _make_mock_st_for_dp()
    monkeypatch.setattr(_dp, "st", mock)
    return mock


# ============================================================================
# data_preview — pure functions (no st needed)
# ============================================================================


class TestPreviewDataframe:
    def test_small_df_returned_whole(self):
        df = pd.DataFrame({"a": range(15)})
        result = preview_dataframe(df, n_rows=10)
        assert len(result) == 15

    def test_large_df_returns_head_and_tail(self):
        df = pd.DataFrame({"a": range(100)})
        result = preview_dataframe(df, n_rows=5)
        assert len(result) == 10
        assert result.iloc[0]["a"] == 0
        assert result.iloc[-1]["a"] == 99


class TestGetDataframeInfo:
    @pytest.fixture
    def mixed_df(self):
        return pd.DataFrame(
            {
                "x": [1, 2, None],
                "y": [4.0, 5.0, 6.0],
                "label": ["a", "b", "c"],
            }
        )

    def test_required_keys(self, mixed_df):
        info = get_dataframe_info(mixed_df)
        for key in (
            "shape",
            "columns",
            "dtypes",
            "memory_usage",
            "missing_values",
            "numeric_columns",
            "categorical_columns",
        ):
            assert key in info

    def test_shape_and_columns(self, mixed_df):
        info = get_dataframe_info(mixed_df)
        assert info["shape"] == (3, 3)
        assert "x" in info["columns"]

    def test_numeric_and_categorical_split(self, mixed_df):
        info = get_dataframe_info(mixed_df)
        assert "y" in info["numeric_columns"]
        assert "label" in info["categorical_columns"]

    def test_missing_values_detected(self, mixed_df):
        info = get_dataframe_info(mixed_df)
        assert info["missing_values"]["x"] == 1


class TestGetArrayInfo:
    def test_basic_keys(self):
        arr = np.array([1, 2, 3])
        info = get_array_info(arr)
        for key in ("shape", "dtype", "size", "memory_usage", "ndim"):
            assert key in info

    def test_numeric_stats(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        info = get_array_info(arr)
        assert info["min"] == 1.0
        assert info["max"] == 4.0
        assert "mean" in info and "std" in info

    def test_non_numeric_no_stats(self):
        arr = np.array(["a", "b", "c"])
        info = get_array_info(arr)
        assert "min" not in info

    def test_2d_array(self):
        arr = np.ones((3, 4))
        info = get_array_info(arr)
        assert info["shape"] == (3, 4)
        assert info["ndim"] == 2


class TestFormatDataSize:
    def test_bytes(self):
        assert format_data_size(512) == "512.0 B"

    def test_kilobytes(self):
        assert format_data_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_data_size(1024**2) == "1.0 MB"

    def test_gigabytes(self):
        assert "GB" in format_data_size(1024**3)

    def test_terabytes(self):
        assert "TB" in format_data_size(1024**4)


class TestGetColumnSuggestions:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(0)
        return pd.DataFrame(
            {
                "time": range(20),
                "value": np.random.randn(20),
                "category": ["A", "B"] * 10,
                "group": [0, 1, 2, 3] * 5,  # numeric, low-cardinality → hue
            }
        )

    def test_result_keys(self, sample_df):
        sugg = get_column_suggestions(sample_df)
        for key in ("x_candidates", "y_candidates", "hue_candidates", "numeric", "categorical"):
            assert key in sugg

    def test_numeric_in_x_and_y(self, sample_df):
        sugg = get_column_suggestions(sample_df)
        assert "time" in sugg["x_candidates"]
        assert "value" in sugg["y_candidates"]

    def test_categorical_in_hue(self, sample_df):
        sugg = get_column_suggestions(sample_df)
        assert "category" in sugg["hue_candidates"]

    def test_low_cardinality_numeric_in_hue(self, sample_df):
        sugg = get_column_suggestions(sample_df)
        assert "group" in sugg["hue_candidates"]


class TestGetPlottableArrays:
    def test_dataframe_returns_numeric_values(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "label": ["a", "b"]})
        arrays, labels = get_plottable_arrays(df)
        assert arrays is not None
        assert set(labels) == {"x", "y"}

    def test_dataframe_no_numeric_returns_none(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        arrays, labels = get_plottable_arrays(df)
        assert arrays is None and labels is None

    def test_numeric_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        arrays, labels = get_plottable_arrays(arr)
        assert arrays is arr
        assert labels is None

    def test_non_numeric_ndarray(self):
        arr = np.array(["a", "b", "c"])
        arrays, labels = get_plottable_arrays(arr)
        assert arrays is None

    def test_dict_with_numeric_arrays(self):
        data = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        arrays, labels = get_plottable_arrays(data)
        assert arrays is not None
        assert set(labels) == {"a", "b"}

    def test_dict_non_numeric_returns_none(self):
        data = {"a": ["x", "y"], "b": ["p", "q"]}
        arrays, labels = get_plottable_arrays(data)
        assert arrays is None

    def test_other_type_returns_none(self):
        arrays, labels = get_plottable_arrays(42)
        assert arrays is None and labels is None


# ============================================================================
# data_preview — Streamlit display helpers (mock_dp_st fixture)
# ============================================================================


class TestDataPreviewDisplay:
    def test_display_dataset_card_dataframe(self, mock_dp_st):
        df = pd.DataFrame({"x": [1, None], "y": [3.0, 4.0]})
        display_dataset_card("test.csv", df, metadata={"source": "test"})

    def test_display_dataset_card_dataframe_no_missing(self, mock_dp_st):
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        display_dataset_card("clean.csv", df)

    def test_display_dataset_card_ndarray(self, mock_dp_st):
        arr = np.array([1.0, 2.0, 3.0])
        display_dataset_card("test.npy", arr)

    def test_display_dataset_card_other_with_len(self, mock_dp_st):
        display_dataset_card("test.pkl", [1, 2, 3])

    def test_display_data_preview_dataframe(self, mock_dp_st):
        df = pd.DataFrame({"x": range(20), "y": range(20, 40)})
        display_data_preview(df, name="MyDF")

    def test_display_data_preview_ndarray_1d(self, mock_dp_st):
        display_data_preview(np.arange(50), name="arr1d")

    def test_display_data_preview_ndarray_2d(self, mock_dp_st):
        display_data_preview(np.ones((10, 3)), name="arr2d")

    def test_display_data_preview_ndarray_3d(self, mock_dp_st):
        display_data_preview(np.ones((2, 3, 4)), name="arr3d")

    def test_display_data_preview_dict(self, mock_dp_st):
        display_data_preview({"a": 1, "b": 2}, name="mydict")

    def test_display_data_preview_list_short(self, mock_dp_st):
        display_data_preview([1, 2, 3], name="short")

    def test_display_data_preview_list_long(self, mock_dp_st):
        display_data_preview(list(range(20)), name="long")

    def test_display_data_preview_other(self, mock_dp_st):
        display_data_preview(42, name="scalar")


# ============================================================================
# session_state — _serialize_data / _deserialize_data (pure, no st needed)
# ============================================================================


class TestSerializeDeserialize:
    def test_serialize_scalar_types(self):
        for val in [1, 2.5, "hello", True, None]:
            assert _serialize_data(val) == val

    def test_serialize_list(self):
        assert _serialize_data([1, 2, 3]) == [1, 2, 3]

    def test_serialize_dict(self):
        assert _serialize_data({"a": 1}) == {"a": 1}

    def test_serialize_ndarray_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0])
        serialized = _serialize_data(arr)
        assert serialized["__type__"] == "ndarray"
        assert np.allclose(_deserialize_data(serialized), arr)

    def test_serialize_dataframe_roundtrip(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        serialized = _serialize_data(df)
        assert serialized["__type__"] == "DataFrame"
        restored = _deserialize_data(serialized)
        assert list(restored.columns) == ["x", "y"]

    def test_serialize_ndarray_preserves_dtype_and_shape(self):
        for arr in [
            np.arange(6, dtype=np.int32).reshape(2, 3),
            np.array([[1.5, 2.5]], dtype=np.float32),
            np.array([True, False, True]),
            np.array([1 + 2j, 3 - 4j]),
        ]:
            restored = _deserialize_data(_serialize_data(arr))
            assert restored.dtype == arr.dtype
            assert restored.shape == arr.shape
            assert np.array_equal(restored, arr)

    def test_deserialized_ndarray_is_writable(self):
        # np.frombuffer returns a read-only view; the decoder must copy.
        restored = _deserialize_data(_serialize_data(np.zeros(3)))
        restored[0] = 1.0  # must not raise
        assert restored[0] == 1.0

    def test_serialize_unknown_type_is_refused_not_pickled(self):
        # Anything without a safe JSON encoding becomes an 'unsupported'
        # placeholder. It must NOT be pickled — see TDEC-009 / audit A-01.
        serialized = _serialize_data(complex(3, 4))
        assert serialized["__type__"] == "unsupported"
        assert "complex" in serialized["__class__"]

        notes = []
        assert _deserialize_data(serialized, notes) is None
        assert notes and "complex" in notes[0]

    def test_serialize_object_dtype_array_is_refused(self):
        arr = np.array([{"a": 1}, "text"], dtype=object)
        serialized = _serialize_data(arr)
        assert serialized["__type__"] == "unsupported"

    def test_deserialize_plain_dict(self):
        assert _deserialize_data({"key": "value"}) == {"key": "value"}

    def test_deserialize_list(self):
        assert _deserialize_data([1, 2, 3]) == [1, 2, 3]

    def test_deserialize_scalar(self):
        assert _deserialize_data(42) == 42


_UNPICKLE_CANARY = []


def _trip_canary(value):
    """Module-level so pickle can reference it by name. Records execution."""
    _UNPICKLE_CANARY.append(value)
    return value


class _CanaryPayload:
    """Pickles to a call of :func:`_trip_canary` — stands in for a malicious
    ``__reduce__`` returning ``os.system``. ``__reduce__`` runs at dump time,
    so the side effect must live in the *returned callable*, not in
    ``__reduce__`` itself.
    """

    def __reduce__(self):
        return (_trip_canary, ("EXECUTED",))


class TestSessionDeserializationIsSafe:
    """Regression tests for audit A-01 — session files must never unpickle.

    ``load_session_from_file`` is reachable from the Export page's file
    uploader, so a pickle-based decoder would make opening a shared session
    file equivalent to executing its author's code.
    """

    def test_session_state_module_does_not_import_pickle(self):
        import plottle.utils.session_state as ss

        assert not hasattr(ss, "pickle"), (
            "session_state must not import pickle — it is reachable from an upload"
        )

    def test_legacy_pickled_entry_is_refused(self):
        import base64
        import pickle

        _UNPICKLE_CANARY.clear()
        blob = base64.b64encode(pickle.dumps(_CanaryPayload())).decode()

        # Sanity-check that the payload really would execute if unpickled,
        # so a green test cannot mean "the canary was inert".
        assert pickle.loads(base64.b64decode(blob)) == "EXECUTED"
        assert _UNPICKLE_CANARY == ["EXECUTED"]
        _UNPICKLE_CANARY.clear()

        notes = []
        assert _deserialize_data({"__type__": "pickled", "__data__": blob}, notes) is None
        assert notes and "refused" in notes[0]
        assert _UNPICKLE_CANARY == [], "the decoder executed a pickle payload"

    def test_legacy_pickle_encoded_ndarray_is_refused(self):
        # Pre-2.0.2 ndarray entries carried a pickle blob and no __dtype__.
        import base64
        import pickle

        blob = base64.b64encode(pickle.dumps(np.zeros(3))).decode()
        notes = []
        assert _deserialize_data({"__type__": "ndarray", "__data__": blob}, notes) is None
        assert notes

    def test_truncated_buffer_is_rejected(self):
        serialized = _serialize_data(np.arange(10, dtype=np.float64))
        serialized["__shape__"] = [11]  # claim more elements than the buffer holds
        notes = []
        assert _deserialize_data(serialized, notes) is None
        assert notes and "corrupt" in notes[0]

    def test_object_dtype_on_decode_is_rejected(self):
        notes = []
        payload = {"__type__": "ndarray", "__dtype__": "|O", "__shape__": [1], "__data__": ""}
        assert _deserialize_data(payload, notes) is None
        assert notes

    def test_unrecognized_type_tag_is_refused(self):
        notes = []
        assert _deserialize_data({"__type__": "something_new"}, notes) is None
        assert notes

    def test_load_reports_skipped_entries_and_clears_selection(self, mock_session_state, tmp_path):
        import base64
        import pickle

        blob = base64.b64encode(pickle.dumps(np.zeros(3))).decode()
        session = {
            "version": "1.0",
            "datasets": {"legacy.npy": {"__type__": "pickled", "__data__": blob}},
            "dataset_metadata": {},
            "current_dataset": "legacy.npy",
            "plot_history": [],
            "analysis_results": [],
            "plot_config": {},
        }
        path = tmp_path / "legacy_session.json"
        path.write_text(json.dumps(session), encoding="utf-8")

        skipped = load_session_from_file(str(path))

        assert len(skipped) == 1
        assert "legacy.npy" in skipped[0]
        assert "legacy.npy" not in mock_session_state.datasets
        # A dropped dataset must not remain the active selection.
        assert mock_session_state.current_dataset is None


# ============================================================================
# session_state — Streamlit-dependent functions (mock_session_state fixture)
# ============================================================================


class TestInitializeSessionState:
    def test_all_keys_created(self, mock_session_state):
        for key in (
            "datasets",
            "dataset_metadata",
            "current_dataset",
            "plot_history",
            "analysis_results",
            "plot_config",
            "export_queue",
        ):
            assert hasattr(mock_session_state, key)

    def test_idempotent(self, mock_session_state):
        mock_session_state.datasets["keep_me"] = 42
        initialize_session_state()
        assert "keep_me" in mock_session_state.datasets


class TestAddAndGetDataset:
    def test_add_ndarray_stored(self, mock_session_state):
        arr = np.array([1, 2, 3])
        add_dataset("arr.npy", arr)
        assert "arr.npy" in mock_session_state.datasets

    def test_add_dataframe_metadata_keys(self, mock_session_state):
        df = pd.DataFrame({"a": [1, 2]})
        add_dataset("df.csv", df)
        meta = mock_session_state.dataset_metadata["df.csv"]
        assert "shape" in meta and "columns" in meta

    def test_add_list_metadata_has_length(self, mock_session_state):
        add_dataset("data.pkl", [1, 2, 3])
        assert "length" in mock_session_state.dataset_metadata["data.pkl"]

    def test_first_dataset_becomes_current(self, mock_session_state):
        add_dataset("first.npy", np.zeros(5))
        assert mock_session_state.current_dataset == "first.npy"

    def test_second_dataset_does_not_change_current(self, mock_session_state):
        add_dataset("first.npy", np.zeros(5))
        add_dataset("second.npy", np.ones(5))
        assert mock_session_state.current_dataset == "first.npy"

    def test_get_current_dataset_returns_data(self, mock_session_state):
        arr = np.array([7, 8, 9])
        add_dataset("test.npy", arr)
        assert np.array_equal(get_current_dataset(), arr)

    def test_get_current_dataset_none_when_empty(self, mock_session_state):
        assert get_current_dataset() is None

    def test_get_dataset_by_name(self, mock_session_state):
        arr = np.array([1, 2])
        add_dataset("named.npy", arr)
        assert np.array_equal(get_dataset("named.npy"), arr)

    def test_get_dataset_missing_returns_none(self, mock_session_state):
        assert get_dataset("nonexistent") is None


class TestDeleteDataset:
    def test_delete_existing_returns_true(self, mock_session_state):
        add_dataset("a.npy", np.zeros(3))
        assert delete_dataset("a.npy") is True
        assert "a.npy" not in mock_session_state.datasets

    def test_delete_nonexistent_returns_false(self, mock_session_state):
        assert delete_dataset("ghost.npy") is False

    def test_delete_current_selects_next(self, mock_session_state):
        add_dataset("a.npy", np.zeros(3))
        add_dataset("b.npy", np.ones(3))
        delete_dataset("a.npy")
        assert mock_session_state.current_dataset == "b.npy"

    def test_delete_last_sets_current_to_none(self, mock_session_state):
        add_dataset("only.npy", np.zeros(3))
        delete_dataset("only.npy")
        assert mock_session_state.current_dataset is None


class TestPlotHistoryAndAnalysis:
    def test_add_plot_gets_timestamp(self, mock_session_state):
        add_plot_to_history({"type": "histogram", "config": {}})
        assert "timestamp" in mock_session_state.plot_history[0]

    def test_add_plot_preserves_existing_timestamp(self, mock_session_state):
        add_plot_to_history({"type": "scatter", "timestamp": "2026-01-01"})
        assert mock_session_state.plot_history[0]["timestamp"] == "2026-01-01"

    def test_clear_plot_history(self, mock_session_state):
        add_plot_to_history({"type": "line"})
        clear_plot_history()
        assert mock_session_state.plot_history == []

    def test_add_analysis_result_gets_timestamp(self, mock_session_state):
        add_analysis_result({"type": "statistics", "results": {}})
        assert "timestamp" in mock_session_state.analysis_results[0]


class TestClearAndSummary:
    def test_clear_session_resets_all(self, mock_session_state):
        add_dataset("x.npy", np.zeros(3))
        add_plot_to_history({"type": "line"})
        clear_session()
        assert mock_session_state.datasets == {}
        assert mock_session_state.plot_history == []
        assert mock_session_state.current_dataset is None

    def test_get_session_summary(self, mock_session_state):
        add_dataset("a.npy", np.zeros(5))
        add_plot_to_history({"type": "hist"})
        summary = get_session_summary()
        assert summary["num_datasets"] == 1
        assert summary["num_plots"] == 1
        assert "a.npy" in summary["dataset_names"]


class TestSaveLoadSession:
    def test_roundtrip_ndarray(self, mock_session_state, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        add_dataset("arr.npy", arr)
        add_plot_to_history({"type": "line", "figure": "EXCLUDED"})
        filepath = tmp_path / "session.json"
        save_session_to_file(str(filepath))
        assert filepath.exists()

        clear_session()
        load_session_from_file(str(filepath))
        assert np.allclose(get_dataset("arr.npy"), arr)
        # 'figure' key must be stripped from saved plot history
        assert "figure" not in mock_session_state.plot_history[0]

    def test_roundtrip_dataframe(self, mock_session_state, tmp_path):
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        add_dataset("df.csv", df)
        filepath = tmp_path / "session.json"
        save_session_to_file(str(filepath))
        clear_session()
        load_session_from_file(str(filepath))
        restored = get_dataset("df.csv")
        assert list(restored.columns) == ["x", "y"]
