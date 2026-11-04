"""Tests for plottle/plugin_loader.py."""

import sys
import types
from pathlib import Path


# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plottle.plugin_loader import (
    discover_plugins,
    get_plugin_analysis_tools,
    get_plugin_plot_types,
    list_plugins,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REAL_PLUGINS_DIR = Path(__file__).parent.parent / "plugins"


def _write_plugin(tmp_path: Path, filename: str, content: str) -> Path:
    """Write a plugin file into tmp_path and return its path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


VALID_PLUGIN_SRC = """\
PLUGIN_NAME = "Test Plugin"
PLUGIN_VERSION = "2.0.0"
PLUGIN_DESCRIPTION = "A test plugin."

def get_plot_types():
    return [{"name": "test_plot", "label": "Test Plot", "function": lambda: None}]

def get_analysis_tools():
    return [{"name": "test_tool", "label": "Test Tool", "function": lambda: None}]
"""

MINIMAL_PLUGIN_SRC = """\
# No metadata, no functions
x = 42
"""

RAISING_PLOT_TYPES_SRC = """\
PLUGIN_NAME = "Raiser"
PLUGIN_VERSION = "0.1.0"

def get_plot_types():
    raise RuntimeError("boom")

def get_analysis_tools():
    return []
"""


# ---------------------------------------------------------------------------
# TestDiscoverPlugins
# ---------------------------------------------------------------------------


class TestDiscoverPlugins:
    def test_discovers_example_plugin_from_real_dir(self):
        """plugin_example.py must appear when scanning the real plugins/ dir."""
        plugins = discover_plugins(REAL_PLUGINS_DIR)
        assert "plugin_example" in plugins

    def test_returns_dict(self):
        plugins = discover_plugins(REAL_PLUGINS_DIR)
        assert isinstance(plugins, dict)

    def test_keys_are_stems_without_py_extension(self):
        plugins = discover_plugins(REAL_PLUGINS_DIR)
        for key in plugins:
            assert not key.endswith(".py")

    def test_custom_plugins_dir_with_valid_plugin(self, tmp_path):
        _write_plugin(tmp_path, "plugin_custom.py", VALID_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        assert "plugin_custom" in plugins

    def test_empty_plugins_dir_returns_empty_dict(self, tmp_path):
        plugins = discover_plugins(tmp_path)
        assert plugins == {}

    def test_nonexistent_plugins_dir_returns_empty_dict(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        plugins = discover_plugins(missing)
        assert plugins == {}

    def test_files_not_matching_plugin_glob_are_ignored(self, tmp_path):
        _write_plugin(tmp_path, "helper.py", VALID_PLUGIN_SRC)
        _write_plugin(tmp_path, "utils_plugin.py", VALID_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        assert plugins == {}

    def test_broken_plugin_with_syntax_error_is_skipped(self, tmp_path):
        _write_plugin(tmp_path, "plugin_good.py", VALID_PLUGIN_SRC)
        _write_plugin(tmp_path, "plugin_bad.py", "def broken(:  # syntax error\n    pass\n")
        plugins = discover_plugins(tmp_path)
        assert "plugin_good" in plugins
        assert "plugin_bad" not in plugins

    def test_plugin_with_import_error_is_skipped(self, tmp_path):
        _write_plugin(tmp_path, "plugin_good.py", VALID_PLUGIN_SRC)
        _write_plugin(
            tmp_path,
            "plugin_import_err.py",
            "import this_package_does_not_exist_xyzzy123\n",
        )
        plugins = discover_plugins(tmp_path)
        assert "plugin_good" in plugins
        assert "plugin_import_err" not in plugins

    def test_values_are_module_objects(self, tmp_path):
        _write_plugin(tmp_path, "plugin_mod.py", VALID_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        assert isinstance(plugins["plugin_mod"], types.ModuleType)

    def test_default_plugins_dir_is_used_when_none_passed(self):
        """discover_plugins() with no arg should scan the real plugins/ dir."""
        plugins = discover_plugins()
        assert isinstance(plugins, dict)
        assert "plugin_example" in plugins


# ---------------------------------------------------------------------------
# TestGetPluginPlotTypes
# ---------------------------------------------------------------------------


class TestGetPluginPlotTypes:
    def test_returns_list(self, tmp_path):
        result = get_plugin_plot_types(plugins={})
        assert isinstance(result, list)

    def test_empty_when_no_plugins_define_get_plot_types(self, tmp_path):
        _write_plugin(tmp_path, "plugin_minimal.py", MINIMAL_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = get_plugin_plot_types(plugins)
        assert result == []

    def test_returns_items_from_plugin_that_defines_get_plot_types(self, tmp_path):
        _write_plugin(tmp_path, "plugin_full.py", VALID_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = get_plugin_plot_types(plugins)
        assert len(result) == 1
        assert result[0]["name"] == "test_plot"

    def test_plugin_raising_in_get_plot_types_is_skipped(self, tmp_path):
        _write_plugin(tmp_path, "plugin_raiser.py", RAISING_PLOT_TYPES_SRC)
        plugins = discover_plugins(tmp_path)
        # Should not raise; result is an empty list (raiser is skipped)
        result = get_plugin_plot_types(plugins)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_aggregates_from_multiple_plugins(self, tmp_path):
        _write_plugin(tmp_path, "plugin_a.py", VALID_PLUGIN_SRC)
        _write_plugin(
            tmp_path,
            "plugin_b.py",
            "def get_plot_types():\n"
            '    return [{"name": "b_plot", "label": "B", "function": None}]\n',
        )
        plugins = discover_plugins(tmp_path)
        result = get_plugin_plot_types(plugins)
        names = [item["name"] for item in result]
        assert "test_plot" in names
        assert "b_plot" in names

    def test_auto_discovers_when_plugins_is_none(self):
        """When plugins=None the function should call discover_plugins itself."""
        result = get_plugin_plot_types(None)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestGetPluginAnalysisTools
# ---------------------------------------------------------------------------


class TestGetPluginAnalysisTools:
    def test_returns_list(self):
        result = get_plugin_analysis_tools(plugins={})
        assert isinstance(result, list)

    def test_empty_when_no_plugins_define_get_analysis_tools(self, tmp_path):
        _write_plugin(tmp_path, "plugin_minimal.py", MINIMAL_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = get_plugin_analysis_tools(plugins)
        assert result == []

    def test_returns_items_from_plugin_that_defines_get_analysis_tools(self, tmp_path):
        _write_plugin(tmp_path, "plugin_full.py", VALID_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = get_plugin_analysis_tools(plugins)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"

    def test_auto_discovers_when_plugins_is_none(self):
        result = get_plugin_analysis_tools(None)
        assert isinstance(result, list)

    def test_aggregates_from_multiple_plugins(self, tmp_path):
        _write_plugin(tmp_path, "plugin_a.py", VALID_PLUGIN_SRC)
        _write_plugin(
            tmp_path,
            "plugin_b.py",
            "def get_analysis_tools():\n"
            '    return [{"name": "b_tool", "label": "B", "function": None}]\n',
        )
        plugins = discover_plugins(tmp_path)
        result = get_plugin_analysis_tools(plugins)
        names = [item["name"] for item in result]
        assert "test_tool" in names
        assert "b_tool" in names


# ---------------------------------------------------------------------------
# TestListPlugins
# ---------------------------------------------------------------------------


class TestListPlugins:
    def test_returns_list(self):
        result = list_plugins(plugins={})
        assert isinstance(result, list)

    def test_each_dict_has_required_keys(self, tmp_path):
        _write_plugin(tmp_path, "plugin_full.py", VALID_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = list_plugins(plugins)
        assert len(result) == 1
        entry = result[0]
        for key in ("name", "version", "description", "file"):
            assert key in entry, f"Missing key: {key}"

    def test_example_plugin_has_correct_name(self):
        plugins = discover_plugins(REAL_PLUGINS_DIR)
        result = list_plugins(plugins)
        names = [e["name"] for e in result]
        assert "Example Plugin" in names

    def test_example_plugin_has_correct_version(self):
        plugins = discover_plugins(REAL_PLUGINS_DIR)
        result = list_plugins(plugins)
        entry = next(e for e in result if e["name"] == "Example Plugin")
        assert entry["version"] == "1.0.0"

    def test_plugin_with_no_plugin_name_uses_stem_as_fallback(self, tmp_path):
        _write_plugin(tmp_path, "plugin_noname.py", MINIMAL_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = list_plugins(plugins)
        assert len(result) == 1
        assert result[0]["name"] == "plugin_noname"

    def test_plugin_with_no_plugin_version_uses_unknown_fallback(self, tmp_path):
        _write_plugin(tmp_path, "plugin_nover.py", MINIMAL_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = list_plugins(plugins)
        assert result[0]["version"] == "unknown"

    def test_plugin_with_no_description_uses_empty_string(self, tmp_path):
        _write_plugin(tmp_path, "plugin_nodesc.py", MINIMAL_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = list_plugins(plugins)
        assert result[0]["description"] == ""

    def test_file_key_is_a_path_object(self, tmp_path):
        _write_plugin(tmp_path, "plugin_full.py", VALID_PLUGIN_SRC)
        plugins = discover_plugins(tmp_path)
        result = list_plugins(plugins)
        assert isinstance(result[0]["file"], Path)

    def test_auto_discovers_when_plugins_is_none(self):
        result = list_plugins(None)
        assert isinstance(result, list)

    def test_empty_plugins_dict_returns_empty_list(self):
        result = list_plugins(plugins={})
        assert result == []
