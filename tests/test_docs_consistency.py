"""Guards against documented counts drifting away from the code.

Audit finding A-25: "26 plot types" was stated in eight places across
``README.md``, ``docs/index.html``, ``Home.py`` and the Help page while
``PLOT_TYPES`` actually held 27 entries. ``inset_plot`` had been added without
anyone updating the prose, and nothing could catch it.

Rather than assert a hardcoded 27 -- which would just move the drift one level
out -- these tests derive the number from ``PLOT_TYPES`` and check that the
documentation agrees with it. Adding a plot type and forgetting the docs fails
here; adding one and updating the docs passes with no test change.
"""

import re
from pathlib import Path

import pytest

from plottle.utils.plot_config import PLOT_TYPES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "plottle"

# Files that state the plot-type count in prose, and must therefore stay in step
# with PLOT_TYPES.
_FILES_STATING_THE_COUNT = [
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "index.html",
    _PACKAGE_ROOT / "Home.py",
    _PACKAGE_ROOT / "pages" / "12_Help.py",
]


class TestPlotTypeCatalogue:
    def test_every_plot_type_resolves_to_a_callable(self):
        """A PLOT_TYPES key with no matching function is a dead GUI menu entry."""
        import plottle.plotting as plotting

        missing = sorted(k for k in PLOT_TYPES if not callable(getattr(plotting, k, None)))
        assert not missing, (
            f"PLOT_TYPES lists entries with no callable in plottle.plotting: {missing}"
        )

    def test_every_entry_has_a_label_and_category(self):
        malformed = sorted(
            key
            for key, meta in PLOT_TYPES.items()
            if not meta.get("label") or not meta.get("category")
        )
        assert not malformed, f"PLOT_TYPES entries missing label/category: {malformed}"

    def test_categories_are_known(self):
        # Matplotlib 15, Seaborn 4, Plotly 7, Specialty 1 as of 2.0.1. `Specialty`
        # holds inset_plot, which composes two axes rather than wrapping a single
        # library call, so it does not belong under a library heading.
        allowed = {"Matplotlib", "Seaborn", "Plotly", "Specialty"}
        unexpected = {
            meta["category"] for meta in PLOT_TYPES.values() if meta["category"] not in allowed
        }
        assert not unexpected, (
            f"unexpected plot categories: {sorted(unexpected)}. The Quick Plot picker "
            "groups by category, so a new one needs a deliberate home in the UI."
        )


class TestDocumentedPlotCountMatchesCode:
    """Regression guard for A-25."""

    @pytest.mark.parametrize(
        "path", _FILES_STATING_THE_COUNT, ids=lambda p: str(p.relative_to(_REPO_ROOT))
    )
    def test_no_stale_plot_count(self, path: Path):
        expected = len(PLOT_TYPES)
        text = path.read_text(encoding="utf-8")

        # Any "<n> plot types" / "<n> plot functions" claim must use the real number.
        stated = {
            int(n) for n in re.findall(r"(\d+)\s+plot\s+(?:types|functions)\b", text, re.IGNORECASE)
        }
        wrong = sorted(n for n in stated if n != expected)
        assert not wrong, (
            f"{path.relative_to(_REPO_ROOT)} claims {wrong} plot types but PLOT_TYPES "
            f"has {expected}. Update the prose, or the catalogue."
        )

    def test_at_least_one_file_actually_states_the_count(self):
        """Keeps the parametrized test honest if the wording ever changes.

        Without this, rewording every mention to something the regex misses would
        leave the guard above vacuously green.
        """
        expected = len(PLOT_TYPES)
        found = [
            path.relative_to(_REPO_ROOT)
            for path in _FILES_STATING_THE_COUNT
            if re.search(
                rf"{expected}\s+plot\s+(?:types|functions)\b",
                path.read_text("utf-8"),
                re.IGNORECASE,
            )
        ]
        assert found, (
            f"no documentation file states '{expected} plot types' -- either the "
            "wording changed (update _FILES_STATING_THE_COUNT and the regex) or the "
            "count is stale everywhere."
        )
