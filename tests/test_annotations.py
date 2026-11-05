"""Tests for eagleplot/annotations.py (M16)."""

import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless backend for CI

from eagleplot.annotations import apply_annotations, describe_overlay, ANNOTATION_COLORS


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def ax():
    """Return a fresh matplotlib Axes pre-populated with a simple line."""
    fig, _ax = plt.subplots()
    _ax.plot([0, 1, 2, 3], [0, 1, 0, 1])
    yield _ax
    plt.close(fig)


# ── apply_annotations ─────────────────────────────────────────────────────────


class TestHLine:
    def test_hline_adds_line(self, ax):
        n_before = len(ax.lines)
        apply_annotations(ax, [{"type": "hline", "y": 0.5}])
        assert len(ax.lines) > n_before

    def test_hline_defaults(self, ax):
        apply_annotations(ax, [{"type": "hline", "y": 1.0}])
        # Should not raise; default color / style applied

    def test_hline_custom_color_and_label(self, ax):
        ann = {"type": "hline", "y": 0.5, "color": "blue", "label": "threshold"}
        apply_annotations(ax, [ann])
        labels = [h.get_label() for h in ax.lines]
        assert "threshold" in labels


class TestVLine:
    def test_vline_adds_line(self, ax):
        n_before = len(ax.lines)
        apply_annotations(ax, [{"type": "vline", "x": 1.5}])
        assert len(ax.lines) > n_before

    def test_vline_linestyle(self, ax):
        apply_annotations(ax, [{"type": "vline", "x": 1.0, "linestyle": ":"}])
        # No assertion on internals — just confirm no exception


class TestHSpan:
    def test_hspan_adds_artist(self, ax):
        # axhspan adds a Polygon patch in mpl >= 3.8, or a collection in older versions
        n_before = len(ax.patches) + len(ax.collections)
        apply_annotations(ax, [{"type": "hspan", "y1": 0.2, "y2": 0.8}])
        assert len(ax.patches) + len(ax.collections) > n_before

    def test_hspan_alpha(self, ax):
        apply_annotations(ax, [{"type": "hspan", "y1": 0.0, "y2": 0.5, "alpha": 0.1}])


class TestVSpan:
    def test_vspan_adds_artist(self, ax):
        n_before = len(ax.patches) + len(ax.collections)
        apply_annotations(ax, [{"type": "vspan", "x1": 0.5, "x2": 1.5}])
        assert len(ax.patches) + len(ax.collections) > n_before

    def test_vspan_label(self, ax):
        ann = {"type": "vspan", "x1": 0.5, "x2": 1.5, "label": "region"}
        apply_annotations(ax, [ann])


class TestText:
    def test_text_no_arrow(self, ax):
        apply_annotations(ax, [{"type": "text", "x": 1.0, "y": 0.5, "text": "hello"}])
        texts = [t.get_text() for t in ax.texts]
        assert "hello" in texts

    def test_text_with_arrow(self, ax):
        ann = {
            "type": "text",
            "x": 1.0,
            "y": 0.5,
            "text": "peak",
            "arrow": True,
            "tx": 1.5,
            "ty": 0.8,
        }
        apply_annotations(ax, [ann])
        # Annotate creates an Annotation artist
        assert len(ax.get_children()) > 0

    def test_text_fontsize(self, ax):
        ann = {"type": "text", "x": 0.0, "y": 0.0, "text": "x", "fontsize": 14}
        apply_annotations(ax, [ann])


class TestRectangle:
    def test_rectangle_added(self, ax):
        n_before = len(ax.patches)
        ann = {"type": "rectangle", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5}
        apply_annotations(ax, [ann])
        assert len(ax.patches) == n_before + 1

    def test_rectangle_facecolor(self, ax):
        apply_annotations(
            ax,
            [
                {
                    "type": "rectangle",
                    "x1": 0,
                    "y1": 0,
                    "x2": 1,
                    "y2": 1,
                    "color": "red",
                    "facecolor": "yellow",
                    "alpha": 0.4,
                }
            ],
        )

    def test_rectangle_label(self, ax):
        apply_annotations(
            ax,
            [{"type": "rectangle", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "label": "box"}],
        )


class TestEllipse:
    def test_ellipse_added(self, ax):
        n_before = len(ax.patches)
        apply_annotations(
            ax, [{"type": "ellipse", "cx": 1.0, "cy": 0.5, "width": 0.5, "height": 0.3}]
        )
        assert len(ax.patches) == n_before + 1

    def test_ellipse_facecolor(self, ax):
        apply_annotations(
            ax,
            [
                {
                    "type": "ellipse",
                    "cx": 1.0,
                    "cy": 0.5,
                    "width": 1.0,
                    "height": 0.5,
                    "color": "blue",
                    "facecolor": "cyan",
                }
            ],
        )


class TestMultipleOverlays:
    def test_multiple_overlays_applied(self, ax):
        overlays = [
            {"type": "hline", "y": 0.3},
            {"type": "vline", "x": 1.0},
            {"type": "hspan", "y1": 0.4, "y2": 0.6},
            {"type": "text", "x": 0.5, "y": 0.5, "text": "note"},
        ]
        apply_annotations(ax, overlays)
        assert len(ax.texts) >= 1

    def test_empty_overlays_no_error(self, ax):
        apply_annotations(ax, [])  # should be a no-op

    def test_unknown_type_skipped(self, ax):
        apply_annotations(ax, [{"type": "banana", "value": 42}])  # silently skipped

    def test_legend_updated_when_labels_present(self, ax):
        apply_annotations(ax, [{"type": "hline", "y": 0.5, "label": "limit"}])
        legend = ax.get_legend()
        assert legend is not None


class TestFloatConversion:
    """Ensure float() conversion works when keys arrive as strings from session state."""

    def test_hline_string_y(self, ax):
        apply_annotations(ax, [{"type": "hline", "y": "0.5"}])

    def test_vline_string_x(self, ax):
        apply_annotations(ax, [{"type": "vline", "x": "2"}])

    def test_hspan_string_bounds(self, ax):
        apply_annotations(ax, [{"type": "hspan", "y1": "0.1", "y2": "0.9"}])


# ── describe_overlay ──────────────────────────────────────────────────────────


class TestDescribeOverlay:
    def test_hline(self):
        s = describe_overlay({"type": "hline", "y": 0.5, "color": "red"})
        assert "H-line" in s
        assert "0.5" in s

    def test_vline(self):
        s = describe_overlay({"type": "vline", "x": 100})
        assert "V-line" in s

    def test_hspan(self):
        s = describe_overlay({"type": "hspan", "y1": 1, "y2": 2})
        assert "H-span" in s

    def test_vspan(self):
        s = describe_overlay({"type": "vspan", "x1": -1, "x2": 1})
        assert "V-span" in s

    def test_text(self):
        s = describe_overlay({"type": "text", "x": 0, "y": 0, "text": "hi"})
        assert "hi" in s

    def test_rectangle(self):
        s = describe_overlay({"type": "rectangle", "x1": 0, "y1": 0, "x2": 1, "y2": 1})
        assert "Rect" in s

    def test_ellipse(self):
        ann = {"type": "ellipse", "cx": 0, "cy": 0, "width": 1, "height": 1}
        s = describe_overlay(ann)
        assert "Ellipse" in s

    def test_label_suffix(self):
        s = describe_overlay({"type": "hline", "y": 0, "label": "baseline"})
        assert "baseline" in s

    def test_unknown_type(self):
        s = describe_overlay({"type": "mystery"})
        assert "mystery" in s


# ── ANNOTATION_COLORS ────────────────────────────────────────────────────────


class TestAnnotationColors:
    def test_nonempty(self):
        assert len(ANNOTATION_COLORS) > 0

    def test_all_strings(self):
        assert all(isinstance(c, str) for c in ANNOTATION_COLORS)

    def test_contains_red(self):
        assert "red" in ANNOTATION_COLORS
