"""Figure annotation helpers.

Provides ``apply_annotations`` for drawing overlay annotations
(reference lines, spans, text labels, and shapes) onto a Matplotlib Axes.

Used by Quick Plot (per-figure annotations) and the Multi-Panel Dashboard
(per-cell annotations before combined export).

Supported overlay types
-----------------------
``hline``
    Horizontal reference line at *y*.
    Keys: ``y``, ``color``, ``linestyle``, ``lw``, ``label``

``vline``
    Vertical reference line at *x*.
    Keys: ``x``, ``color``, ``linestyle``, ``lw``, ``label``

``hspan``
    Horizontal band shading between *y1* and *y2*.
    Keys: ``y1``, ``y2``, ``color``, ``alpha``, ``label``

``vspan``
    Vertical band shading between *x1* and *x2*.
    Keys: ``x1``, ``x2``, ``color``, ``alpha``, ``label``

``text``
    Text label placed at (*x*, *y*), with optional annotating arrow.
    Keys: ``x``, ``y``, ``text``, ``color``, ``fontsize``,
          ``arrow`` (bool), ``tx``, ``ty`` (arrow-tail coordinates)

``rectangle``
    Filled/outlined rectangle spanning (x1, y1) → (x2, y2).
    Keys: ``x1``, ``y1``, ``x2``, ``y2``,
          ``color``, ``facecolor``, ``lw``, ``alpha``, ``label``

``ellipse``
    Filled/outlined ellipse centred at (*cx*, *cy*).
    Keys: ``cx``, ``cy``, ``width``, ``height``,
          ``color``, ``facecolor``, ``lw``, ``alpha``, ``label``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

# Colour choices available in the GUI colour picker
ANNOTATION_COLORS: list[str] = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "black",
    "gray",
    "cyan",
    "magenta",
    "yellow",
]

# Line-style choices available in the GUI selector
ANNOTATION_LINESTYLES: list[str] = ["--", "-", ":", "-."]


def apply_annotations(ax: "Axes", overlays: list) -> None:  # noqa: C901
    """Apply a list of annotation overlays to a Matplotlib Axes.

    Parameters
    ----------
    ax:
        Target Matplotlib ``Axes`` object.
    overlays:
        Sequence of annotation dicts.  Each dict **must** contain a
        ``'type'`` key; see module docstring for required/optional keys
        per type.  Unknown types are silently skipped.
    """
    if not overlays:
        return

    for ann in overlays:
        t = ann.get("type")
        color: str = str(ann.get("color", "red"))
        label = ann.get("label") or None
        lw: float = float(ann.get("lw", 1.5))
        ls: str = str(ann.get("linestyle", "--"))
        alpha: float = float(ann.get("alpha", 0.3))

        if t == "hline":
            ax.axhline(
                float(ann["y"]),
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=label,
            )

        elif t == "vline":
            ax.axvline(
                float(ann["x"]),
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=label,
            )

        elif t == "hspan":
            ax.axhspan(
                float(ann["y1"]),
                float(ann["y2"]),
                alpha=alpha,
                color=color,
                label=label,
            )

        elif t == "vspan":
            ax.axvspan(
                float(ann["x1"]),
                float(ann["x2"]),
                alpha=alpha,
                color=color,
                label=label,
            )

        elif t == "text":
            kw: dict = {
                "fontsize": int(ann.get("fontsize", 10)),
                "color": color,
            }
            if ann.get("arrow"):
                ax.annotate(
                    str(ann["text"]),
                    xy=(float(ann["x"]), float(ann["y"])),
                    xytext=(
                        float(ann.get("tx", ann["x"])),
                        float(ann.get("ty", ann["y"])),
                    ),
                    arrowprops=dict(arrowstyle="->", color=color),
                    **kw,
                )
            else:
                ax.text(float(ann["x"]), float(ann["y"]), str(ann["text"]), **kw)

        elif t == "rectangle":
            from matplotlib.patches import Rectangle  # local import

            w = float(ann["x2"]) - float(ann["x1"])
            h = float(ann["y2"]) - float(ann["y1"])
            rect = Rectangle(
                (float(ann["x1"]), float(ann["y1"])),
                w,
                h,
                linewidth=lw,
                edgecolor=color,
                facecolor=ann.get("facecolor", "none"),
                alpha=alpha,
                label=label,
            )
            ax.add_patch(rect)

        elif t == "ellipse":
            from matplotlib.patches import Ellipse  # local import

            el = Ellipse(
                (float(ann["cx"]), float(ann["cy"])),
                float(ann["width"]),
                float(ann["height"]),
                linewidth=lw,
                edgecolor=color,
                facecolor=ann.get("facecolor", "none"),
                alpha=alpha,
                label=label,
            )
            ax.add_patch(el)

    # Refresh legend if any overlay carries a label
    if any(a.get("label") for a in overlays):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best")


def describe_overlay(ann: dict) -> str:
    """Return a human-readable one-line description of an annotation dict.

    Used in the Quick Plot annotations list UI.
    """
    t = ann.get("type", "unknown")
    color = ann.get("color", "red")
    label = ann.get("label", "")
    suffix = f" [{label}]" if label else ""

    if t == "hline":
        return f"H-line  y = {ann.get('y')}  ({color}){suffix}"
    if t == "vline":
        return f"V-line  x = {ann.get('x')}  ({color}){suffix}"
    if t == "hspan":
        return f"H-span  y = {ann.get('y1')} – {ann.get('y2')}  ({color}){suffix}"
    if t == "vspan":
        return f"V-span  x = {ann.get('x1')} – {ann.get('x2')}  ({color}){suffix}"
    if t == "text":
        return f"Text '{ann.get('text')}'  @ ({ann.get('x')}, {ann.get('y')}){suffix}"
    if t == "rectangle":
        return (
            f"Rect  ({ann.get('x1')}, {ann.get('y1')}) → "
            f"({ann.get('x2')}, {ann.get('y2')})  ({color}){suffix}"
        )
    if t == "ellipse":
        return (
            f"Ellipse  c=({ann.get('cx')}, {ann.get('cy')})  "
            f"w={ann.get('width')} h={ann.get('height')}  ({color}){suffix}"
        )
    return f"{t}{suffix}"
