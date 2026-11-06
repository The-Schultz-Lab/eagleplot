"""Build the EaglePlot brand SVGs in brand/: the mark, the app icon and the lockups.

    pip install fonttools uharfbuzz
    python scripts/brand/build_brand.py FONT_DIR [OUT_DIR]     # OUT_DIR defaults to brand
    node scripts/brand/render_brand.js                         # then the PNGs and logo.ico

FONT_DIR holds Archivo's variable fonts from Google Fonts (SIL Open Font License):
Archivo[wdth,wght].ttf and Archivo-Italic[wdth,wght].ttf, saved as Archivo.ttf and
Archivo-Italic.ttf. Text is converted to outlines, so no output file needs the font.

The six head pieces below are also inlined in packaging/web/index.html (the loading
splash, which animates them one by one) together with the wordmark outlines, which this
script writes to build/brand-wordmark.json. Change the head here and there together.
"""

import io
import json
import sys
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

REPO = Path(__file__).resolve().parents[2]
FONT_DIR = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "brand"
OUT.mkdir(parents=True, exist_ok=True)

MAROON, GRAY, BEAK, OUTLINE, INK, ROSE = (
    "#8b2233",
    "#a2a9ad",
    "#d7dbde",
    "#24090f",
    "#1d2226",
    "#e0a3a3",
)

# The six pieces of the head, in drawing order: (name, path, fill, outline width).
# paint-order="stroke" puts half of each outline under the fill.
PIECES = [
    (
        "head",
        "M10 36C34 22 60 18 82 25C100 29 112 38 116 52C119 62 115 72 108 80C108 72 104 66 96 64"
        "L82 64C72 76 58 88 40 96L54 84L20 92L40 76L8 76L32 62L4 56L30 48Z",
        "#fff",
        9,
    ),
    ("gray", "M30 48L58 46C66 46 72 50 76 56C66 58 54 62 40 70L60 60L24 66Z", GRAY, 5.5),
    ("maroon", "M82 64C72 76 58 88 40 96L54 84L36 86C52 78 66 70 76 62Z", MAROON, 5.5),
    (
        "beak",
        "M84 26C100 29 112 38 116 52C119 62 115 72 108 80C108 72 104 66 96 64"
        "L84 64C88 58 90 50 88 42C87 36 86 30 84 26Z",
        BEAK,
        5.5,
    ),
    ("brow", "M54 36L84 32L80 40C72 40 62 40 54 36Z", OUTLINE, 5),
    ("eye", "M64 42C70 40 76 40 80 42C76 46 70 47 64 45Z", MAROON, 4.5),
]
# Extent of the head including its outline, in head units.
HEAD_BOX = (-0.5, 13.5, 124.0, 100.5)  # x0, y0, x1, y1


def head_svg(indent="  "):
    out = []
    for name, d, fill, sw in PIECES:
        out.append(
            f'{indent}<path data-piece="{name}" d="{d}" fill="{fill}" stroke="{OUTLINE}" '
            f'stroke-width="{sw}" stroke-linejoin="round" paint-order="stroke"/>'
        )
    return "\n".join(out)


def outline_text(font_file, text, size, wght, wdth, tracking=0.0):
    """Return (svg path d, advance width, (xMin, yMin, xMax, yMax)) with the baseline at y=0."""
    vf = TTFont(font_file)
    inst = instantiateVariableFont(vf, {"wght": wght, "wdth": wdth}, inplace=False)
    buf = io.BytesIO()
    inst.save(buf)
    data = buf.getvalue()
    font = TTFont(io.BytesIO(data))
    upem = font["head"].unitsPerEm
    glyphset = font.getGlyphSet()
    order = font.getGlyphOrder()

    hb_font = hb.Font(hb.Face(data))
    hb_buf = hb.Buffer()
    hb_buf.add_str(text)
    hb_buf.guess_segment_properties()
    hb.shape(hb_font, hb_buf, {"kern": True, "liga": True})

    s = size / upem
    pen = SVGPathPen(glyphset, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    bounds = BoundsPen(glyphset)
    x = 0.0
    for info, pos in zip(hb_buf.glyph_infos, hb_buf.glyph_positions):
        name = order[info.codepoint]
        t = (s, 0, 0, -s, (x + pos.x_offset) * s, -pos.y_offset * s)
        glyphset[name].draw(TransformPen(pen, t))
        glyphset[name].draw(TransformPen(bounds, t))
        x += pos.x_advance + tracking * upem
    adv = x * s - tracking * size
    return pen.getCommands(), adv, bounds.bounds


ITALIC = FONT_DIR / "Archivo-Italic.ttf"
UPRIGHT = FONT_DIR / "Archivo.ttf"

# Wordmark: Archivo Italic, width 115, weight 850. "Eagle" and "Plot" are separate paths
# so each can take its own colour.
WM_SIZE = 100
eagle_d, eagle_w, eagle_b = outline_text(ITALIC, "Eagle", WM_SIZE, 850, 115)
plot_d, plot_w, plot_b = outline_text(ITALIC, "Plot", WM_SIZE, 850, 115)
# Kerning between the two words is lost by shaping them separately; take it from the whole word.
_, whole_w, whole_b = outline_text(ITALIC, "EaglePlot", WM_SIZE, 850, 115)
plot_dx = whole_w - plot_w
cap_top = min(eagle_b[1], plot_b[1])  # most negative y = cap height (y is flipped)
descent = max(eagle_b[3], plot_b[3])  # "g" descender
wm_x0, wm_x1 = min(eagle_b[0], 0), plot_dx + plot_b[2]

lab_d, lab_w, lab_b = outline_text(UPRIGHT, "Schultz Lab", 30, 650, 105)
uni_d, uni_w, uni_b = outline_text(
    UPRIGHT, "NORTH CAROLINA CENTRAL UNIVERSITY", 15.5, 600, 100, tracking=0.14
)


def wordmark_group(eagle_fill, plot_fill, x, y):
    return (
        f'  <g transform="translate({x:.2f} {y:.2f})">\n'
        f'    <path d="{eagle_d}" fill="{eagle_fill}"/>\n'
        f'    <path transform="translate({plot_dx:.2f} 0)" d="{plot_d}" fill="{plot_fill}"/>\n'
        f"  </g>"
    )


SHADOW = (
    '  <defs>\n    <filter id="lift" x="-10%" y="-10%" width="130%" height="140%">\n'
    '      <feDropShadow dx="0" dy="2" stdDeviation="1.4"'
    ' flood-color="#24090f" flood-opacity=".28"/>\n'
    '      <feDropShadow dx="0" dy="6" stdDeviation="5"'
    ' flood-color="#24090f" flood-opacity=".18"/>\n'
    "    </filter>\n  </defs>\n"
)


def svg(w, h, body, title, vb=None):
    vb = vb or f"0 0 {w:.2f} {h:.2f}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{w:.0f}" height="{h:.0f}" '
        f'role="img" aria-label="{title}">\n  <title>{title}</title>\n{body}\n</svg>\n'
    )


def lockup(theme, endorsed=False):
    dark = theme == "dark"
    eagle_fill, plot_fill = ("#ffffff", GRAY) if dark else (MAROON, INK)
    pad = 24
    head_scale = 1.45  # head height on the canvas = cap height * ~1.5
    hx0, hy0, hx1, hy1 = HEAD_BOX
    head_w, head_h = (hx1 - hx0) * head_scale, (hy1 - hy0) * head_scale
    cap_h = -cap_top
    canvas_h = max(head_h, cap_h - (-descent)) + 2 * pad
    mid = canvas_h / 2
    head_x = pad - hx0 * head_scale
    head_y = mid - (hy0 + hy1) / 2 * head_scale
    wm_x = pad + head_w + 22 - wm_x0
    baseline = mid + cap_h / 2
    parts = []
    if not dark:
        parts.append(SHADOW)
    filt = ' filter="url(#lift)"' if not dark else ""
    parts.append(
        f'  <g transform="translate({head_x:.2f} {head_y:.2f}) scale({head_scale})"{filt}>\n'
        f"{head_svg('    ')}\n  </g>"
    )
    parts.append(wordmark_group(eagle_fill, plot_fill, wm_x, baseline))
    right = wm_x + wm_x1
    if endorsed:
        rule_x = right + 26
        line_c, lab_c, uni_c = (GRAY, "#ffffff", ROSE) if dark else (GRAY, INK, MAROON)
        tx = rule_x + 22
        text_w = max(lab_b[2], uni_b[2])
        parts.append(
            f'  <rect x="{rule_x:.2f}" y="{mid - 32:.2f}" width="3" height="64" rx="1.5"'
            f' fill="{line_c}"/>'
        )
        parts.append(
            f'  <path transform="translate({tx:.2f} {mid - 4:.2f})" d="{lab_d}" fill="{lab_c}"/>'
        )
        parts.append(
            f'  <path transform="translate({tx:.2f} {mid + 24:.2f})" d="{uni_d}" fill="{uni_c}"/>'
        )
        right = tx + text_w
    w = right + pad
    title = "EaglePlot" + (" · Schultz Lab, North Carolina Central University" if endorsed else "")
    return svg(w, canvas_h, "\n".join(parts), title)


# Mark: the head alone, square canvas, transparent.
hx0, hy0, hx1, hy1 = HEAD_BOX
side = 136
cx, cy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
mark = svg(
    1024, 1024, head_svg(), "EaglePlot", vb=f"{cx - side / 2:.2f} {cy - side / 2:.2f} {side} {side}"
)
(OUT / "logo.svg").write_text(mark)

# App icon: the head on a maroon rounded square.
icon_body = (
    f'  <rect width="128" height="128" rx="28" fill="{MAROON}"/>\n'
    f'  <g transform="translate(6 10) scale(.9)">\n{head_svg("    ")}\n  </g>'
)
(OUT / "icon.svg").write_text(svg(1024, 1024, icon_body, "EaglePlot", vb="0 0 128 128"))

for theme in ("light", "dark"):
    (OUT / f"lockup-{theme}.svg").write_text(lockup(theme))
    (OUT / f"lockup-endorsed-{theme}.svg").write_text(lockup(theme, endorsed=True))

# Wordmark alone, for the web splash (inline SVG there).
wm_h = -cap_top + descent
(REPO / "build").mkdir(exist_ok=True)
(REPO / "build" / "brand-wordmark.json").write_text(
    json.dumps(
        {
            "eagle": eagle_d,
            "plot": plot_d,
            "plot_dx": round(plot_dx, 2),
            "viewBox": [
                round(wm_x0, 2),
                round(cap_top, 2),
                round(wm_x1 - wm_x0, 2),
                round(wm_h, 2),
            ],
        }
    )
)
print(f"wrote {OUT} and build/brand-wordmark.json")
