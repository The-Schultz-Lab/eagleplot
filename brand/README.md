# EaglePlot brand files

| File | Use it for |
| --- | --- |
| `logo.svg` | The eagle head on its own, transparent. Master for `eagleplot/assets/logo.png`. |
| `icon.svg` | The head on a maroon rounded square: app icon, favicon. Master for `eagleplot/assets/icon.png` and `logo.ico`. |
| `lockup-light.svg`, `lockup-light@2x.png` | Head and wordmark on white or light backgrounds. |
| `lockup-dark.svg`, `lockup-dark@2x.png` | Head and wordmark on dark backgrounds, such as the app's own. |
| `lockup-endorsed-*.svg`, `*@2x.png` | The same, followed by "Schultz Lab · North Carolina Central University". |

Use the SVGs wherever you can. The PNGs are for tools that cannot place SVG (some slide and
poster software). All text is converted to outlines, so no font is needed.

## Colours

| Name | Hex | Where |
| --- | --- | --- |
| Maroon | `#8B2233` | Throat, eye, the "Eagle" of the wordmark on light backgrounds, app-icon tile |
| Gray | `#A2A9AD` | Feather band, the "Plot" of the wordmark on dark backgrounds |
| Beak | `#D7DBDE` | Beak |
| Outline | `#24090F` | Outlines and brow |
| Ink | `#1D2226` | The "Plot" of the wordmark on light backgrounds |

Maroon and gray match NCCU's (sampled from `eagleplot/assets/nccu-horiz-logo.png`). The
wordmark is set in Archivo Italic (width 115, weight 850); the endorsement line in Archivo.
Archivo is by Omnibus-Type, under the SIL Open Font License.

## Rules

- Keep the head's outline. On white, the head is a white shape: the outline is what holds it.
  The light lockups add a soft drop shadow for the same reason.
- Don't recolour, stretch or redraw the head, and don't put it inside other emblems.
- Keep NCCU's own logos separate. Show the university's marks as NCCU supplies them, next to
  EaglePlot's, never merged with it.

## Regenerating

The head is defined once, in `scripts/brand/build_brand.py` (six pieces, drawn in order).
After changing it:

```bash
pip install fonttools uharfbuzz
python scripts/brand/build_brand.py path/to/archivo-fonts   # SVGs in brand/
node scripts/brand/render_brand.js                          # PNGs and logo.ico
```

The web app's loading splash (`packaging/web/index.html`) inlines the same six pieces and
the wordmark outlines (`build/brand-wordmark.json`), so update it too.
