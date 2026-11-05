# Quick Plot latency benchmarks (prototype tooling)

Playwright scripts used on 2026-11-04/05 to compare Quick Plot variants. They need
Node with `playwright` installed and a running app. Screenshots go to `$OUT` (default `.`).

| Script | Measures |
| --- | --- |
| `apptest_bench.py REPO PAGE` | Server-side cost of each interaction via `streamlit.testing` (full reruns only) |
| `st_e2e.js PORT LABEL [--live]` | Browser end to end on `streamlit run`: change title → visible on plot, Generate |
| `stlite_e2e.js PORT LABEL --live` | Same scenario against the static stlite site (starts at `/`) |
| `dash_e2e.js` | Same scenario against the Dash prototype on port 8050 |
| `features.js [PORT]` | Export download and Plotly click-to-mark still work |

Pitfalls: in live-preview mode an identical figure keeps the same image URL, so time
Generate by app idle, not image change; and stlite does not show the "running" status
widget, so idle detection passes early there. Only the "change → visible on plot"
metric (waits for the image itself to change) is comparable across all variants.
