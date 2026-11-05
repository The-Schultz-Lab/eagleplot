# Quick Plot latency benchmarks

Scripts behind the 2026-11 Quick Plot speed work (roadmap D1). They time what a user
feels: a setting change until the plot on screen shows it. The **Benchmark** workflow
(`.github/workflows/bench.yml`, run manually from the Actions tab) runs them in Chromium
and writes the numbers to the run summary.

Run locally (Node with `playwright` installed, app running on `PORT`):

```bash
streamlit run eagleplot/Home.py --server.headless true --server.port 8501 &
node scripts/bench/st_e2e.js 8501 quickplot --live   # JSON timings; screenshot to $OUT
node scripts/bench/features.js 8501                   # export download + click-to-mark still work
python scripts/bench/apptest_bench.py . eagleplot/pages/2_Quick_Plot.py   # server-side cost per rerun
```

| Script | Measures |
| --- | --- |
| `st_e2e.js PORT LABEL [--live]` | Upload `ir_spectrum.csv`, Matplotlib line plot, then "change title → visible on plot" ×6 and Generate ×3. `--live` for pages with live preview (no Generate click needed to see a change) |
| `features.js [PORT]` | A PNG 300 dpi export downloads, and clicking a Plotly point adds a marker |
| `apptest_bench.py REPO PAGE` | Server-side time per rerun via `streamlit.testing` (full reruns only, so it cannot show `st.fragment` savings) |

## Pitfalls (see also the planning notes on latency measurement)

- With live preview on, **Generate** redraws identical pixels and the image URL does not
  change, so Generate is timed until the app is idle, not until the image changes. Only
  "change → visible on plot" is comparable across page variants.
- Idle detection waits for Streamlit's "running" status widget to disappear. The stlite
  (browser) build never shows that widget.
- The first run after a server start is slower (imports, caches). Run twice and use the
  second result.
