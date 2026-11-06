# EaglePlot in the browser (stlite)

`build_site.py` builds a static web site that runs the full EaglePlot Streamlit app in the
browser, on [stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide/WebAssembly).
Nothing is installed and no data leaves the computer.

## Where it is published

`.github/workflows/docs.yml` builds the documentation site and this app into one GitHub
Pages site: the app is at **https://the-schultz-lab.github.io/eagleplot/app/**. Every pull
request that touches `eagleplot/`, the site or this folder builds it with CDN Pyodide and
runs the smoke test (below) in Chromium; pushes to `main` also deploy it.

## Build and serve

```bash
python packaging/web/build_site.py build/web          # Pyodide from the jsDelivr CDN (default)
python -m http.server -d build/web 8000               # then open http://localhost:8000/
```

Options:

| Option | Effect |
| --- | --- |
| `OUT_DIR` | Output directory. It is deleted and rebuilt. `build/` is git-ignored. |
| `--pyodide-dir DIR` | Copy an unpacked **full** Pyodide distribution into the site and load it from there (offline or filtered networks; about 530 MB on disk, browsers fetch only the packages they need). |
| `--pyodide-url URL` | Use a `pyodide.mjs` that is already served at this URL or site-relative path. |

The build needs network access to the npm registry (stlite) and PyPI (wheels). With CDN
Pyodide the site is about 45 MB: the stlite runtime (~33 MB, source maps stripped), the
bundled wheels (~11 MB) and the app zip (~0.7 MB).

Smoke test (Node with `playwright`, against a served site):

```bash
node scripts/web/smoke.js http://localhost:8000/
```

It opens every sidebar page with an empty session and again with an example dataset,
downloads a Quick Plot PNG built on click (this exercises the download workaround below),
and checks that no request goes anywhere but the site and the Pyodide CDN. It fails on any
Streamlit exception, uncaught page error, failed download or request to another origin.

## What goes into the site

| File | Content |
| --- | --- |
| `index.html` | Loading splash (below), the stlite mount call, and a download workaround (below) |
| `404.html` | A copy of `index.html`, for hosts that serve the nearest `404.html` (below) |
| `site-config.js` | Written by the build: entrypoint, packages, wheels, Pyodide URL, Streamlit flags, storage mount, environment |
| `eagleplot-app.zip` | `eagleplot/` (code, assets, gallery, example data) and `.streamlit/config.toml`; entrypoint `eagleplot/Home.py` |
| `wheels/` | Pure-Python packages not in Pyodide: seaborn, plotly, narwhals, openpyxl, et-xmlfile, jcamp (`.jdx` uploads), and Streamlit's python-multipart and itsdangerous |
| `stlite/` | The `@stlite/browser` build |
| `pyodide/` | Only with `--pyodide-dir` |

numpy, scipy, pandas, matplotlib, PyYAML and Pillow come from the Pyodide distribution.
Optional extras (h5py, xarray, netCDF4, statsmodels, requests) are not loaded.

## Pins (they move together)

| Component | Version |
| --- | --- |
| `@stlite/browser` | 1.12.0 |
| Streamlit (bundled by stlite) | 1.65 |
| Pyodide | 314.0.3 (Python 3.14) |

Upgrading one means checking the other two. `pyodideUrl` must point at **`pyodide.mjs`**:
stlite's worker is an ES module and cannot load the classic `pyodide.js`.

## Why `file://` does not work

Opening `index.html` from disk fails: the app runs in a module Web Worker and loads the
app zip, wheels and Pyodide with `fetch()`, and browsers block both for `file://` pages.
Serve the directory over HTTP (`python -m http.server`, GitHub Pages, any static host).

## Loading splash

`index.html` draws the splash before anything else loads: a classic inline script, inline
CSS and SVG, and no other files, so it is on screen while `stlite.js` and Pyodide download.

- **The logo.** The pieces of `eagleplot/assets/logo.svg` (three rings, the axes, four bars,
  the line, four points) are inlined as separate `<g class="ep-pc">` groups. Each carries
  `data-from="x y rotate scale"` (its off-screen start, in logo units) and `data-order`
  (it leaves at order × 120 ms), and flies in with the Web Animations API; the points land
  without overshoot (`data-ease`). While Python loads, a highlight sweeps across the disc
  and a pulse runs along the points. If the logo changes, update the pieces here too. No
  web fonts: the smoke test fails on requests to other origins.
- **Steps.** `window.eaglePlotSplash` exposes `setStage(i)`, `ready()` and `fail(message)`.
  The module script calls them from stlite's kernel events: `loadProgress` messages are
  matched by keyword (`/pyodide/`, `/install|package/`, `/streamlit|server|boot/`),
  `loadFinished` is the last step, and `loadError` shows the message and a Reload button.
  Steps only move forward, so a reworded stlite message leaves the step where it is. The
  events come from `_kernel`, which stlite calls unstable: re-check them when upgrading.
- **Leaving.** The existing sidebar observer calls `ready()`, which waits for the logo to
  finish building, fades the splash out over 450 ms and removes `#loading`. It also sets
  `localStorage["eagleplot.started"]`, which hides the first-visit download note next time.
- **Reduced motion.** With `prefers-reduced-motion`, the logo is drawn complete and nothing
  moves.

The smoke test checks that the splash shows a step, reaches the last one and is gone once
the app is up.

## Browser-specific behaviour

- **Downloads.** Streamlit 1.65 starts a deferred download
  (`st.download_button(data=callable)`) at a placeholder origin, `https://stlite.invalid`,
  that stlite 1.12 does not route to the Python worker. The tab then navigates to a dead
  URL and the session is lost. `index.html` routes those requests through the worker.
  Remove the workaround once stlite handles deferred downloads itself.
- **Storage.** `/home/pyodide/.eagleplot` is mounted on IndexedDB (stlite
  `idbfsMountpoints`) and survives a reload. `EAGLEPLOT_CONFIG_DIR` points at it, so
  `config.json` (Settings defaults, Quick Plot presets, Batch workflows) is kept in the
  browser. Other files the app writes are in memory and are lost on reload.
- **What the app turns off** (`eagleplot/utils/runtime.py`, `capabilities()`): folder
  import, NIST fetching (no CORS; the page links to NIST and explains how to load the
  downloaded `.jdx`), and the Settings theme form (the theme comes from `site-config.js`).
  Errors about a missing optional package say it isn't in this version instead of
  suggesting `pip install`.
- **Reloading a page other than Home.** Each Streamlit page has its own path
  (`…/app/settings`), which is not a file in the site. GitHub Pages answers a missing
  path with the `404.html` at the root of the *whole* site, so
  `.github/workflows/docs.yml` copies `pages-404.html` there: under `…/app/` it sends the
  browser to the app (which opens on Home); anywhere else it shows "page not found". The
  build also writes `app/404.html` (a copy of `index.html`) for hosts that use the
  nearest `404.html`, such as Cloudflare Pages. `python -m http.server` has neither:
  reload from the app root when testing locally.
- **Startup** takes about 15 s on every visit (Python start-up and imports), longer on
  the first visit over a slow network.
