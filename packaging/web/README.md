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
| `manifest.webmanifest`, `icons/` | Web app manifest and icons: the app can be installed (below) |
| `sw.js` | Service worker: caches the app and Pyodide, so it opens offline (below) |
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

## Installable app and offline use

`manifest.webmanifest` (written by the build, icons in `icons/`) and the service worker
`sw.js` make the app installable: Chrome and Edge show **Install EaglePlot** in the
address bar or menu. It then opens in its own window from a desktop or Start-menu icon,
with no admin rights. After one online visit it also opens offline.

- **First visit:** `index.html` registers `sw.js` and waits (at most 3 s) until it
  controls the page before starting Python, so this visit's downloads, the Python
  worker's included, pass through it and are cached. The page and the stlite runtime load
  before that; the page sends their URLs to the worker to cache.
- **Site files:** network first (a new deploy shows up at once), the cached copy offline.
  The cache is per build (`build_site.py` stamps `sw.js` with a hash of the build), and
  older builds' caches are removed when a new one activates.
- **Pyodide from the CDN:** cache first; its URLs contain the version and never change.
- `window.eaglePlotOffline` says what happened: `ready`, `next-visit` (the worker didn't
  take over within 3 s), `failed`, or `unsupported` (no service workers, or not https).
- `scripts/web/smoke.js` checks both: Chromium's installability check, then a reload
  with the network off.
- The icons are PNGs made from `eagleplot/assets/logo.png` (192 and 512 px, maskable
  versions on the dark background, a 180 px Apple touch icon). Regenerate them if the
  logo changes.

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
- **Start-up** (roadmap D4.1). Measured locally (self-hosted Pyodide, no network), the
  app is ready in about 7 s:

  | Stage | Time |
  | --- | --- |
  | Loading Pyodide | ~1.5 s |
  | Installing packages (numpy, pandas, matplotlib, Pillow, PyYAML, the wheels) | ~3.2 s |
  | Loading Streamlit | ~1.0 s |
  | Home.py | ~0.5 s |

  What keeps it there:
  - **scipy is not loaded at start-up** (it alone added ~2 s). Home starts loading it in
    the background (`eagleplot.utils.runtime.DEFERRED_PACKAGES`); a page that imports it
    shows "Loading the science libraries…" until it is there. Home and Quick Plot don't
    wait.
  - **Home doesn't import pandas** (~2.3 s under Pyodide): `eagleplot.utils` loads its
    submodules on first use. After Home is drawn, pandas, matplotlib and the plotting
    modules are imported in the background (`runtime.WARM_IMPORTS`), one per event-loop
    turn, so a click is handled between two imports.
  - **`eagleplot.plotting` imports seaborn and plotly when a plot type needs them**, not
    at import.

  Keep it that way: `tests/test_web_build.py` checks that Home doesn't import pandas and
  that `eagleplot.plotting` doesn't import seaborn, plotly or scipy.
  `node scripts/web/boot_timing.js URL` prints the start-up timeline (cold and warm), and
  the docs workflow reports it with the real CDN on every run.
