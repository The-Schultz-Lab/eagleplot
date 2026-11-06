# EaglePlot in the browser (stlite)

`build_site.py` builds a static web site that runs the full EaglePlot Streamlit app in the
browser, on [stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide/WebAssembly).
Nothing is installed and no data leaves the computer.

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

It opens every sidebar page with an empty session and again with an example dataset, and
fails on any Streamlit exception or uncaught page error.

## What goes into the site

| File | Content |
| --- | --- |
| `index.html` | Loading splash, the stlite mount call, and a download workaround (below) |
| `site-config.js` | Written by the build: entrypoint, packages, wheels, Pyodide URL, Streamlit flags, storage mount, environment |
| `eagleplot-app.zip` | `eagleplot/` (code, assets, gallery, example data) and `.streamlit/config.toml`; entrypoint `eagleplot/Home.py` |
| `wheels/` | Pure-Python packages not in Pyodide: seaborn, plotly, narwhals, openpyxl, et-xmlfile, and Streamlit's python-multipart and itsdangerous |
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

## Browser-specific behaviour

- **Downloads.** Streamlit 1.65 starts a deferred download
  (`st.download_button(data=callable)`) at a placeholder origin, `https://stlite.invalid`,
  that stlite 1.12 does not route to the Python worker. The tab then navigates to a dead
  URL and the session is lost. `index.html` routes those requests through the worker.
  Remove the workaround once stlite handles deferred downloads itself.
- **Storage.** `/home/pyodide/.eagleplot` is mounted on IndexedDB (stlite
  `idbfsMountpoints`) and survives a reload; `EAGLEPLOT_CONFIG_DIR` points at it. Other
  files the app writes (for example `config.json` next to the app) are in memory and are
  lost on reload.
- **Startup** takes about 15 s on every visit (Python start-up and imports), longer on
  the first visit over a slow network.
