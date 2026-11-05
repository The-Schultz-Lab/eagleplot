# Plottle in the browser (stlite) — throwaway prototype, not for merge

The tuned Streamlit app (branch `claude/proto-streamlit-quickplot`) running **entirely in the
browser**: [stlite](https://github.com/whitphx/stlite) runs Streamlit on Pyodide (Python
compiled to WebAssembly), so the site is static files and can be hosted on GitHub Pages
like the QuantUI viewer. Data never leaves the user's computer.

Build and try it locally:

    pip install -e ".[formats,nist]"
    python stlite/build_site.py site          # Pyodide loads from the jsDelivr CDN
    python -m http.server -d site 8000        # open http://localhost:8000 (file:// won't work)

`--pyodide-dir DIR` self-hosts an unpacked *full* Pyodide 314.0.3 distribution instead
(offline-capable, ~500 MB). The site itself is ~45 MB: the stlite runtime, Plottle as a zip,
and the pure-Python wheels (seaborn, plotly, openpyxl, …) so nothing comes from PyPI.

Browser-build differences: batch import from a folder is off (there is no local disk),
NIST lookups depend on the site allowing cross-origin requests, and plots are embedded
inline instead of served as media files.
