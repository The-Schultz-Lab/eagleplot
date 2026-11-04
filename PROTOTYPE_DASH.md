# Dash Quick Plot prototype (throwaway, not for merge)

This branch carries a cut-down port of the April 2026 Dash app (`dash_app/`), so the Dash version
of **Quick Plot** can be compared side by side with the Streamlit one (smoothness, speed, look and feel).
It is a comparison prototype only: it is not meant to be merged or maintained.

Pages: **Home**, **Data Upload** (file upload + built-in example datasets), **Quick Plot**.

Run from the repo root:

    pip install -e ".[formats,nist]" dash dash-bootstrap-components
    python -m dash_app.app

Then open http://127.0.0.1:8050. Set `PLOTTLE_DEBUG=1` for Dash dev tools.
