# Tuned Streamlit Quick Plot prototype (throwaway, not for merge)

This branch changes only `plottle/pages/2_Quick_Plot.py`, to show how fast and smooth the
existing Streamlit Quick Plot can get without changing frameworks. Compare it with `main`
and with the Dash prototype on branch `claude/proto-dash-quickplot`.

What changed:

- **Settings left, plot right.** The plot stays in view while the settings scroll, so every
  change is visible immediately.
- **Live preview.** The plot redraws as settings change. *Generate Plot* still saves to the
  plot history.
- **Partial reruns.** The configuration and plot area is an `st.fragment`, so a change there
  reruns only that part of the page.
- **Exports on demand.** `main` re-encodes five export formats (including a 600 dpi PNG) on
  every interaction. Here the format you pick is encoded only when you click Download
  (Streamlit ≥ 1.52), or once per figure on older versions.
- Click-to-mark shows the marker immediately, and old figures are closed when replaced.

Run it like `main`: `pip install -e ".[formats,nist]"`, then `streamlit run plottle/Home.py`.
Load a dataset on **Data Upload**, then open **Plot → Basic**.
