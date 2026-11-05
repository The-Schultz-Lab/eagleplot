"""Molecular Visualization page.

Upload a Gaussian (.log), ORCA (.out), or Molden (.molden) output file,
parse the vibrational frequency data, and visualize the molecular structure
and normal mode displacement vectors interactively in 3D using Plotly.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import os

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.utils.figures import LAZY_DOWNLOAD  # noqa: E402
from eagleplot.utils.session_state import initialize_session_state
from eagleplot.molecular import (
    VibrationalData,
    build_molecule_figure,
    parse_vibrations,
)

initialize_session_state()

st.title("Molecular Visualization")
st.caption(
    "Upload a quantum chemistry output file to visualize molecular structure "
    "and normal-mode vibrational displacements in 3D."
)

# ─────────────────────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a vibrational frequency output file",
    type=["log", "out", "molden"],
    help="Supported formats: Gaussian .log/.out, ORCA .out, Molden .molden",
)

if uploaded is None:
    st.info("Upload a Gaussian `.log`, ORCA `.out`, or Molden `.molden` file to begin.")
    st.stop()

# ── Parse the uploaded file ────────────────────────────────────────────────
ext = Path(uploaded.name).suffix.lower()
suffix_map = {".log": ".log", ".out": ".out", ".molden": ".molden"}
file_suffix = suffix_map.get(ext, ".out")

tmp_path = None
vib_data: VibrationalData | None = st.session_state.mol_vib_data

# Re-parse if the uploaded filename changed (or first load).
if vib_data is None or st.session_state.get("mol_vib_filename") != uploaded.name:
    with tempfile.NamedTemporaryFile(suffix=file_suffix, mode="wb", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        vib_data = parse_vibrations(tmp_path)
        st.session_state.mol_vib_data = vib_data
        st.session_state["mol_vib_filename"] = uploaded.name
        st.success(
            f"Parsed **{uploaded.name}** — {vib_data.program.title()} format, "
            f"{len(vib_data.atomic_numbers)} atoms, {len(vib_data.modes)} modes."
        )
    except Exception as exc:
        st.error(f"Failed to parse file: {exc}")
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        st.stop()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if vib_data is None:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Vibrational Frequency Summary", expanded=False):
    if vib_data.modes:
        freq_df = pd.DataFrame(
            {
                "Mode #": [m.mode_number for m in vib_data.modes],
                "Frequency (cm⁻¹)": [m.frequency for m in vib_data.modes],
                "IR Intensity (km/mol)": [
                    (f"{m.ir_intensity:.2f}" if m.ir_intensity is not None else "—")
                    for m in vib_data.modes
                ],
                "Imaginary": ["Yes" if m.is_imaginary else "No" for m in vib_data.modes],
            }
        )
        st.dataframe(freq_df, width="stretch")
    else:
        st.info("No vibrational modes found in the file.")

# ─────────────────────────────────────────────────────────────────────────────
# Visualization controls
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("3D Molecular Structure")

show_arrows = st.checkbox("Show vibrational displacement arrows", value=False)

mode_number = None
arrow_color = "red"
arrow_scale = 1.0
amplitude = 1.0

if show_arrows and vib_data.modes:
    mode_numbers = [m.mode_number for m in vib_data.modes]
    freqs = [m.frequency for m in vib_data.modes]
    mode_labels = [f"Mode {n}: {f:.1f} cm⁻¹" for n, f in zip(mode_numbers, freqs)]

    selected_label = st.selectbox("Vibrational mode", mode_labels, key="mol_mode_sel")
    mode_number = mode_numbers[mode_labels.index(selected_label)]

    col1, col2, col3 = st.columns(3)
    with col1:
        arrow_color = st.color_picker("Arrow color", value="#FF0000", key="mol_arrow_color")
    with col2:
        arrow_scale = st.slider(
            "Arrow visual scale",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="mol_arrow_scale",
        )
    with col3:
        amplitude = st.slider(
            "Displacement amplitude",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="mol_amplitude",
        )
elif show_arrows and not vib_data.modes:
    st.warning("No vibrational modes available in this file.")

# ─────────────────────────────────────────────────────────────────────────────
# Build and display figure
# ─────────────────────────────────────────────────────────────────────────────
try:
    fig = build_molecule_figure(
        vib_data,
        mode_number=mode_number,
        arrow_color=arrow_color,
        arrow_scale=float(arrow_scale),
        amplitude=float(amplitude),
    )
    title_text = uploaded.name
    if mode_number is not None:
        mode_obj = vib_data.get_mode(mode_number)
        if mode_obj:
            title_text += f" — Mode {mode_number} ({mode_obj.frequency:.1f} cm⁻¹)"
    fig.update_layout(title=title_text, height=600)
    st.plotly_chart(fig, width="stretch")
except Exception as exc:
    st.error(f"Error building figure: {exc}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Export")

col1, col2 = st.columns(2)


def _freq_table_csv(modes) -> bytes:
    return (
        pd.DataFrame(
            {
                "mode": [m.mode_number for m in modes],
                "frequency_cm-1": [m.frequency for m in modes],
                "ir_intensity_km_per_mol": [
                    (m.ir_intensity if m.ir_intensity is not None else float("nan")) for m in modes
                ],
                "is_imaginary": [m.is_imaginary for m in modes],
            }
        )
        .to_csv(index=False)
        .encode()
    )


# On Streamlit >= 1.52 each export is one click: the file is built when Download is
# pressed. Older Streamlit keeps the two-step "prepare, then download" flow, so no
# export is built on every rerun there either (roadmap D1.2).
with col1:
    if LAZY_DOWNLOAD:
        st.download_button(
            "Download interactive HTML",
            data=lambda f=fig: f.to_html(full_html=True, include_plotlyjs="cdn"),
            file_name=f"{Path(uploaded.name).stem}_mol.html",
            mime="text/html",
            key="mol_dl_html",
        )
    elif st.button("Export as interactive HTML", key="mol_export_html"):
        try:
            html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
            st.download_button(
                "Download HTML",
                data=html_str,
                file_name=f"{Path(uploaded.name).stem}_mol.html",
                mime="text/html",
                key="mol_dl_html",
            )
        except Exception as exc:
            st.error(str(exc))

with col2:
    if vib_data.modes and LAZY_DOWNLOAD:
        st.download_button(
            "Download frequency table (CSV)",
            data=lambda modes=list(vib_data.modes): _freq_table_csv(modes),
            file_name=f"{Path(uploaded.name).stem}_frequencies.csv",
            mime="text/csv",
            key="mol_dl_csv_btn",
        )
    elif vib_data.modes and st.button("Download frequency table (CSV)", key="mol_dl_csv"):
        st.download_button(
            "Download CSV",
            data=_freq_table_csv(vib_data.modes),
            file_name=f"{Path(uploaded.name).stem}_frequencies.csv",
            mime="text/csv",
            key="mol_dl_csv_btn",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Atom coordinates table
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Atomic Coordinates", expanded=False):
    from eagleplot.molecular.atom_data import get_atom_symbol

    coord_df = pd.DataFrame(
        {
            "Atom": [get_atom_symbol(z) for z in vib_data.atomic_numbers],
            "X (Å)": vib_data.coordinates[:, 0],
            "Y (Å)": vib_data.coordinates[:, 1],
            "Z (Å)": vib_data.coordinates[:, 2],
        }
    )
    st.dataframe(coord_df, width="stretch")
