"""Spectroscopy Analysis page.

Four-tab spectroscopy analysis page: IR/Raman, NMR, UV-Vis, and Mass Spec.
All operations are non-destructive and save results as new named datasets.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eagleplot.utils.session_state import initialize_session_state, add_dataset
from eagleplot.spectroscopy import (
    absorbance_to_transmittance,
    apply_line_broadening,
    assign_bands,
    atr_correction,
    beer_lambert,
    calibrate_ppm_axis,
    centroid_spectrum,
    find_mz_peaks,
    integrate_nmr_regions,
    molar_absorptivity_series,
    nmr_fft,
    pick_nmr_peaks,
    remove_cosmic_rays,
    spectral_overlap_integral,
    spectral_subtraction,
    transmittance_to_absorbance,
    zero_fill,
)

initialize_session_state()

st.title("Spectroscopy")
st.caption("Spectral processing and analysis for IR/Raman, NMR, UV-Vis, and Mass Spectrometry.")

# ── Dataset selector ───────────────────────────────────────────────────────────
dataset_names = list(st.session_state.datasets.keys())
if not dataset_names:
    st.warning("No dataset loaded. Please upload data on the **Data Upload** page first.")
    st.stop()

selected_name = st.selectbox("Source dataset", dataset_names, key="spec_source_ds")
data = st.session_state.datasets[selected_name]

if not isinstance(data, pd.DataFrame):
    st.error(
        "Spectroscopy Analysis requires a tabular (DataFrame) dataset. "
        "Please load a CSV or Excel file."
    )
    st.stop()

numeric_cols = data.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.warning("No numeric columns found in the current dataset.")
    st.stop()

st.caption(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")


def _save(result_df: pd.DataFrame, suffix: str) -> None:
    base = selected_name.rsplit(".", 1)[0]
    new_name = f"{base}_{suffix}"
    add_dataset(new_name, result_df)
    st.success(f"Saved as **{new_name}**")
    st.dataframe(result_df.head(20), width="stretch")


# ── Tabs ────────────────────────────────────────────────────────────────────
tab_ir, tab_nmr, tab_uv, tab_ms = st.tabs(["IR / Raman", "NMR", "UV-Vis", "Mass Spec"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — IR / Raman
# ─────────────────────────────────────────────────────────────────────────────
with tab_ir:
    st.header("IR / Raman Analysis")

    col1, col2 = st.columns(2)
    with col1:
        wn_col = st.selectbox("Wavenumber column (cm⁻¹)", numeric_cols, key="ir_wn")
    with col2:
        ab_col_ir = st.selectbox(
            "Absorbance / Intensity column",
            numeric_cols,
            index=min(1, len(numeric_cols) - 1),
            key="ir_ab",
        )

    ir_op = st.radio(
        "Operation",
        [
            "Absorbance ↔ Transmittance",
            "ATR Correction",
            "Spectral Subtraction",
            "Remove Cosmic Rays",
            "Band Assignment",
        ],
        horizontal=True,
        key="ir_op",
    )

    wn_arr = data[wn_col].dropna().to_numpy(dtype=float)
    ab_arr = data[ab_col_ir].dropna().to_numpy(dtype=float)
    n_ir = min(len(wn_arr), len(ab_arr))
    wn_arr, ab_arr = wn_arr[:n_ir], ab_arr[:n_ir]

    # ── Absorbance ↔ Transmittance ─────────────────────────────────────────
    if ir_op == "Absorbance ↔ Transmittance":
        direction = st.radio(
            "Direction",
            ["Absorbance → Transmittance (%)", "Transmittance (%) → Absorbance"],
            horizontal=True,
            key="ir_conv_dir",
        )
        if st.button("Convert", key="ir_conv_btn"):
            try:
                if direction == "Absorbance → Transmittance (%)":
                    result_arr = absorbance_to_transmittance(ab_arr)
                    result_label = "Transmittance (%)"
                else:
                    result_arr = transmittance_to_absorbance(ab_arr)
                    result_label = "Absorbance"
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(wn_arr, result_arr)
                ax.set_xlabel(wn_col)
                ax.set_ylabel(result_label)
                ax.set_title("Converted Spectrum")
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                out_df = pd.DataFrame({wn_col: wn_arr, result_label: result_arr})
                _save(out_df, f"conv_{ab_col_ir}")
            except Exception as exc:
                st.error(str(exc))

    # ── ATR Correction ─────────────────────────────────────────────────────
    elif ir_op == "ATR Correction":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_atr = st.number_input(
                "Crystal refractive index (n_atr)",
                value=1.5,
                min_value=1.0,
                step=0.1,
                key="ir_natr",
            )
        with col2:
            angle_deg = st.number_input(
                "Angle of incidence (°)",
                value=45.0,
                min_value=1.0,
                max_value=89.0,
                step=1.0,
                key="ir_angle",
            )
        with col3:
            n_sample = st.number_input(
                "Sample refractive index",
                value=1.0,
                min_value=0.1,
                step=0.1,
                key="ir_nsamp",
            )
        if st.button("Apply ATR Correction", key="ir_atr_btn"):
            try:
                result_arr = atr_correction(
                    wn_arr,
                    ab_arr,
                    n_atr=float(n_atr),
                    angle_deg=float(angle_deg),
                    n_sample=float(n_sample),
                )
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(wn_arr, ab_arr)
                axes[0].set_title("Original")
                axes[0].set_xlabel(wn_col)
                axes[1].plot(wn_arr, result_arr, color="darkorange")
                axes[1].set_title("ATR Corrected")
                axes[1].set_xlabel(wn_col)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                out_df = pd.DataFrame({wn_col: wn_arr, f"{ab_col_ir}_atr": result_arr})
                _save(out_df, f"atr_{ab_col_ir}")
            except Exception as exc:
                st.error(str(exc))

    # ── Spectral Subtraction ───────────────────────────────────────────────
    elif ir_op == "Spectral Subtraction":
        col1, col2 = st.columns(2)
        with col1:
            ref_col = st.selectbox("Reference column", numeric_cols, key="ir_ref")
        with col2:
            scale = st.number_input("Scale factor", value=1.0, step=0.05, key="ir_scale")
        if st.button("Subtract", key="ir_sub_btn"):
            try:
                ref_arr = data[ref_col].dropna().to_numpy(dtype=float)[:n_ir]
                result_arr = spectral_subtraction(wn_arr, ab_arr, ref_arr, scale=float(scale))
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(wn_arr, result_arr)
                ax.set_xlabel(wn_col)
                ax.set_ylabel("Difference")
                ax.set_title(f"{ab_col_ir} − {scale:.2f}×{ref_col}")
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                out_df = pd.DataFrame({wn_col: wn_arr, "difference": result_arr})
                _save(out_df, f"sub_{ab_col_ir}")
            except Exception as exc:
                st.error(str(exc))

    # ── Remove Cosmic Rays ─────────────────────────────────────────────────
    elif ir_op == "Remove Cosmic Rays":
        col1, col2 = st.columns(2)
        with col1:
            thresh_sigma = st.number_input(
                "Threshold (σ)", value=5.0, min_value=1.0, step=0.5, key="ir_cr_thresh"
            )
        with col2:
            cr_window = st.number_input(
                "Window half-width", value=5, min_value=1, step=1, key="ir_cr_window"
            )
        if st.button("Remove Cosmic Rays", key="ir_cr_btn"):
            result_arr = remove_cosmic_rays(
                ab_arr, threshold_sigma=float(thresh_sigma), window=int(cr_window)
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(wn_arr, ab_arr, alpha=0.5, label="Original")
            ax.plot(wn_arr, result_arr, label="Cleaned")
            ax.set_xlabel(wn_col)
            ax.legend()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            out_df = pd.DataFrame({wn_col: wn_arr, f"{ab_col_ir}_cleaned": result_arr})
            _save(out_df, f"cr_{ab_col_ir}")

    # ── Band Assignment ────────────────────────────────────────────────────
    elif ir_op == "Band Assignment":
        threshold_ab = st.number_input(
            "Min absorbance for peaks",
            value=0.05,
            min_value=0.0,
            step=0.01,
            key="ir_ba_thresh",
        )
        if st.button("Assign Bands", key="ir_ba_btn"):
            from scipy.signal import find_peaks as _sp_peaks

            peak_idx, _ = _sp_peaks(ab_arr, height=float(threshold_ab))
            if len(peak_idx) == 0:
                st.info("No bands found above threshold.")
            else:
                rows = []
                for idx in peak_idx:
                    peak_wn = float(wn_arr[idx])
                    groups = assign_bands(peak_wn)
                    rows.append(
                        {
                            "Wavenumber (cm⁻¹)": f"{peak_wn:.1f}",
                            "Absorbance": f"{ab_arr[idx]:.4f}",
                            "Possible Assignments": (", ".join(groups) if groups else "—"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")

    # ── NIST WebBook Lookup ────────────────────────────────────────────────────
    with st.expander("🔍 NIST WebBook Lookup"):
        st.markdown(
            "Download an IR spectrum directly from the "
            "[NIST Chemistry WebBook](https://webbook.nist.gov) by CAS number."
        )
        col_nist1, col_nist2 = st.columns([3, 1])
        with col_nist1:
            nist_cas = st.text_input(
                "CAS Registry Number",
                value="64-17-5",
                key="nist_cas",
                help="Example: 64-17-5 (ethanol), 67-63-0 (isopropanol)",
            )
        with col_nist2:
            nist_idx = st.number_input(
                "Spectrum index",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="nist_idx",
                help="0 = first available IR spectrum",
            )
        from eagleplot.nist import fetch_ir_spectrum as _nist_fetch, get_compound_url as _nist_url

        nist_page_url = _nist_url(nist_cas) if nist_cas.strip() else ""
        if nist_page_url:
            st.markdown(f"NIST page: [{nist_cas}]({nist_page_url})")
        if st.button("Fetch IR spectrum from NIST", key="nist_fetch_btn", type="primary"):
            if not nist_cas.strip():
                st.warning("Please enter a CAS Registry Number.")
            else:
                try:
                    nist_df = _nist_fetch(nist_cas.strip(), index=int(nist_idx))
                    title = nist_df.attrs.get("TITLE", nist_cas.strip())
                    ds_name = (
                        f"NIST_{nist_cas.strip().replace('-', '')}_{title[:20].replace(' ', '_')}"
                    )
                    add_dataset(ds_name, nist_df)
                    st.success(f"Spectrum fetched and saved as **{ds_name}**")
                    st.dataframe(nist_df.head(20), width="stretch")
                    fig_nist, ax_nist = plt.subplots(figsize=(8, 4))
                    ax_nist.plot(nist_df["x"], nist_df["y"], color="steelblue", linewidth=1.2)
                    ax_nist.set_xlabel("Wavenumber (cm⁻¹)")
                    ax_nist.set_ylabel("Intensity")
                    ax_nist.set_title(f"IR Spectrum: {title}")
                    ax_nist.invert_xaxis()
                    st.pyplot(fig_nist, use_container_width=False)
                    plt.close(fig_nist)
                except ImportError as e:
                    st.warning(str(e))
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — NMR
# ─────────────────────────────────────────────────────────────────────────────
with tab_nmr:
    st.header("NMR Processing")

    col1, col2 = st.columns(2)
    with col1:
        fid_col = st.selectbox("FID / Spectrum column", numeric_cols, key="nmr_fid")
    with col2:
        nmr_op = st.selectbox(
            "Operation",
            [
                "Line Broadening + FFT",
                "Peak Picking",
                "Region Integration",
                "Calibrate Hz → ppm",
            ],
            key="nmr_op",
        )

    fid_arr = data[fid_col].dropna().to_numpy(dtype=float)

    # ── Line Broadening + FFT ──────────────────────────────────────────────
    if nmr_op == "Line Broadening + FFT":
        col1, col2, col3 = st.columns(3)
        with col1:
            dt = st.number_input(
                "Dwell time (s)",
                value=0.0001,
                format="%.6f",
                step=0.0001,
                key="nmr_dt",
            )
        with col2:
            lb = st.number_input(
                "Line broadening (Hz)",
                value=0.5,
                min_value=0.0,
                step=0.1,
                key="nmr_lb",
            )
        with col3:
            n_zf = st.number_input(
                "Zero-fill to (pts; 0=none)",
                value=0,
                min_value=0,
                step=1024,
                key="nmr_zf",
            )
        lb_mode = st.radio(
            "Apodisation", ["lorentzian", "gaussian"], horizontal=True, key="nmr_lb_mode"
        )
        if st.button("Process FID → Spectrum", key="nmr_fft_btn"):
            try:
                proc = apply_line_broadening(fid_arr, dt=float(dt), lb=float(lb), mode=lb_mode)
                if int(n_zf) > len(proc):
                    proc = zero_fill(proc, int(n_zf))
                result = nmr_fft(proc, dt=float(dt))
                spectrum = result["spectrum"]
                freq_hz = result["frequencies_hz"]
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(freq_hz, spectrum)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Intensity")
                ax.set_title("NMR Spectrum (FFT)")
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                out_df = pd.DataFrame({"frequency_hz": freq_hz, "spectrum": spectrum})
                _save(out_df, f"fft_{fid_col}")
            except Exception as exc:
                st.error(str(exc))

    # ── Peak Picking ───────────────────────────────────────────────────────
    elif nmr_op == "Peak Picking":
        ppm_col_nmr = st.selectbox("Chemical shift (ppm) column", numeric_cols, key="nmr_pk_ppm")
        col1, col2 = st.columns(2)
        with col1:
            thresh_frac = st.number_input(
                "Threshold (fraction of max)",
                value=0.01,
                min_value=0.001,
                step=0.005,
                key="nmr_pk_thresh",
            )
        with col2:
            min_sep_nmr = st.number_input(
                "Min separation (ppm)",
                value=0.05,
                min_value=0.001,
                step=0.01,
                key="nmr_pk_sep",
            )
        if st.button("Pick Peaks", key="nmr_pk_btn"):
            ppm_arr = data[ppm_col_nmr].dropna().to_numpy(dtype=float)
            n_use = min(len(ppm_arr), len(fid_arr))
            result = pick_nmr_peaks(
                ppm_arr[:n_use],
                fid_arr[:n_use],
                threshold_fraction=float(thresh_frac),
                min_separation_ppm=float(min_sep_nmr),
            )
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(ppm_arr[:n_use], fid_arr[:n_use])
            if result["n_peaks"] > 0:
                ax.scatter(
                    result["ppm_positions"],
                    result["intensities"],
                    color="red",
                    zorder=5,
                )
            ax.invert_xaxis()
            ax.set_xlabel("Chemical Shift (ppm)")
            ax.set_ylabel("Intensity")
            ax.set_title(f"NMR Peaks ({result['n_peaks']} found)")
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            if result["n_peaks"] > 0:
                pk_df = pd.DataFrame(
                    {
                        "ppm": result["ppm_positions"],
                        "intensity": result["intensities"],
                    }
                )
                st.dataframe(pk_df, width="stretch")

    # ── Region Integration ─────────────────────────────────────────────────
    elif nmr_op == "Region Integration":
        ppm_col_int = st.selectbox("Chemical shift (ppm) column", numeric_cols, key="nmr_int_ppm")
        st.markdown("Enter integration regions — one per line as `ppm_low, ppm_high`:")
        regions_text = st.text_area(
            "Regions", "0.0, 2.0\n2.0, 4.0\n7.0, 8.0", key="nmr_int_regions"
        )
        if st.button("Integrate Regions", key="nmr_int_btn"):
            try:
                regions = []
                for line in regions_text.strip().split("\n"):
                    parts = line.split(",")
                    if len(parts) == 2:
                        regions.append((float(parts[0]), float(parts[1])))
                if not regions:
                    st.warning("No valid regions entered.")
                else:
                    ppm_arr = data[ppm_col_int].dropna().to_numpy(dtype=float)
                    n_use = min(len(ppm_arr), len(fid_arr))
                    result = integrate_nmr_regions(ppm_arr[:n_use], fid_arr[:n_use], regions)
                    int_df = pd.DataFrame(
                        {
                            "Region": [f"{r[0]:.2f}–{r[1]:.2f} ppm" for r in result["regions"]],
                            "Integral": result["integrals"],
                            "Ratio": result["normalized_ratios"],
                        }
                    )
                    st.dataframe(int_df, width="stretch")
            except Exception as exc:
                st.error(str(exc))

    # ── Calibrate Hz → ppm ─────────────────────────────────────────────────
    elif nmr_op == "Calibrate Hz → ppm":
        col1, col2 = st.columns(2)
        with col1:
            spec_freq = st.number_input(
                "Spectrometer frequency (MHz)",
                value=400.0,
                step=1.0,
                key="nmr_cal_freq",
            )
        with col2:
            ref_hz = st.number_input(
                "Reference offset (Hz)", value=0.0, step=0.1, key="nmr_cal_ref"
            )
        if st.button("Convert Hz → ppm", key="nmr_cal_btn"):
            ppm_out = calibrate_ppm_axis(
                fid_arr,
                spectrometer_freq_mhz=float(spec_freq),
                reference_hz=float(ref_hz),
            )
            out_df = pd.DataFrame({"ppm": ppm_out, "intensity": fid_arr})
            _save(out_df, f"ppm_{fid_col}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — UV-Vis
# ─────────────────────────────────────────────────────────────────────────────
with tab_uv:
    st.header("UV-Vis Analysis")

    uv_op = st.radio(
        "Operation",
        ["Beer-Lambert Calculator", "Calibration Series", "Spectral Overlap (FRET)"],
        horizontal=True,
        key="uv_op",
    )

    # ── Beer-Lambert ───────────────────────────────────────────────────────
    if uv_op == "Beer-Lambert Calculator":
        st.markdown("Solve **A = ε × c × l** for the unknown quantity.")
        col1, col2, col3 = st.columns(3)
        with col1:
            ab_val = st.number_input(
                "Absorbance (A)", value=0.5, min_value=0.0, step=0.01, key="uv_bl_ab"
            )
        with col2:
            solve_for = st.radio(
                "Solve for",
                ["Concentration (c)", "Molar absorptivity (ε)"],
                key="uv_bl_solve",
            )
        with col3:
            path_l = st.number_input(
                "Path length (cm)",
                value=1.0,
                min_value=0.001,
                step=0.1,
                key="uv_bl_path",
            )
        if solve_for == "Concentration (c)":
            eps_val = st.number_input(
                "ε (L·mol⁻¹·cm⁻¹)",
                value=1000.0,
                min_value=0.001,
                step=100.0,
                key="uv_bl_eps",
            )
            if st.button("Calculate", key="uv_bl_btn"):
                try:
                    result = beer_lambert(
                        absorbance=float(ab_val),
                        epsilon=float(eps_val),
                        concentration=None,
                        path_length=float(path_l),
                    )
                    st.success(f"**Concentration = {result['concentration']:.6g} mol/L**")
                except Exception as exc:
                    st.error(str(exc))
        else:
            conc_val = st.number_input(
                "Concentration (mol/L)",
                value=0.001,
                min_value=1e-10,
                format="%.6f",
                step=0.0001,
                key="uv_bl_conc",
            )
            if st.button("Calculate", key="uv_bl_btn2"):
                try:
                    result = beer_lambert(
                        absorbance=float(ab_val),
                        epsilon=None,
                        concentration=float(conc_val),
                        path_length=float(path_l),
                    )
                    st.success(f"**ε = {result['epsilon']:.4g} L·mol⁻¹·cm⁻¹**")
                except Exception as exc:
                    st.error(str(exc))

    # ── Calibration Series ─────────────────────────────────────────────────
    elif uv_op == "Calibration Series":
        col1, col2, col3 = st.columns(3)
        with col1:
            conc_col_uv = st.selectbox("Concentration column", numeric_cols, key="uv_cs_conc")
        with col2:
            abs_col_uv = st.selectbox(
                "Absorbance column",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="uv_cs_abs",
            )
        with col3:
            path_l2 = st.number_input(
                "Path length (cm)",
                value=1.0,
                min_value=0.001,
                step=0.1,
                key="uv_cs_path",
            )
        if st.button("Fit Calibration", key="uv_cs_btn"):
            try:
                c_arr = data[conc_col_uv].dropna().to_numpy(dtype=float)
                a_arr = data[abs_col_uv].dropna().to_numpy(dtype=float)
                n_use = min(len(c_arr), len(a_arr))
                result = molar_absorptivity_series(
                    c_arr[:n_use], a_arr[:n_use], path_length=float(path_l2)
                )
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(c_arr[:n_use], a_arr[:n_use], label="Data", zorder=5)
                ax.plot(
                    c_arr[:n_use],
                    result["fitted_absorbances"],
                    color="red",
                    label="Fit",
                )
                ax.set_xlabel("Concentration (mol/L)")
                ax.set_ylabel("Absorbance")
                r2 = result["r_squared"]
                eps = result["epsilon"]
                ax.set_title(f"ε = {eps:.2f} L·mol⁻¹·cm⁻¹ | R² = {r2:.4f}")
                ax.legend()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                lin_str = "✅ Linear" if result["linearity_ok"] else "⚠️ Non-linear"
                st.info(f"**ε = {eps:.4g} L·mol⁻¹·cm⁻¹** | **R² = {r2:.5f}** | {lin_str}")
            except Exception as exc:
                st.error(str(exc))

    # ── Spectral Overlap (FRET) ────────────────────────────────────────────
    elif uv_op == "Spectral Overlap (FRET)":
        st.markdown("Calculate the spectral overlap integral **J** for FRET efficiency.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Donor emission**")
            wl_d_col = st.selectbox("Wavelength (nm)", numeric_cols, key="uv_fret_wld")
            em_d_col = st.selectbox(
                "Emission intensity",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="uv_fret_emd",
            )
        with col2:
            st.markdown("**Acceptor absorption**")
            wl_a_col = st.selectbox(
                "Wavelength (nm)",
                numeric_cols,
                index=min(2, len(numeric_cols) - 1),
                key="uv_fret_wla",
            )
            abs_a_col = st.selectbox(
                "ε (L·mol⁻¹·cm⁻¹)",
                numeric_cols,
                index=min(3, len(numeric_cols) - 1),
                key="uv_fret_absa",
            )
        if st.button("Calculate J", key="uv_fret_btn"):
            try:
                wd = data[wl_d_col].dropna().to_numpy(dtype=float)
                ed = data[em_d_col].dropna().to_numpy(dtype=float)
                wa = data[wl_a_col].dropna().to_numpy(dtype=float)
                aa = data[abs_a_col].dropna().to_numpy(dtype=float)
                n_d = min(len(wd), len(ed))
                n_a = min(len(wa), len(aa))
                result = spectral_overlap_integral(wd[:n_d], ed[:n_d], wa[:n_a], aa[:n_a])
                st.success(f"**J = {result['J']:.4e} M⁻¹·cm⁻¹·nm⁴**")
                if len(result["wavelengths_common"]) > 0:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax2 = ax.twinx()
                    ax.plot(
                        result["wavelengths_common"],
                        result["donor_norm"],
                        color="green",
                        label="Donor (norm.)",
                    )
                    ax2.plot(
                        result["wavelengths_common"],
                        result["acceptor_interp"],
                        color="red",
                        label="Acceptor ε",
                    )
                    ax.set_xlabel("Wavelength (nm)")
                    ax.set_ylabel("Donor emission (norm.)")
                    ax2.set_ylabel("ε (L·mol⁻¹·cm⁻¹)")
                    ax.legend(loc="upper left")
                    ax2.legend(loc="upper right")
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
            except Exception as exc:
                st.error(str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Mass Spec
# ─────────────────────────────────────────────────────────────────────────────
with tab_ms:
    st.header("Mass Spectrometry")

    col1, col2 = st.columns(2)
    with col1:
        mz_col = st.selectbox("m/z column", numeric_cols, key="ms_mz")
    with col2:
        int_col_ms = st.selectbox(
            "Intensity column",
            numeric_cols,
            index=min(1, len(numeric_cols) - 1),
            key="ms_int",
        )

    mz_arr = data[mz_col].dropna().to_numpy(dtype=float)
    int_arr_ms = data[int_col_ms].dropna().to_numpy(dtype=float)
    n_ms = min(len(mz_arr), len(int_arr_ms))

    ms_op = st.radio(
        "Operation",
        ["Find Peaks", "Centroid Spectrum"],
        horizontal=True,
        key="ms_op",
    )

    # ── Find Peaks ─────────────────────────────────────────────────────────
    if ms_op == "Find Peaks":
        col1, col2 = st.columns(2)
        with col1:
            min_int_frac = st.number_input(
                "Min intensity (fraction of base)",
                value=0.01,
                min_value=0.001,
                step=0.005,
                key="ms_pk_thresh",
            )
        with col2:
            min_sep_ms = st.number_input(
                "Min m/z separation",
                value=0.3,
                min_value=0.01,
                step=0.1,
                key="ms_pk_sep",
            )
        if st.button("Find Peaks", key="ms_pk_btn"):
            result = find_mz_peaks(
                mz_arr[:n_ms],
                int_arr_ms[:n_ms],
                min_intensity_fraction=float(min_int_frac),
                min_separation=float(min_sep_ms),
            )
            import plotly.graph_objects as go

            fig = go.Figure()
            for mv, iv in zip(mz_arr[:n_ms], int_arr_ms[:n_ms]):
                fig.add_trace(
                    go.Scatter(
                        x=[mv, mv],
                        y=[0, iv],
                        mode="lines",
                        line=dict(color="steelblue", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            if result["n_peaks"] > 0:
                fig.add_trace(
                    go.Scatter(
                        x=result["mz_positions"],
                        y=result["intensities"],
                        mode="markers+text",
                        text=[f"{m:.2f}" for m in result["mz_positions"]],
                        textposition="top center",
                        marker=dict(color="red", size=8),
                        name="Peaks",
                    )
                )
            fig.update_layout(xaxis_title="m/z", yaxis_title="Intensity", height=400)
            st.plotly_chart(fig, width="stretch")

            if result["n_peaks"] > 0:
                st.markdown(
                    f"**Base peak:** m/z = {result['base_peak_mz']:.4f} "
                    f"(intensity = {result['base_peak_intensity']:.1f})"
                )
                pk_df = pd.DataFrame(
                    {
                        "m/z": result["mz_positions"],
                        "Intensity": result["intensities"],
                        "Relative Intensity (%)": result["relative_intensities"],
                    }
                )
                st.dataframe(pk_df, width="stretch")
            else:
                st.info("No peaks found above threshold.")

    # ── Centroid Spectrum ──────────────────────────────────────────────────
    elif ms_op == "Centroid Spectrum":
        width_da = st.number_input(
            "m/z grouping window (Da)",
            value=0.05,
            min_value=0.001,
            step=0.01,
            key="ms_cent_width",
        )
        if st.button("Centroid", key="ms_cent_btn"):
            result = centroid_spectrum(mz_arr[:n_ms], int_arr_ms[:n_ms], width=float(width_da))
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Bar(
                    x=result["mz_centroids"],
                    y=result["intensities"],
                    name="Centroids",
                    marker_color="steelblue",
                )
            )
            fig.update_layout(xaxis_title="m/z", yaxis_title="Intensity", height=400)
            st.plotly_chart(fig, width="stretch")
            st.info(f"{result['n_peaks']} centroided peaks")
            if result["n_peaks"] > 0:
                out_df = pd.DataFrame(
                    {
                        "mz": result["mz_centroids"],
                        "intensity": result["intensities"],
                    }
                )
                _save(out_df, f"centroid_{int_col_ms}")
