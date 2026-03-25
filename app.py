import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="UV-Vis Analyzer", layout="wide")

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "uv_results_ready" not in st.session_state:
    st.session_state.uv_results_ready = False
if "uv_review_df" not in st.session_state:
    st.session_state.uv_review_df = pd.DataFrame()
if "uv_parsed_data" not in st.session_state:
    st.session_state.uv_parsed_data = {}
if "uv_warnings_df" not in st.session_state:
    st.session_state.uv_warnings_df = pd.DataFrame()
if "uv_editor_df" not in st.session_state:
    st.session_state.uv_editor_df = pd.DataFrame()


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_name(name: str) -> str:
    stem = Path(name).stem.strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def smart_detect_type(name: str) -> str:
    n = normalize_name(name).upper()
    ref_keywords = ["REF", "REFERENCE", "BASELINE", "BLANK", "BACKGROUND", "100%", "0 ABSORBANCE"]
    if any(k in n for k in ref_keywords):
        return "Reference"
    return "Sample"


def smart_reference_choice(editor_df: pd.DataFrame):
    refs = editor_df.loc[editor_df["Type"] == "Reference", "Dataset"].tolist()
    if refs:
        return refs[0]
    return None


def read_text_lines(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    return text.splitlines()


def parse_uvvis_file(uploaded_file):
    """
    Generic parser for UV-Vis exported CSV/TXT files.
    Attempts to find the first 2-column numeric table corresponding to wavelength + value.
    Returns wavelength array and measured signal array.
    """
    lines = read_text_lines(uploaded_file)

    candidates = []
    for delimiter in [",", ";", "\t"]:
        xs = []
        ys = []
        for line in lines:
            parts = [p.strip().strip('"') for p in line.split(delimiter)]
            if len(parts) < 2:
                continue
            nums = []
            for p in parts[:4]:
                try:
                    nums.append(float(p))
                except Exception:
                    nums.append(None)
            found = False
            for i in range(len(nums) - 1):
                if nums[i] is not None and nums[i + 1] is not None:
                    x = nums[i]
                    y = nums[i + 1]
                    if 150 <= x <= 2000:
                        xs.append(x)
                        ys.append(y)
                        found = True
                        break
            if not found:
                continue
        if len(xs) >= 10:
            candidates.append((delimiter, np.array(xs, dtype=float), np.array(ys, dtype=float)))

    if not candidates:
        raise ValueError("Could not find a numeric wavelength/value table in the file.")

    delimiter, x, y = max(candidates, key=lambda t: len(t[1]))

    # sort by wavelength and drop duplicates
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[idx]

    if len(x) < 10:
        raise ValueError("Parsed too few usable spectral points.")

    return x, y, delimiter


def detect_signal_kind(y: np.ndarray):
    """
    Heuristic:
    - if mostly between 0 and 100 or 0 and 1.2, likely transmission-like
    - otherwise could still be absorbance or arbitrary signal.
    """
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return "unknown"

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))

    if 0 <= y_min and y_max <= 100.5:
        return "percent_transmission"
    if 0 <= y_min and y_max <= 1.2:
        return "fraction_transmission"
    if -0.5 <= y_min and y_max <= 5.5:
        return "absorbance_like"
    return "unknown"


def convert_signal(y: np.ndarray, source_kind: str, output_mode: str):
    """
    output_mode: 'Transmission (%)' or 'Absorbance'
    """
    eps = 1e-12

    if output_mode == "Transmission (%)":
        if source_kind == "percent_transmission":
            return y.copy()
        if source_kind == "fraction_transmission":
            return y * 100.0
        if source_kind == "absorbance_like":
            return (10 ** (-y)) * 100.0
        return y.copy()

    # output absorbance
    if source_kind == "percent_transmission":
        t = np.clip(y / 100.0, eps, None)
        return -np.log10(t)
    if source_kind == "fraction_transmission":
        t = np.clip(y, eps, None)
        return -np.log10(t)
    if source_kind == "absorbance_like":
        return y.copy()
    return y.copy()


def build_review_table(measurement_files):
    rows = []
    parsed_data = {}
    warnings = []

    for f in measurement_files:
        dataset_name = normalize_name(f.name)
        try:
            wl, signal, detected_delim = parse_uvvis_file(f)
            kind = detect_signal_kind(signal)
            parsed_data[dataset_name] = {
                "wavelength": wl,
                "signal_raw": signal,
                "signal_kind": kind,
                "source_file": f.name,
                "delimiter": detected_delim,
            }
            rows.append({
                "Dataset": dataset_name,
                "Type": smart_detect_type(dataset_name),
                "Reference": None,
                "Detected signal": kind,
                "Points": len(wl),
                "Min wl": float(np.min(wl)),
                "Max wl": float(np.max(wl)),
            })
        except Exception as e:
            warnings.append({
                "Dataset": dataset_name,
                "Type": "Parsing error",
                "Message": str(e),
            })

    review_df = pd.DataFrame(rows)
    if not review_df.empty:
        default_ref = smart_reference_choice(review_df)
        if default_ref is not None:
            sample_mask = review_df["Type"] == "Sample"
            review_df.loc[sample_mask, "Reference"] = default_ref

    warnings_df = pd.DataFrame(warnings)
    return review_df, parsed_data, warnings_df


def interpolate_to_reference_grid(x_ref, x, y):
    return np.interp(x_ref, x, y)


def compute_processed_data(editor_df, parsed_data, output_mode, subtraction_mode):
    """
    subtraction_mode: 'No subtraction' or 'Subtract selected reference'
    Returns details dict + warnings.
    """
    details = {}
    warnings = []

    for _, row in editor_df.iterrows():
        dataset = row.get("Dataset")
        dtype = row.get("Type")
        ref_name = row.get("Reference")

        if dtype != "Sample":
            continue
        if dataset not in parsed_data:
            warnings.append({
                "Dataset": dataset,
                "Type": "Missing parsed data",
                "Message": "Dataset was not successfully parsed.",
            })
            continue

        d = parsed_data[dataset]
        wl = d["wavelength"]
        y_raw = d["signal_raw"]
        source_kind = d["signal_kind"]
        y_mode = convert_signal(y_raw, source_kind, output_mode)

        ref_mode = None
        y_processed = y_mode.copy()
        if subtraction_mode == "Subtract selected reference":
            if pd.isna(ref_name) or ref_name is None or str(ref_name).strip() == "":
                warnings.append({
                    "Dataset": dataset,
                    "Type": "Missing reference",
                    "Message": "No reference selected for subtraction.",
                })
                continue
            if ref_name not in parsed_data:
                warnings.append({
                    "Dataset": dataset,
                    "Type": "Reference not parsed",
                    "Message": f"Reference '{ref_name}' was not parsed successfully.",
                })
                continue

            ref_d = parsed_data[ref_name]
            ref_wl = ref_d["wavelength"]
            ref_y_raw = ref_d["signal_raw"]
            ref_kind = ref_d["signal_kind"]
            ref_mode_full = convert_signal(ref_y_raw, ref_kind, output_mode)
            ref_mode = interpolate_to_reference_grid(wl, ref_wl, ref_mode_full)
            y_processed = y_mode - ref_mode

        details[dataset] = {
            "dataset": dataset,
            "reference": None if pd.isna(ref_name) else ref_name,
            "wavelength": wl,
            "raw_mode_signal": y_mode,
            "reference_mode_signal": ref_mode,
            "processed_signal": y_processed,
            "signal_kind": source_kind,
            "output_mode": output_mode,
        }

    return details, pd.DataFrame(warnings)


def build_plotly_figure(details_dict, selected_datasets, plot_variant, x_range=None):
    fig = go.Figure()

    for name in selected_datasets:
        d = details_dict[name]
        x = d["wavelength"]

        if plot_variant == "Processed":
            y = d["processed_signal"]
            trace_name = name if d["reference"] is None else f"{name} | ref: {d['reference']}"
        elif plot_variant == "Raw":
            y = d["raw_mode_signal"]
            trace_name = name
        else:
            continue

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=trace_name,
            )
        )

    y_label = selected_datasets and details_dict[selected_datasets[0]]["output_mode"] or "Signal"
    if plot_variant == "Processed" and selected_datasets:
        if details_dict[selected_datasets[0]]["reference"] is not None:
            y_label += " (difference)"

    fig.update_layout(
        title=f"{plot_variant} UV-Vis spectra",
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Datasets",
    )

    if x_range is not None:
        fig.update_xaxes(range=x_range)

    return fig


def probe_values(details_dict, selected_datasets, wavelength_nm, plot_variant):
    rows = []
    for name in selected_datasets:
        d = details_dict[name]
        wl = d["wavelength"]
        y = d["processed_signal"] if plot_variant == "Processed" else d["raw_mode_signal"]
        idx = int(np.argmin(np.abs(wl - wavelength_nm)))
        rows.append({
            "Dataset": name,
            "Reference": d["reference"],
            "Nearest wavelength (nm)": float(wl[idx]),
            "Value": float(y[idx]),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("UV-Vis Analyzer")
st.caption(
    "Upload a folder of UV-Vis datasets, switch between transmission and absorbance, optionally subtract a selected reference, and interactively inspect spectra."
)

left, right = st.columns([1, 1.7], gap="large")

with left:
    st.subheader("Inputs")

    measurement_files = st.file_uploader(
        "1. Drop all UV-Vis files",
        type=["csv", "txt", "dat"],
        accept_multiple_files=True,
        key="uv_measurement_files",
    )

    matching_mode = st.radio(
        "2. Reference assignment mode",
        options=["Smart mode", "Manual mode"],
        index=0,
    )

    output_mode = st.radio(
        "3. Display mode",
        options=["Transmission (%)", "Absorbance"],
        index=0,
    )

    subtraction_mode = st.radio(
        "4. Reference handling",
        options=["No subtraction", "Subtract selected reference"],
        index=0,
    )

    preview = st.button("Preview parsing", type="secondary", width="stretch")
    run_analysis = st.button("Run UV-Vis analysis", type="primary", width="stretch")

with right:
    st.subheader("Review and results")

    if preview or (measurement_files and st.session_state.uv_editor_df.empty):
        if not measurement_files:
            st.warning("Upload the UV-Vis files first.")
        else:
            review_df, parsed_data, warnings_df = build_review_table(measurement_files)
            st.session_state.uv_review_df = review_df
            st.session_state.uv_parsed_data = parsed_data
            st.session_state.uv_warnings_df = warnings_df
            st.session_state.uv_editor_df = review_df.copy()

    if not st.session_state.uv_editor_df.empty:
        st.subheader("Editable dataset review")

        if matching_mode == "Smart mode":
            disabled_cols = ["Dataset", "Type", "Detected signal", "Points", "Min wl", "Max wl"]
        else:
            disabled_cols = ["Dataset", "Detected signal", "Points", "Min wl", "Max wl"]

        dataset_options = st.session_state.uv_editor_df["Dataset"].tolist()

        edited_df = st.data_editor(
            st.session_state.uv_editor_df,
            width="stretch",
            num_rows="fixed",
            key="uv_data_editor",
            disabled=disabled_cols,
            column_config={
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["Sample", "Reference"],
                    required=True,
                ),
                "Reference": st.column_config.SelectboxColumn(
                    "Reference",
                    options=dataset_options,
                ),
            },
        )
        st.session_state.uv_editor_df = edited_df.copy()

        st.download_button(
            "Download review CSV",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="uvvis_review_table.csv",
            mime="text/csv",
            width="stretch",
        )

    if run_analysis:
        try:
            if not measurement_files:
                st.error("Please upload the UV-Vis files.")
                st.stop()

            editor_df = st.session_state.uv_editor_df.copy()
            if editor_df.empty:
                review_df, parsed_data, warnings_df = build_review_table(measurement_files)
                editor_df = review_df.copy()
                st.session_state.uv_parsed_data = parsed_data
                st.session_state.uv_warnings_df = warnings_df

            parsed_data = st.session_state.uv_parsed_data
            parse_warnings_df = st.session_state.uv_warnings_df

            details, process_warnings_df = compute_processed_data(
                editor_df=editor_df,
                parsed_data=parsed_data,
                output_mode=output_mode,
                subtraction_mode=subtraction_mode,
            )

            all_warnings = pd.concat([parse_warnings_df, process_warnings_df], ignore_index=True) if not parse_warnings_df.empty or not process_warnings_df.empty else pd.DataFrame()

            st.session_state.uv_review_df = editor_df
            st.session_state.uv_warnings_df = all_warnings
            st.session_state.uv_parsed_data = parsed_data
            st.session_state.uv_results_ready = True

        except Exception as e:
            st.error(f"Error while running UV-Vis analysis: {e}")

    if st.session_state.uv_results_ready:
        review_df = st.session_state.uv_review_df
        warnings_df = st.session_state.uv_warnings_df
        parsed_data = st.session_state.uv_parsed_data

        details, _ = compute_processed_data(
            editor_df=review_df,
            parsed_data=parsed_data,
            output_mode=output_mode,
            subtraction_mode=subtraction_mode,
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary",
            "Interactive plot",
            "Wavelength probe",
            "Warnings",
        ])

        with tab1:
            st.subheader("Parsed files and reference assignment")
            st.dataframe(review_df, width="stretch")

        with tab2:
            st.subheader("Interactive UV-Vis plot")
            if not details:
                st.info("No graphable datasets are available.")
            else:
                dataset_options = sorted(details.keys())
                selected_datasets = st.multiselect(
                    "Select one or more datasets to display",
                    options=dataset_options,
                    default=dataset_options[: min(4, len(dataset_options))],
                    key="uv_graph_multiselect",
                )

                if not selected_datasets:
                    st.warning("Select at least one dataset.")
                else:
                    plot_variant = st.radio(
                        "Plot variant",
                        options=["Processed", "Raw"],
                        horizontal=True,
                        key="uv_plot_variant",
                    )

                    x_min, x_max = st.slider(
                        "Displayed wavelength range (nm)",
                        min_value=150,
                        max_value=1100,
                        value=(300, 900),
                        step=1,
                        key="uv_graph_range",
                    )

                    fig = build_plotly_figure(
                        details_dict=details,
                        selected_datasets=selected_datasets,
                        plot_variant=plot_variant,
                        x_range=[x_min, x_max],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    detail_rows = []
                    for name in selected_datasets:
                        d = details[name]
                        detail_rows.append({
                            "Dataset": d["dataset"],
                            "Reference": d["reference"],
                            "Detected signal": d["signal_kind"],
                            "Output mode": d["output_mode"],
                        })
                    st.dataframe(pd.DataFrame(detail_rows), width="stretch")

        with tab3:
            st.subheader("Wavelength probe")
            if not details:
                st.info("No processed datasets are available.")
            else:
                dataset_options = sorted(details.keys())
                selected_probe_datasets = st.multiselect(
                    "Datasets to probe",
                    options=dataset_options,
                    default=dataset_options[: min(4, len(dataset_options))],
                    key="uv_probe_multiselect",
                )

                probe_variant = st.radio(
                    "Probe values from",
                    options=["Processed", "Raw"],
                    horizontal=True,
                    key="uv_probe_variant",
                )

                probe_nm = st.number_input(
                    "Wavelength to inspect (nm)",
                    min_value=150.0,
                    max_value=1100.0,
                    value=550.0,
                    step=1.0,
                )

                if selected_probe_datasets:
                    probe_df = probe_values(details, selected_probe_datasets, probe_nm, probe_variant)
                    st.dataframe(probe_df, width="stretch")
                    st.download_button(
                        "Download probe table CSV",
                        data=probe_df.to_csv(index=False).encode("utf-8"),
                        file_name="uvvis_probe_values.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                else:
                    st.warning("Select at least one dataset to probe.")

        with tab4:
            st.subheader("Warnings")
            if warnings_df.empty:
                st.success("No warnings.")
            else:
                st.dataframe(warnings_df, width="stretch")
    else:
        st.info("Upload files, preview parsing, review the assignments, then run the UV-Vis analysis.")
