from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.charts import CHART_CONFIG, candle_chart, full_split_chart
from src.cryptomamba_ui.data import (
    CandleDataError,
    add_chronological_split,
    build_predict_payload,
    daily_ohlcv,
    model_tensor_preview,
    normalize_candles,
    split_summary,
)
from src.cryptomamba_ui.dataset_service import DatasetBundle, DatasetService
from src.cryptomamba_ui.ui import date_range, stat_card
from src.cryptomamba_ui.upload_state import clear_retained_upload, remember_uploaded_csv, retained_uploaded_csv_name


def render_data_source_controls() -> tuple[str, Any | None]:
    if st.session_state.get("data_mode") == "Paper sample":
        st.session_state["data_mode"] = "Paper dataset"
    st.session_state.setdefault("data_mode", "Paper dataset")
    st.session_state.setdefault("replace_uploaded_csv", False)
    st.session_state.setdefault("uploaded_csv_uploader_version", 0)

    st.markdown("#### Data source")
    data_mode = st.radio(
        "Data source",
        ["Paper dataset", "Upload CSV"],
        horizontal=True,
        key="data_mode",
    )
    uploaded = None
    if data_mode == "Upload CSV":
        retained_name = retained_uploaded_csv_name()
        replace_upload = bool(st.session_state.get("replace_uploaded_csv", False))

        if retained_name and not replace_upload:
            uploaded = remember_uploaded_csv(None)
            st.markdown(f"**Uploaded CSV in use:** `{retained_name}`")
            col_replace, col_clear = st.columns([1, 1])
            if col_replace.button("Replace CSV", key="replace_uploaded_csv_button"):
                st.session_state["replace_uploaded_csv"] = True
                st.rerun()
            if col_clear.button("Clear CSV", key="clear_uploaded_csv_button"):
                clear_retained_upload()
                uploaded = None
                st.rerun()
        else:
            uploader_key = f"uploaded_csv_{st.session_state['uploaded_csv_uploader_version']}"
            uploaded_widget = st.file_uploader(
                "Upload CSV dataset",
                type=["csv"],
                key=uploader_key,
                help="Required columns: date/timestamp, open, high, low, close, volume.",
            )
            uploaded = remember_uploaded_csv(uploaded_widget)
            if uploaded_widget is not None:
                st.session_state["replace_uploaded_csv"] = False
                st.session_state["uploaded_csv_uploader_version"] += 1
                st.rerun()
            if retained_name:
                st.caption(f"`{retained_name}` remains active until a replacement CSV is uploaded.")
            else:
                st.caption("Upload a BTC-compatible OHLCV CSV to test the data processing pipeline. This does not change the thesis scope from BTC.")
    return data_mode, uploaded


def render_data_source_indicator() -> tuple[str, Any | None]:
    """Read-only mirror of the dataset chosen on the Data screen.

    Unlike ``render_data_source_controls`` this renders no radio/uploader: downstream
    screens (Predict) inherit the active dataset from shared session state instead of
    re-asking for it. Returns the same ``(data_mode, uploaded)`` tuple so callers can
    load the dataset identically.
    """
    if st.session_state.get("data_mode") == "Paper sample":
        st.session_state["data_mode"] = "Paper dataset"
    data_mode = str(st.session_state.get("data_mode", "Paper dataset"))

    uploaded = None
    if data_mode == "Upload CSV":
        retained_name = retained_uploaded_csv_name()
        if retained_name:
            uploaded = remember_uploaded_csv(None)
            st.caption(f"Dataset: **Upload CSV** — `{retained_name}` · chosen on the **Data** screen.")
        else:
            st.caption("Dataset: **Upload CSV** selected but no file uploaded — upload it on the **Data** screen.")
    else:
        st.caption("Dataset: **Paper dataset** · chosen on the **Data** screen.")
    return data_mode, uploaded


def render_empty_upload(dataset: DatasetBundle) -> None:
    st.markdown("### 1. Data")
    st.caption(f"Dataset source: {dataset.source_label}. {dataset.source_detail}")
    st.info("No CSV uploaded yet, so no placeholder data is rendered. Upload a BTC-compatible OHLCV file or select Paper dataset.")
    st.dataframe(
        pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]),
        hide_index=True,
        width="stretch",
    )


def model_window_preview(source_df: pd.DataFrame) -> pd.DataFrame:
    """Show the concrete 14-input window and target row when available."""
    if len(source_df) < 14:
        raise CandleDataError("Need at least 14 valid daily candles to preview model input")
    row_count = 15 if len(source_df) >= 15 else 14
    sample = source_df.tail(row_count).head(row_count).copy()
    sample = sample[["date", "open", "high", "low", "close", "volume"]].copy()
    roles = [f"input_{i:02d}" for i in range(1, min(14, row_count) + 1)]
    if row_count == 15:
        roles.append("target_close")
    sample.insert(0, "role", roles)
    return sample


def processing_summary(
    raw_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    tensor_shape: list[int],
    split_strategy: str,
) -> pd.DataFrame:
    in_scope = processed_df[processed_df["dataset_split"].isin(["train", "validation", "test"])]
    split_stage = "Apply fixed paper-date split" if split_strategy == "paper_date" else "Apply chronological 70/15/15 split"
    return pd.DataFrame(
        [
            {"status": "PASS", "stage": "Read CSV", "rows": len(raw_df), "artifact": "raw dataframe", "message": ""},
            {"status": "PASS", "stage": "Validate schema", "rows": len(normalized_df), "artifact": "date/open/high/low/close/volume", "message": ""},
            {"status": "PASS", "stage": "Build daily OHLCV", "rows": len(daily_df), "artifact": "1 row per day", "message": ""},
            {"status": "PASS", "stage": split_stage, "rows": len(in_scope), "artifact": "train/validation/test", "message": ""},
            {"status": "PASS", "stage": "Build tensor", "rows": 14, "artifact": str(tensor_shape), "message": ""},
        ]
    )


def skip_row(stage: str, blocked_by: str) -> dict[str, object]:
    return {"status": "SKIP", "stage": stage, "rows": "—", "artifact": "—", "message": f"blocked by {blocked_by}"}


def uploaded_pipeline_report(uploaded_file: Any) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows: list[dict[str, object]] = []
    raw_df: pd.DataFrame | None = None

    try:
        raw_df = DatasetService.read_csv(uploaded_file)
        rows.append({"status": "PASS", "stage": "Read CSV", "rows": len(raw_df), "artifact": "raw dataframe", "message": ""})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        rows.append({"status": "FAIL", "stage": "Read CSV", "rows": "—", "artifact": "raw dataframe", "message": str(exc)})
        rows.extend(
            [
                skip_row("Validate schema", "Read CSV"),
                skip_row("Build daily OHLCV", "Read CSV"),
                skip_row("Apply chronological 70/15/15 split", "Read CSV"),
                skip_row("Build tensor", "Read CSV"),
            ]
        )
        return pd.DataFrame(rows), raw_df

    try:
        normalized_df = normalize_candles(raw_df)
        rows.append({"status": "PASS", "stage": "Validate schema", "rows": len(normalized_df), "artifact": "date/open/high/low/close/volume", "message": ""})
    except CandleDataError as exc:
        rows.append({"status": "FAIL", "stage": "Validate schema", "rows": len(raw_df), "artifact": "date/open/high/low/close/volume", "message": str(exc)})
        rows.extend(
            [
                skip_row("Build daily OHLCV", "Validate schema"),
                skip_row("Apply chronological 70/15/15 split", "Validate schema"),
                skip_row("Build tensor", "Validate schema"),
            ]
        )
        return pd.DataFrame(rows), raw_df

    try:
        daily_df = daily_ohlcv(raw_df)
        rows.append({"status": "PASS", "stage": "Build daily OHLCV", "rows": len(daily_df), "artifact": "1 row per day", "message": ""})
    except CandleDataError as exc:
        rows.append({"status": "FAIL", "stage": "Build daily OHLCV", "rows": len(normalized_df), "artifact": "1 row per day", "message": str(exc)})
        rows.extend(
            [
                skip_row("Apply chronological 70/15/15 split", "Build daily OHLCV"),
                skip_row("Build tensor", "Build daily OHLCV"),
            ]
        )
        return pd.DataFrame(rows), raw_df

    try:
        processed_df = add_chronological_split(daily_df)
        rows.append({"status": "PASS", "stage": "Apply chronological 70/15/15 split", "rows": len(processed_df), "artifact": "train/validation/test", "message": ""})
    except CandleDataError as exc:
        rows.append({"status": "FAIL", "stage": "Apply chronological 70/15/15 split", "rows": len(daily_df), "artifact": "train/validation/test", "message": str(exc)})
        rows.append(skip_row("Build tensor", "Apply split"))
        return pd.DataFrame(rows), raw_df

    try:
        _, tensor_payload = model_tensor_preview(processed_df)
        rows.append({"status": "PASS", "stage": "Build tensor", "rows": 14, "artifact": str(tensor_payload["tensor_shape_sent_to_model"]), "message": ""})
    except (CandleDataError, ValueError, KeyError) as exc:
        rows.append({"status": "FAIL", "stage": "Build tensor", "rows": len(processed_df), "artifact": "[1, 6, 14]", "message": str(exc)})
    return pd.DataFrame(rows), raw_df


def pipeline_status_class(status: object) -> str:
    normalized_status = str(status).strip().upper()
    if normalized_status == "PASS":
        return "status-pass"
    if normalized_status == "FAIL":
        return "status-fail"
    if normalized_status == "SKIP":
        return "status-skip"
    return "status-unknown"


def pipeline_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return escape(str(value))


def render_pipeline_summary(report: pd.DataFrame) -> None:
    display_columns = ["status", "stage", "rows", "artifact"]
    if "message" in report.columns and report["message"].fillna("").astype(str).str.strip().any():
        display_columns.append("message")

    colgroup = "".join(
        [
            '<col style="width: 10%;" />',
            '<col style="width: 28%;" />',
            '<col style="width: 9%;" />',
            '<col style="width: 24%;" />',
            '<col style="width: 29%;" />' if "message" in display_columns else "",
        ]
    )
    header_html = "".join(f"<th>{escape(column)}</th>" for column in display_columns)
    rows_html = []
    for _, row in report[display_columns].iterrows():
        cell_html = []
        for column in display_columns:
            value = row[column]
            css_class = f' class="{pipeline_status_class(value)}"' if column == "status" else ""
            cell_html.append(f"<td{css_class}>{pipeline_cell(value)}</td>")
        rows_html.append(f"<tr>{''.join(cell_html)}</tr>")

    st.markdown(
        f"""
<style>
.crypto-pipeline-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    table-layout: fixed;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    font-size: 0.95rem;
}}
.crypto-pipeline-table th,
.crypto-pipeline-table td {{
    padding: 0.75rem 0.9rem;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
    white-space: normal;
    overflow-wrap: anywhere;
}}
.crypto-pipeline-table tr:last-child td {{
    border-bottom: 0;
}}
.crypto-pipeline-table th {{
    background: #f8fafc;
    color: #64748b;
    font-weight: 700;
    text-align: left;
}}
.crypto-pipeline-table td:nth-child(3) {{
    text-align: right;
}}
.crypto-pipeline-table .status-pass {{
    color: #047857;
    font-weight: 800;
    letter-spacing: 0.02em;
}}
.crypto-pipeline-table .status-fail {{
    color: #dc2626;
    font-weight: 800;
    letter-spacing: 0.02em;
}}
.crypto-pipeline-table .status-skip {{
    color: #64748b;
    font-weight: 800;
    letter-spacing: 0.02em;
}}
.crypto-pipeline-table .status-unknown {{
    color: #334155;
    font-weight: 700;
}}
</style>
<table class="crypto-pipeline-table">
    <colgroup>{colgroup}</colgroup>
    <thead><tr>{header_html}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
</table>
""",
        unsafe_allow_html=True,
    )


def render_invalid_upload_page(uploaded_file: Any, source_detail: str) -> None:
    report, raw_df = uploaded_pipeline_report(uploaded_file)

    st.markdown("### 1. Data")
    st.caption(f"Dataset source: Uploaded CSV. {source_detail}")
    st.warning("Uploaded CSV failed the data processing pipeline. Fix the FAIL stage and re-upload the file.")

    st.markdown("### Data processing pipeline")
    st.caption("The pipeline runs step-by-step so invalid input shows exactly where processing failed.")
    summary_tab, raw_tab = st.tabs(["Summary", "Raw uploaded data"])
    with summary_tab:
        render_pipeline_summary(report)
    with raw_tab:
        if raw_df is None:
            st.info("Raw rows are unavailable because the CSV could not be read.")
        else:
            st.dataframe(raw_df.head(20), hide_index=True, width="stretch")


def render_data_processing_pipeline(
    raw_source_df: pd.DataFrame,
    daily_source_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    source_label: str,
    source_detail: str,
    split_strategy: str,
) -> None:
    st.markdown("### Data processing pipeline")
    st.caption(
        f"Source: {source_label}. {source_detail}. "
        "The selected source is normalized into daily OHLCV rows, assigned to train/validation/test, then converted into the 14-day input window."
    )

    try:
        normalized_df = normalize_candles(raw_source_df)
        _, tensor_payload = model_tensor_preview(processed_df)
    except (CandleDataError, ValueError, KeyError) as exc:
        st.error(f"Invalid current dataset: {exc}")
        return

    required_view = ["date", "open", "high", "low", "close", "volume"]
    cleaned_daily = daily_source_df[required_view].copy()
    cleaned_with_split = processed_df[[*required_view, "dataset_split"]].copy()
    stage_summary = processing_summary(
        raw_source_df,
        normalized_df,
        daily_source_df,
        processed_df,
        tensor_payload["tensor_shape_sent_to_model"],
        split_strategy,
    )

    summary_tab, compare_tab, split_tab = st.tabs(["Summary", "Before / After", "Split"])

    with summary_tab:
        render_pipeline_summary(stage_summary)
    with compare_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("##### Raw selected dataset")
            st.dataframe(raw_source_df.head(12), hide_index=True, height=280, width="stretch")
        with right:
            st.markdown("##### After processing: daily OHLCV")
            st.dataframe(cleaned_daily.head(12), hide_index=True, height=280, width="stretch")
    with split_tab:
        st.dataframe(split_summary(processed_df), hide_index=True, width="stretch")
        st.dataframe(cleaned_with_split.head(20), hide_index=True, height=260, width="stretch")


def render_data_page(dataset: DatasetBundle, default_prediction_date: str, risk: float) -> None:
    raw_df = dataset.raw_df
    daily_df = dataset.daily_df
    df = dataset.processed_df
    dataset_split_strategy = dataset.split_strategy

    st.markdown("### 1. Data")
    st.caption(f"Dataset source: {dataset.source_label}. {dataset.source_detail}")
    train_df = df[df["dataset_split"] == "train"]
    val_df = df[df["dataset_split"] == "validation"]
    test_df = df[df["dataset_split"] == "test"]
    out_of_scope_df = df[df["dataset_split"] == "out_of_scope"]

    show_out_of_scope = dataset_split_strategy == "paper_date" and not out_of_scope_df.empty
    metric_columns = st.columns(5 if show_out_of_scope else 4)
    with metric_columns[0]:
        stat_card("Total daily candles", f"{len(df):,}", date_range(df))
    with metric_columns[1]:
        stat_card("Train", f"{len(train_df):,}", date_range(train_df))
    with metric_columns[2]:
        stat_card("Validation", f"{len(val_df):,}", date_range(val_df))
    with metric_columns[3]:
        stat_card("Test", f"{len(test_df):,}", date_range(test_df))
    if show_out_of_scope:
        with metric_columns[4]:
            stat_card("Outside paper split", f"{len(out_of_scope_df):,}", date_range(out_of_scope_df))

    if dataset_split_strategy == "paper_date":
        st.caption("Split strategy: fixed paper-date split for CryptoMamba-v reproduction.")
    else:
        st.caption("Split strategy: chronological ratio split for uploaded data: train 70%, validation 15%, test 15%.")

    chart_title = (
        "Paper dataset close price — train/validation/test split"
        if dataset_split_strategy == "paper_date"
        else "Uploaded dataset close price — chronological train/validation/test split"
    )
    st.plotly_chart(full_split_chart(df, chart_title), use_container_width=True, config=CHART_CONFIG)

    st.markdown("#### Latest 14-day model input")
    st.dataframe(model_window_preview(df.tail(14)), hide_index=True, width="stretch")

    payload_col, tensor_col = st.columns(2)
    with payload_col:
        with st.expander("Prediction request payload", expanded=False):
            st.caption("Request body built from the latest 14 daily candles. The inference service converts this payload into the model tensor.")
            st.json(build_predict_payload(df, prediction_date=default_prediction_date, risk=risk))
    with tensor_col:
        with st.expander("Model tensor contract", expanded=False):
            _, tensor_payload = model_tensor_preview(df)
            st.caption("CryptoMamba-v feature order and shape used after API conversion.")
            st.json({key: tensor_payload[key] for key in ["feature_order", "tensor_shape_before_batch", "tensor_shape_sent_to_model", "volume_rule", "normalize"]})

    st.divider()
    render_data_processing_pipeline(raw_df, daily_df, df, dataset.source_label, dataset.source_detail, dataset_split_strategy)

    with st.expander("Dataset detail", expanded=False):
        st.caption(f"Source: {dataset.source_label}. {dataset.source_detail}")
        selected_name = st.radio(
            "Split",
            ["Train", "Validation", "Test"],
            horizontal=True,
            key="selected_data_split_label",
        )
        selected_df = {"Train": train_df, "Validation": val_df, "Test": test_df}[selected_name]

        left, right = st.columns([1, 2])
        with left:
            st.dataframe(
                selected_df[["date", "open", "high", "low", "close", "volume"]].tail(15),
                hide_index=True,
                height=360,
                width="stretch",
            )
        with right:
            detail_df = selected_df.tail(90 if selected_name != "Train" else 180)
            st.plotly_chart(candle_chart(detail_df, f"{selected_name} candles", height=430), use_container_width=True, config=CHART_CONFIG)

        raw_tab, daily_tab, split_detail_tab = st.tabs(["Raw", "Daily", "Split summary"])
        with raw_tab:
            st.dataframe(raw_df.head(12), hide_index=True, height=240, width="stretch")
        with daily_tab:
            st.dataframe(daily_df.head(12), hide_index=True, height=240, width="stretch")
        with split_detail_tab:
            st.dataframe(split_summary(df), hide_index=True, width="stretch")
