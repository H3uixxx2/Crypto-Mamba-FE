from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.charts import CHART_CONFIG, candle_chart, full_split_chart
from src.cryptomamba_ui.data import (
    CandleDataError,
    build_predict_payload,
    model_tensor_preview,
    normalize_candles,
    split_summary,
)
from src.cryptomamba_ui.dataset_service import DatasetBundle
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
            {"status": "passed", "stage": "Read CSV", "rows": len(raw_df), "artifact": "raw dataframe"},
            {"status": "passed", "stage": "Validate schema", "rows": len(normalized_df), "artifact": "date/open/high/low/close/volume"},
            {"status": "passed", "stage": "Build daily OHLCV", "rows": len(daily_df), "artifact": "1 row per day"},
            {"status": "passed", "stage": split_stage, "rows": len(in_scope), "artifact": "train/validation/test"},
            {"status": "passed", "stage": "Build tensor", "rows": 14, "artifact": str(tensor_shape)},
        ]
    )


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
        st.dataframe(stage_summary, hide_index=True, width="stretch")
    with compare_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("##### Raw selected dataset")
            st.dataframe(raw_source_df.head(12), hide_index=True, height=280, width="stretch")
        with right:
            st.markdown("##### After processing: daily OHLCV")
            st.dataframe(cleaned_daily.head(12), hide_index=True, height=280, width="stretch")
    with split_tab:
        st.dataframe(split_summary(processed_df), hide_index=True, height=170, width="stretch")
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
    st.table(model_window_preview(df.tail(14)))

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
            st.dataframe(split_summary(df), hide_index=True, height=170, width="stretch")
