from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.cryptomamba_ui.data import CandleDataError
from src.cryptomamba_ui.dataset_service import DatasetBundle, DatasetService
from src.cryptomamba_ui.pages.data_page import render_data_page, render_data_source_controls, render_empty_upload, render_invalid_upload_page
from src.cryptomamba_ui.pages.plan_page import render_plan_page
from src.cryptomamba_ui.pages.predict_page import render_predict_page
from src.cryptomamba_ui.pages.reproduce_page import render_reproduce_page
from src.cryptomamba_ui.pages.trading_page import render_trading_page
from src.cryptomamba_ui.ui import render_header, render_style
from src.cryptomamba_ui.upload_state import remember_uploaded_csv


def resolve_app_root(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parent


ROOT = resolve_app_root(__file__)
load_dotenv(ROOT / ".env")
SAMPLE_PATH = ROOT / "sample_data" / "btc_ohlcv_paper_splits.csv"
DATASET_SERVICE = DatasetService(sample_path=SAMPLE_PATH, display_root=ROOT)
CORE_ROOT = Path(os.getenv("CRYPTO_MAMBA_CORE_ROOT", ROOT.parent / "CryptoMamba")).expanduser().resolve()
EVALUATION_DIR = CORE_ROOT / "output" / "evaluation"
REPRODUCE_DIR = CORE_ROOT / "output" / "reproduce_colab_train"
FORECAST_METRICS_PATH = EVALUATION_DIR / "forecast_metrics.csv"
BASELINE_METRICS_PATH = EVALUATION_DIR / "baseline_metrics.csv"
TRADING_REPLAY_METRICS_PATH = EVALUATION_DIR / "trading_replay_metrics.csv"
REPRODUCE_PROVENANCE_DIR = REPRODUCE_DIR / "provenance"
SELECTED_CHECKPOINT_PATH = REPRODUCE_DIR / "checkpoints" / "cmamba_v_best_colab_train.ckpt"
TRADING_METRICS_PATH = EVALUATION_DIR / "trading_metrics.csv"
REGIME_METRICS_PATH = EVALUATION_DIR / "regime_metrics.csv"
OFFLINE_PREDICTION_PATH = EVALUATION_DIR / "offline_prediction.json"


def initialize_session_state() -> None:
    st.session_state.setdefault("data_mode", "Paper dataset")
    st.session_state.setdefault("inference_mode", "Live Model API")
    st.session_state.setdefault("api_url", os.getenv("CRYPTO_MAMBA_API_URL", ""))
    st.session_state.setdefault("risk", 2.0)
    st.session_state.setdefault("replace_uploaded_csv", False)
    st.session_state.setdefault("uploaded_csv_uploader_version", 0)
    st.session_state.setdefault("prediction", None)
    st.session_state.setdefault("prediction_payload", None)
    st.session_state.setdefault("prediction_response", None)
    st.session_state.setdefault("prediction_mode", st.session_state["inference_mode"])


def load_selected_dataset(data_mode: str, uploaded: object | None) -> tuple[bool, DatasetBundle]:
    if data_mode == "Paper dataset":
        return True, DATASET_SERVICE.load_paper_sample()
    if uploaded is None:
        return False, DATASET_SERVICE.empty_upload()
    return True, DATASET_SERVICE.load_uploaded_csv(uploaded)


def load_current_dataset() -> tuple[bool, DatasetBundle]:
    return load_selected_dataset(st.session_state.get("data_mode", "Paper dataset"), remember_uploaded_csv(None))


def safe_load_current_dataset() -> tuple[bool, DatasetBundle]:
    try:
        return load_current_dataset()
    except (CandleDataError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Invalid dataset: {exc}")
        st.stop()


def reset_prediction_if_mode_changed(inference_mode: str) -> None:
    if st.session_state.prediction_mode == inference_mode:
        return
    st.session_state.prediction = None
    st.session_state.prediction_payload = None
    st.session_state.prediction_response = None
    st.session_state.prediction_mode = inference_mode


def prepared_dataset_or_stop() -> tuple[DatasetBundle, pd.DataFrame, pd.DataFrame, float, str]:
    dataset_ready, dataset = safe_load_current_dataset()
    if not dataset_ready:
        render_empty_upload(dataset)
        st.stop()

    df = dataset.processed_df
    last_14 = df.tail(14).copy()
    last_close = float(last_14["close"].iloc[-1])
    default_prediction_date = (pd.to_datetime(last_14["date"].iloc[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return dataset, df, last_14, last_close, default_prediction_date


def data_page() -> None:
    data_mode, uploaded = render_data_source_controls()
    try:
        dataset_ready, dataset = load_selected_dataset(data_mode, uploaded)
    except (CandleDataError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        if data_mode == "Upload CSV" and uploaded is not None:
            render_invalid_upload_page(uploaded, source_detail=f"File: {getattr(uploaded, 'name', 'uploaded.csv')}")
            return
        st.error(f"Invalid dataset: {exc}")
        st.stop()
    if not dataset_ready:
        render_empty_upload(dataset)
        st.stop()

    df = dataset.processed_df
    default_prediction_date = (pd.to_datetime(df.tail(14)["date"].iloc[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    render_data_page(dataset, default_prediction_date=default_prediction_date, risk=float(st.session_state.get("risk", 2.0)))


def reproduce_page() -> None:
    render_reproduce_page(
        evaluation_dir=EVALUATION_DIR,
        provenance_dir=REPRODUCE_PROVENANCE_DIR,
        selected_checkpoint_path=SELECTED_CHECKPOINT_PATH,
    )


def predict_page() -> None:
    _, df, last_14, last_close, default_prediction_date = prepared_dataset_or_stop()
    inference_mode = st.session_state.get("inference_mode", "Live Model API")
    reset_prediction_if_mode_changed(inference_mode)
    render_predict_page(
        df=df,
        last_14=last_14,
        last_close=last_close,
        default_prediction_date=default_prediction_date,
        inference_mode=inference_mode,
        api_url=str(st.session_state.get("api_url", "") or "").strip(),
        risk=float(st.session_state.get("risk", 2.0)),
        offline_prediction_path=OFFLINE_PREDICTION_PATH,
    )


def trading_page() -> None:
    render_trading_page(risk=float(st.session_state.get("risk", 2.0)))


def plan_page() -> None:
    render_plan_page(
        core_root=CORE_ROOT,
        forecast_metrics_path=FORECAST_METRICS_PATH,
        baseline_metrics_path=BASELINE_METRICS_PATH,
        trading_metrics_path=TRADING_METRICS_PATH,
        regime_metrics_path=REGIME_METRICS_PATH,
    )


def main() -> None:
    st.set_page_config(page_title="CryptoMamba Bitcoin Forecast", page_icon="₿", layout="wide")
    render_style()
    render_header()
    initialize_session_state()

    page = st.navigation(
        [
            st.Page(data_page, title="1 · Data"),
            st.Page(reproduce_page, title="2 · Reproduce"),
            st.Page(predict_page, title="3 · Predict"),
            st.Page(trading_page, title="4 · Trading"),
            st.Page(plan_page, title="5 · Plan"),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
