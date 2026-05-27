from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.cryptomamba_ui.api_client import ApiClientError, CryptoMambaApiClient
from src.cryptomamba_ui.data import (
    CandleDataError,
    build_predict_payload,
    model_tensor_preview,
    normalize_candles,
    split_summary,
)
from src.cryptomamba_ui.dataset_service import DatasetService

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "sample_data" / "btc_ohlcv_paper_splits.csv"
DATASET_SERVICE = DatasetService(sample_path=SAMPLE_PATH, display_root=ROOT)
CORE_ROOT = ROOT.parent / "CryptoMamba"
EVALUATION_DIR = CORE_ROOT / "output" / "evaluation"
FORECAST_METRICS_PATH = EVALUATION_DIR / "forecast_metrics.csv"
BASELINE_METRICS_PATH = EVALUATION_DIR / "baseline_metrics.csv"
TRADING_METRICS_PATH = EVALUATION_DIR / "trading_metrics.csv"
REGIME_METRICS_PATH = EVALUATION_DIR / "regime_metrics.csv"

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="CryptoMamba Bitcoin Demo", page_icon="₿", layout="wide")


# -----------------------------
# Style: minimal, demo-first
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px;}
    h1, h2, h3 {letter-spacing: -0.03em;}
    .topbar {
        padding: 1.1rem 1.25rem;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 8px 28px rgba(15, 23, 42, .06);
        margin-bottom: 1rem;
    }
    .eyebrow {font-size: .82rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;}
    .lead {color: #475569; font-size: 1.02rem; line-height: 1.55; margin: .25rem 0 0 0;}
    .card {
        padding: 1rem 1.05rem;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .04);
    }
    .soft {
        padding: .9rem 1rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .number {font-size: 1.75rem; font-weight: 800; color: #0f172a; line-height: 1.15;}
    .label {font-size: .88rem; color: #64748b; margin-bottom: .2rem;}
    .ok {color:#059669; font-weight:800;}
    .warn {color:#d97706; font-weight:800;}
    .bad {color:#dc2626; font-weight:800;}
    .muted {color:#64748b;}
    .pill {
        display:inline-block;
        padding:.22rem .55rem;
        border:1px solid #cbd5e1;
        border-radius:999px;
        background:#f8fafc;
        color:#334155;
        font-size:.78rem;
        font-weight:700;
        margin-right:.25rem;
    }
    .legend-row {display:flex; align-items:center; gap:.45rem; margin:.38rem 0;}
    .legend-box {width:14px; height:22px; border-radius:3px; display:inline-block;}
    .up-candle {background:#16a34a;}
    .down-candle {background:#dc2626;}
    .wick {
        width: 2px;
        height: 24px;
        background: #64748b;
        display: inline-block;
        margin: 0 6px;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: .78rem .9rem;
        background: #ffffff;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .04);
    }
    .stTabs [data-baseweb="tab-list"] {gap: .25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Small utilities
# -----------------------------
def money(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def pct(predicted: float, current: float) -> float:
    if current == 0:
        return 0.0
    return (predicted - current) / current * 100


def date_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "—"
    return f"{df['date'].min()} → {df['date'].max()}"


def stat_card(title: str, value: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div class='muted' style='margin-top:.35rem;'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="label">{title}</div>
          <div class="number">{value}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_optional_csv(path: Path, fallback: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        return fallback.copy(), False
    try:
        return pd.read_csv(path), True
    except (OSError, pd.errors.ParserError) as exc:
        st.warning(f"Cannot read artifact {path.name}: {exc}")
        return fallback.copy(), False


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
) -> pd.DataFrame:
    in_scope = processed_df[processed_df["dataset_split"].isin(["train", "validation", "test"])]
    return pd.DataFrame(
        [
            {"status": "passed", "stage": "Read CSV", "rows": len(raw_df), "artifact": "raw dataframe"},
            {"status": "passed", "stage": "Validate schema", "rows": len(normalized_df), "artifact": "date/open/high/low/close/volume"},
            {"status": "passed", "stage": "Build daily OHLCV", "rows": len(daily_df), "artifact": "1 row per day"},
            {"status": "passed", "stage": "Apply paper split", "rows": len(in_scope), "artifact": "train/validation/test"},
            {"status": "passed", "stage": "Build tensor", "rows": 14, "artifact": str(tensor_shape)},
        ]
    )


def render_current_dataset_processing(raw_source_df: pd.DataFrame, daily_source_df: pd.DataFrame, processed_df: pd.DataFrame, source_label: str, source_detail: str) -> None:
    st.markdown("### Current dataset processing")
    st.caption(
        f"Source: {source_label}. {source_detail}. "
        "This uses the dataset selected on the Data screen. Processing runs locally with pandas; no model inference is executed on the Data screen."
    )

    try:
        normalized_df = normalize_candles(raw_source_df)
        tensor_table, tensor_payload = model_tensor_preview(processed_df)
    except (CandleDataError, ValueError, KeyError) as exc:
        st.error(f"Invalid current dataset: {exc}")
        return

    required_view = ["date", "open", "high", "low", "close", "volume"]
    cleaned_daily = daily_source_df[required_view].copy()
    cleaned_with_split = processed_df[[*required_view, "dataset_split"]].copy()
    in_scope = processed_df[processed_df["dataset_split"].isin(["train", "validation", "test"])]
    source = in_scope if len(in_scope) >= 15 else processed_df
    model_input = model_window_preview(source)
    stage_summary = processing_summary(
        raw_source_df,
        normalized_df,
        daily_source_df,
        processed_df,
        tensor_payload["tensor_shape_sent_to_model"],
    )

    summary_tab, compare_tab, split_tab, window_tab, tensor_tab = st.tabs(
        ["Summary", "Before / After", "Split", "Window", "Tensor"]
    )

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

    with window_tab:
        st.dataframe(model_input, hide_index=True, height=330, width="stretch")

    with tensor_tab:
        st.markdown(
            """
            <div class="soft">
            <code>[1, 6, 14]</code> · <code>Timestamp, Open, High, Low, Close, Volume</code> · Volume / 1e9
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(tensor_table, hide_index=True, height=330, width="stretch")
        with st.expander("Tensor payload", expanded=False):
            st.json(tensor_payload)



def vanilla_signal(current: float, predicted: float, threshold: float = 0.01) -> str:
    move = (predicted - current) / current
    if move > threshold:
        return "buy"
    if move < -threshold:
        return "sell"
    return "hold"


def smart_signal(current: float, predicted: float, risk_pct: float) -> tuple[str, float]:
    band = predicted * risk_pct / 100
    if band <= 0:
        return "hold", 0.0
    if current > predicted + band:
        return "sell", 100.0
    if current > predicted:
        return "sell", round(max(0.0, min(100.0, (current - predicted) / band * 100)), 1)
    if current > predicted - band:
        return "buy", round(max(0.0, min(100.0, (predicted - current) / band * 100)), 1)
    return "buy", 100.0


def mock_predict(df: pd.DataFrame, current_price: float, bias_pct: float, risk: float, prediction_date: str) -> dict[str, Any]:
    candles = normalize_candles(df)
    closes = candles["close"].astype(float)
    momentum_pct = ((closes.iloc[-1] / closes.iloc[-7]) - 1) * 100 if len(closes) >= 7 and closes.iloc[-7] else 0.0
    dampened = max(-3.0, min(3.0, momentum_pct * 0.35))
    predicted = current_price * (1 + (bias_pct + dampened) / 100)
    smart_action, smart_pct = smart_signal(current_price, predicted, risk)
    return {
        "model_id": "demo_mock_no_checkpoint",
        "inference_type": "mock",
        "prediction_date": prediction_date,
        "last_close": round(current_price, 2),
        "predicted_close": round(predicted, 2),
        "predicted_change_pct": round(pct(predicted, current_price), 3),
        "vanilla_action": vanilla_signal(current_price, predicted),
        "smart_action": smart_action,
        "smart_pct": smart_pct,
        "note": "Demo mock only. No CryptoMamba checkpoint is loaded.",
    }


def action_html(action: str, amount: float | None = None) -> str:
    css = "ok" if action == "buy" else "bad" if action == "sell" else "warn"
    label = action.upper()
    suffix = f" · {amount:.1f}%" if amount is not None and action != "hold" else ""
    return f"<span class='{css}'>{label}{suffix}</span>"


# -----------------------------
# Charts
# -----------------------------
CHART_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": False,
    "showTips": False,
}


def full_split_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    split_colors = {
        "train": "#16a34a",
        "validation": "#f59e0b",
        "test": "#dc2626",
        "out_of_scope": "#64748b",
    }

    if "dataset_split" in df.columns:
        for split_name, label in [
            ("train", "Train"),
            ("validation", "Validation"),
            ("test", "Test"),
            ("out_of_scope", "Out of paper range"),
        ]:
            part = df[df["dataset_split"] == split_name]
            if part.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=part["date"],
                    y=part["close"],
                    mode="lines",
                    name=label,
                    line=dict(color=split_colors[split_name], width=1.9),
                    hovertemplate=f"{label}<br>Ngày=%{{x}}<br>Close=%{{y:,.2f}}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(color="#2563eb", width=1.9),
                hovertemplate="Ngày=%{x}<br>Close=%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=410,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        yaxis_title="Close price",
        xaxis_title="Ngày",
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def candle_chart(df: pd.DataFrame, title: str, prediction: dict[str, Any] | None = None, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC candle",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    if prediction:
        fig.add_trace(
            go.Scatter(
                x=[df["date"].iloc[-1], prediction["prediction_date"]],
                y=[float(prediction["last_close"]), float(prediction["predicted_close"])],
                mode="lines+markers",
                name="Prediction",
                line=dict(color="#f97316", width=3, dash="dash"),
                marker=dict(size=9),
            )
        )
    fig.update_layout(
        height=height,
        title=title,
        margin=dict(l=12, r=12, t=48, b=12),
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def roi_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#2563eb", "#0f766e"]
    fig.add_trace(go.Bar(x=sim_df["strategy"], y=sim_df["roi_pct"], marker_color=colors, text=sim_df["roi_pct"].map(lambda x: f"{x:+.2f}%"), textposition="outside"))
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=28, b=12), yaxis_title="ROI (%)", showlegend=False)
    return fig


# -----------------------------
# Static, verified reproduction summary from current notebook run
# -----------------------------
REPRO_METRICS = pd.DataFrame(
    [
        {"Metric": "RMSE", "Paper": 1598.10, "Observed": 1600.53, "Delta": "+0.15%"},
        {"Metric": "MAE", "Paper": 1120.70, "Observed": 1119.20, "Delta": "-0.13%"},
    ]
)

TRADE_METRICS = pd.DataFrame(
    [
        {"Split": "Validation", "Mode": "Vanilla", "Paper balance": 124.09, "Observed": 124.90},
        {"Split": "Validation", "Mode": "Smart", "Paper balance": 127.12, "Observed": 127.12},
        {"Split": "Test", "Mode": "Vanilla", "Paper balance": 246.58, "Observed": 246.58},
        {"Split": "Test", "Mode": "Smart", "Paper balance": 213.20, "Observed": 213.20},
    ]
)


def simulate_trade(current: float, predicted: float, capital: float, btc: float, risk: float, realized_price: float) -> pd.DataFrame:
    start_value = capital + btc * current
    rows: list[dict[str, Any]] = []

    # Vanilla: all-in or all-out
    action = vanilla_signal(current, predicted)
    cash_v, btc_v = capital, btc
    if action == "buy" and cash_v > 0:
        btc_v += cash_v / current
        cash_v = 0.0
    elif action == "sell" and btc_v > 0:
        cash_v += btc_v * current
        btc_v = 0.0
    end_value = cash_v + btc_v * realized_price
    rows.append({
        "strategy": "Vanilla",
        "action": action.upper(),
        "trade_size": "100%" if action != "hold" else "0%",
        "end_value": end_value,
        "pnl": end_value - start_value,
        "roi_pct": (end_value - start_value) / start_value * 100 if start_value else 0,
    })

    # Smart: partial trade by risk band
    action, amount = smart_signal(current, predicted, risk)
    cash_s, btc_s = capital, btc
    fraction = amount / 100
    if action == "buy" and cash_s > 0:
        spend = cash_s * fraction
        btc_s += spend / current
        cash_s -= spend
    elif action == "sell" and btc_s > 0:
        sold = btc_s * fraction
        cash_s += sold * current
        btc_s -= sold
    end_value = cash_s + btc_s * realized_price
    rows.append({
        "strategy": "Smart",
        "action": action.upper(),
        "trade_size": f"{amount:.1f}%",
        "end_value": end_value,
        "pnl": end_value - start_value,
        "roi_pct": (end_value - start_value) / start_value * 100 if start_value else 0,
    })
    return pd.DataFrame(rows)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="topbar">
      <div class="eyebrow">CryptoMamba Graduation Project</div>
      <h1 style="margin:.15rem 0 .2rem 0;">Bitcoin forecast + trading simulation</h1>
      <p class="lead">Demo tối giản: đi từ dữ liệu → reproduce paper → dự đoán 1 ngày → mô phỏng giao dịch → kế hoạch mở rộng.</p>
      <div style="margin-top:.7rem;">
        <span class="pill">BTC/USD daily</span>
        <span class="pill">CryptoMamba-v</span>
        <span class="pill">Streamlit local</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Top-level navigation + screen-local controls
# -----------------------------
screen_options = [
    "1 · Data",
    "2 · Reproduce",
    "3 · Predict",
    "4 · Trading",
    "5 · Plan",
]

screen = st.radio(
    "Demo path",
    screen_options,
    horizontal=True,
    key="screen",
)

st.session_state.setdefault("data_mode", "Paper sample")
st.session_state.setdefault("inference_mode", "Live Model API")
st.session_state.setdefault("api_url", os.getenv("CRYPTO_MAMBA_API_URL", ""))
st.session_state.setdefault("risk", 2.0)

if screen == "1 · Data":
    st.markdown("#### Dataset source")
    data_mode = st.radio(
        "Dataset source",
        ["Paper sample", "Upload CSV"],
        horizontal=True,
        key="data_mode",
    )
    uploaded = None
    if data_mode == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            key="uploaded_csv",
            help="Required columns: date/timestamp, open, high, low, close, volume.",
        )
        st.caption("Upload CSV để chạy pipeline. Nếu chưa upload, Data screen sẽ không render dữ liệu giả.")
else:
    data_mode = st.session_state.get("data_mode", "Paper sample")
    uploaded = st.session_state.get("uploaded_csv")

inference_mode = st.session_state.get("inference_mode", "Live Model API")
api_url = str(st.session_state.get("api_url", "") or "").strip()
risk = float(st.session_state.get("risk", 2.0))


# -----------------------------
# Load data
# -----------------------------
dataset_ready = True
try:
    if data_mode == "Paper sample":
        dataset = DATASET_SERVICE.load_paper_sample()
    else:
        if uploaded is None:
            dataset_ready = False
            dataset = DATASET_SERVICE.empty_upload()
        else:
            dataset = DATASET_SERVICE.load_uploaded_csv(uploaded)
except (CandleDataError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Invalid dataset: {exc}")
    st.stop()

raw_df = dataset.raw_df
daily_df = dataset.daily_df
df = dataset.processed_df
dataset_source_label = dataset.source_label
dataset_source_detail = dataset.source_detail
dataset_split_strategy = dataset.split_strategy

if not dataset_ready:
    st.markdown("### 1. Data")
    st.caption(f"Current dataset source: {dataset_source_label}. {dataset_source_detail}")
    st.info("Chưa có CSV nên không hiển thị data/pipeline. Upload file hoặc chọn Paper sample.")
    st.dataframe(
        pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]),
        hide_index=True,
        width="stretch",
    )
    st.stop()

last_14 = df.tail(14).copy()
last_close = float(last_14["close"].iloc[-1])
default_prediction_date = (pd.to_datetime(last_14["date"].iloc[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "prediction_payload" not in st.session_state:
    st.session_state.prediction_payload = None
if "prediction_response" not in st.session_state:
    st.session_state.prediction_response = None
if "prediction_mode" not in st.session_state:
    st.session_state.prediction_mode = inference_mode
if st.session_state.prediction_mode != inference_mode:
    st.session_state.prediction = None
    st.session_state.prediction_payload = None
    st.session_state.prediction_response = None
    st.session_state.prediction_mode = inference_mode


# -----------------------------
# Screens
# -----------------------------
if screen == "1 · Data":
    st.markdown("### 1. Data")
    st.caption(f"Current dataset source: {dataset_source_label}. {dataset_source_detail}")
    train_df = df[df["dataset_split"] == "train"]
    val_df = df[df["dataset_split"] == "validation"]
    test_df = df[df["dataset_split"] == "test"]
    out_of_scope_df = df[df["dataset_split"] == "out_of_scope"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stat_card("Total daily candles", f"{len(df):,}", date_range(df))
    with c2:
        stat_card("Train", f"{len(train_df):,}", date_range(train_df))
    with c3:
        stat_card("Validation", f"{len(val_df):,}", date_range(val_df))
    with c4:
        stat_card("Test", f"{len(test_df):,}", date_range(test_df))
    with c5:
        stat_card("Out of paper range", f"{len(out_of_scope_df):,}", date_range(out_of_scope_df))

    if dataset_split_strategy == "paper_date":
        st.caption("Split strategy: fixed paper-date split for BTC reproduction.")
    else:
        st.caption("Split strategy: chronological ratio split for uploaded data: train 70%, validation 15%, test 15%.")

    chart_title = (
        "Paper sample close price — train/validation/test split"
        if dataset_source_label == "Paper sample"
        else "Uploaded CSV close price — processed split coverage"
    )
    st.plotly_chart(full_split_chart(df, chart_title), use_container_width=True, config=CHART_CONFIG)

    st.markdown("#### 14-day input")
    st.table(model_window_preview(df.tail(15)))

    payload_col, tensor_col = st.columns(2)
    with payload_col:
        with st.expander("Predict/API payload", expanded=False):
            st.caption("Transport format sent by Streamlit. The API converts these 14 candles into the model tensor.")
            st.json(build_predict_payload(df, prediction_date=default_prediction_date, risk=risk))
    with tensor_col:
        with st.expander("Model tensor contract", expanded=False):
            _, tensor_payload = model_tensor_preview(df)
            st.caption("CryptoMamba-v feature order and shape used after API conversion.")
            st.json({key: tensor_payload[key] for key in ["feature_order", "tensor_shape_before_batch", "tensor_shape_sent_to_model", "volume_rule", "normalize"]})

    st.divider()
    render_current_dataset_processing(raw_df, daily_df, df, dataset_source_label, dataset_source_detail)

    with st.expander("Current dataset detail", expanded=False):
        st.caption(f"Source: {dataset_source_label}. {dataset_source_detail}")
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


elif screen == "2 · Reproduce":
    st.markdown("### 2. Reproduce — checkpoint/paper có khớp không?")
    st.markdown(
        """
        <div class="soft">
        <b>Confirmed:</b> official CryptoMamba-v checkpoint replay has been checked against the paper trading table.
        Full retraining, baseline comparison, transaction-cost backtesting, and regime robustness are tracked as separate evaluation artifacts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Target model", "CryptoMamba-v")
    c2.metric("Checkpoint replay", "Verified")
    c3.metric("Baselines", "Pending")

    forecast_metrics, forecast_from_artifact = load_optional_csv(FORECAST_METRICS_PATH, REPRO_METRICS)
    trading_metrics, trading_from_artifact = load_optional_csv(TRADING_METRICS_PATH, TRADE_METRICS)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Forecast metrics")
        st.dataframe(forecast_metrics, hide_index=True, width="stretch")
        st.caption("Source: forecast_metrics.csv" if forecast_from_artifact else "Source: notebook summary fallback; replace with forecast_metrics.csv.")
    with right:
        st.markdown("#### Trading replay")
        table = trading_metrics.copy()
        if {"Observed", "Paper balance"}.issubset(table.columns):
            table["Delta"] = table["Observed"] - table["Paper balance"]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Source: trading_metrics.csv" if trading_from_artifact else "Source: checkpoint replay fallback; replace with trading_metrics.csv.")

    st.info("Next evidence needed: baseline_metrics.csv and transaction-cost-aware trading_metrics.csv.")

elif screen == "3 · Predict":
    st.markdown("### 3. Predict")

    previous_inference_mode = st.session_state.get("prediction_mode", inference_mode)
    st.markdown("#### Inference setup")
    cfg_mode, cfg_api, cfg_risk = st.columns([1, 2, 1])
    with cfg_mode:
        inference_mode = st.radio(
            "Mode",
            ["Live Model API", "Demo Mock"],
            key="inference_mode",
        )
    with cfg_api:
        if inference_mode == "Live Model API":
            api_url = st.text_input(
                "Colab API URL",
                key="api_url",
                placeholder="https://xxx.ngrok-free.app",
            ).strip()
            if st.button("Check API", disabled=not api_url):
                try:
                    st.json(CryptoMambaApiClient(api_url).health())
                except (ValueError, ApiClientError) as exc:
                    st.error(str(exc))
        else:
            api_url = ""
            st.caption("Demo Mock không load checkpoint; chỉ dùng để kiểm thử UI flow.")
    with cfg_risk:
        risk = st.slider("Risk band", 0.5, 10.0, key="risk", step=0.5)

    if previous_inference_mode != inference_mode:
        st.session_state.prediction = None
        st.session_state.prediction_payload = None
        st.session_state.prediction_response = None
        st.session_state.prediction_mode = inference_mode

    mode_col, input_col = st.columns([1, 2])
    with mode_col:
        if inference_mode == "Live Model API":
            stat_card("Inference", "Live API", "CryptoMamba checkpoint")
        else:
            stat_card("Inference", "Demo Mock", "no checkpoint loaded")
            st.warning("Demo Mock does not load CryptoMamba. Use only to validate the UI flow.")
    with input_col:
        st.plotly_chart(candle_chart(last_14, "Last 14 candles used as model input", height=320), use_container_width=True, config=CHART_CONFIG)

    c1, c2, c3 = st.columns(3)
    with c1:
        current_price = st.number_input("Current BTC price", min_value=1.0, value=last_close, step=100.0, format="%.2f")
    with c2:
        prediction_date = st.date_input("Prediction date", value=pd.to_datetime(default_prediction_date)).strftime("%Y-%m-%d")
    with c3:
        if inference_mode == "Demo Mock":
            mock_bias = st.slider("Mock scenario", -8.0, 8.0, 1.5, 0.25)
        else:
            st.metric("Payload", "14 candles")
            mock_bias = 0.0

    payload = build_predict_payload(df, prediction_date=prediction_date, risk=risk)
    payload["candles"][-1]["close"] = float(current_price)

    with st.expander("Request payload", expanded=False):
        st.json(payload)

    if st.button("Run prediction", type="primary", width="stretch"):
        if inference_mode == "Live Model API":
            if not api_url:
                st.error("Live Model API requires Colab API URL.")
                st.stop()
            try:
                result = CryptoMambaApiClient(api_url).predict(payload)
            except (ValueError, ApiClientError) as exc:
                st.error(str(exc))
                st.stop()
        else:
            patched = df.copy()
            patched.loc[patched.index[-1], "close"] = current_price
            result = mock_predict(patched, current_price=current_price, bias_pct=mock_bias, risk=risk, prediction_date=prediction_date)
        st.session_state.prediction = result
        st.session_state.prediction_payload = payload
        st.session_state.prediction_response = result

    prediction = st.session_state.prediction
    if prediction:
        p_last = float(prediction["last_close"])
        p_pred = float(prediction["predicted_close"])
        move = pct(p_pred, p_last)
        m1, m2, m3 = st.columns(3)
        m1.metric("Last close", money(p_last))
        m2.metric("Predicted close", money(p_pred), f"{move:+.2f}%")
        m3.metric("Source", prediction.get("model_id", inference_mode))

        if prediction.get("inference_type") == "mock":
            st.warning("Demo Mock result only — do not present as CryptoMamba model output.")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"<div class='card'><div class='label'>Vanilla signal</div><div class='number'>{action_html(prediction['vanilla_action'])}</div></div>", unsafe_allow_html=True)
        with s2:
            st.markdown(f"<div class='card'><div class='label'>Smart signal</div><div class='number'>{action_html(prediction['smart_action'], float(prediction.get('smart_pct', 0) or 0))}</div></div>", unsafe_allow_html=True)

        st.plotly_chart(candle_chart(last_14, "Last 14 candles + next-day prediction", prediction), use_container_width=True, config=CHART_CONFIG)
        with st.expander("Response", expanded=False):
            st.json(prediction)
    else:
        if inference_mode == "Live Model API" and not api_url:
            st.info("Paste Colab API URL, then Check API / Run prediction.")
        else:
            st.info("Run prediction to create a result.")

elif screen == "4 · Trading":
    st.markdown("### 4. Trading — one-day decision simulator")
    st.caption(
        "This screen explains how one prediction becomes a trading action. "
        "Thesis-grade chronological backtesting with transaction costs is tracked separately in evaluation artifacts."
    )
    prediction = st.session_state.prediction
    if not prediction:
        st.warning("Chưa có prediction. Qua màn 3 chạy prediction trước.")
        st.stop()

    p_last = float(prediction["last_close"])
    p_pred = float(prediction["predicted_close"])
    default_move = pct(p_pred, p_last)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        capital = st.number_input("Cash", min_value=0.0, value=10_000.0, step=500.0)
    with c2:
        btc = st.number_input("BTC holding", min_value=0.0, value=0.0, step=0.01, format="%.4f")
    with c3:
        realized_move = st.slider("Next-day actual move", -15.0, 15.0, default_move, 0.25)
    with c4:
        realized_price = p_last * (1 + realized_move / 100)
        st.metric("Assumed next close", money(realized_price))

    sim = simulate_trade(p_last, p_pred, capital, btc, risk, realized_price)
    display = sim.copy()
    display["end_value"] = display["end_value"].map(money)
    display["pnl"] = display["pnl"].map(money)
    display["roi_pct"] = display["roi_pct"].map(lambda x: f"{x:+.2f}%")
    st.dataframe(display, hide_index=True, width="stretch")
    st.plotly_chart(roi_chart(sim), use_container_width=True)
    st.caption("Vanilla = all-in/all-out nếu forecast lệch >1%. Smart = mua/bán từng phần theo risk band. Đây chưa phải full chronological backtest.")

else:
    st.markdown("### 5. Plan — roadmap theo từng màn")
    st.info("App chưa complete. Các màn hiện là scaffold/demo shell, chỉ được claim thesis result khi có artifact thật.")

    st.markdown("#### Screen status")
    st.dataframe(
        pd.DataFrame(
            [
                {"Screen": "1 · Data", "Status": "Ready for demo", "Current truth": "Paper split + 14-day window shown", "Next work": "CSV validation tests + final visual review"},
                {"Screen": "2 · Reproduce", "Status": "Scaffolded", "Current truth": "Checkpoint replay summary/fallback", "Next work": "Read forecast/baseline artifacts instead of fallback"},
                {"Screen": "3 · Predict", "Status": "Blocked by API", "Current truth": "Demo Mock + request payload scaffold", "Next work": "Colab/FastAPI real cmamba_v inference"},
                {"Screen": "4 · Trading", "Status": "Scaffolded", "Current truth": "One-day decision simulator only", "Next work": "Chronological backtest with transaction costs"},
                {"Screen": "5 · Plan", "Status": "Scaffolded", "Current truth": "Roadmap/status dashboard", "Next work": "Keep synced with ROADMAP.md and artifacts"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="card">
            <div class="label">Scaffolded / partially verified</div>
            <b>1. Data explorer</b>: ready for demo<br>
            <b>2. Checkpoint replay</b>: partial evidence<br>
            <b>3. Predict screen</b>: UI/API scaffold<br>
            <b>4. Trading screen</b>: one-day simulator only<br>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
            <div class="label">Research pending before thesis claim</div>
            <b>1. Baselines</b>: naive done; ARIMA/LSTM/GRU/iTransformer pending<br>
            <b>2. Transaction-cost backtest</b>: 0%, 0.1%, 0.2%<br>
            <b>3. BTC regime robustness</b>: bull/bear/sideways or volatility buckets<br>
            <b>4. Usability evaluation</b>: 5–10 users<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Evaluation artifact slots")
    st.dataframe(
        pd.DataFrame(
            [
                {"Artifact": str(FORECAST_METRICS_PATH.relative_to(CORE_ROOT)), "Purpose": "CryptoMamba-v forecast metrics", "Status": "Ready" if FORECAST_METRICS_PATH.exists() else "Pending"},
                {"Artifact": str(BASELINE_METRICS_PATH.relative_to(CORE_ROOT)), "Purpose": "Naive/ARIMA/LSTM/GRU/iTransformer comparison", "Status": "Ready" if BASELINE_METRICS_PATH.exists() else "Pending"},
                {"Artifact": str(TRADING_METRICS_PATH.relative_to(CORE_ROOT)), "Purpose": "Chronological backtest with costs", "Status": "Ready" if TRADING_METRICS_PATH.exists() else "Pending"},
                {"Artifact": str(REGIME_METRICS_PATH.relative_to(CORE_ROOT)), "Purpose": "BTC regime robustness", "Status": "Ready" if REGIME_METRICS_PATH.exists() else "Pending"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Demo story khi bảo vệ")
    st.markdown(
        """
        1. **Data**: đây là split paper, không dùng 2025 để claim reproduce.  
        2. **Reproduce**: official checkpoint replay là nền tảng kiểm chứng ban đầu.  
        3. **Predict**: UI gửi 14 nến gần nhất; Live API mới là inference thật.  
        4. **Trading**: one-day simulator giải thích logic action, còn full backtest nằm trong artifact.  
        5. **Research evidence**: baseline, transaction cost, regime robustness, usability là phần phải hoàn thiện cho thesis.
        """
    )
