from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.api_client import ApiClientError, CryptoMambaApiClient
from src.cryptomamba_ui.charts import CHART_CONFIG, candle_chart
from src.cryptomamba_ui.data import (
    MODEL_TRAIN_HORIZON,
    MODEL_WINDOW_SIZE,
    CandleDataError,
    build_predict_payload,
    is_out_of_distribution,
    normalize_candles,
    prediction_date_bounds,
    select_window,
)
from src.cryptomamba_ui.trading_logic import pct
from src.cryptomamba_ui.ui import action_html, money, stat_card


def render_predict_page(
    df: pd.DataFrame,
    source_label: str,
    source_detail: str,
    api_url: str,
    risk: float,
) -> None:
    st.markdown("### 3. Predict")
    st.caption(
        f"Model nhận {MODEL_WINDOW_SIZE} nến ngày liên tiếp → dự đoán giá ĐÓNG CỬA của ngày kế tiếp."
    )

    try:
        normalized = normalize_candles(df)
    except CandleDataError as exc:
        st.error(f"Dataset chưa sẵn sàng: {exc}")
        st.stop()

    # ── 1. Nguồn dữ liệu (read-only) ────────────────────────────────────────────
    range_start = pd.to_datetime(normalized["date"].iloc[0]).strftime("%Y-%m-%d")
    range_end = pd.to_datetime(normalized["date"].iloc[-1]).strftime("%Y-%m-%d")
    st.markdown("#### Nguồn dữ liệu")
    d1, d2, d3 = st.columns(3)
    with d1:
        stat_card("Dataset", source_label, source_detail)
    with d2:
        stat_card("Khoảng dữ liệu", f"{range_start} → {range_end}", f"{len(normalized)} nến ngày")
    with d3:
        stat_card("Cửa sổ model", f"{MODEL_WINDOW_SIZE} nến", "input → 1 ngày dự đoán")

    # ── 2. Thiết lập & input ────────────────────────────────────────────────────
    st.markdown("#### Thiết lập dự đoán")
    min_pred, max_pred = prediction_date_bounds(df)
    s1, s2, s3 = st.columns([2, 1, 1])
    with s1:
        api_url = st.text_input(
            "Colab API URL", key="api_url", placeholder="https://xxx.ngrok-free.app"
        ).strip()
    with s2:
        prediction_date = st.date_input(
            "Ngày dự đoán",
            value=max_pred,
            min_value=min_pred,
            max_value=max_pred,
            help=(
                "Ngày cần dự đoán giá đóng cửa. Giới hạn trong khoảng dataset hỗ trợ "
                f"(cần đủ {MODEL_WINDOW_SIZE} nến trước đó)."
            ),
        ).strftime("%Y-%m-%d")
    with s3:
        risk = st.slider("Risk band", 0.5, 10.0, key="risk", step=0.5)

    if st.button("Check API", disabled=not api_url):
        try:
            st.json(CryptoMambaApiClient(api_url).health())
        except (ValueError, ApiClientError) as exc:
            st.error(str(exc))

    # Build the exact 14-candle window that ends the day before prediction_date.
    try:
        window = select_window(df, prediction_date)
    except CandleDataError as exc:
        st.error(str(exc))
        st.stop()

    window_last_close = float(window["close"].iloc[-1])
    window_last_date = pd.to_datetime(window["date"].iloc[-1]).strftime("%Y-%m-%d")

    # Honesty gate: predictions on data beyond the validated paper horizon are
    # extrapolation and must never be shown as validated research evidence.
    out_of_dist = is_out_of_distribution(window)
    if out_of_dist:
        st.warning(
            f"⚠️ **Demo định tính — KHÔNG phải bằng chứng nghiên cứu đã kiểm chứng.** "
            f"Cửa sổ input kết thúc {window_last_date}, vượt ngoài phạm vi model được "
            f"huấn luyện/đánh giá (paper test đến {MODEL_TRAIN_HORIZON}). Checkpoint học "
            f"trên BTC tới 2024 với feature giá/timestamp thô (normalize=False) nên dự đoán "
            f"ngoài vùng này thường sai lệch (extrapolation)."
        )
    elif source_label != "Paper dataset":
        st.info(
            f"Dataset tùy chọn ({source_label}) nằm trong phạm vi ngày đã kiểm chứng "
            f"(đến {MODEL_TRAIN_HORIZON}). Không phải paper split nhưng in-distribution."
        )

    i1, i2 = st.columns([1, 2])
    with i1:
        stat_card("Giá đóng cửa nến cuối", money(window_last_close), f"ngày {window_last_date}")
        stat_card("Dự đoán cho ngày", prediction_date, "giá đóng cửa")
    with i2:
        st.plotly_chart(
            candle_chart(window, f"{MODEL_WINDOW_SIZE} nến input (đến {window_last_date})", height=300),
            use_container_width=True,
            config=CHART_CONFIG,
        )

    payload = build_predict_payload(window, prediction_date=prediction_date, risk=risk)
    with st.expander("Request payload", expanded=False):
        st.json(payload)

    # Drop a stale result if the request changed (dataset / date / risk).
    signature = f"{source_detail}|{prediction_date}|{risk}"
    if st.session_state.get("predict_signature") != signature:
        st.session_state.prediction = None

    if st.button("Run prediction", type="primary", width="stretch"):
        if not api_url:
            st.error("Cần Colab API URL để chạy model thật.")
            st.stop()
        try:
            result = CryptoMambaApiClient(api_url).predict(payload)
        except (ValueError, ApiClientError) as exc:
            st.error(str(exc))
            st.stop()
        st.session_state.prediction = result
        st.session_state.prediction_payload = payload
        st.session_state.prediction_response = result
        st.session_state.predict_signature = signature

    # ── 3. Kết quả ──────────────────────────────────────────────────────────────
    prediction = st.session_state.get("prediction")
    if not prediction:
        st.info("Dán Colab API URL, chọn ngày dự đoán, rồi bấm Run prediction.")
        return

    p_last = float(prediction["last_close"])
    p_pred = float(prediction["predicted_close"])
    move = pct(p_pred, p_last)
    inference_type = str(prediction.get("inference_type", "live"))

    st.markdown("#### Kết quả")
    m1, m2, m3 = st.columns(3)
    m1.metric("Giá đóng cửa gần nhất", money(p_last))
    m2.metric("Dự đoán Close ngày kế", money(p_pred), f"{move:+.2f}%")
    m3.metric("Nguồn", f"{prediction.get('model_id', 'cmamba_v')} · {inference_type}")
    st.caption(
        "⚠️ Đây chỉ là dự đoán giá ĐÓNG CỬA (point forecast). Model không dự đoán "
        "dao động trong ngày (high/low/đường đi của giá)."
    )

    if inference_type not in ("live", "offline"):
        st.warning("Kết quả không phải từ model thật — không dùng làm bằng chứng nghiên cứu.")
    if out_of_dist:
        st.caption(
            "⚠️ Input ngoài phạm vi huấn luyện — demo định tính, không phải bằng chứng nghiên cứu."
        )

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            f"<div class='card'><div class='label'>Vanilla signal</div>"
            f"<div class='number'>{action_html(prediction['vanilla_action'])}</div></div>",
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            f"<div class='card'><div class='label'>Smart signal</div>"
            f"<div class='number'>{action_html(prediction['smart_action'], float(prediction.get('smart_pct', 0) or 0))}</div></div>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(
        candle_chart(window, f"{MODEL_WINDOW_SIZE} nến + điểm dự đoán", prediction),
        use_container_width=True,
        config=CHART_CONFIG,
    )
    with st.expander("Response", expanded=False):
        st.json(prediction)
