from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.api_client import ApiClientError, CryptoMambaApiClient
from src.cryptomamba_ui.charts import CHART_CONFIG, candle_chart
from src.cryptomamba_ui.data import build_predict_payload
from src.cryptomamba_ui.trading_logic import mock_predict, pct
from src.cryptomamba_ui.ui import action_html, money, stat_card


def render_predict_page(
    df: pd.DataFrame,
    last_14: pd.DataFrame,
    last_close: float,
    default_prediction_date: str,
    inference_mode: str,
    api_url: str,
    risk: float,
) -> None:
    st.markdown("### 3. Predict")

    previous_inference_mode = st.session_state.get("prediction_mode", inference_mode)
    st.markdown("#### Inference setup")
    cfg_mode, cfg_api, cfg_risk = st.columns([1, 2, 1])
    with cfg_mode:
        inference_mode = st.radio("Mode", ["Live Model API", "Demo Mock"], key="inference_mode")
    with cfg_api:
        if inference_mode == "Live Model API":
            api_url = st.text_input("Colab API URL", key="api_url", placeholder="https://xxx.ngrok-free.app").strip()
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
