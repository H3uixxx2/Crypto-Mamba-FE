from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.api_client import ApiClientError, CryptoMambaApiClient
from src.cryptomamba_ui.charts import CHART_CONFIG, candle_chart
from src.cryptomamba_ui.data import build_predict_payload
from src.cryptomamba_ui.trading_logic import mock_predict, pct
from src.cryptomamba_ui.ui import action_html, money, stat_card

# Inference mode labels — exactly three states the spec requires.
_MODE_LIVE = "Live Model API"
_MODE_OFFLINE = "Offline prediction"
_MODE_MOCK = "Demo Mock"
_ALL_MODES = [_MODE_LIVE, _MODE_OFFLINE, _MODE_MOCK]


def _load_offline_artifact(path: Path) -> dict[str, Any]:
    """Load and validate the frozen offline prediction artifact.

    Returns the inner response dict with inference_type overridden to 'offline'.
    Raises ValueError with a user-readable message on any problem.
    """
    if not path.exists():
        raise ValueError(
            f"Offline prediction artifact not found: {path}\n"
            "Run the Phase 3 Colab export cell first, then copy "
            "offline_prediction.json to CryptoMamba/output/evaluation/."
        )
    try:
        artifact = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read offline prediction artifact: {exc}") from exc

    if artifact.get("artifact_type") != "offline_prediction":
        raise ValueError(
            "File does not look like a Phase 3 offline prediction artifact "
            f"(artifact_type={artifact.get('artifact_type')!r}). "
            "Regenerate it from the Colab export cell."
        )
    response = artifact.get("response")
    if not isinstance(response, dict):
        raise ValueError("Offline artifact is missing 'response' field.")

    required = {"prediction_date", "last_close", "predicted_close", "vanilla_action", "smart_action"}
    missing = required - response.keys()
    if missing:
        raise ValueError(f"Offline artifact response is missing fields: {missing}")

    result = dict(response)
    result["inference_type"] = "offline"
    result["_offline_generated_at"] = artifact.get("generated_at_utc", "unknown")
    result["_offline_warning"] = artifact.get("warning", "")
    return result


def render_predict_page(
    df: pd.DataFrame,
    last_14: pd.DataFrame,
    last_close: float,
    default_prediction_date: str,
    inference_mode: str,
    api_url: str,
    risk: float,
    offline_prediction_path: Path | None = None,
) -> None:
    st.markdown("### 3. Predict")

    previous_inference_mode = st.session_state.get("prediction_mode", inference_mode)
    st.markdown("#### Inference setup")
    cfg_mode, cfg_api, cfg_risk = st.columns([1, 2, 1])

    with cfg_mode:
        inference_mode = st.radio("Mode", _ALL_MODES, key="inference_mode")

    with cfg_api:
        if inference_mode == _MODE_LIVE:
            api_url = st.text_input(
                "Colab API URL", key="api_url", placeholder="https://xxx.ngrok-free.app"
            ).strip()
            if st.button("Check API", disabled=not api_url):
                try:
                    st.json(CryptoMambaApiClient(api_url).health())
                except (ValueError, ApiClientError) as exc:
                    st.error(str(exc))

        elif inference_mode == _MODE_OFFLINE:
            offline_path = offline_prediction_path or Path(
                "CryptoMamba/output/evaluation/offline_prediction.json"
            )
            if offline_path.exists():
                st.success(f"Offline artifact found: `{offline_path.name}`")
            else:
                st.warning(
                    "No offline artifact found at "
                    f"`{offline_path}`.\n\n"
                    "Run the Phase 3 Colab export cell, download `offline_prediction.json`, "
                    "and copy it to `CryptoMamba/output/evaluation/`."
                )

        else:
            st.caption("Demo Mock does not load a CryptoMamba checkpoint. Use only to test the UI flow.")

    with cfg_risk:
        risk = st.slider("Risk band", 0.5, 10.0, key="risk", step=0.5)

    # Clear stale prediction when the user switches inference mode.
    if previous_inference_mode != inference_mode:
        st.session_state.prediction = None
        st.session_state.prediction_payload = None
        st.session_state.prediction_response = None
        st.session_state.prediction_mode = inference_mode

    # Source badge + candle chart.
    mode_col, input_col = st.columns([1, 2])
    with mode_col:
        if inference_mode == _MODE_LIVE:
            stat_card("Inference", "Live API", "CryptoMamba checkpoint")
        elif inference_mode == _MODE_OFFLINE:
            stat_card("Inference", "Offline", "frozen prediction — not a live call")
        else:
            stat_card("Inference", "Demo Mock", "no checkpoint loaded")
            st.warning("Demo Mock does not load CryptoMamba. Use only to validate the UI flow.")

    with input_col:
        st.plotly_chart(
            candle_chart(last_14, "Last 14 candles used as model input", height=320),
            use_container_width=True,
            config=CHART_CONFIG,
        )

    # Input controls (only relevant for Live and Mock; Offline uses the frozen payload).
    if inference_mode != _MODE_OFFLINE:
        c1, c2, c3 = st.columns(3)
        with c1:
            current_price = st.number_input(
                "Current BTC price", min_value=1.0, value=last_close, step=100.0, format="%.2f"
            )
        with c2:
            prediction_date = st.date_input(
                "Prediction date", value=pd.to_datetime(default_prediction_date)
            ).strftime("%Y-%m-%d")
        with c3:
            if inference_mode == _MODE_MOCK:
                mock_bias = st.slider("Mock scenario", -8.0, 8.0, 1.5, 0.25)
            else:
                st.metric("Payload", "14 candles")
                mock_bias = 0.0

        payload = build_predict_payload(df, prediction_date=prediction_date, risk=risk)
        payload["candles"][-1]["close"] = float(current_price)

        with st.expander("Request payload", expanded=False):
            st.json(payload)
    else:
        # Offline mode: inputs are frozen inside the artifact.
        current_price = last_close
        prediction_date = default_prediction_date
        mock_bias = 0.0
        payload = {}

    # Run prediction button.
    if st.button("Run prediction", type="primary", use_container_width=True):
        if inference_mode == _MODE_LIVE:
            if not api_url:
                st.error("Live Model API requires a Colab API URL.")
                st.stop()
            try:
                result = CryptoMambaApiClient(api_url).predict(payload)
            except (ValueError, ApiClientError) as exc:
                st.error(str(exc))
                st.stop()

        elif inference_mode == _MODE_OFFLINE:
            offline_path = offline_prediction_path or Path(
                "CryptoMamba/output/evaluation/offline_prediction.json"
            )
            try:
                result = _load_offline_artifact(offline_path)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        else:
            patched = df.copy()
            patched.loc[patched.index[-1], "close"] = current_price
            result = mock_predict(
                patched,
                current_price=current_price,
                bias_pct=mock_bias,
                risk=risk,
                prediction_date=prediction_date,
            )

        st.session_state.prediction = result
        st.session_state.prediction_payload = payload
        st.session_state.prediction_response = result

    # Result display.
    prediction = st.session_state.prediction
    if prediction:
        p_last = float(prediction["last_close"])
        p_pred = float(prediction["predicted_close"])
        move = pct(p_pred, p_last)

        itype = prediction.get("inference_type", "unknown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Last close", money(p_last))
        m2.metric("Predicted close", money(p_pred), f"{move:+.2f}%")
        m3.metric("Source", prediction.get("model_id", itype))

        # Honesty banners — never let offline or mock pass as live.
        if itype == "offline":
            generated_at = prediction.get("_offline_generated_at", "unknown")
            st.info(
                f"Offline prediction loaded from frozen artifact (generated {generated_at}). "
                "This is NOT a live API call."
            )
            warning_text = prediction.get("_offline_warning", "")
            if warning_text:
                st.warning(warning_text)
        elif itype == "mock":
            st.warning("Demo Mock result only — do not present as CryptoMamba model output.")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                f"<div class='card'><div class='label'>Vanilla signal</div>"
                f"<div class='number'>{action_html(prediction['vanilla_action'])}</div></div>",
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"<div class='card'><div class='label'>Smart signal</div>"
                f"<div class='number'>{action_html(prediction['smart_action'], float(prediction.get('smart_pct', 0) or 0))}</div></div>",
                unsafe_allow_html=True,
            )

        st.plotly_chart(
            candle_chart(last_14, "Last 14 candles + next-day prediction", prediction),
            use_container_width=True,
            config=CHART_CONFIG,
        )
        with st.expander("Response", expanded=False):
            st.json({k: v for k, v in prediction.items() if not k.startswith("_")})

    else:
        if inference_mode == _MODE_LIVE and not api_url:
            st.info("Paste the Colab API URL above, then Check API / Run prediction.")
        elif inference_mode == _MODE_OFFLINE:
            offline_path = offline_prediction_path or Path(
                "CryptoMamba/output/evaluation/offline_prediction.json"
            )
            if not offline_path.exists():
                st.info(
                    "Copy `offline_prediction.json` to `CryptoMamba/output/evaluation/`, "
                    "then click Run prediction."
                )
            else:
                st.info("Click Run prediction to load the offline artifact.")
        else:
            st.info("Run prediction to create a result.")
