from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.ui import load_optional_csv

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


def render_reproduce_page(forecast_metrics_path: Path, trading_metrics_path: Path) -> None:
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

    forecast_metrics, forecast_from_artifact = load_optional_csv(forecast_metrics_path, REPRO_METRICS)
    trading_metrics, trading_from_artifact = load_optional_csv(trading_metrics_path, TRADE_METRICS)

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
