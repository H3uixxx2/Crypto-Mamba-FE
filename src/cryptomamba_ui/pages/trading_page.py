from __future__ import annotations

import streamlit as st

from src.cryptomamba_ui.charts import roi_chart
from src.cryptomamba_ui.trading_logic import pct, simulate_trade
from src.cryptomamba_ui.ui import money


def render_trading_page(risk: float) -> None:
    st.markdown("### 4. Trading — one-day decision simulator")
    st.caption(
        "This page explains how one prediction becomes a trading action. "
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
