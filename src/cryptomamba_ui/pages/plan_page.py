from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_plan_page(core_root: Path, forecast_metrics_path: Path, baseline_metrics_path: Path, trading_metrics_path: Path, regime_metrics_path: Path) -> None:
    st.markdown("### 5. Plan — roadmap theo từng màn")
    st.info("App chưa complete. Các màn hiện là scaffold/demo shell, chỉ được claim thesis result khi có artifact thật.")

    st.markdown("#### Page status")
    st.dataframe(
        pd.DataFrame(
            [
                {"Page": "1 · Data", "Status": "Ready for demo", "Current truth": "Paper split + 14-day window shown", "Next work": "CSV validation tests + final visual review"},
                {"Page": "2 · Reproduce", "Status": "Done", "Current truth": "Validated forecast/replay/baseline/provenance artifacts", "Next work": "Use frozen checkpoint and fixture in Phase 3"},
                {"Page": "3 · Predict", "Status": "Blocked by API", "Current truth": "Demo Mock + request payload scaffold", "Next work": "Colab/FastAPI real cmamba_v inference"},
                {"Page": "4 · Trading", "Status": "Scaffolded", "Current truth": "One-day decision simulator only", "Next work": "Chronological backtest with transaction costs"},
                {"Page": "5 · Plan", "Status": "Scaffolded", "Current truth": "Status dashboard", "Next work": "Keep synced with generated artifacts"},
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
            <div class="label">Validated / scaffolded</div>
            <b>1. Data explorer</b>: ready for demo<br>
            <b>2. Reproduce</b>: artifact-backed and validated<br>
            <b>3. Predict page</b>: UI/API scaffold<br>
            <b>4. Trading page</b>: one-day simulator only<br>
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
                {"Artifact": str(forecast_metrics_path.relative_to(core_root)), "Purpose": "CryptoMamba-v forecast metrics", "Status": "Ready" if forecast_metrics_path.exists() else "Pending"},
                {"Artifact": str(baseline_metrics_path.relative_to(core_root)), "Purpose": "Naive/ARIMA/LSTM/GRU/iTransformer comparison", "Status": "Ready" if baseline_metrics_path.exists() else "Pending"},
                {"Artifact": str(trading_metrics_path.relative_to(core_root)), "Purpose": "Chronological backtest with costs", "Status": "Ready" if trading_metrics_path.exists() else "Pending"},
                {"Artifact": str(regime_metrics_path.relative_to(core_root)), "Purpose": "BTC regime robustness", "Status": "Ready" if regime_metrics_path.exists() else "Pending"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Demo story khi bảo vệ")
    st.markdown(
        """
        1. **Data**: đây là split paper, không dùng 2025 để claim reproduce.  
        2. **Reproduce**: official/retrained forecast, replay, baseline, checkpoint, and provenance are artifact-backed.
        3. **Predict**: UI gửi 14 nến gần nhất; Live API mới là inference thật.  
        4. **Trading**: one-day simulator giải thích logic action, còn full backtest nằm trong artifact.  
        5. **Research evidence**: baseline, transaction cost, regime robustness, usability là phần phải hoàn thiện cho thesis.
        """
    )
