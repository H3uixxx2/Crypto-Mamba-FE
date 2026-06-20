from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.reproduce_artifacts import ReproduceArtifacts, load_reproduce_artifacts


def render_reproduce_page(
    evaluation_dir: Path,
    provenance_dir: Path,
    selected_checkpoint_path: Path,
) -> None:
    artifacts = load_reproduce_artifacts(evaluation_dir, provenance_dir, selected_checkpoint_path)

    st.markdown("### 2. Reproduce")
    st.caption("Artifact-backed CryptoMamba-v reproduction using the pinned released source and chronological paper split.")

    if artifacts.status != "READY":
        st.error("Phase 2 artifacts: NOT READY")
        for error in artifacts.errors:
            st.markdown(f"- {error}")
        st.caption(
            f"Searched: evaluation={evaluation_dir} · provenance={provenance_dir} · checkpoint={selected_checkpoint_path}"
        )
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Phase 2 evidence", artifacts.status)
    metric_columns[1].metric("Forecast reproduction", artifacts.forecast_status)
    metric_columns[2].metric("Official replay", "VERIFIED")
    metric_columns[3].metric("Baselines", artifacts.baseline_status)

    selection = artifacts.model_selection
    st.markdown(
        f"**Selected checkpoint:** `{selection['selected_checkpoint_type']}`  \n"
        f"**SHA-256:** `{selection['checkpoint_sha256']}`  \n"
        f"**Source commit:** `{selection['source_commit']}`"
    )

    forecast_tab, replay_tab, baseline_tab, evidence_tab = st.tabs(
        ["Forecast", "Trading replay", "Baseline", "Evidence"]
    )
    with forecast_tab:
        _render_forecast(artifacts)
    with replay_tab:
        _render_replay(artifacts)
    with baseline_tab:
        _render_baseline(artifacts)
    with evidence_tab:
        _render_evidence(artifacts, selected_checkpoint_path)


def _render_forecast(artifacts: ReproduceArtifacts) -> None:
    st.markdown(
        "**How to read:** Paper target is the published benchmark. "
        "The official checkpoint verifies the evaluation pipeline. "
        "Our retrained checkpoint is the result used to decide reproduction success. "
        "Largest gap must stay below 5%."
    )
    st.dataframe(forecast_comparison_table(artifacts.forecast_metrics), hide_index=True, width="stretch")
    st.success("Retrained checkpoint passes the agreed 5% forecast-metric tolerance.")
    st.caption("Protocol: global aggregation over the released test split; formulas match scripts/evaluation.py.")


def forecast_comparison_table(forecast_metrics: pd.DataFrame) -> pd.DataFrame:
    official = forecast_metrics.loc[forecast_metrics["result_type"] == "official_checkpoint"].iloc[0]
    retrained = forecast_metrics.loc[forecast_metrics["result_type"] == "retrained_checkpoint"].iloc[0]

    return pd.DataFrame(
        [
            {
                "Source": "Paper target",
                "RMSE ↓": f"{official['paper_RMSE']:,.3f}",
                "MAE ↓": f"{official['paper_MAE']:,.3f}",
                "MAPE ↓": f"{official['paper_MAPE_pct']:.3f}%",
                "Largest gap": "—",
                "Verdict": "REFERENCE",
            },
            {
                "Source": "Official checkpoint",
                "RMSE ↓": f"{official['RMSE']:,.3f}",
                "MAE ↓": f"{official['MAE']:,.3f}",
                "MAPE ↓": f"{official['MAPE_pct']:.3f}%",
                "Largest gap": f"{_largest_forecast_gap(official):.3f}%",
                "Verdict": "VERIFIED",
            },
            {
                "Source": "Our retrained checkpoint",
                "RMSE ↓": f"{retrained['RMSE']:,.3f}",
                "MAE ↓": f"{retrained['MAE']:,.3f}",
                "MAPE ↓": f"{retrained['MAPE_pct']:.3f}%",
                "Largest gap": f"{_largest_forecast_gap(retrained):.3f}%",
                "Verdict": "PASS (<5%)" if retrained["status"] == "PASS" else "FAIL",
            },
        ]
    )


def _largest_forecast_gap(result: pd.Series) -> float:
    return max(result["RMSE_gap_pct"], result["MAE_gap_pct"], result["MAPE_gap_pct"])


def _render_replay(artifacts: ReproduceArtifacts) -> None:
    replay = artifacts.trading_replay_metrics
    st.markdown(
        "**How to read:** Each row is one trading strategy. "
        "Paper is the published balance, Official verifies the replay pipeline, "
        "and Our retrain shows the balance produced by the newly trained checkpoint."
    )
    validation_tab, test_tab = st.tabs(["Validation period", "Test period"])
    for tab, split in ((validation_tab, "val"), (test_tab, "test")):
        with tab:
            st.markdown("**Final balance — starting balance: 100**")
            st.dataframe(
                trading_balance_comparison_table(replay, split),
                hide_index=True,
                width="stretch",
            )
            with st.expander("Risk detail — maximum drawdown"):
                st.caption("MDD is the largest peak-to-trough loss. Lower is better.")
                st.dataframe(
                    trading_drawdown_comparison_table(replay, split),
                    hide_index=True,
                    width="stretch",
                )
    st.warning(
        "Official checkpoint: VERIFIED. Our retrained checkpoint: NOT MATCHED on trading balances. "
        "Forecast reproduction still passes; the trading mismatch remains an explicit limitation."
    )
    st.caption("Protocol: released utils.trade.trade, chronological order, zero transaction cost, initial balance 100.")


def trading_balance_comparison_table(replay: pd.DataFrame, split: str) -> pd.DataFrame:
    official = replay[
        (replay["result_type"] == "official_checkpoint") & (replay["split"] == split)
    ].set_index("trade_mode")
    retrained = replay[
        (replay["result_type"] == "retrained_checkpoint") & (replay["split"] == split)
    ].set_index("trade_mode")

    rows: list[dict[str, str]] = []
    for mode, strategy in (
        ("vanilla", "Vanilla"),
        ("smart", "Smart"),
        ("smart_w_short", "Smart + short"),
    ):
        official_row = official.loc[mode]
        retrained_row = retrained.loc[mode]
        rows.append(
            {
                "Strategy": strategy,
                "Paper": f"{official_row['paper_final_balance']:.2f}",
                "Official": f"{official_row['final_balance']:.2f}",
                "Our retrain": f"{retrained_row['final_balance']:.2f}",
                "Our gap": f"{retrained_row['balance_gap_pct']:+.2f}%",
                "Verdict": "VERIFIED" if retrained_row["status"] == "VERIFIED" else "NOT MATCHED",
            }
        )
    return pd.DataFrame(rows)


def trading_drawdown_comparison_table(replay: pd.DataFrame, split: str) -> pd.DataFrame:
    official = replay[
        (replay["result_type"] == "official_checkpoint") & (replay["split"] == split)
    ].set_index("trade_mode")
    retrained = replay[
        (replay["result_type"] == "retrained_checkpoint") & (replay["split"] == split)
    ].set_index("trade_mode")

    rows: list[dict[str, str]] = []
    for mode, strategy in (
        ("vanilla", "Vanilla"),
        ("smart", "Smart"),
        ("smart_w_short", "Smart + short"),
    ):
        official_row = official.loc[mode]
        retrained_row = retrained.loc[mode]
        difference = retrained_row["max_drawdown_pct"] - official_row["paper_max_drawdown_pct"]
        rows.append(
            {
                "Strategy": strategy,
                "Paper MDD": f"{official_row['paper_max_drawdown_pct']:.2f}%",
                "Official MDD": f"{official_row['max_drawdown_pct']:.2f}%",
                "Our MDD": f"{retrained_row['max_drawdown_pct']:.2f}%",
                "Difference": f"{difference:+.2f} pp",
            }
        )
    return pd.DataFrame(rows)


def _render_baseline(artifacts: ReproduceArtifacts) -> None:
    baseline = artifacts.baseline_metrics[artifacts.baseline_metrics["split"] == "test"][
        ["model", "samples", "RMSE", "MAE", "MAPE_pct", "protocol"]
    ].copy()
    baseline = baseline.rename(
        columns={
            "model": "Model",
            "samples": "Samples",
            "MAPE_pct": "MAPE %",
            "protocol": "Protocol",
        }
    )
    st.dataframe(
        baseline.style.format({"RMSE": "{:,.3f}", "MAE": "{:,.3f}", "MAPE %": "{:.3f}"}),
        hide_index=True,
        width="stretch",
    )

    retrained_rmse = float(
        artifacts.forecast_metrics.loc[
            artifacts.forecast_metrics["result_type"] == "retrained_checkpoint",
            "RMSE",
        ].iloc[0]
    )
    naive_rmse = float(
        artifacts.baseline_metrics.loc[
            (artifacts.baseline_metrics["model"] == "naive_persistence")
            & (artifacts.baseline_metrics["split"] == "test"),
            "RMSE",
        ].iloc[0]
    )
    if naive_rmse < retrained_rmse:
        st.warning(
            f"Naive persistence has lower test RMSE ({naive_rmse:,.3f}) than the retrained checkpoint "
            f"({retrained_rmse:,.3f}). This result must not be hidden."
        )
    st.info("Baseline scope is PARTIAL: only naive persistence is currently available.")


def _render_evidence(artifacts: ReproduceArtifacts, selected_checkpoint_path: Path) -> None:
    evidence = evidence_table(artifacts, selected_checkpoint_path)
    st.dataframe(evidence, hide_index=True, width="stretch")
    st.caption(artifacts.model_selection["selection_reason"])
    with st.expander("Released source contract"):
        st.json(artifacts.source_contract)


def evidence_table(artifacts: ReproduceArtifacts, selected_checkpoint_path: Path) -> pd.DataFrame:
    validation = artifacts.artifact_validation
    selection = artifacts.model_selection
    fixture = artifacts.inference_fixture
    evidence = pd.DataFrame(
        [
            {"Evidence": "Artifact validation", "Value": validation["status"]},
            {"Evidence": "Forecast rows", "Value": validation["forecast_rows"]},
            {"Evidence": "Historical prediction rows", "Value": validation["prediction_rows"]},
            {"Evidence": "Trading replay rows", "Value": validation["replay_rows"]},
            {"Evidence": "Selected checkpoint", "Value": str(selected_checkpoint_path)},
            {"Evidence": "Checkpoint SHA-256", "Value": selection["checkpoint_sha256"]},
            {"Evidence": "Golden fixture shape", "Value": str(fixture["expected_tensor_shape"])},
            {"Evidence": "Golden fixture prediction", "Value": fixture["expected_predicted_close"]},
        ]
    )
    evidence["Value"] = evidence["Value"].astype(str)
    return evidence
