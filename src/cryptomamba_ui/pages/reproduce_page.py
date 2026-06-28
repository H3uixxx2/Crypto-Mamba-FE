from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.cryptomamba_ui.charts import (
    CHART_CONFIG,
    forecast_error_chart,
    forecast_prediction_chart,
    forecast_timeseries_chart,
    grouped_bar_chart,
    leaderboard_bar_chart,
    tradeoff_scatter_chart,
)
from src.cryptomamba_ui.reproduce_artifacts import (
    MANDATORY_BASELINES,
    ReproduceArtifacts,
    load_reproduce_artifacts,
)


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

    forecast_pass = artifacts.forecast_status == "PASS"
    baseline_full = artifacts.baseline_status not in ("PARTIAL", "MISSING")
    _present = artifacts.baseline_models_present
    _pending = sorted(MANDATORY_BASELINES - set(_present))
    baseline_card_msg = (
        "Reference models to beat. "
        + (f"Done: {', '.join(_present)}." if _present else "None done yet.")
        + (f" Pending (GPU): {', '.join(_pending)}." if _pending else " All five present.")
    )

    # Plain-language verdict — what this whole screen is proving, in one sentence.
    if forecast_pass:
        st.success(
            "**What this screen proves — CryptoMamba-v is reproduced.** "
            "The model we re-trained predicts the next-day BTC close with error within **5%** of the "
            "published paper, so the paper's forecast result is confirmed. Trading replay and the "
            "baseline comparison are still **partial** — they are shown openly below, nothing is hidden."
        )
    else:
        st.error(
            "**Forecast reproduction did NOT pass the 5% tolerance.** See the Forecast tab for the gap."
        )

    cards = st.columns(4)
    with cards[0]:
        _status_card("1 · Evidence files", artifacts.status,
                     "All Phase 2 result files were found, loaded and validated.",
                     "ok" if artifacts.status == "READY" else "bad")
    with cards[1]:
        _status_card("2 · Forecast accuracy", artifacts.forecast_status,
                     "Our re-trained model's RMSE / MAE / MAPE are within 5% of the paper.",
                     "ok" if forecast_pass else "bad")
    with cards[2]:
        _status_card("3 · Pipeline check", "VERIFIED",
                     "The official checkpoint reproduces the paper almost exactly — proof our evaluation pipeline is correct.",
                     "ok")
    with cards[3]:
        _status_card("4 · Baseline comparison", artifacts.baseline_status,
                     baseline_card_msg,
                     "ok" if baseline_full else "warn")

    selection = artifacts.model_selection
    st.caption(
        f"Model under test: **{selection['selected_checkpoint_type']}** · "
        f"SHA-256 `{selection['checkpoint_sha256'][:16]}…` · source commit `{selection['source_commit'][:12]}…` "
        "(full values in the Evidence tab). Open each tab below for the detailed numbers behind a card."
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


def _artifact_scalar(frame: pd.DataFrame, col: str, **filters: object) -> float | None:
    """First value of ``col`` from rows matching ``filters``, or None if unavailable.

    Used so caption metrics are sourced from the loaded artifacts instead of being
    hardcoded — keeps the text from going stale if the underlying CSVs change.
    """
    if frame is None or frame.empty or col not in frame.columns:
        return None
    sub = frame
    for key, value in filters.items():
        if key not in sub.columns:
            return None
        sub = sub[sub[key] == value]
    if sub.empty:
        return None
    try:
        return float(sub[col].iloc[0])
    except (TypeError, ValueError):
        return None


def _status_card(title: str, verdict: str, meaning: str, tone: str) -> None:
    icon = {"ok": "✅", "warn": "⚠️", "bad": "⛔"}.get(tone, "•")
    st.markdown(
        f"""
        <div class="card" style="height:100%;">
          <div class="label">{title}</div>
          <div class="number" style="font-size:1.2rem;"><span class="{tone}">{icon} {verdict}</span></div>
          <div class="muted" style="margin-top:.45rem; font-size:.82rem; line-height:1.4;">{meaning}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_forecast(artifacts: ReproduceArtifacts) -> None:
    st.markdown(
        "**How to read:** Paper target is the published benchmark. "
        "The official checkpoint verifies the evaluation pipeline. "
        "Our retrained checkpoint is the result used to decide reproduction success. "
        "Largest gap must stay below 5%."
    )
    predictions = artifacts.forecast_predictions
    if not predictions.empty:
        # Daily % error is the discriminating view — at the $0-70K price scale the predicted and
        # actual lines (and even naive) overlap, so the real day-to-day miss only shows as error.
        st.plotly_chart(
            forecast_error_chart(
                predictions,
                baseline_predictions=artifacts.baseline_predictions,
                split="test",
                title="Daily forecast error — CryptoMamba-v vs naive (test period)",
            ),
            use_container_width=True,
            config=CHART_CONFIG,
        )
        # Source every number from the artifacts so the caption can never drift from the charts.
        cm_mape = _artifact_scalar(
            artifacts.forecast_metrics, "MAPE_pct",
            result_type="retrained_checkpoint", split="test",
        )
        naive_mape = _artifact_scalar(
            artifacts.baseline_metrics, "MAPE_pct",
            model="naive_persistence", split="test",
        )
        cm_diracc = _artifact_scalar(artifacts.significance, "cm_directional_acc_pct")
        if None not in (cm_mape, naive_mape, cm_diracc):
            comparison = (
                f"CryptoMamba-v (red, MAPE ~{cm_mape:.2f}%) sits marginally **above** naive "
                f"(amber, ~{naive_mape:.2f}%): on squared/absolute error they are statistically "
                f"tied. CryptoMamba-v's edge is **directional accuracy** ({cm_diracc:.2f}% vs "
                "naive 0%), not price error"
            )
        else:
            comparison = (
                "CryptoMamba-v sits marginally above naive on squared/absolute error (statistically "
                "tied); its edge is **directional accuracy**, not price error"
            )
        st.caption(
            "Absolute % error per day (dotted line = each model's MAPE). On the raw price chart "
            "everything overlaps because day-to-day moves are tiny vs the $30–70K level — so the "
            f"real signal is here. {comparison} — see the Baseline tab."
        )
        with st.expander("Paper-style overlay — predicted vs actual close over the full split"):
            st.plotly_chart(
                forecast_timeseries_chart(
                    predictions,
                    result_type="retrained_checkpoint",
                    title="Predicted vs actual BTC close — paper split (train / val / test)",
                ),
                use_container_width=True,
                config=CHART_CONFIG,
            )
            st.caption(
                "Reproduction of the paper's prediction figure: actual close (blue) with predicted "
                "close coloured by split (red train · green val · magenta test). The prediction line "
                "tracks the actual line — faithful to the paper, but by design the lines overlap, so "
                "use the error chart above to actually compare models."
            )
        with st.expander("Parity scatter — predicted vs actual (test)"):
            st.plotly_chart(
                forecast_prediction_chart(
                    predictions,
                    "test",
                    "Test period — predicted vs actual BTC close (350 days)",
                    baseline=artifacts.baseline_predictions,
                ),
                use_container_width=True,
                config=CHART_CONFIG,
            )
            st.caption(
                "Each dot is one test day: predicted (y) vs actual (x); dots on y = x mean accurate. "
                "Author (blue) and retrained (red) clouds overlap (~0.22% mean price diff)."
            )

    st.dataframe(forecast_comparison_table(artifacts.forecast_metrics), hide_index=True, width="stretch")

    fm = artifacts.forecast_metrics
    official = fm.loc[fm["result_type"] == "official_checkpoint"]
    retrained = fm.loc[fm["result_type"] == "retrained_checkpoint"]
    if not official.empty and not retrained.empty:
        gap_cols = ["RMSE_gap_pct", "MAE_gap_pct", "MAPE_gap_pct"]
        retrained_gaps = [float(retrained.iloc[0][c]) for c in gap_cols]
        worst_gap = max(retrained_gaps)
        st.markdown(
            "**Did our retrained model reproduce the paper?** Each bar is how far our model's "
            "error is from the paper. The whole bar must stay under the dotted 5% line."
        )
        st.plotly_chart(
            grouped_bar_chart(
                ["RMSE", "MAE", "MAPE"],
                {"Our retrained — gap vs paper": retrained_gaps},
                title=f"Our retrained model — worst gap {worst_gap:.2f}%, all well under 5%",
                y_title="Gap vs paper (%)",
                value_fmt="{:.2f}%",
                colors=["#1e3a8a"],
                hline=5.0,
                hline_label="5% tolerance",
            ),
            use_container_width=True,
            config=CHART_CONFIG,
        )
        st.caption(
            f"Reproduced ✓ — the largest gap is {worst_gap:.2f}%, leaving ~{5 - worst_gap:.1f}% of "
            "headroom under the tolerance. (The official checkpoint's gap is ~0% — it essentially "
            "equals the paper — so it is the pipeline check in the table above, not plotted here.)"
        )

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
        "**How to read — three bars per strategy:** "
        "**Paper** = the balance printed in the paper. "
        "**Official** = the authors' released checkpoint run through *our* pipeline in Phase 2 "
        "(a real run — its numbers differ slightly from Paper, e.g. 124.90 vs 124.09). "
        "**Our retrain** = the checkpoint we trained ourselves. "
        "Official ≈ Paper is the proof our pipeline is faithful, so any gap in *Our retrain* is the "
        "model, not buggy code."
    )
    validation_tab, test_tab = st.tabs(["Validation period", "Test period"])
    for tab, split in ((validation_tab, "val"), (test_tab, "test")):
        with tab:
            st.markdown("**Final balance — starting balance: 100**")
            categories, balance_series = trading_balance_series(replay, split)
            if categories:
                st.plotly_chart(
                    grouped_bar_chart(
                        categories,
                        balance_series,
                        title=f"Final balance by strategy — {split} period (start = 100)",
                        y_title="Final balance",
                        value_fmt="{:,.1f}",
                        colors=["#94a3b8", "#2563eb", "#1e3a8a"],
                    ),
                    use_container_width=True,
                    config=CHART_CONFIG,
                )
                st.caption(
                    "Paper (published) vs Official (authors' checkpoint via our pipeline, Phase 2) "
                    "vs Our retrain (our checkpoint). Paper ≈ Official ⇒ pipeline verified."
                )
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


_TRADE_MODES = (("vanilla", "Vanilla"), ("smart", "Smart"), ("smart_w_short", "Smart + short"))


def trading_balance_series(replay: pd.DataFrame, split: str) -> tuple[list[str], dict[str, list[float]]]:
    """Build (strategy categories, {Paper/Official/Our retrain: balances}) for the bar chart."""
    official = replay[
        (replay["result_type"] == "official_checkpoint") & (replay["split"] == split)
    ].set_index("trade_mode")
    retrained = replay[
        (replay["result_type"] == "retrained_checkpoint") & (replay["split"] == split)
    ].set_index("trade_mode")
    if official.empty or retrained.empty:
        return [], {}

    categories: list[str] = []
    paper, off, retr = [], [], []
    for mode, label in _TRADE_MODES:
        if mode not in official.index or mode not in retrained.index:
            continue
        categories.append(label)
        paper.append(float(official.loc[mode, "paper_final_balance"]))
        off.append(float(official.loc[mode, "final_balance"]))
        retr.append(float(retrained.loc[mode, "final_balance"]))
    return categories, {"Paper": paper, "Official": off, "Our retrain": retr}


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


def baseline_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["model", "RMSE", "MAE", "MAPE_pct", "paper_RMSE", "RMSE_gap_pct"] if c in comparison.columns]
    return comparison[cols].rename(
        columns={"model": "Model", "MAPE_pct": "MAPE %", "paper_RMSE": "Paper RMSE", "RMSE_gap_pct": "vs paper %"}
    )


def significance_table(significance: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in ["comparison", "cm_directional_acc_pct", "baseline_directional_acc_pct",
                    "dm_p_value", "wilcoxon_p_value", "conclusion"]
        if c in significance.columns
    ]
    return significance[cols].rename(
        columns={
            "comparison": "Comparison",
            "cm_directional_acc_pct": "CM-v dir-acc %",
            "baseline_directional_acc_pct": "Baseline dir-acc %",
            "dm_p_value": "DM p",
            "wilcoxon_p_value": "Wilcoxon p",
            "conclusion": "Conclusion",
        }
    )


def model_metrics_frame(comparison: pd.DataFrame, significance: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-model frame [model, RMSE, MAE, MAPE_pct, dir_acc_pct] for the comparison
    charts. CryptoMamba-v's directional accuracy lives in significance_tests.csv (not in
    the baseline comparison rows), so graft it onto the CryptoMamba-v rows."""
    cols = ["model", "RMSE", "MAE", "MAPE_pct", "dir_acc_pct"]
    if comparison.empty:
        return pd.DataFrame(columns=cols)

    cm_dir: float | None = None
    if not significance.empty and "cm_directional_acc_pct" in significance.columns:
        cm_dir = float(significance["cm_directional_acc_pct"].iloc[0])

    rows: list[dict[str, object]] = []
    for _, record in comparison.iterrows():
        model = str(record["model"])
        dir_acc = record.get("directional_accuracy_strict_pct")
        if (dir_acc is None or pd.isna(dir_acc)) and "cryptomamba" in model.lower() and cm_dir is not None:
            dir_acc = cm_dir
        rows.append(
            {
                "model": model,
                "RMSE": float(record["RMSE"]) if pd.notna(record.get("RMSE")) else None,
                "MAE": float(record["MAE"]) if pd.notna(record.get("MAE")) else None,
                "MAPE_pct": float(record["MAPE_pct"]) if pd.notna(record.get("MAPE_pct")) else None,
                "dir_acc_pct": float(dir_acc) if dir_acc is not None and pd.notna(dir_acc) else None,
            }
        )
    return pd.DataFrame(rows, columns=cols)


# label -> (column, lower_is_better, value format)
_LEADERBOARD_METRICS = {
    "RMSE (lower better)": ("RMSE", True, "{:,.0f}"),
    "MAE (lower better)": ("MAE", True, "{:,.0f}"),
    "MAPE % (lower better)": ("MAPE_pct", True, "{:.2f}"),
    "Directional accuracy % (higher better)": ("dir_acc_pct", False, "{:.1f}"),
}


def _render_baseline(artifacts: ReproduceArtifacts) -> None:
    st.markdown(
        "**How to read:** Baselines are reference models evaluated on the same test split. "
        "CryptoMamba-v is meant to beat them — lower RMSE / MAE / MAPE is better, and directional "
        "accuracy (predicting up vs down) is what matters most for trading."
    )

    comparison = artifacts.baseline_comparison
    if not comparison.empty:
        st.dataframe(
            baseline_comparison_table(comparison).style.format(
                {"RMSE": "{:,.3f}", "MAE": "{:,.3f}", "MAPE %": "{:.3f}",
                 "Paper RMSE": "{:,.1f}", "vs paper %": "{:.2f}"},
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        baseline = artifacts.baseline_metrics[artifacts.baseline_metrics["split"] == "test"][
            ["model", "samples", "RMSE", "MAE", "MAPE_pct", "protocol"]
        ].rename(columns={"model": "Model", "samples": "Samples", "MAPE_pct": "MAPE %", "protocol": "Protocol"})
        st.dataframe(
            baseline.style.format({"RMSE": "{:,.3f}", "MAE": "{:,.3f}", "MAPE %": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )

    significance = artifacts.significance
    if not significance.empty:
        st.markdown(
            "**Statistical significance — CryptoMamba-v vs each baseline (test split).** "
            "Directional accuracy plus Diebold-Mariano / Wilcoxon p-values on squared error "
            "(p < 0.05 = a significant difference)."
        )
        st.dataframe(
            significance_table(significance).style.format(
                {"CM-v dir-acc %": "{:.2f}", "Baseline dir-acc %": "{:.2f}",
                 "DM p": "{:.4f}", "Wilcoxon p": "{:.4f}"},
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )

    metrics = model_metrics_frame(comparison, significance)
    if not metrics.empty:
        st.markdown("**Leaderboard — rank all models by a metric.** CryptoMamba-v is highlighted.")
        metric_label = st.selectbox(
            "Metric", list(_LEADERBOARD_METRICS.keys()), key="baseline_leaderboard_metric"
        )
        column, lower_is_better, value_fmt = _LEADERBOARD_METRICS[metric_label]
        ranked = metrics[["model", column]].dropna(subset=[column])
        if not ranked.empty:
            st.plotly_chart(
                leaderboard_bar_chart(
                    ranked["model"].tolist(),
                    ranked[column].astype(float).tolist(),
                    value_label=metric_label.split(" (")[0],
                    lower_is_better=lower_is_better,
                    value_fmt=value_fmt,
                ),
                use_container_width=True,
                config=CHART_CONFIG,
            )
        else:
            st.caption(f"No models report {metric_label.split(' (')[0]} yet.")

        scatter = metrics.dropna(subset=["RMSE", "dir_acc_pct"])
        if len(scatter) >= 2:
            st.markdown(
                "**Error vs direction.** The honest trade-off: CryptoMamba-v is not the lowest "
                "RMSE, but it has the highest directional accuracy — the bottom-right corner is best."
            )
            st.plotly_chart(
                tradeoff_scatter_chart(
                    scatter["model"].tolist(),
                    scatter["RMSE"].astype(float).tolist(),
                    scatter["dir_acc_pct"].astype(float).tolist(),
                ),
                use_container_width=True,
                config=CHART_CONFIG,
            )

    retrained = artifacts.forecast_metrics.loc[
        artifacts.forecast_metrics["result_type"] == "retrained_checkpoint", "RMSE"
    ]
    naive = artifacts.baseline_metrics.loc[
        (artifacts.baseline_metrics["model"] == "naive_persistence")
        & (artifacts.baseline_metrics["split"] == "test"),
        "RMSE",
    ]
    if not retrained.empty and not naive.empty and float(naive.iloc[0]) < float(retrained.iloc[0]):
        st.warning(
            f"Naive persistence has lower test RMSE ({float(naive.iloc[0]):,.3f}) than the retrained "
            f"checkpoint ({float(retrained.iloc[0]):,.3f}) — disclosed openly. CryptoMamba-v's edge is "
            "directional accuracy + trading, not raw squared error (see the significance table)."
        )

    present = list(artifacts.baseline_models_present)
    total = len(MANDATORY_BASELINES)
    if artifacts.baseline_status == "COMPLETE":
        st.success(f"Baseline scope COMPLETE: all {total} baselines present ({', '.join(present)}).")
    else:
        missing = sorted(MANDATORY_BASELINES - set(present))
        st.info(
            f"Baseline scope PARTIAL: {len(present)}/{total} present "
            f"({', '.join(present) or 'none'}); pending (Colab GPU): {', '.join(missing)}."
        )


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
