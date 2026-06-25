from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

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
            ("out_of_scope", "Outside paper split"),
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
                    hovertemplate=f"{label}<br>Date=%{{x}}<br>Close=%{{y:,.2f}}<extra></extra>",
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
                hovertemplate="Date=%{x}<br>Close=%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=410,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        yaxis_title="Close price",
        xaxis_title="Date",
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_prediction_chart(
    predictions: pd.DataFrame,
    split: str,
    title: str,
    baseline: pd.DataFrame | None = None,
) -> go.Figure:
    """Parity scatter: predicted vs actual close. Points on the diagonal = accurate.

    Author (official) and our retrained predictions are overlaid; both hugging the
    y=x line shows the retrain reproduced the paper model. If ``baseline`` is given,
    the naive-persistence baseline is added as a third cloud for comparison.
    """
    part = predictions[predictions["split"] == split]
    official = part[part["result_type"] == "official_checkpoint"]
    retrained = part[part["result_type"] == "retrained_checkpoint"]

    naive = pd.DataFrame()
    if baseline is not None and not baseline.empty:
        naive = baseline[
            (baseline["split"] == split) & (baseline["model"] == "naive_persistence")
        ]

    lo = float(part["target_close"].min())
    hi = float(part["target_close"].max())
    pad = (hi - lo) * 0.03

    fig = go.Figure()
    # Perfect-prediction reference line (y = x).
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            name="Perfect prediction (y = x)",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=official["target_close"],
            y=official["predicted_close"],
            mode="markers",
            name="Author model (official)",
            marker=dict(color="#2563eb", size=6, opacity=0.45, line=dict(width=0)),
            hovertemplate="Author<br>actual=%{x:,.0f}<br>pred=%{y:,.0f}<extra></extra>",
        )
    )
    if not naive.empty:
        fig.add_trace(
            go.Scatter(
                x=naive["target_close"],
                y=naive["predicted_close"],
                mode="markers",
                name="Naive baseline (yesterday's close)",
                marker=dict(color="#f59e0b", size=5, opacity=0.35, line=dict(width=0)),
                hovertemplate="Naive<br>actual=%{x:,.0f}<br>pred=%{y:,.0f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=retrained["target_close"],
            y=retrained["predicted_close"],
            mode="markers",
            name="Our retrained model",
            marker=dict(color="#dc2626", size=6, opacity=0.45, line=dict(width=0)),
            hovertemplate="Retrained<br>actual=%{x:,.0f}<br>pred=%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=460,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        xaxis_title="Actual BTC close (USD)",
        yaxis_title="Predicted BTC close (USD)",
        dragmode=False,
        xaxis=dict(range=[lo - pad, hi + pad], fixedrange=True, constrain="domain"),
        yaxis=dict(range=[lo - pad, hi + pad], fixedrange=True, scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_timeseries_chart(
    predictions: pd.DataFrame,
    result_type: str = "retrained_checkpoint",
    title: str = "Predicted vs actual BTC close — paper split (train / val / test)",
) -> go.Figure:
    """Paper-style time series: actual close (blue) overlaid with the model's predicted
    close colored by split (train / val / test), over the full paper timeline.

    Mirrors the CryptoMamba paper's prediction figure (scripts/evaluation.py pred plot):
    the predicted line hugging the actual line is the visual proof of reproduction.
    """
    part = predictions[predictions["result_type"] == result_type].copy()
    if part.empty:
        return go.Figure()
    part["_d"] = pd.to_datetime(part["prediction_date"], errors="coerce")
    part = part.dropna(subset=["_d"]).sort_values("_d")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=part["_d"], y=part["target_close"], mode="lines", name="Actual close",
            line=dict(color="#2563eb", width=1.9),
            hovertemplate="Actual<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    for split_name, label, color in (
        ("train", "Predicted · Train", "#dc2626"),
        ("val", "Predicted · Val", "#16a34a"),
        ("test", "Predicted · Test", "#db2777"),
    ):
        seg = part[part["split"] == split_name]
        if seg.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=seg["_d"], y=seg["predicted_close"], mode="lines", name=label,
                line=dict(color=color, width=1.4),
                hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        height=460,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        yaxis_title="BTC close (USD)",
        xaxis_title="Date",
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def candle_chart(df: pd.DataFrame, title: str, prediction: dict | None = None, height: int = 430) -> go.Figure:
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
    fig.add_trace(
        go.Bar(
            x=sim_df["strategy"],
            y=sim_df["roi_pct"],
            marker_color=colors,
            text=sim_df["roi_pct"].map(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=28, b=12), yaxis_title="ROI (%)", showlegend=False)
    return fig
