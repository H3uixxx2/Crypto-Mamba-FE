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
