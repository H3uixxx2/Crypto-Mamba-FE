from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px;}
        h1, h2, h3 {letter-spacing: -0.03em;}
        .topbar {
            padding: 1.1rem 1.25rem;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 8px 28px rgba(15, 23, 42, .06);
            margin-bottom: 1rem;
        }
        .eyebrow {font-size: .82rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;}
        .lead {color: #475569; font-size: 1.02rem; line-height: 1.55; margin: .25rem 0 0 0;}
        .card {
            padding: 1rem 1.05rem;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 5px 18px rgba(15, 23, 42, .04);
        }
        .soft {
            padding: .9rem 1rem;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .number {font-size: 1.75rem; font-weight: 800; color: #0f172a; line-height: 1.15;}
        .label {font-size: .88rem; color: #64748b; margin-bottom: .2rem;}
        .ok {color:#059669; font-weight:800;}
        .warn {color:#d97706; font-weight:800;}
        .bad {color:#dc2626; font-weight:800;}
        .muted {color:#64748b;}
        .pill {
            display:inline-block;
            padding:.22rem .55rem;
            border:1px solid #cbd5e1;
            border-radius:999px;
            background:#f8fafc;
            color:#334155;
            font-size:.78rem;
            font-weight:700;
            margin-right:.25rem;
        }
        .legend-row {display:flex; align-items:center; gap:.45rem; margin:.38rem 0;}
        .legend-box {width:14px; height:22px; border-radius:3px; display:inline-block;}
        .up-candle {background:#16a34a;}
        .down-candle {background:#dc2626;}
        .wick {
            width: 2px;
            height: 24px;
            background: #64748b;
            display: inline-block;
            margin: 0 6px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: .78rem .9rem;
            background: #ffffff;
            box-shadow: 0 5px 18px rgba(15, 23, 42, .04);
        }
        .stTabs [data-baseweb="tab-list"] {gap: .25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="eyebrow">CryptoMamba Graduation Project</div>
          <h1 style="margin:.15rem 0 .2rem 0;">Bitcoin forecast + trading simulation</h1>
          <p class="lead">Experimental workflow: BTC/USD data preprocessing → CryptoMamba-v reproduction → next-day close prediction → trading decision simulation.</p>
          <div style="margin-top:.7rem;">
            <span class="pill">BTC/USD daily</span>
            <span class="pill">CryptoMamba-v</span>
            <span class="pill">Trading simulation</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def money(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def date_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "—"
    return f"{df['date'].min()} → {df['date'].max()}"


def stat_card(title: str, value: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div class='muted' style='margin-top:.35rem;'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="label">{title}</div>
          <div class="number">{value}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_html(action: str, amount: float | None = None) -> str:
    css = "ok" if action == "buy" else "bad" if action == "sell" else "warn"
    label = action.upper()
    suffix = f" · {amount:.1f}%" if amount is not None and action != "hold" else ""
    return f"<span class='{css}'>{label}{suffix}</span>"


def load_optional_csv(path: Path, fallback: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        return fallback.copy(), False
    try:
        return pd.read_csv(path), True
    except (OSError, pd.errors.ParserError) as exc:
        st.warning(f"Cannot read artifact {path.name}: {exc}")
        return fallback.copy(), False
