"""Tab 3 — how the portfolio has performed over time.

The organising question is "was any of this worth it?", which needs three
comparisons the holdings table cannot make: against the money actually put in,
against the same money passively invested in an index, and against the risk
taken to get there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an
from utils.pipeline import Analysis, pct, signed_usd, usd

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
)
GRID = dict(gridcolor="rgba(128,128,128,.18)", zeroline=False)

PORTFOLIO_COLOR = "#2e6fd6"
BENCH_COLOR = "#e08a3c"
INVESTED_COLOR = "rgba(128,128,128,.85)"


def _growth_index(a: Analysis) -> pd.Series:
    """Contribution-free growth of $1, which is what drawdown must be measured on.

    Using raw market value would let a large deposit look like a recovery from a
    crash, understating how deep the fall actually was.
    """
    if a.value_series.empty:
        return pd.Series(dtype=float)
    flows = an.daily_net_cashflow(a.transactions, a.value_series.index)
    returns = an.time_weighted_returns(a.value_series, flows)
    return (1 + returns).cumprod()


def _value_chart(a: Analysis) -> None:
    st.markdown("##### Portfolio value over time")
    st.caption(
        f"The dashed line replays your exact cashflows into {a.benchmark_ticker} on the "
        "same dates. That is the honest comparison — quoting the index's total return "
        "would ignore when your money actually went in."
    )

    invested = an.daily_net_cashflow(a.transactions, a.value_series.index).cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a.value_series.index, y=a.value_series.values, name="Your portfolio",
        line=dict(color=PORTFOLIO_COLOR, width=2.5),
        fill="tozeroy", fillcolor="rgba(46,111,214,.10)",
        hovertemplate="%{y:$,.0f}<extra>Your portfolio</extra>",
    ))
    if not a.bench_series.empty:
        fig.add_trace(go.Scatter(
            x=a.bench_series.index, y=a.bench_series.values,
            name=f"Same cash in {a.benchmark_ticker}",
            line=dict(color=BENCH_COLOR, width=2, dash="dash"),
            hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=invested.index, y=invested.values, name="Net invested",
        line=dict(color=INVESTED_COLOR, width=1.5, dash="dot"),
        hovertemplate="%{y:$,.0f}<extra>Net invested</extra>",
    ))

    fig.update_layout(height=430, legend=dict(orientation="h", y=1.12, x=0), **CHART_LAYOUT)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(tickformat="$,.0f", **GRID)
    st.plotly_chart(fig, width="stretch")


def _benchmark_scorecard(a: Analysis) -> None:
    if not a.bench_summary:
        st.info(f"Benchmark data for {a.benchmark_ticker} could not be fetched.", icon="📡")
        return

    b = a.bench_summary
    diff = b["difference"]
    beat = bool(diff > 0) if np.isfinite(diff) else None

    c = st.columns(4)
    c[0].metric("Your portfolio today", usd(b["portfolio_value"], 0))
    c[1].metric(f"Same cash in {a.benchmark_ticker}", usd(b["benchmark_value"], 0))
    c[2].metric("Difference", signed_usd(diff),
                "ahead of the index" if beat else "behind the index",
                delta_color="normal" if beat else "inverse")

    gap = None
    if a.xirr is not None and a.bench_xirr is not None:
        gap = (a.xirr - a.bench_xirr) * 100
    c[3].metric("XIRR vs benchmark",
                pct(a.xirr * 100 if a.xirr is not None else None, 2),
                f"{pct(a.bench_xirr * 100 if a.bench_xirr is not None else None, 2)} for "
                f"{a.benchmark_ticker}", delta_color="off")

    if gap is not None:
        verdict = (
            f"Your stock picking added **{gap:+.2f} percentage points a year** over simply "
            f"buying {a.benchmark_ticker} with the same cash on the same dates."
            if gap >= 0 else
            f"Your stock picking cost **{abs(gap):.2f} percentage points a year** against "
            f"simply buying {a.benchmark_ticker} with the same cash on the same dates."
        )
        (st.success if gap >= 0 else st.warning)(verdict, icon="📈" if gap >= 0 else "📉")


def _risk(a: Analysis) -> None:
    st.markdown("##### Risk")
    if not a.risk:
        st.info("Not enough price history to compute risk metrics.")
        return

    r, dd = a.risk, a.drawdown
    c = st.columns(4)
    c[0].metric("Volatility (annualised)", pct(r.get("volatility"), 1),
                "std dev of daily returns", delta_color="off")
    c[1].metric("Sharpe ratio",
                "—" if pd.isna(r.get("sharpe")) else f"{r['sharpe']:.2f}",
                "return per unit of risk", delta_color="off")
    c[2].metric("Beta vs " + a.benchmark_ticker,
                "—" if pd.isna(r.get("beta")) else f"{r['beta']:.2f}",
                "1.0 = moves with the index", delta_color="off")
    c[3].metric("Max drawdown", pct(dd.get("pct"), 1),
                "recovered" if dd.get("recovered") else "not yet recovered",
                delta_color="off")

    if dd.get("peak_date") is not None:
        st.caption(
            f"Deepest fall ran from {dd['peak_date']:%d %b %Y} to {dd['trough_date']:%d %b %Y}. "
            "It is measured on contribution-free growth, so adding money mid-fall cannot "
            "disguise the size of the decline."
        )

    growth = _growth_index(a)
    if growth.empty:
        return
    underwater = (growth / growth.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(
        x=underwater.index, y=underwater.values, name="Drawdown",
        line=dict(color="#d64545", width=1.5), fill="tozeroy",
        fillcolor="rgba(214,69,69,.15)",
        hovertemplate="%{y:.1f}%<extra>Below peak</extra>",
    ))
    fig.update_layout(height=240, showlegend=False, **CHART_LAYOUT)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(ticksuffix="%", **GRID)
    st.plotly_chart(fig, width="stretch")


def _realized(a: Analysis) -> None:
    st.markdown("##### Closed trades")
    if not a.fifo.sales:
        st.info("No sales yet — every share bought is still held.")
        return

    rows = []
    for s in a.fifo.sales:
        short, long = s.split_by_term()
        rows.append({
            "Ticker": s.ticker, "Sold": s.date, "Quantity": s.qty,
            "Sell Price": s.price, "Proceeds": s.proceeds,
            "FIFO Cost Basis": s.cost_basis, "Realized P/L": s.realized,
            "Return %": s.realized_pct, "Short-term": short, "Long-term": long,
        })
    df = pd.DataFrame(rows).sort_values("Sold", ascending=False)

    c = st.columns(4)
    c[0].metric("Realised P/L", signed_usd(df["Realized P/L"].sum()))
    c[1].metric("Winners", int((df["Realized P/L"] > 0).sum()),
                f"of {len(df)} trades", delta_color="off")
    c[2].metric("Short-term gains", signed_usd(df["Short-term"].sum()),
                "taxed as income", delta_color="off")
    c[3].metric("Long-term gains", signed_usd(df["Long-term"].sum()),
                "lower tax rate", delta_color="off")

    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "Sold": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Quantity": st.column_config.NumberColumn(format="%.4g"),
            "Sell Price": st.column_config.NumberColumn(format="$%.2f"),
            "Proceeds": st.column_config.NumberColumn(format="$%.2f"),
            "FIFO Cost Basis": st.column_config.NumberColumn(format="$%.2f"),
            "Realized P/L": st.column_config.NumberColumn(format="$%.2f"),
            "Return %": st.column_config.NumberColumn(format="%.2f%%"),
            "Short-term": st.column_config.NumberColumn(format="$%.2f"),
            "Long-term": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    w = a.what_if
    if w and w.get("hold_value"):
        delta = w["delta"]
        st.markdown(
            f"**Had you never sold anything**, every share ever bought would be worth "
            f"{usd(w['hold_value'], 0)} today, against {usd(w['actual_total'], 0)} for what "
            f"you actually hold plus what you took out — a difference of "
            f"**{signed_usd(delta)}**. "
            + ("Selling left money on the table." if delta > 0 else "Selling was the better call.")
        )
        if w.get("missing_prices"):
            st.caption("Excluded (no price): " + ", ".join(w["missing_prices"]))


def _dividends(a: Analysis) -> None:
    st.markdown("##### Dividend income")
    if a.dividends.empty:
        st.info("No dividends recorded for these holdings over the period.")
        return

    df = a.dividends
    by_year = (
        df.assign(Year=df["Ex-Date"].dt.year)
        .groupby("Year", as_index=False)["Income"].sum()
    )

    c = st.columns(3)
    c[0].metric("Total received", usd(df["Income"].sum()))
    c[1].metric("Payments", len(df))
    c[2].metric("Paying tickers", int(df["Ticker"].nunique()))
    st.caption(
        "Computed from the shares you actually held on each ex-dividend date, not from "
        "the current position — so a stock bought last month is not credited with years "
        "of past payouts."
    )

    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure(go.Bar(
            x=by_year["Year"].astype(str), y=by_year["Income"],
            marker_color=PORTFOLIO_COLOR,
            hovertemplate="%{x}<br>%{y:$,.2f}<extra></extra>",
        ))
        fig.update_layout(height=280, showlegend=False, **CHART_LAYOUT)
        fig.update_xaxes(title=None, **GRID)
        fig.update_yaxes(tickformat="$,.0f", **GRID)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.dataframe(
            df.head(50), width="stretch", hide_index=True, height=280,
            column_config={
                "Ex-Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Shares Held": st.column_config.NumberColumn(format="%.4g"),
                "Per Share": st.column_config.NumberColumn(format="$%.4f"),
                "Income": st.column_config.NumberColumn(format="$%.2f"),
            },
        )


def render(a: Analysis) -> None:
    st.subheader("Historical performance")

    if a.value_series.empty:
        st.warning(
            "Price history could not be fetched, so the performance timeline is "
            "unavailable. Check your network connection and reload.",
            icon="📡",
        )
        return

    _benchmark_scorecard(a)
    st.divider()
    _value_chart(a)
    st.divider()
    _risk(a)
    st.divider()
    _realized(a)
    st.divider()
    _dividends(a)
