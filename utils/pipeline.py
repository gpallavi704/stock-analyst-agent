"""Runs the full analysis once and hands every tab the same result object.

Keeping this in one place means the numbers a chart shows, the numbers a metric
card shows, and the numbers the AI reasons about are guaranteed to be identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

from utils import analytics as an
from utils import market_data as md
from utils import portfolio_math as pm


@dataclass
class Analysis:
    transactions: pd.DataFrame
    fifo: pm.FifoResult
    holdings: pd.DataFrame
    totals: dict
    xirr: float | None
    quotes: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    value_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bench_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bench_summary: dict = field(default_factory=dict)
    bench_xirr: float | None = None
    sectors: pd.DataFrame = field(default_factory=pd.DataFrame)
    flags: list = field(default_factory=list)
    tax_lots: pd.DataFrame = field(default_factory=pd.DataFrame)
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk: dict = field(default_factory=dict)
    drawdown: dict = field(default_factory=dict)
    attribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    what_if: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)
    price_failures: list = field(default_factory=list)
    benchmark_ticker: str = "SPY"


def _splits_after_purchase(splits: dict, df: pd.DataFrame) -> dict:
    """Keep only splits that post-date the first purchase of that ticker.

    A split that happened before you ever bought the stock is already baked into
    the price you paid, so warning about it would be a false alarm. Only a split
    you held *through* leaves the CSV on a different share basis than the quote.
    """
    first_buy = df.groupby("ticker")["date"].min()
    out = {}
    for ticker, events in splits.items():
        if ticker not in first_buy.index:
            continue
        after = events[events.index > first_buy[ticker]]
        if not after.empty:
            out[ticker] = after
    return out


@st.cache_data(show_spinner=False, ttl=600)
def build_analysis(df: pd.DataFrame, benchmark: str, risk_free: float) -> Analysis:
    """Compute everything. Cached on the transaction data so tabs are instant."""
    fifo = pm.run_fifo(df)

    all_tickers = tuple(sorted(df["ticker"].unique()))
    open_tickers = tuple(sorted(fifo.open_lots))

    quotes = md.get_quotes(all_tickers)
    holdings = pm.build_holdings(fifo, quotes)
    totals = pm.portfolio_totals(df, holdings, fifo)

    start = (df["date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    prices = md.get_history(all_tickers, start)
    bench_hist = md.get_history((benchmark,), start)
    bench_px = bench_hist[benchmark] if benchmark in bench_hist.columns else pd.Series(dtype=float)

    value_series = an.portfolio_value_series(df, prices)
    bench_series = an.benchmark_shadow(df, bench_px)

    bench_summary, bench_xirr = {}, None
    if not bench_series.empty:
        bench_summary = an.benchmark_summary(
            totals["current_value"], bench_series, benchmark
        )
        bench_xirr = pm.xirr(pm.cashflows(df, float(bench_series.iloc[-1])))

    risk = drawdown = {}
    if not value_series.empty:
        flows = an.daily_net_cashflow(df, value_series.index)
        port_ret = an.time_weighted_returns(value_series, flows)
        bench_ret = None
        if not bench_series.empty:
            b_flows = an.daily_net_cashflow(df, bench_series.index)
            bench_ret = an.time_weighted_returns(bench_series, b_flows)
        risk = an.risk_metrics(port_ret, bench_ret, risk_free)
        # Drawdown is measured on a contribution-free growth index, otherwise a
        # large deposit would masquerade as a recovery.
        growth = (1 + port_ret).cumprod()
        drawdown = an.max_drawdown(growth)

    profiles = md.get_profiles(open_tickers)
    sectors = an.sector_breakdown(holdings, profiles)
    flags = an.concentration_flags(holdings, sectors)
    tax_lots = pm.tax_lot_table(fifo, quotes)
    divs = md.get_dividends(all_tickers, start)
    dividends = an.dividend_income(df, divs)
    attribution = an.day_attribution(holdings)
    what_if = an.what_if_never_sold(
        df, quotes, totals["current_value"] + totals["total_sells"]
    )
    splits = _splits_after_purchase(md.get_splits(all_tickers, start), df)

    return Analysis(
        transactions=df, fifo=fifo, holdings=holdings, totals=totals,
        xirr=pm.xirr(pm.cashflows(df, totals["current_value"])),
        quotes=quotes, profiles=profiles, prices=prices,
        value_series=value_series, bench_series=bench_series,
        bench_summary=bench_summary, bench_xirr=bench_xirr,
        sectors=sectors, flags=flags, tax_lots=tax_lots, dividends=dividends,
        risk=risk, drawdown=drawdown, attribution=attribution, what_if=what_if,
        splits=splits,
        price_failures=[t for t in open_tickers if t not in quotes],
        benchmark_ticker=benchmark,
    )


def settings() -> dict:
    return {
        "benchmark": os.getenv("BENCHMARK_TICKER", "SPY").upper(),
        "risk_free": float(os.getenv("RISK_FREE_RATE", "0.045") or 0.045),
    }


def agent_context(a: Analysis):
    """Package the analysis into the shape the LLM tools expect."""
    from utils.llm_agent import AgentContext

    return AgentContext(
        transactions=a.transactions, holdings=a.holdings, totals=a.totals,
        sectors=a.sectors, flags=a.flags, tax_lots=a.tax_lots, sales=a.fifo.sales,
        benchmark={**a.bench_summary,
                   "benchmark_xirr_pct": a.bench_xirr * 100 if a.bench_xirr else None,
                   "portfolio_xirr_pct": a.xirr * 100 if a.xirr else None},
        risk=a.risk, drawdown=a.drawdown, xirr=a.xirr,
        attribution=a.attribution, dividends=a.dividends,
    )


# -- shared formatting helpers used across tabs -----------------------------

def usd(v, decimals: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"${v:,.{decimals}f}"


def pct(v, decimals: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.{decimals}f}%"


def signed_usd(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def days_since(d: date) -> int:
    return (date.today() - d).days
