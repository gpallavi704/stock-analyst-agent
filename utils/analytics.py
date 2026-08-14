"""Analysis that goes beyond basic holdings: benchmark comparison, risk,
sector concentration, dividend income, and counterfactual scenarios.

The theme here is answering questions a plain holdings table can't:
was the stock picking actually worth it, where is the hidden concentration,
and what did the decision to sell really cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Daily reconstruction
# --------------------------------------------------------------------------

def daily_shares(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Share count held in each ticker on every day of ``index``.

    Signed quantities are accumulated on transaction dates then forward-filled,
    which reproduces the position as it stood on any given day.
    """
    signed = df.assign(
        signed=np.where(df["transaction_type"] == "BUY", df["quantity"], -df["quantity"])
    )
    pivot = signed.pivot_table(
        index="date", columns="ticker", values="signed", aggfunc="sum"
    ).fillna(0.0)
    return pivot.reindex(index.union(pivot.index)).fillna(0.0).cumsum().reindex(index).ffill()


def portfolio_value_series(df: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Total market value of open holdings on each day since the first trade."""
    if prices.empty:
        return pd.Series(dtype=float)
    idx = prices.index[prices.index >= df["date"].min()]
    if len(idx) == 0:
        return pd.Series(dtype=float)
    shares = daily_shares(df, idx)
    cols = [c for c in shares.columns if c in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    return (shares[cols] * prices.loc[idx, cols]).sum(axis=1)


def daily_net_cashflow(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Net cash added to the portfolio each day (buys positive, sells negative)."""
    signed = df.assign(
        cf=np.where(df["transaction_type"] == "BUY", df["amount"], -df["amount"])
    )
    daily = signed.groupby("date")["cf"].sum()
    return daily.reindex(index).fillna(0.0)


def time_weighted_returns(values: pd.Series, flows: pd.Series) -> pd.Series:
    """Daily returns with the effect of deposits and withdrawals removed.

    Without this correction, paying money in looks like a gain. Time-weighted
    return isolates investment performance from the timing of contributions,
    which is what makes it comparable to an index.
    """
    v_prev = values.shift(1)
    r = (values - flows - v_prev) / v_prev.replace(0, np.nan)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def benchmark_shadow(df: pd.DataFrame, bench: pd.Series) -> pd.Series:
    """Value over time had every cashflow gone into the benchmark instead.

    Each buy purchases benchmark units at that day's price; each sell redeems
    them. This makes the comparison money-weighted, so it accounts for *when*
    cash was invested rather than just quoting the index's total return.
    """
    if bench.empty:
        return pd.Series(dtype=float)
    idx = bench.index[bench.index >= df["date"].min()]
    flows = daily_net_cashflow(df, idx)
    px = bench.reindex(idx).ffill()
    units = (flows / px).replace([np.inf, -np.inf], 0.0).fillna(0.0).cumsum()
    return units * px


def benchmark_summary(port_value: float, shadow: pd.Series, ticker: str) -> dict:
    bench_value = float(shadow.iloc[-1]) if not shadow.empty else np.nan
    diff = port_value - bench_value if np.isfinite(bench_value) else np.nan
    return {
        "benchmark": ticker,
        "benchmark_value": bench_value,
        "portfolio_value": port_value,
        "difference": diff,
        "beat": bool(diff > 0) if np.isfinite(diff) else None,
    }


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------

def max_drawdown(series: pd.Series) -> dict:
    """Deepest peak-to-trough fall, and whether it has since recovered."""
    if series.empty or series.max() <= 0:
        return {"pct": np.nan, "peak_date": None, "trough_date": None, "recovered": None}
    running_peak = series.cummax()
    dd = (series - running_peak) / running_peak
    trough = dd.idxmin()
    peak = series.loc[:trough].idxmax()
    after = series.loc[trough:]
    recovered = bool((after >= series.loc[peak]).any())
    return {
        "pct": float(dd.min() * 100),
        "peak_date": peak,
        "trough_date": trough,
        "recovered": recovered,
    }


def risk_metrics(port_ret: pd.Series, bench_ret: pd.Series | None, rf: float = 0.045) -> dict:
    """Annualised volatility, Sharpe, and beta from daily time-weighted returns."""
    r = port_ret.replace(0.0, np.nan).dropna()
    if len(r) < 20:
        return {"volatility": np.nan, "sharpe": np.nan, "beta": np.nan, "ann_return": np.nan}

    vol = float(r.std() * np.sqrt(TRADING_DAYS) * 100)
    ann = float((1 + r.mean()) ** TRADING_DAYS - 1) * 100
    sharpe = (ann / 100 - rf) / (vol / 100) if vol else np.nan

    beta = np.nan
    if bench_ret is not None and not bench_ret.empty:
        joined = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
        joined = joined[(joined != 0).any(axis=1)]
        if len(joined) >= 20:
            var = joined.iloc[:, 1].var()
            if var and np.isfinite(var) and var > 0:
                beta = float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / var)

    return {"volatility": vol, "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "beta": beta, "ann_return": ann}


# --------------------------------------------------------------------------
# Concentration
# --------------------------------------------------------------------------

def sector_breakdown(holdings: pd.DataFrame, profiles: dict[str, dict]) -> pd.DataFrame:
    """Allocation grouped by sector.

    A portfolio can look diversified stock-by-stock while being heavily exposed
    to a single sector — this is the check that catches it.
    """
    if holdings.empty:
        return pd.DataFrame()
    tmp = holdings.copy()
    tmp["Sector"] = tmp["Ticker"].map(lambda t: profiles.get(t, {}).get("sector", "Unknown"))
    g = (
        tmp.groupby("Sector", as_index=False)
        .agg(**{
            "Market Value": ("Market Value", "sum"),
            "Unrealized P/L": ("Unrealized P/L", "sum"),
            "Holdings": ("Ticker", lambda s: ", ".join(sorted(s))),
        })
        .sort_values("Market Value", ascending=False)
    )
    total = g["Market Value"].sum()
    g["Allocation %"] = g["Market Value"] / total * 100 if total else np.nan
    return g.reset_index(drop=True)


def concentration_flags(holdings: pd.DataFrame, sectors: pd.DataFrame) -> list[str]:
    """Plain-language risk observations derived purely from allocation numbers."""
    flags: list[str] = []
    if holdings.empty:
        return flags

    top = holdings.iloc[0]
    if pd.notna(top.get("Allocation %")) and top["Allocation %"] >= 25:
        flags.append(
            f"{top['Ticker']} alone is {top['Allocation %']:.1f}% of the portfolio — "
            "a single-stock shock would move your total materially."
        )

    top3 = holdings.head(3)["Allocation %"].sum()
    if pd.notna(top3) and top3 >= 60:
        names = ", ".join(holdings.head(3)["Ticker"])
        flags.append(f"Your top 3 positions ({names}) are {top3:.1f}% of the portfolio.")

    if not sectors.empty:
        s = sectors.iloc[0]
        if pd.notna(s["Allocation %"]) and s["Allocation %"] >= 40:
            flags.append(
                f"{s['Allocation %']:.1f}% sits in a single sector ({s['Sector']}: "
                f"{s['Holdings']}). These names tend to fall together, so the "
                "stock-level spread overstates how diversified you are."
            )

    n = len(holdings)
    if n < 5:
        flags.append(f"Only {n} open position(s) — limited room to absorb a single bad outcome.")

    # Herfindahl index: the equivalent number of equally-weighted positions.
    w = (holdings["Allocation %"].dropna() / 100) ** 2
    if len(w):
        eff = 1 / w.sum()
        flags.append(
            f"Effective diversification is about {eff:.1f} equally-weighted positions "
            f"(you hold {n})."
        )
    return flags


# --------------------------------------------------------------------------
# Dividends
# --------------------------------------------------------------------------

def dividend_income(df: pd.DataFrame, divs: dict[str, pd.Series]) -> pd.DataFrame:
    """Dividends actually received, based on shares held on each ex-date."""
    rows = []
    for t, series in divs.items():
        tx = df[df["ticker"] == t]
        if tx.empty:
            continue
        for ex_date, per_share in series.items():
            prior = tx[tx["date"] <= ex_date]
            if prior.empty:
                continue
            shares = float(
                np.where(prior["transaction_type"] == "BUY", prior["quantity"], -prior["quantity"]).sum()
            )
            if shares > 1e-9:
                rows.append({
                    "Ticker": t, "Ex-Date": ex_date, "Shares Held": shares,
                    "Per Share": float(per_share), "Income": shares * float(per_share),
                })
    out = pd.DataFrame(rows)
    return out.sort_values("Ex-Date", ascending=False).reset_index(drop=True) if not out.empty else out


# --------------------------------------------------------------------------
# Counterfactuals
# --------------------------------------------------------------------------

def what_if_never_sold(df: pd.DataFrame, quotes: dict[str, dict], actual_total: float) -> dict:
    """Value today if every share ever bought had simply been held.

    Compared against actual value *plus* realised proceeds, so both branches
    account for the same money.
    """
    buys = df[df["transaction_type"] == "BUY"].groupby("ticker")["quantity"].sum()
    missing = [t for t in buys.index if t not in quotes]
    value = sum(q * quotes[t]["price"] for t, q in buys.items() if t in quotes)
    return {
        "hold_value": value,
        "actual_total": actual_total,
        "delta": value - actual_total,
        "missing_prices": missing,
    }


def day_attribution(holdings: pd.DataFrame) -> pd.DataFrame:
    """Which holdings drove today's move, largest contribution first.

    This is what lets the assistant answer 'why is my portfolio down today'
    from arithmetic instead of inventing market news.
    """
    if holdings.empty or "Day Change" not in holdings:
        return pd.DataFrame()
    d = holdings[["Ticker", "Day Change", "Day Change %", "Market Value"]].dropna(
        subset=["Day Change"]
    ).copy()
    if d.empty:
        return d
    total = d["Day Change"].sum()
    d["Share of Move %"] = d["Day Change"] / total * 100 if total else np.nan
    return d.reindex(d["Day Change"].abs().sort_values(ascending=False).index).reset_index(drop=True)
