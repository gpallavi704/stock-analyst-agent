"""FIFO lot accounting, holdings valuation, and portfolio-level return maths.

FIFO is the US default cost-basis method: a sale consumes the oldest shares
first. Tracking individual lots (rather than a running average) is what lets the
app report realised gains correctly and split a position into short- and
long-term holding periods for tax purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

EPS = 1e-9
LONG_TERM_DAYS = 365  # US: held > 1 year qualifies for long-term capital gains


@dataclass
class Lot:
    """An open parcel of shares bought on one date at one price."""

    qty: float
    price: float
    date: pd.Timestamp

    @property
    def cost(self) -> float:
        return self.qty * self.price


@dataclass
class Sale:
    """A realised disposal, with the lots it consumed."""

    ticker: str
    date: pd.Timestamp
    qty: float
    price: float
    proceeds: float
    cost_basis: float
    consumed: list[tuple[float, float, pd.Timestamp]] = field(default_factory=list)

    @property
    def realized(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def realized_pct(self) -> float:
        return (self.realized / self.cost_basis * 100) if self.cost_basis else 0.0

    def split_by_term(self) -> tuple[float, float]:
        """Realised gain split into (short_term, long_term) by holding period."""
        short = long = 0.0
        for qty, px, bought in self.consumed:
            gain = qty * (self.price - px)
            held = (self.date - bought).days
            if held > LONG_TERM_DAYS:
                long += gain
            else:
                short += gain
        return short, long


@dataclass
class FifoResult:
    open_lots: dict[str, list[Lot]] = field(default_factory=dict)
    sales: list[Sale] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)

    def realized_by_ticker(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.sales:
            out[s.ticker] = out.get(s.ticker, 0.0) + s.realized
        return out

    @property
    def total_realized(self) -> float:
        return sum(s.realized for s in self.sales)


def run_fifo(df: pd.DataFrame) -> FifoResult:
    """Replay transactions in date order, maintaining a FIFO lot queue per ticker.

    Oversells are recorded as errors and the offending row is skipped, so one bad
    line doesn't invalidate the rest of the portfolio.
    """
    res = FifoResult()
    lots: dict[str, list[Lot]] = {}

    for row in df.sort_values("date", kind="stable").itertuples():
        t = row.ticker
        lots.setdefault(t, [])

        if row.transaction_type == "BUY":
            lots[t].append(Lot(qty=float(row.quantity), price=float(row.price), date=row.date))
            continue

        held = sum(lot.qty for lot in lots[t])
        want = float(row.quantity)
        if want > held + EPS:
            res.errors.append(
                f"{row.date:%Y-%m-%d} — cannot sell {want:g} shares of {t}: only "
                f"{held:g} held at that point. This row was skipped."
                + (" No prior purchase of this ticker exists." if held <= EPS else "")
            )
            continue

        remaining, cost, consumed = want, 0.0, []
        while remaining > EPS:
            lot = lots[t][0]
            take = min(lot.qty, remaining)
            cost += take * lot.price
            consumed.append((take, lot.price, lot.date))
            lot.qty -= take
            remaining -= take
            if lot.qty <= EPS:
                lots[t].pop(0)

        res.sales.append(
            Sale(
                ticker=t, date=row.date, qty=want, price=float(row.price),
                proceeds=want * float(row.price), cost_basis=cost, consumed=consumed,
            )
        )

    for t, parcels in lots.items():
        live = [lot for lot in parcels if lot.qty > EPS]
        if live:
            res.open_lots[t] = live
        else:
            res.closed.append(t)  # fully sold: excluded from current holdings
    return res


def build_holdings(fifo: FifoResult, quotes: dict[str, dict]) -> pd.DataFrame:
    """Value each open position at its latest price.

    Tickers whose quote could not be fetched are still returned, with NaN price
    so the UI can flag them rather than silently dropping the position.
    """
    realized = fifo.realized_by_ticker()
    rows = []
    for t, parcels in sorted(fifo.open_lots.items()):
        qty = sum(l.qty for l in parcels)
        basis = sum(l.cost for l in parcels)
        q = quotes.get(t, {})
        price = q.get("price")
        prev = q.get("prev_close")
        mv = price * qty if price is not None else np.nan
        rows.append(
            {
                "Ticker": t,
                "Quantity": qty,
                "Avg Cost Basis": basis / qty if qty else np.nan,
                "Current Price": price if price is not None else np.nan,
                "Cost Basis": basis,
                "Market Value": mv,
                "Unrealized P/L": mv - basis if price is not None else np.nan,
                "Unrealized P/L %": ((mv - basis) / basis * 100) if price and basis else np.nan,
                "Day Change": (price - prev) * qty if (price and prev) else np.nan,
                "Day Change %": ((price - prev) / prev * 100) if (price and prev) else np.nan,
                "Realized P/L": realized.get(t, 0.0),
                "Lots": len(parcels),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        total = df["Market Value"].sum(skipna=True)
        df["Allocation %"] = df["Market Value"] / total * 100 if total else np.nan
        df = df.sort_values("Market Value", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def portfolio_totals(df: pd.DataFrame, holdings: pd.DataFrame, fifo: FifoResult) -> dict:
    """Lifetime cash totals plus current value and return."""
    buys = float(df.loc[df["transaction_type"] == "BUY", "amount"].sum())
    sells = float(df.loc[df["transaction_type"] == "SELL", "amount"].sum())
    value = float(holdings["Market Value"].sum(skipna=True)) if not holdings.empty else 0.0
    basis = float(holdings["Cost Basis"].sum()) if not holdings.empty else 0.0
    total_return = sells + value - buys
    return {
        "total_investment": buys,
        "total_sells": sells,
        "current_value": value,
        "open_cost_basis": basis,
        "total_return": total_return,
        "total_return_pct": (total_return / buys * 100) if buys else np.nan,
        "realized_pl": fifo.total_realized,
        "unrealized_pl": value - basis,
        "day_change": float(holdings["Day Change"].sum(skipna=True)) if not holdings.empty else 0.0,
    }


# --------------------------------------------------------------------------
# XIRR
# --------------------------------------------------------------------------

def xnpv(rate: float, flows: list[tuple[date, float]]) -> float:
    t0 = flows[0][0]
    return sum(cf / (1.0 + rate) ** ((d - t0).days / 365.0) for d, cf in flows)


def xirr(flows: list[tuple[date, float]]) -> float | None:
    """Money-weighted annualised return for irregularly spaced cashflows.

    Returns None when no rate can be found — which legitimately happens if the
    flows never change sign, or all land on one day. Bisection is used rather
    than Newton's method because it cannot diverge; scipy's brentq is tried
    first since it converges faster.
    """
    if len(flows) < 2:
        return None
    flows = sorted(flows, key=lambda x: x[0])
    if flows[0][0] == flows[-1][0]:
        return None
    if not (any(cf < 0 for _, cf in flows) and any(cf > 0 for _, cf in flows)):
        return None

    lo, hi = -0.9999, 10.0
    try:
        f_lo, f_hi = xnpv(lo, flows), xnpv(hi, flows)
    except (OverflowError, ZeroDivisionError):
        return None
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return None

    try:
        from scipy.optimize import brentq

        return float(brentq(lambda r: xnpv(r, flows), lo, hi, maxiter=200, xtol=1e-10))
    except Exception:
        pass

    for _ in range(300):  # bisection fallback
        mid = (lo + hi) / 2
        f_mid = xnpv(mid, flows)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def cashflows(df: pd.DataFrame, terminal_value: float, as_of: date | None = None):
    """Buys are cash out, sells are cash in, and today's holdings are a final
    inflow representing what you'd receive if you liquidated now."""
    flows = [
        (r.date.date(), (-1 if r.transaction_type == "BUY" else 1) * float(r.amount))
        for r in df.itertuples()
    ]
    if terminal_value > 0:
        flows.append((as_of or date.today(), terminal_value))
    return flows


def tax_lot_table(fifo: FifoResult, quotes: dict[str, dict], today: date | None = None):
    """Open lots classified short- vs long-term, with the date each one flips.

    A lot crossing the one-year mark moves from ordinary income rates to the
    lower long-term capital gains rate, so the flip date is actionable.
    """
    today = today or date.today()
    rows = []
    for t, parcels in sorted(fifo.open_lots.items()):
        price = quotes.get(t, {}).get("price")
        for lot in parcels:
            bought = lot.date.date()
            held = (today - bought).days
            long_term = held > LONG_TERM_DAYS
            gain = (price - lot.price) * lot.qty if price is not None else np.nan
            rows.append(
                {
                    "Ticker": t,
                    "Purchased": lot.date,
                    "Quantity": lot.qty,
                    "Cost/Share": lot.price,
                    "Cost Basis": lot.cost,
                    "Market Value": price * lot.qty if price is not None else np.nan,
                    "Unrealized P/L": gain,
                    "Days Held": held,
                    "Term": "Long-term" if long_term else "Short-term",
                    "Long-term On": pd.Timestamp(bought + timedelta(days=LONG_TERM_DAYS + 1)),
                    "Days To Long-term": max(0, LONG_TERM_DAYS + 1 - held) if not long_term else 0,
                }
            )
    return pd.DataFrame(rows)
