"""All yfinance access lives here, wrapped in Streamlit caches.

Network calls are the slowest and least reliable part of the app, so every
function degrades to an empty/partial result instead of raising. Callers are
expected to check for missing tickers and tell the user which ones failed.

TTLs are deliberately short for quotes (prices move) and long for metadata
(a company's sector does not change during a session).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

QUOTE_TTL = 300  # 5 min
HISTORY_TTL = 3600  # 1 hr
META_TTL = 86400  # 1 day


def _naive(s: pd.Series) -> pd.Series:
    """Strip timezone and time-of-day so batch and per-ticker results align.

    ``yf.download`` returns naive dates while ``Ticker.history`` returns
    tz-aware ones; mixing them in one frame would misalign every row.
    """
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out = s.copy()
    out.index = idx.normalize()
    return out


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def get_quotes(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Latest close and previous close per ticker.

    Returns ``{ticker: {"price": float, "prev_close": float|None}}``. Tickers
    that fail are simply absent from the mapping.
    """
    if not tickers:
        return {}
    out: dict[str, dict] = {}
    try:
        raw = yf.download(
            list(tickers), period="5d", progress=False,
            auto_adjust=False, group_by="ticker", threads=True,
        )
    except Exception:
        raw = None

    for t in tickers:
        closes = None
        try:
            if raw is not None and not raw.empty:
                # yfinance flattens columns when only one ticker is requested.
                col = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
                closes = col.dropna()
        except Exception:
            closes = None

        if closes is None or closes.empty:
            try:  # per-ticker retry; a single bad symbol shouldn't sink the batch
                closes = yf.Ticker(t).history(period="5d")["Close"].dropna()
            except Exception:
                continue
        if closes is None or closes.empty:
            continue

        out[t] = {
            "price": float(closes.iloc[-1]),
            "prev_close": float(closes.iloc[-2]) if len(closes) > 1 else None,
        }
    return out


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def get_history(tickers: tuple[str, ...], start: str, end: str | None = None) -> pd.DataFrame:
    """Daily close prices, one column per ticker, forward-filled across gaps.

    Uses unadjusted closes so the series is comparable to the prices a user
    actually typed into their CSV.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(tickers), start=start, end=end, progress=False,
            auto_adjust=False, group_by="ticker", threads=True,
        )
    except Exception:
        raw = None

    frame = {}
    if raw is not None and not raw.empty:
        for t in tickers:
            try:
                col = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
                s = col.dropna()
                if not s.empty:
                    frame[t] = _naive(s)
            except Exception:
                continue

    # The batch endpoint silently drops symbols when it is rate-limited, which
    # would quietly remove the benchmark and with it the comparison that matters
    # most. Retry the stragglers one at a time.
    for t in tickers:
        if t in frame:
            continue
        try:
            s = yf.Ticker(t).history(start=start, end=end, auto_adjust=False)["Close"].dropna()
            if not s.empty:
                frame[t] = _naive(s)
        except Exception:
            continue

    if not frame:
        return pd.DataFrame()

    out = pd.DataFrame(frame)
    # Reindex onto every calendar day so portfolio value can be evaluated on any
    # date; ffill carries the last close through weekends and holidays.
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    return out.reindex(full).ffill()


@st.cache_data(ttl=META_TTL, show_spinner=False)
def get_profiles(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Company name / sector / industry, used for the diversification view."""
    out: dict[str, dict] = {}
    for t in tickers:
        info = {}
        try:
            info = yf.Ticker(t).get_info() or {}
        except Exception:
            info = {}
        out[t] = {
            "name": info.get("longName") or info.get("shortName") or t,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "quote_type": info.get("quoteType") or "",
        }
    return out


@st.cache_data(ttl=META_TTL, show_spinner=False)
def get_dividends(tickers: tuple[str, ...], start: str) -> dict[str, pd.Series]:
    """Per-share dividend payments indexed by ex-dividend date."""
    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            s = yf.Ticker(t).dividends
            if s is None or s.empty:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[s.index >= pd.Timestamp(start)]
            if not s.empty:
                out[t] = s
        except Exception:
            continue
    return out


@st.cache_data(ttl=META_TTL, show_spinner=False)
def get_splits(tickers: tuple[str, ...], start: str) -> dict[str, pd.Series]:
    """Split events since ``start``.

    This exists to catch a silent correctness bug: yfinance reports today's
    price on a post-split basis, but a CSV records the price actually paid. A
    position held through a split will otherwise show a fake catastrophic loss.
    """
    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            s = yf.Ticker(t).splits
            if s is None or s.empty:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[s.index >= pd.Timestamp(start)]
            if not s.empty:
                out[t] = s
        except Exception:
            continue
    return out
