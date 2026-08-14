"""Tab 1 — upload, validate, and preview the transaction CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_processing import REQUIRED_COLUMNS, upload_stats, validate_transactions
from utils.pipeline import usd

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"


def _accept(df: pd.DataFrame, source: str) -> None:
    """Validate a raw dataframe and, if it passes, promote it to session state."""
    result = validate_transactions(df)

    for w in result.warnings:
        st.warning(w, icon="⚠️")

    if not result.ok:
        st.error("This file can't be used yet — fix the issues below and re-upload.", icon="🚫")
        for e in result.errors:
            st.markdown(f"- {e}")
        st.session_state.pop("transactions", None)
        return

    st.session_state["transactions"] = result.clean
    st.session_state["source_name"] = source
    st.success(f"Loaded {len(result.clean)} transactions from **{source}**.", icon="✅")


def render() -> None:
    st.subheader("Upload your transaction history")
    st.caption(
        "Your portfolio is reconstructed entirely from this file — there is no manual "
        "entry. Nothing is uploaded anywhere; everything is computed on your machine."
    )

    left, right = st.columns([2, 1])

    with left:
        uploaded = st.file_uploader(
            "Transaction CSV", type=["csv"], label_visibility="collapsed",
            help=f"Required columns: {', '.join(REQUIRED_COLUMNS)}",
        )
    with right:
        st.write("")
        if st.button("Load sample portfolio", width="stretch"):
            if SAMPLE.exists():
                _accept(pd.read_csv(SAMPLE), "sample_transactions.csv")
            else:
                st.error("Sample file is missing from data/.")

    with st.expander("Required CSV format"):
        st.markdown(
            "Exactly these five columns, in any order. `transaction_type` may be any "
            "capitalisation. Dates in any common format."
        )
        st.code(
            "ticker,date,transaction_type,quantity,price\n"
            "AAPL,2023-02-14,BUY,30,153.20\n"
            "MSFT,2023-03-21,Buy,15,273.78\n"
            "AAPL,2025-01-15,sell,25,237.87",
            language="csv",
        )

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read that file as CSV: {exc}", icon="🚫")
            return
        _accept(raw, uploaded.name)

    df = st.session_state.get("transactions")
    if df is None:
        st.info("Upload a CSV or load the sample portfolio to begin.", icon="📄")
        return

    st.divider()
    s = upload_stats(df)
    st.markdown("#### Upload summary")
    c = st.columns(5)
    c[0].metric("Transactions", s["transactions"])
    c[1].metric("Unique tickers", s["tickers"])
    c[2].metric("Buys", s["buys"])
    c[3].metric("Sells", s["sells"])
    c[4].metric("Date range", f"{(s['end'] - s['start']).days // 365}y",
                f"{s['start']:%b %Y} → {s['end']:%b %Y}", delta_color="off")

    c2 = st.columns(2)
    c2[0].metric("Total bought", usd(s["invested"], 0))
    c2[1].metric("Total sold", usd(s["proceeds"], 0))

    st.markdown("#### Cleaned data")
    st.caption(
        "Tickers upper-cased, dates parsed, rows sorted oldest first. `amount` is "
        "quantity × price, added by the app."
    )
    st.dataframe(
        df.assign(date=df["date"].dt.strftime("%Y-%m-%d")),
        width="stretch", hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn(format="$%.2f"),
            "amount": st.column_config.NumberColumn(format="$%.2f"),
            "quantity": st.column_config.NumberColumn(format="%.4g"),
        },
    )

    st.download_button(
        "Download cleaned CSV", df.to_csv(index=False).encode(),
        file_name="cleaned_transactions.csv", mime="text/csv",
    )
