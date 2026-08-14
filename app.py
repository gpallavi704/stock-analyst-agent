"""US Stock Portfolio Analyst — Streamlit entry point.

The whole app runs off one uploaded CSV. Analysis is computed once per rerun in
``utils.pipeline`` and handed to every tab, so a number shown on a chart, in a
metric card, and in the AI's answer can never disagree with each other.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from components import chat_tab, data_upload_tab, performance_tab, portfolio_view_tab
from utils.pipeline import build_analysis, settings

load_dotenv()

CONFIG_KEYS = ("GROQ_API_KEY", "GROQ_MODEL", "BENCHMARK_TICKER", "RISK_FREE_RATE")


def load_secrets() -> None:
    """Bridge Streamlit Cloud's secrets into the environment.

    Deployed there is no .env file — configuration is pasted into the app's
    Secrets box instead. Copying it into os.environ means every module keeps
    reading config the one way, whether it is running locally or hosted.
    """
    try:
        for key in CONFIG_KEYS:
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # No secrets configured at all: normal when running locally from .env.
        pass


load_secrets()

st.set_page_config(
    page_title="US Stock Portfolio Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def sidebar(cfg: dict) -> None:
    with st.sidebar:
        st.markdown("### 📈 Portfolio Analyst")
        st.caption("Your transaction CSV, turned into a portfolio you can interrogate.")
        st.divider()

        if source := st.session_state.get("source_name"):
            df = st.session_state.get("transactions")
            st.markdown(f"**Loaded:** `{source}`")
            if df is not None:
                st.caption(
                    f"{len(df)} transactions · {df['ticker'].nunique()} tickers · "
                    f"{df['date'].min():%b %Y} → {df['date'].max():%b %Y}"
                )
        else:
            st.info("No file loaded yet.", icon="📄")

        st.divider()
        st.markdown("**Settings**")
        st.caption(
            f"Benchmark `{cfg['benchmark']}` · risk-free rate "
            f"{cfg['risk_free'] * 100:.2f}%\n\nChange these in `.env` "
            "(`BENCHMARK_TICKER`, `RISK_FREE_RATE`)."
        )

        if st.button("Refresh market data", width="stretch"):
            st.cache_data.clear()
            st.session_state.pop("health_summary", None)
            st.session_state.pop("insights", None)
            st.rerun()
        st.caption("Quotes cache for 5 minutes, price history for an hour.")

        st.divider()
        st.caption(
            "Prices via Yahoo Finance and may be delayed. This tool analyses your own "
            "data and is not financial advice."
        )


def locked(label: str) -> None:
    st.info(
        f"{label} unlocks once a transaction CSV is loaded — head to the **Data Upload** "
        "tab and upload your file or load the sample portfolio.",
        icon="🔒",
    )


def main() -> None:
    cfg = settings()
    sidebar(cfg)

    st.title("US Stock Portfolio Analyst")

    tabs = st.tabs([
        "📄 Data Upload",
        "📊 Consolidated Portfolio View",
        "📈 Historical Performance",
        "🤖 AI Analyst Chat",
    ])

    with tabs[0]:
        data_upload_tab.render()

    df = st.session_state.get("transactions")
    analysis = None
    if df is not None:
        with st.spinner("Fetching prices and computing your portfolio…"):
            try:
                analysis = build_analysis(df, cfg["benchmark"], cfg["risk_free"])
            except Exception as exc:  # a failed analysis must not blank the whole app
                st.error(
                    f"Analysis failed: {exc}\n\nThis is usually a market-data timeout — "
                    "try **Refresh market data** in the sidebar.",
                    icon="💥",
                )

    for tab, module, label in (
        (tabs[1], portfolio_view_tab, "The portfolio view"),
        (tabs[2], performance_tab, "Performance history"),
        (tabs[3], chat_tab, "The AI analyst"),
    ):
        with tab:
            if analysis is None:
                locked(label)
            else:
                module.render(analysis)


if __name__ == "__main__":
    main()
