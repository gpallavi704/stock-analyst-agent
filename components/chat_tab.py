"""Tab 4 — conversational analyst backed by tool calls, not a stuffed prompt.

The model never sees a giant dump of the portfolio. It gets a toolbox and picks
what it needs, which keeps answers grounded in computed numbers and makes it
obvious — via the "data used" note under each reply — where a figure came from.
"""

from __future__ import annotations

import streamlit as st

from utils.llm_agent import DISCLAIMER, GroqNotConfigured, PortfolioAgent
from utils.pipeline import Analysis, agent_context

SUGGESTIONS = [
    "How is my portfolio doing overall?",
    "Did I actually beat the market?",
    "Where am I most concentrated?",
    "Which positions are dragging me down?",
    "Which lots are close to becoming long-term?",
    "Why did my portfolio move today?",
]

TOOL_LABELS = {
    "get_portfolio_summary": "portfolio summary",
    "get_holdings": "holdings",
    "get_sector_breakdown": "sector breakdown",
    "get_day_attribution": "today's attribution",
    "get_benchmark_comparison": "benchmark comparison",
    "get_risk_metrics": "risk metrics",
    "get_tax_lots": "tax lots",
    "get_realized_trades": "closed trades",
    "get_transactions": "transaction history",
    "get_price_history": "price history",
}


def _setup_help() -> None:
    st.warning("The AI analyst needs a Groq API key before it can answer.", icon="🔑")
    st.markdown(
        "1. Create a free key at [console.groq.com](https://console.groq.com/keys)\n"
        "2. Add it to a `.env` file in the project root:\n"
    )
    st.code("GROQ_API_KEY=gsk_your_key_here\nGROQ_MODEL=llama-3.3-70b-versatile", language="bash")
    st.markdown("3. Restart the app.")
    st.caption("Every other tab works fully without a key — only this one is gated.")


def _insights(agent: PortfolioAgent) -> None:
    st.markdown("##### Proactive insights")
    st.caption(
        "Nothing is asked here — the analyst investigates on its own and reports what "
        "it thinks you may not have noticed."
    )
    if st.button("Analyse my portfolio", key="insights_btn"):
        with st.spinner("Investigating — checking benchmark, sectors, risk and tax lots…"):
            try:
                st.session_state["insights"] = agent.proactive_insights()
            except GroqNotConfigured as exc:
                st.error(str(exc), icon="🔑")

    if text := st.session_state.get("insights"):
        st.markdown(text)


def _ask(agent: PortfolioAgent, question: str) -> None:
    history = st.session_state["chat"]
    history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply, tools = agent.chat(history)
            except GroqNotConfigured as exc:
                reply, tools = f"🔑 {exc}", []
        st.markdown(reply)
        if tools:
            st.caption("Data used: " + ", ".join(TOOL_LABELS.get(t, t) for t in dict.fromkeys(tools)))

    history.append({"role": "assistant", "content": reply, "tools": tools})


def render(a: Analysis) -> None:
    st.subheader("AI analyst chat")

    agent = PortfolioAgent(agent_context(a))
    if not agent.configured:
        _setup_help()
        return

    st.caption(
        f"Answers are grounded in tools that read your computed portfolio — the model "
        f"cannot invent a number. It has no news access, so it will say so rather than "
        f"explain *why* a stock moved. Model: `{agent.model}`."
    )

    _insights(agent)
    st.divider()

    st.session_state.setdefault("chat", [])

    if not st.session_state["chat"]:
        st.markdown("##### Start with one of these")
        cols = st.columns(3)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 3].button(s, key=f"suggest_{i}", width="stretch"):
                st.session_state["pending_question"] = s
                st.rerun()

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                st.caption("Data used: " + ", ".join(
                    TOOL_LABELS.get(t, t) for t in dict.fromkeys(msg["tools"])))

    if pending := st.session_state.pop("pending_question", None):
        _ask(agent, pending)

    if question := st.chat_input("Ask about your portfolio…"):
        _ask(agent, question)

    if st.session_state["chat"]:
        if st.button("Clear conversation"):
            st.session_state["chat"] = []
            st.rerun()

    st.caption(DISCLAIMER)
