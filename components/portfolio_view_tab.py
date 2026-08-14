"""Tab 2 — the consolidated view of what is owned right now.

Everything here describes the present: positions rebuilt from FIFO lots, valued
at the latest price, plus the concentration and tax-lot structure hiding inside
those positions. Anything time-series lives in the performance tab.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.llm_agent import GroqNotConfigured, PortfolioAgent
from utils.pipeline import Analysis, agent_context, pct, signed_usd, usd

HOLDINGS_FORMAT = {
    "Quantity": st.column_config.NumberColumn(format="%.4g"),
    "Avg Cost Basis": st.column_config.NumberColumn(format="$%.2f"),
    "Current Price": st.column_config.NumberColumn(format="$%.2f"),
    "Cost Basis": st.column_config.NumberColumn(format="$%.2f"),
    "Market Value": st.column_config.NumberColumn(format="$%.2f"),
    "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
    "Unrealized P/L %": st.column_config.NumberColumn(format="%.2f%%"),
    "Day Change": st.column_config.NumberColumn(format="$%.2f"),
    "Day Change %": st.column_config.NumberColumn(format="%.2f%%"),
    "Realized P/L": st.column_config.NumberColumn(format="$%.2f"),
    "Allocation %": st.column_config.ProgressColumn(
        format="%.1f%%", min_value=0, max_value=100
    ),
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
)


def _notices(a: Analysis) -> None:
    """Surface anything that would silently distort the numbers below."""
    for err in a.fifo.errors:
        st.error(err, icon="🚫")

    if a.price_failures:
        st.warning(
            "No live price for " + ", ".join(a.price_failures) + ". Those positions "
            "are listed but excluded from every total.",
            icon="📡",
        )

    held = set(a.fifo.open_lots)
    split_names = [t for t in a.splits if t in held]
    if split_names:
        st.warning(
            "Stock split since purchase: "
            + ", ".join(
                f"{t} ({', '.join(f'{r:g}-for-1 on {d:%b %Y}' for d, r in a.splits[t].items())})"
                for t in split_names
            )
            + ". Your CSV records the pre-split price while the quote is post-split, "
            "so restate those rows (multiply quantity, divide price) or the position "
            "will show a false loss.",
            icon="✂️",
        )

    if a.fifo.closed:
        st.info(
            "Fully sold, so excluded from holdings: " + ", ".join(sorted(a.fifo.closed))
            + ". Their realised gains still count in the totals.",
            icon="📁",
        )


def _headline(a: Analysis) -> None:
    t = a.totals
    day = t["day_change"]
    prior = t["current_value"] - day

    c = st.columns(4)
    c[0].metric(
        "Portfolio value", usd(t["current_value"], 0),
        f"{signed_usd(day)} today ({pct(day / prior * 100 if prior else None, 2)})",
    )
    c[1].metric("Total invested", usd(t["total_investment"], 0),
                f"{usd(t['total_sells'], 0)} sold back", delta_color="off")
    c[2].metric("Total return", signed_usd(t["total_return"]), pct(t["total_return_pct"], 2))
    c[3].metric("XIRR (annualised)", pct(a.xirr * 100 if a.xirr else None, 2),
                "money-weighted", delta_color="off")

    c2 = st.columns(4)
    c2[0].metric("Unrealised P/L", signed_usd(t["unrealized_pl"]),
                 pct(t["unrealized_pl"] / t["open_cost_basis"] * 100
                     if t["open_cost_basis"] else None, 2))
    c2[1].metric("Realised P/L", signed_usd(t["realized_pl"]),
                 f"{len(a.fifo.sales)} closed trade(s)", delta_color="off")
    c2[2].metric("Open cost basis", usd(t["open_cost_basis"], 0))
    c2[3].metric("Positions", len(a.holdings),
                 f"{int(a.holdings['Lots'].sum()) if not a.holdings.empty else 0} lots",
                 delta_color="off")

    st.caption(
        "Total return counts realised proceeds plus today's market value against every "
        "dollar ever put in. XIRR annualises that same cash, so it accounts for *when* "
        "each purchase was made rather than just how much."
    )


def _allocation(a: Analysis) -> None:
    priced = a.holdings.dropna(subset=["Market Value"])
    if priced.empty:
        return

    left, right = st.columns(2)

    with left:
        st.markdown("##### Allocation by position")
        fig = px.pie(priced, names="Ticker", values="Market Value", hole=0.55)
        fig.update_traces(
            textposition="outside", textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:$,.0f}<br>%{percent}<extra></extra>",
        )
        fig.update_layout(showlegend=False, height=380, **CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("##### Unrealised gain / loss")
        d = priced.sort_values("Unrealized P/L")
        fig = px.bar(
            d, x="Unrealized P/L", y="Ticker", orientation="h",
            color="Unrealized P/L", color_continuous_scale=["#d64545", "#9e9e9e", "#2e9e6b"],
            color_continuous_midpoint=0,
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>%{x:$,.0f}<extra></extra>"
        )
        fig.update_layout(
            height=380, coloraxis_showscale=False,
            xaxis_title=None, yaxis_title=None, **CHART_LAYOUT,
        )
        fig.update_xaxes(tickformat="$,.0f", zeroline=True, zerolinecolor="rgba(128,128,128,.5)")
        st.plotly_chart(fig, width="stretch")


def _holdings_table(a: Analysis) -> None:
    st.markdown("##### Current holdings")
    st.caption(
        "Cost basis is FIFO: a sale consumes the oldest shares first, so the average "
        "cost shown is the average of the lots you still hold."
    )

    names = {t: p.get("name", t) for t, p in a.profiles.items()}
    table = a.holdings.copy()
    table.insert(1, "Company", table["Ticker"].map(names).fillna(table["Ticker"]))

    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={**HOLDINGS_FORMAT,
                       "Company": st.column_config.TextColumn(width="medium"),
                       "Lots": st.column_config.NumberColumn(help="Open FIFO lots")},
    )
    st.download_button(
        "Download holdings CSV", table.to_csv(index=False).encode(),
        file_name="holdings.csv", mime="text/csv",
    )


def _movers(a: Analysis) -> None:
    if a.attribution.empty:
        return
    st.markdown("##### What moved today")
    st.caption(
        "Today's portfolio change attributed to each holding, so the move is explained "
        "by arithmetic rather than by guessing at news."
    )
    st.dataframe(
        a.attribution, width="stretch", hide_index=True,
        column_config={
            "Day Change": st.column_config.NumberColumn(format="$%.2f"),
            "Day Change %": st.column_config.NumberColumn(format="%.2f%%"),
            "Market Value": st.column_config.NumberColumn(format="$%.2f"),
            "Share of Move %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


def _concentration(a: Analysis) -> None:
    st.markdown("##### Diversification")
    if a.sectors.empty:
        st.info("Sector data unavailable for these tickers.")
        return

    left, right = st.columns([1, 1])
    with left:
        fig = px.pie(a.sectors, names="Sector", values="Market Value", hole=0.55)
        fig.update_traces(
            textposition="outside", textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:$,.0f}<br>%{percent}<extra></extra>",
        )
        fig.update_layout(showlegend=False, height=330, **CHART_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.dataframe(
            a.sectors, width="stretch", hide_index=True,
            column_config={
                "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
                "Allocation %": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100),
                "Holdings": st.column_config.TextColumn(width="medium"),
            },
        )

    for flag in a.flags:
        st.markdown(f"- {flag}")
    st.caption(
        "A portfolio can look spread out stock-by-stock while sitting almost entirely "
        "in one sector — those names fall together, so the sector row is the honest test."
    )


def _tax_lots(a: Analysis) -> None:
    st.markdown("##### Tax lots")
    if a.tax_lots.empty:
        st.info("No open lots.")
        return

    lots = a.tax_lots
    short = lots[lots["Term"] == "Short-term"]
    long = lots[lots["Term"] == "Long-term"]

    c = st.columns(4)
    c[0].metric("Short-term lots", len(short), signed_usd(short["Unrealized P/L"].sum()),
                delta_color="off")
    c[1].metric("Long-term lots", len(long), signed_usd(long["Unrealized P/L"].sum()),
                delta_color="off")

    soon = short[short["Days To Long-term"] <= 90].sort_values("Days To Long-term")
    c[2].metric("Flip within 90 days", len(soon))
    c[3].metric("Gains still short-term",
                signed_usd(short.loc[short["Unrealized P/L"] > 0, "Unrealized P/L"].sum()))

    if not soon.empty:
        nearest = soon.iloc[0]
        st.info(
            f"{nearest['Ticker']} bought {nearest['Purchased']:%d %b %Y} becomes long-term "
            f"in {int(nearest['Days To Long-term'])} days "
            f"({nearest['Long-term On']:%d %b %Y}) — selling after that date taxes the "
            f"{signed_usd(nearest['Unrealized P/L'])} gain at the lower long-term rate.",
            icon="⏳",
        )

    st.dataframe(
        lots, width="stretch", hide_index=True,
        column_config={
            "Purchased": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Long-term On": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Quantity": st.column_config.NumberColumn(format="%.4g"),
            "Cost/Share": st.column_config.NumberColumn(format="$%.2f"),
            "Cost Basis": st.column_config.NumberColumn(format="$%.2f"),
            "Market Value": st.column_config.NumberColumn(format="$%.2f"),
            "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.caption(
        "US rule: shares held more than one year are taxed at the long-term capital "
        "gains rate. This is informational, not tax advice."
    )


def _ai_summary(a: Analysis) -> None:
    st.markdown("##### AI portfolio health check")
    agent = PortfolioAgent(agent_context(a))

    if not agent.configured:
        st.info(
            "Set `GROQ_API_KEY` in your `.env` to enable AI commentary. "
            "The rest of this tab works without it.",
            icon="🔑",
        )
        return

    if st.button("Generate health check", key="health_btn"):
        with st.spinner("The analyst is reading your portfolio…"):
            try:
                st.session_state["health_summary"] = agent.health_summary()
            except GroqNotConfigured as exc:
                st.error(str(exc), icon="🔑")

    if summary := st.session_state.get("health_summary"):
        st.markdown(summary)


def render(a: Analysis) -> None:
    st.subheader("Consolidated portfolio")
    st.caption(
        "Rebuilt from your transaction history alone — every share, lot and dollar "
        "below traces back to a row in the CSV."
    )

    _notices(a)

    if a.holdings.empty:
        st.warning("Every position has been fully sold, so there are no open holdings.")
        return

    _headline(a)
    st.divider()
    _allocation(a)
    _holdings_table(a)
    st.divider()
    _movers(a)
    st.divider()
    _concentration(a)
    st.divider()
    _tax_lots(a)
    st.divider()
    _ai_summary(a)
