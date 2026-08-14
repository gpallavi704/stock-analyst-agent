"""Groq-backed portfolio analyst with real tool calling.

Rather than flattening the whole portfolio into one prompt, the model is given
a set of callable tools and decides which data it needs. That keeps the base
prompt small, lets it drill into a specific ticker or tax lot on demand, and —
most importantly — means answers about market moves come from fetched prices
rather than from the model's imagination.

Every tool returns computed numbers. The model's job is to explain them, not to
produce them.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "qwen/qwen3.6-27b"
# The insights prompt asks for five separate tools, and some models fetch only
# one per round, so this ceiling has to clear that comfortably.
MAX_TOOL_ROUNDS = 8

DISCLAIMER = (
    "This is informational analysis of your own uploaded data, not financial advice."
)

SYSTEM_PROMPT = f"""You are a portfolio analyst assistant examining the user's real \
brokerage transaction history.

Hard rules:
- Ground every claim in data returned by your tools. Call a tool before answering \
anything about numbers, holdings, prices, or performance.
- Never invent market news, analyst opinions, earnings results, or reasons a stock \
moved. If asked "why" a stock moved and you only have price data, say what the price \
did and state plainly that you don't have news access.
- If the tools don't cover something, say so instead of guessing.
- Quote the tools' figures as they come back. Never re-group holdings into your own \
categories or recompute a percentage the tools already report — if the sector tool says \
44.1% Technology, that is the number, even if you would have classified the companies \
differently. Contradicting the app's own displayed figures is the worst failure you can \
make.
- No buy/sell recommendations. You may describe risks that follow arithmetically from \
the data, such as concentration.
- Be concise and specific. Use real figures with tickers. Prefer 3-6 sentences or a \
short list over an essay.
- Format money as $12,345 and percentages to one decimal.
- {DISCLAIMER} Mention it when giving an overall assessment, not in every reply.
"""

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": (
                "Overall portfolio state: total value, cost basis, total invested, "
                "realized and unrealized P/L, total return, XIRR, and today's change. "
                "Start here for broad questions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_holdings",
            "description": (
                "Every current holding with quantity, average cost, current price, "
                "market value, allocation %, unrealized and realized P/L. Use for "
                "questions about which stocks are winning or losing."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_breakdown",
            "description": (
                "Allocation grouped by sector plus computed concentration warnings. "
                "Use for diversification and concentration-risk questions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_day_attribution",
            "description": (
                "Today's dollar move broken down by holding, largest mover first. "
                "This is the correct tool for 'why is my portfolio up/down today'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_benchmark_comparison",
            "description": (
                "Compares the portfolio against investing the identical cashflows "
                "into the benchmark index on the same dates. Use for 'did I beat "
                "the market' or 'was my stock picking worth it'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_metrics",
            "description": (
                "Annualized volatility, Sharpe ratio, beta vs the benchmark, and "
                "maximum drawdown with its peak and trough dates."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tax_lots",
            "description": (
                "Open lots with purchase date, cost, unrealized gain, and whether "
                "each is short-term or long-term for US capital gains, including "
                "days until a short-term lot becomes long-term."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Optional ticker filter."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_realized_trades",
            "description": (
                "Closed trades with proceeds, FIFO cost basis, realized gain/loss, "
                "and short/long-term split. Use for questions about past selling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Optional ticker filter."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": "The raw uploaded transaction history, optionally filtered by ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Optional ticker filter."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "Recent daily closing prices for one ticker, with the change over the "
                "period. Call this before commenting on how a stock has moved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol."},
                    "days": {
                        "type": "integer",
                        "description": "Calendar days of history, default 30, max 365.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


PLAIN_TEXT_NUDGE = (
    "Answer now, in plain prose, using the tool results already provided. Do not "
    "write function-call syntax in your reply."
)

# Provider-side flakes that succeed when the identical request is retried.
TRANSIENT_ERRORS = ("output_parse_failed", "tool_use_failed")

WRAP_UP_NUDGE = (
    "Stop calling tools and write your final answer now, using only the tool "
    "results already gathered above."
)

# Llama models occasionally emit a tool call as literal text instead of using the
# tool-calling channel. Left alone it reaches the user as raw XML.
_PSEUDO_CALL = re.compile(r"<function=.*?(?:</function>|$)", re.DOTALL)


def _strip_pseudo_calls(text: str) -> str:
    return _PSEUDO_CALL.sub("", text).strip()


def _readable_wait(raw: str) -> str:
    """Groq reports waits like '11m16.512s' or '2.182499999s'. Round them off."""
    m = re.match(r"(?:(\d+)m)?([\d.]+)s", raw)
    if not m:
        return raw
    minutes, seconds = int(m.group(1) or 0), round(float(m.group(2)))
    if minutes:
        return f"{minutes} min {seconds} sec" if seconds else f"{minutes} min"
    return f"{max(seconds, 1)} sec"


def _friendly_error(exc: Exception) -> str:
    """Turn provider errors into something the user can act on.

    A raw 429 payload is a wall of JSON that buries the one useful fact: free
    tier limits are per model, so switching models is an instant fix.
    """
    text = str(exc)
    if "rate_limit" in text or "429" in text:
        wait = re.search(r"try again in ([\w.]+)", text)
        when = _readable_wait(wait.group(1)) if wait else None

        # A per-minute burst limit clears on its own in seconds; the daily token
        # budget does not, and needs a different model. Telling the user to edit
        # config when they only need to pause would be actively unhelpful.
        if "per day" in text or "TPD" in text:
            return (
                "⚠️ **Groq's free daily token budget for this model is used up.**"
                + (f" It resets in about **{when}**." if when else "")
                + "\n\nThe budget is *per model*, so the quickest fix is to change "
                "`GROQ_MODEL` in your `.env` and restart the app — try "
                "`openai/gpt-oss-120b` or `llama-3.3-70b-versatile`. Every other "
                "tab works without the AI."
            )
        return (
            "⚠️ **Too many AI requests in a short window.** Wait "
            + (f"**{when}**" if when else "a moment")
            + " and ask again — this one clears on its own."
        )
    if "invalid_api_key" in text or "401" in text:
        return (
            "⚠️ **Groq rejected the API key.** Check `GROQ_API_KEY` in your `.env` "
            "against https://console.groq.com/keys, then restart the app."
        )
    if "model_not_found" in text or "does not exist" in text:
        return (
            f"⚠️ **Groq doesn't recognise that model.** Check `GROQ_MODEL` in your "
            f"`.env` against https://console.groq.com/docs/models.\n\n`{text[:200]}`"
        )
    return f"⚠️ The AI request failed: {exc}"


class GroqNotConfigured(RuntimeError):
    """Raised when no API key is available, so the UI can show setup help."""


@dataclass
class AgentContext:
    """Everything the tools can read. Populated once per rerun by the app."""

    transactions: pd.DataFrame | None = None
    holdings: pd.DataFrame | None = None
    totals: dict = field(default_factory=dict)
    sectors: pd.DataFrame | None = None
    flags: list[str] = field(default_factory=list)
    tax_lots: pd.DataFrame | None = None
    sales: list = field(default_factory=list)
    benchmark: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    drawdown: dict = field(default_factory=dict)
    xirr: float | None = None
    attribution: pd.DataFrame | None = None
    dividends: pd.DataFrame | None = None


def _frame(df: pd.DataFrame | None, limit: int = 40) -> Any:
    """Compact, token-cheap serialisation of a dataframe for the model.

    CSV rather than JSON records: a record-per-row payload repeats every column
    name on every row, which roughly doubles the token cost of a wide table for
    no gain in comprehension.
    """
    if df is None or df.empty:
        return "no data available"
    out = df.head(limit).copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(2)
    text = out.to_csv(index=False).strip()
    if len(df) > limit:
        text += f"\n({len(df) - limit} more rows omitted)"
    return text


def _clean(d: dict) -> dict:
    """Round floats and stringify timestamps so json.dumps never chokes."""
    out = {}
    for k, v in d.items():
        if isinstance(v, pd.Timestamp):
            out[k] = v.strftime("%Y-%m-%d")
        elif isinstance(v, float):
            out[k] = None if pd.isna(v) else round(v, 2)
        else:
            out[k] = v
    return out


class PortfolioAgent:
    """Thin wrapper over Groq chat completions with a tool-execution loop."""

    def __init__(self, ctx: AgentContext, api_key: str | None = None, model: str | None = None):
        self.ctx = ctx
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or ""
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

    def _get_client(self):
        if not self.configured:
            raise GroqNotConfigured(
                "GROQ_API_KEY is not set. Create a free key at console.groq.com and "
                "put it in a .env file at the project root as GROQ_API_KEY=..."
            )
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    # -- tool implementations ------------------------------------------------

    def _tools(self) -> dict[str, Callable[..., Any]]:
        c = self.ctx
        return {
            "get_portfolio_summary": lambda: _clean({**c.totals, "xirr_pct": (
                c.xirr * 100 if c.xirr is not None else None)}),
            "get_holdings": lambda: _frame(c.holdings),
            "get_sector_breakdown": lambda: {
                "by_sector": _frame(c.sectors),
                "concentration_observations": c.flags,
            },
            "get_day_attribution": lambda: _frame(c.attribution),
            "get_benchmark_comparison": lambda: _clean(c.benchmark) if c.benchmark
            else "benchmark comparison unavailable",
            "get_risk_metrics": lambda: {
                **_clean(c.risk), "max_drawdown": _clean(c.drawdown)},
            "get_tax_lots": self._tax_lots,
            "get_realized_trades": self._realized,
            "get_transactions": self._transactions,
            "get_price_history": self._price_history,
        }

    def _tax_lots(self, ticker: str | None = None):
        df = self.ctx.tax_lots
        if df is None or df.empty:
            return "no open lots"
        if ticker:
            df = df[df["Ticker"] == ticker.upper()]
        return _frame(df, limit=60)

    def _realized(self, ticker: str | None = None):
        sales = self.ctx.sales or []
        if ticker:
            sales = [s for s in sales if s.ticker == ticker.upper()]
        if not sales:
            return "no closed trades"
        rows = []
        for s in sales:
            st, lt = s.split_by_term()
            rows.append({
                "ticker": s.ticker, "date": s.date.strftime("%Y-%m-%d"),
                "qty": round(s.qty, 4), "sell_price": round(s.price, 2),
                "proceeds": round(s.proceeds, 2), "cost_basis": round(s.cost_basis, 2),
                "realized": round(s.realized, 2), "realized_pct": round(s.realized_pct, 2),
                "short_term": round(st, 2), "long_term": round(lt, 2),
            })
        return rows

    def _transactions(self, ticker: str | None = None):
        df = self.ctx.transactions
        if df is None or df.empty:
            return "no transactions"
        if ticker:
            df = df[df["ticker"] == ticker.upper()]
        return _frame(df[["ticker", "date", "transaction_type", "quantity", "price", "amount"]], 60)

    def _price_history(self, ticker: str, days: int = 30):
        from utils.market_data import get_history

        days = max(5, min(int(days or 30), 365))
        start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        hist = get_history((ticker.upper(),), start)
        if hist.empty or ticker.upper() not in hist.columns:
            return f"no price history available for {ticker.upper()}"
        s = hist[ticker.upper()].dropna()
        if s.empty:
            return f"no price history available for {ticker.upper()}"
        first, last = float(s.iloc[0]), float(s.iloc[-1])
        step = max(1, len(s) // 12)  # thin the series to keep the payload small
        return {
            "ticker": ticker.upper(),
            "period_days": days,
            "start_price": round(first, 2),
            "latest_price": round(last, 2),
            "change_pct": round((last - first) / first * 100, 2) if first else None,
            "high": round(float(s.max()), 2),
            "low": round(float(s.min()), 2),
            "samples": {d.strftime("%Y-%m-%d"): round(float(v), 2)
                        for d, v in s.iloc[::step].items()},
        }

    # -- conversation --------------------------------------------------------

    def _create(self, client, kwargs: dict, attempts: int = 3):
        """Call the API, retrying the provider's own non-deterministic failures.

        Groq intermittently rejects a completion with ``output_parse_failed``
        when it cannot parse the model's tool-call output — the same request
        succeeds on a retry, so failing the whole answer would be needless.
        """
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not any(code in str(exc) for code in TRANSIENT_ERRORS):
                    raise
                last = exc
        raise last  # type: ignore[misc]

    def _complete(self, messages: list[dict], use_tools: bool = True, temperature: float = 0.3):
        client = self._get_client()
        tools = self._tools()
        convo = list(messages)
        last_used: list[str] = []

        for round_no in range(MAX_TOOL_ROUNDS):
            if round_no == MAX_TOOL_ROUNDS - 1:
                # Last permitted round: ask for the answer now. The tool schema
                # stays attached, because a model that emits a call when no tools
                # are advertised gets a 400 back from the provider.
                convo.append({"role": "user", "content": WRAP_UP_NUDGE})

            kwargs: dict[str, Any] = {
                "model": self.model, "messages": convo,
                "temperature": temperature, "max_tokens": 1200,
            }
            if use_tools:
                kwargs["tools"] = TOOL_SCHEMA
                kwargs["tool_choice"] = "auto"

            msg = self._create(client, kwargs).choices[0].message
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                content = _strip_pseudo_calls(msg.content or "")
                if content:
                    return content, last_used
                # The model printed only tool-call syntax as prose. Show it the
                # mistake and ask again rather than handing the user raw XML.
                convo.append({"role": "assistant", "content": msg.content or ""})
                convo.append({"role": "user", "content": PLAIN_TEXT_NUDGE})
                continue

            convo.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls
                ],
            })

            used = []
            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                # No-argument tools come back as the literal "null", which parses
                # to None and would blow up the ** unpack — turning every such
                # call into a tool error the model then reports as "no data".
                if not isinstance(args, dict):
                    args = {}
                used.append(name)
                fn = tools.get(name)
                try:
                    result = fn(**args) if fn else f"unknown tool: {name}"
                except Exception as exc:  # a broken tool shouldn't kill the chat
                    result = f"tool error: {exc}"
                # Strings (already CSV) go through raw; JSON-encoding them would
                # only escape every newline and quote back into the payload.
                payload = result if isinstance(result, str) else json.dumps(result, default=str)
                convo.append({
                    "role": "tool", "tool_call_id": call.id, "name": name,
                    "content": payload[:12000],
                })

            # Loop back so the model can answer using the tool output, or call more.
            last_used = used

        # Still calling tools after the wrap-up nudge. Everything fetched is in
        # the transcript, so say so rather than surfacing a provider error.
        return (
            "I gathered the data but ran out of steps before summarising it. "
            "Try asking a narrower question.", last_used,
        )

    def chat(self, history: list[dict]) -> tuple[str, list[str]]:
        """Answer the latest user message. Returns (reply, tools_used)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m["role"], "content": m["content"]} for m in history
        ]
        try:
            reply, used = self._complete(messages)
        except GroqNotConfigured:
            raise
        except Exception as exc:
            return _friendly_error(exc), []
        if not reply:
            return "⚠️ The model returned an empty response. Try rephrasing your question.", used
        return reply, used

    def health_summary(self) -> str:
        """Short portfolio-health paragraph for the holdings tab."""
        prompt = (
            "Write 2-3 sentences on the health of this portfolio. Call the summary, "
            "holdings and sector tools first. Cover total value, the balance of "
            "realized vs unrealized gains, and explicitly address concentration risk "
            "(including sector concentration, which can be high even when no single "
            "stock is). No recommendations."
        )
        try:
            reply, _ = self._complete(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}]
            )
            return reply or "The model returned an empty summary."
        except GroqNotConfigured:
            raise
        except Exception as exc:
            return _friendly_error(exc)

    def proactive_insights(self) -> str:
        """Unprompted findings: the agent decides what's worth flagging."""
        prompt = (
            "Investigate this portfolio and report the 3-4 most important findings the "
            "owner may not have noticed. Use several tools before answering — at "
            "minimum the summary, sector breakdown, benchmark comparison, risk metrics "
            "and tax lots. Prioritise things that are actionable or surprising, such as "
            "underperforming the index despite gains, hidden sector concentration, "
            "short-term lots close to becoming long-term, or an outsized drawdown. "
            "Format as a markdown list where each item is a bold headline followed by "
            "one sentence with the supporting numbers. End with the disclaimer."
        )
        try:
            reply, used = self._complete(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}], temperature=0.4
            )
            return reply or "The model returned no insights."
        except GroqNotConfigured:
            raise
        except Exception as exc:
            return _friendly_error(exc)
