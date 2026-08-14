# US Stock Portfolio Analyst

An AI-powered Streamlit app that turns a CSV of stock transactions into a full portfolio
analysis — current holdings, performance against the market, risk, tax-lot structure — and
lets you interrogate it in plain English.

The portfolio is reconstructed **entirely from the uploaded transaction history**. There is
no manual stock entry anywhere in the app.

```bash
uv sync
cp .env.example .env      # add your Groq key
uv run streamlit run app.py
```

Then open http://localhost:8501 and click **Load sample portfolio** to see it working
immediately, or upload your own CSV.

---

## The four tabs

| Tab | What it answers |
|---|---|
| **📄 Data Upload** | Is my file valid? Every problem is reported with the CSV row number that caused it. |
| **📊 Consolidated Portfolio View** | What do I own right now, what is it worth, and where is my risk concentrated? |
| **📈 Historical Performance** | Was any of this worth it — against the money I put in, against the index, against the risk I took? |
| **🤖 AI Analyst Chat** | Anything else, asked conversationally and answered from computed numbers. |

## What it computes

**FIFO cost basis.** A sale consumes the oldest shares first, matching the US default
method. Individual lots are tracked rather than a running average, which is what makes
realised gains correct and lets each open position be split into short- and long-term
holding periods. Overselling a position is reported as an error on that row and skipped,
so one bad line cannot invalidate the rest of the file.

**XIRR.** Money-weighted annualised return over irregularly spaced cashflows, solved by
Brent's method with a bisection fallback that cannot diverge. Unlike a simple percentage
gain, it accounts for *when* each dollar was invested.

**Benchmark shadow portfolio.** The headline comparison. Rather than quoting "the S&P is up
X% since 2023", the app replays your *exact* cashflows into SPY on the *same dates* — each
buy purchases index units, each sell redeems them. That is the only honest answer to "would
I have done better just buying the index?", and it is the number most portfolio trackers
get wrong.

**Time-weighted returns and risk.** Daily returns with the effect of deposits and
withdrawals stripped out, so paying money in doesn't masquerade as a gain. Annualised
volatility, Sharpe, and beta are derived from these. Maximum drawdown is measured on a
contribution-free growth index for the same reason — otherwise a large deposit during a
crash would look like a recovery and understate the fall.

**Concentration.** Allocation is checked stock-by-stock *and* by sector, because a
portfolio can look well spread across a dozen names while sitting almost entirely in one
sector — those names fall together. A Herfindahl index reports the effective number of
equally-weighted positions you actually hold.

**Tax lots.** Each open lot is classified short- or long-term, with the date it flips and
the gain that would be re-rated when it does.

**Dividends.** Credited from the shares actually held on each ex-dividend date, so a stock
bought last month is not retroactively credited with years of payouts.

**Split detection.** yfinance quotes post-split prices while your CSV records what you
actually paid. A position held *through* a split would otherwise show a catastrophic fake
loss, so the app detects and warns about exactly that case — and stays quiet when the split
predates your first purchase.

**Counterfactual.** What every share ever bought would be worth today had you never sold,
compared against what you actually hold plus what you took out.

## The AI analyst

The model is given **callable tools**, not a prompt stuffed with the whole portfolio. It
decides what data it needs and fetches it, so answers are grounded in computed numbers
rather than recalled ones. Each reply lists the data it used.

Ten tools are exposed: portfolio summary, holdings, sector breakdown, day attribution,
benchmark comparison, risk metrics, tax lots, realised trades, raw transactions, and price
history.

It is constrained to say "I don't have news access" rather than inventing a reason a stock
moved, and it gives no buy/sell recommendations — only risks that follow arithmetically
from the data.

## Input format

```csv
ticker,date,transaction_type,quantity,price
AAPL,2023-02-14,BUY,30,153.20
MSFT,2023-03-21,Buy,15,273.78
AAPL,2025-01-15,sell,25,237.87
```

Exactly these five columns, in any order. `transaction_type` is case-insensitive, dates
parse from any common format, and `$`/`,` are stripped from numbers. Extra columns are
ignored with a warning.

## Configuration

All optional except the API key — see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required only for the AI tab; every other tab works without it. |
| `GROQ_MODEL` | `qwen/qwen3.6-27b` | Must support tool calling well — see below. |
| `BENCHMARK_TICKER` | `SPY` | The "just buy the index" baseline. |
| `RISK_FREE_RATE` | `0.045` | Used for the Sharpe ratio. |

### Choosing a model

The agent lives or dies on tool calling, and the models differ more than their
benchmarks suggest. Tested against this portfolio:

| Model | Verdict |
|---|---|
| `qwen/qwen3.6-27b` | **Default.** Grounded, well formatted, no failures in testing. |
| `openai/gpt-oss-120b` | Equally reliable, spends the token budget faster. |
| `llama-3.3-70b-versatile` | Works, heaviest on tokens. Occasionally prints tool-call syntax as prose (handled). |
| `openai/gpt-oss-20b` | **Avoid** — intermittently returns unparseable output. |
| `llama-3.1-8b-instant` | **Avoid** — ignores the "no recommendations" rule and gives investment advice. |

### If the AI tab hits a rate limit

Groq's free tier caps tokens **per day, per model**. On a 429 the app says which
limit was hit and how long it lasts: a per-minute burst clears itself in seconds,
while an exhausted daily budget needs a different `GROQ_MODEL` — switching gives
you a fresh allowance. Tool payloads are serialised as CSV rather than JSON
records, which cuts 38–58% off the tokens a wide table costs.

## Deploying

The app is deployable to [Streamlit Community Cloud](https://share.streamlit.io) for free
from this repo.

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `gpallavi704/stock-analyst-agent`, branch `main`, main file `app.py`.
4. Under **Advanced settings**, set Python to **3.12** and paste into **Secrets**:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   GROQ_MODEL = "qwen/qwen3.6-27b"
   BENCHMARK_TICKER = "SPY"
   RISK_FREE_RATE = "0.045"
   ```

5. **Deploy.** First build takes a few minutes.

Two things this repo does to make that work:

- **`requirements.txt` is committed.** Community Cloud does not read `uv.lock`, and it
  assumes any `pyproject.toml` is Poetry format — this one is PEP 621. The file is
  generated from the lockfile, so the deployed versions match local exactly:

  ```bash
  uv export --format requirements-txt --no-hashes --no-emit-project --no-dev -o requirements.txt
  ```

  Re-run it whenever dependencies change.

- **Secrets are bridged into the environment** in `app.py`, since there is no `.env` file
  on the host.

**Caveat worth knowing:** Yahoo Finance rate-limits datacenter IPs more aggressively than
home connections, so a hosted instance may see slower or occasionally failed price fetches
where local runs are fine. The app degrades honestly — missing prices are flagged, not
silently dropped — and **Refresh market data** retries. If it proves a problem, running
locally is unaffected.

## Layout

```
app.py                      Entry point: page config, sidebar, the four tabs
components/                 One module per tab, each exposing render()
utils/
  data_processing.py        CSV validation and normalisation
  market_data.py            All yfinance access, wrapped in caches with retries
  portfolio_math.py         FIFO lots, holdings valuation, XIRR, tax lots
  analytics.py              Benchmark shadow, risk, sectors, dividends, counterfactuals
  llm_agent.py              Groq agent with tool calling
  pipeline.py               Runs everything once, hands every tab the same result
data/sample_transactions.csv
```

`pipeline.build_analysis()` is the spine: the full analysis is computed once per rerun and
cached, then passed to every tab. A number on a chart, a number in a metric card, and a
number in the AI's answer therefore cannot disagree with each other.

## Notes and limitations

- Prices come from Yahoo Finance and may be delayed. Quotes cache for 5 minutes, price
  history for an hour; **Refresh market data** in the sidebar clears both.
- A ticker whose price cannot be fetched is still listed but excluded from totals, and
  flagged — never silently dropped.
- US equities only. No support for options, bonds, FX, or multi-currency.
- Cash is not modelled: the app tracks positions, not an account balance.
- Informational analysis of your own data. Not financial advice.

## Built with

Streamlit · yfinance · Groq · pandas · numpy · Plotly · scipy · uv
