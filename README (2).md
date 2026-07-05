# 📐 Nifty 50 Confluence Strategy — Signal Engine & Backtest

A free, no-API-key Streamlit app that adapts a TradingView Pine Script v5
strategy — **"Nifty Confluence: BOS + Retest + 9EMA"** (1H bias/zone, 15m
break-of-structure, 5m entry) — into Python, fetches Nifty 50 (`^NSEI`) data
from Yahoo Finance, generates buy/sell signals from the indicator logic,
picks an in-the-money option strike for each signal, and backtests the
strategy.

> ⚠️ **Educational tool only. Not investment advice.** Options premiums are
> **simulated** with a Black-Scholes model — this app does not use a live
> options-chain data feed (none is available for free/historically for NSE
> index options).

---

## Why two tabs? (please read this first)

Yahoo Finance's free intraday history is limited:

| Interval | Max history via `yfinance` |
|----------|------------------------------|
| 5-minute / 15-minute | ~60 days |
| 60-minute (1H) | ~730 days (~2 years) |
| Daily | Full history (decades) |

The original strategy needs **1H + 15m + 5m data at the same time**. That
combination simply doesn't exist going back 3 years for free — Yahoo doesn't
retain that intraday history. So this app gives you two honest, distinct
things instead of one dishonest one:

- **🔴 Live / Recent Signal tab** — uses *real* 1H, 15m and 5m data (as far
  back as Yahoo allows) to run the exact multi-timeframe logic: 1H bias +
  supply/demand zone, 15m BOS, 5m 9-EMA entry/retest. Shows the current
  signal state and a genuine recent (~60-day) trade log.
- **📊 3-Year Backtest tab** — runs the **same rules adapted to daily bars**
  (a slower pivot lookback stands in for the "1H" role, a faster pivot
  lookback stands in for the "15m" role, and the 9 EMA/retest trigger runs
  on daily closes) so a genuine multi-year backtest is possible. It is the
  same logic, a coarser timeframe — not literally the same trade-by-trade
  signals you'd see intraday.

---

## How the option leg is simulated

Since no free historical NSE options-chain data exists:

1. **Strike selection** — for every signal, the nearest at-the-money strike
   (rounded to the nearest 50-point step by default) is shifted a
   configurable number of steps into-the-money: **Call ITM strikes sit
   below** the spot price, **Put ITM strikes sit above** it.
2. **Premium** — priced with the **Black-Scholes model** using:
   - trailing **realized volatility** of the index (rolling daily-return
     std, annualized) as the volatility input,
   - a flat, user-set **risk-free rate**,
   - a user-set **assumed number of days to expiry** at entry, decaying as
     the trade is held.
3. **P&L** — always modeled as **buying** the option (call for a long/bias
   signal, put for a short/bias signal) for the configured number of
   lots — `PnL = (exit premium − entry premium) × lot size × lots`.

This ignores bid/ask spread, liquidity, slippage, and the volatility
smile/skew real options exhibit. Treat results as **directionally
indicative**, not a precise P&L estimate.

**Lot size**: NSE periodically revises index F&O lot sizes by circular (for
example, Nifty 50 moved from 75 to 65 in a January 2026 revision). The lot
size in the sidebar defaults to a recent value but **you should verify the
current lot size on the NSE website before relying on the numbers.**

### Live NSE option chain (Live tab only)

The **Live / Recent Signal** tab additionally calls NSE India's public
option-chain endpoint (`nseindia.com/api/option-chain-indices`) to show the
**real current premium and IV** for the suggested strike, alongside the
Black-Scholes estimate, so you can compare the two.

- This is an *unofficial*, undocumented NSE endpoint. NSE actively blocks
  non-browser and many datacenter/cloud IPs — **it may well fail when
  deployed on Streamlit Cloud**, even though it can work fine from a home
  connection. The app detects this and falls back to showing only the
  Black-Scholes estimate with a warning, rather than crashing.
- It only ever returns the **current** live chain — NSE has no free API for
  *historical* strike-level option prices, which is why the 3-year backtest
  still uses the simulated Black-Scholes premiums described above.

---

## Run locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free — GitHub + share.streamlit.io

1. Push `app.py`, `engine.py`, `requirements.txt`, and this `README.md` to a
   new GitHub repository.
2. Go to **https://share.streamlit.io**, sign in with GitHub, click
   **"New app"**.
3. Pick your repo, branch `main`, main file path `app.py`, click **Deploy**.
4. You'll get a public `https://<your-app>.streamlit.app` URL. Every push to
   the connected branch auto-redeploys.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI: data fetch, sidebar parameters, both tabs |
| `engine.py` | Data-agnostic strategy engine: pivot detection, bias/zone/BOS state, the trade state machine, Black-Scholes option pricing, performance stats. Fully unit-tested with synthetic data (no network needed) |
| `requirements.txt` | Python dependencies |

## Strategy parameters (sidebar)

| Parameter | Pine equivalent |
|-----------|-----------------|
| Bias/Zone pivot lookback | `swingLB` on the 1H timeframe |
| BOS/SL pivot lookback | `swingLB` on the 15m timeframe |
| Zone tolerance % | `zonetol` |
| EMA length | `emaLen` |
| SL buffer (points) | `slBuffer` |
| Risk:Reward ratio | `rrRatio` |
| Max trades per day | `maxTrades` |
| Max holding period (bars) | new — forces a time-based exit if neither SL nor Target is hit (mimics weekly-option expiry risk, not present in the original intraday-only script) |
| Lot size / Lots per trade / ITM depth / Strike step / Days to expiry / Risk-free rate / Vol lookback | option-leg simulation inputs, not present in the original Pine script (which trades the index directly) |

## Limitations / disclaimers

- Yahoo Finance (`yfinance`) is an unofficial, free data source and can
  occasionally rate-limit or return incomplete data — retry if a fetch fails.
- The daily-bar adaptation is a genuine reinterpretation of the original
  intraday logic, not an identical replica — a 3-year *intraday* backtest of
  the exact 5m/15m/1H rules is not possible with free data.
- Option premiums are a Black-Scholes simulation, not real market prices.
- Backtested (simulated or real) performance never guarantees future results.
- This is an educational project, not financial advice.
