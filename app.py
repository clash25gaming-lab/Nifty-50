"""
Nifty 50 Confluence Strategy — Live Signal + 3-Year Backtest
=============================================================
Adapts a TradingView Pine Script v5 strategy ("Nifty Confluence: BOS + Retest
+ 9EMA" — 1H bias/zones, 15m BOS, 5m entry) into a Python/Streamlit app that:

  1. Pulls Nifty 50 (^NSEI) price data from Yahoo Finance
  2. Reproduces the indicator's logic (9 EMA, pivot-based bias, supply/demand
     zones, break-of-structure, retest entries) in `engine.py`
  3. Selects an in-the-money option strike for every signal and simulates
     the option leg with a Black-Scholes premium model (2 lots per trade)
  4. Backtests the strategy over the last 3 years
  5. Also shows a "live" multi-timeframe snapshot (1H/15m/5m) for recent
     signals, within Yahoo Finance's intraday history limits

IMPORTANT — please read the "Methodology & Limitations" panel in the app.
Yahoo Finance only retains 5m/15m intraday history for ~60 days and 60m
(1H) history for ~730 days, so a genuine 3-year backtest of the *exact*
5m/15m/1H strategy is not possible with free data. The 3-year backtest
therefore runs a faithful **daily-bar adaptation** of the same confluence
logic (see the panel for details); the live tab uses the real intraday
multi-timeframe data for as far back as Yahoo allows.

Educational tool only. Not investment advice. Options premiums are modeled,
not sourced from a live options chain.
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from engine import (
    build_structure_state, build_bos_state, add_ema_and_touch,
    run_confluence_backtest, attach_option_pnl, compute_performance, get_itm_strike,
)

TICKER = "^NSEI"

st.set_page_config(page_title="Nifty 50 Confluence Strategy", page_icon="📐", layout="wide")

# --------------------------------------------------------------------------------------
# NSE INDIA — LIVE OPTION CHAIN (real premiums / IV where reachable)
# --------------------------------------------------------------------------------------
# NSE's option-chain JSON API is unofficial and NOT documented/supported by NSE. It also
# does not provide historical strike-level prices - only the CURRENT live chain. It's used
# here only in the Live tab, purely to show a real premium alongside the simulated one.
# NSE aggressively blocks non-browser / datacenter traffic on this endpoint, so failures
# here are common and expected on some hosts (including, likely, Streamlit Cloud) — the
# app must always degrade gracefully to the Black-Scholes estimate when this fails.

_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/option-chain",
}


@st.cache_data(ttl=60 * 3, show_spinner=False)
def fetch_nse_option_chain(symbol: str = "NIFTY"):
    """
    Fetches the LIVE option chain (nearest expiry) from NSE's public JSON API.
    Returns {'spot': float, 'nearest_expiry': str, 'expiries': list, 'rows': DataFrame}
    or None on any failure (network block, non-200, unexpected JSON shape, etc).
    Never raises.
    """
    try:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        session.get("https://www.nseindia.com", timeout=6)
        session.get("https://www.nseindia.com/option-chain", timeout=6)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        resp = session.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        data = resp.json()

        records = data.get("records", {})
        spot = records.get("underlyingValue")
        expiries = records.get("expiryDates", [])
        if not expiries:
            return None
        nearest_expiry = expiries[0]

        rows = []
        for item in records.get("data", []):
            if item.get("expiryDate") != nearest_expiry:
                continue
            ce, pe = item.get("CE", {}), item.get("PE", {})
            rows.append({
                "strike": item.get("strikePrice"),
                "CE_LTP": ce.get("lastPrice"), "CE_IV": ce.get("impliedVolatility"), "CE_OI": ce.get("openInterest"),
                "PE_LTP": pe.get("lastPrice"), "PE_IV": pe.get("impliedVolatility"), "PE_OI": pe.get("openInterest"),
            })
        if not rows:
            return None
        df_rows = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        return {"spot": spot, "expiries": expiries, "nearest_expiry": nearest_expiry, "rows": df_rows}
    except Exception:
        return None


def lookup_nse_premium(chain: dict, strike: float, option_type: str):
    """Returns (ltp, iv_pct) for the closest listed strike matching `strike`, or (None, None)."""
    if not chain or chain.get("rows") is None or chain["rows"].empty:
        return None, None
    rows = chain["rows"]
    idx = (rows["strike"] - strike).abs().idxmin()
    row = rows.loc[idx]
    if abs(row["strike"] - strike) > 1e-6:
        return None, None
    if option_type == "call":
        return row.get("CE_LTP"), row.get("CE_IV")
    return row.get("PE_LTP"), row.get("PE_IV")

# --------------------------------------------------------------------------------------
# DATA FETCH  (cached, never raises)
# --------------------------------------------------------------------------------------

@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_history(interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(TICKER).history(interval=interval, period=period, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close"]].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    return df.sort_index()


def build_vol_lookup(daily_close: pd.Series, window: int):
    """Returns a function timestamp -> annualized realized volatility."""
    daily_ret = daily_close.pct_change()
    rolling_vol = daily_ret.rolling(window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    fallback = rolling_vol.mean() if len(rolling_vol) else 0.13

    def lookup(ts):
        if ts.tzinfo is not None:
            ts_naive_date = ts.tz_localize(None).normalize()
        else:
            ts_naive_date = ts.normalize()
        idx = rolling_vol.index
        idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
        pos = idx_naive.searchsorted(ts_naive_date, side="right") - 1
        if pos < 0:
            return fallback
        val = rolling_vol.iloc[pos]
        return val if not np.isnan(val) else fallback

    return lookup


# --------------------------------------------------------------------------------------
# STRATEGY PARAMETERS  (sidebar - shared by both tabs)
# --------------------------------------------------------------------------------------

st.title("📐 Nifty 50 Confluence Strategy — Signal Engine & Backtest")
st.caption(
    "Adapted from a TradingView Pine Script (9 EMA + Pivot Structure, BOS + Retest). "
    "Data: Yahoo Finance (`yfinance`), free & no API key. Option premiums are "
    "**simulated** via Black-Scholes — see Methodology below."
)

with st.expander("📖 Methodology & Limitations — please read before trusting the numbers", expanded=False):
    st.markdown("""
**Why two tabs?** Yahoo Finance's free intraday history is limited:
- 5-minute / 15-minute bars → last **~60 days** only
- 60-minute (1H) bars → last **~730 days** (~2 years)
- Daily bars → full history (decades)

The original indicator needs 1H + 15m + 5m data simultaneously, so a *literal*
3-year backtest of it is impossible with free data — Yahoo simply doesn't retain
that intraday history. To meet your two goals (faithful signal **and** a genuine
3-year backtest), this app does both, honestly:

- **🔴 Live / Recent Signal tab** — uses real 1H, 15m and 5m data (as far back
  as Yahoo allows, ~60 days) to reproduce the *exact* multi-timeframe logic:
  1H bias + supply/demand zone, 15m break-of-structure, 5m 9-EMA entry/retest.
  This gives you today's live signal state and a genuine recent trade log.
- **📊 3-Year Backtest tab** — since 3 years of intraday data doesn't exist for
  free, this tab runs the **same confluence logic adapted to daily bars**:
  a slower pivot lookback stands in for the "1H bias/zone" role, a faster
  pivot lookback stands in for the "15m BOS/SL" role, and the 9 EMA / retest
  trigger runs on daily closes. It is the same rules, a coarser timeframe —
  not the same trade-by-trade signals you'd get intraday.

**Options are simulated, not sourced from a live chain.** NSE index options
historical tick data isn't available for free. Each signal picks an
in-the-money strike (nearest 50-point step, offset by your chosen ITM depth)
and prices it with the **Black-Scholes model**, using trailing realized
volatility of the index as the volatility input, a flat assumed risk-free
rate, and an assumed number of calendar days to expiry. This ignores bid/ask
spreads, liquidity, slippage, and the vol smile/skew real options exhibit —
treat the P&L as directional, not exact.

**This is an educational backtest, not investment advice.** Past performance
(simulated or real) does not guarantee future results.
""")

with st.sidebar:
    st.header("⚙️ Strategy Parameters")

    st.subheader("Data source")
    nse_symbol = st.text_input("NSE option-chain symbol (live premiums)", value="NIFTY",
                                help="Used only in the Live tab to fetch real current option "
                                     "premiums/IV from NSE's public site. NIFTY, BANKNIFTY, FINNIFTY, etc.")

    st.subheader("Structure")
    slow_lb = st.slider("Bias/Zone pivot lookback (slow — '1H' role)", 5, 30, 10)
    fast_lb = st.slider("BOS/SL pivot lookback (fast — '15m' role)", 3, 20, 5)
    zone_tol_pct = st.slider("Zone tolerance (%)", 0.05, 2.0, 0.3, 0.05) / 100
    ema_len = st.slider("EMA length", 3, 50, 9)

    st.subheader("Risk")
    sl_buffer = st.number_input("SL buffer (index points)", 1.0, 200.0, 15.0, 1.0)
    rr_ratio = st.slider("Risk:Reward ratio", 1.0, 5.0, 2.0, 0.1)
    max_trades_per_day = st.slider("Max trades per day", 1, 10, 2)
    max_holding_bars = st.slider("Max holding period (bars)", 1, 20, 7,
                                  help="Force-close a trade after this many bars if neither "
                                       "SL nor Target is hit (mimics weekly-option expiry risk).")

    st.subheader("Option Leg")
    lot_size = st.number_input("Lot size", 1, 500, 65,
                                help="Nifty 50 lot size changes periodically by NSE circular — "
                                     "verify the current lot size before relying on this.")
    n_lots = st.number_input("Lots per trade", 1, 50, 2)
    itm_depth = st.slider("ITM depth (strikes into the money)", 1, 5, 1)
    strike_step = st.number_input("Strike step (points)", 10, 500, 50, 10)
    days_to_expiry = st.slider("Assumed days to expiry at entry", 1, 30, 7)
    risk_free_rate = st.slider("Risk-free rate (%)", 0.0, 12.0, 7.0, 0.25) / 100
    vol_window = st.slider("Realized volatility lookback (days)", 5, 60, 20)
    initial_capital = st.number_input("Reference capital (₹)", 10000, 100_000_000, 500_000, 10_000)

shared_params = dict(
    sl_buffer=sl_buffer, rr_ratio=rr_ratio, max_trades_per_day=max_trades_per_day,
    max_holding_bars=max_holding_bars,
)
option_params = dict(
    strike_step=strike_step, itm_depth=itm_depth, risk_free_rate=risk_free_rate,
    days_to_expiry=days_to_expiry, lot_size=lot_size, n_lots=n_lots,
)

tab_live, tab_backtest = st.tabs(["🔴 Live / Recent Signal (Intraday)", "📊 3-Year Backtest (Daily-Adapted)"])

# ========================================================================================
# TAB 1 — LIVE / RECENT MULTI-TIMEFRAME SIGNAL
# ========================================================================================
with tab_live:
    st.markdown("Uses real 1H (bias/zone), 15m (BOS/SL) and 5m (EMA entry/retest) data — "
                 "the maximum history Yahoo Finance provides for these intervals.")

    with st.spinner("Fetching 1H / 15m / 5m Nifty 50 data from Yahoo Finance..."):
        df_1h = fetch_history("60m", "730d")
        df_15m = fetch_history("15m", "60d")
        df_5m = fetch_history("5m", "60d")
        df_daily_for_vol = fetch_history("1d", "2y")

    if df_1h.empty or df_15m.empty or df_5m.empty:
        st.error(
            "Could not fetch intraday data for ^NSEI from Yahoo Finance right now "
            "(this can happen from rate limiting, or outside market hours on some hosts). "
            "Try again in a minute, or check the 3-Year Backtest tab which uses daily data."
        )
    else:
        structure_1h = build_structure_state(df_1h, slow_lb, slow_lb, zone_tol_pct)
        bos_15m = build_bos_state(df_15m, fast_lb, fast_lb)
        ema_5m = add_ema_and_touch(df_5m, ema_len)

        # Align the slower timeframes onto the 5m grid with an as-of (backward) merge —
        # i.e. at each 5m bar we only ever see 1H/15m information confirmed *before* it.
        base = df_5m.join(ema_5m)[["Open", "High", "Low", "Close", "EMA", "EmaTouchLong", "EmaTouchShort"]]
        base = base.reset_index()
        base = base.rename(columns={base.columns[0]: "Datetime"}).sort_values("Datetime")

        struct_reset = structure_1h.reset_index()
        struct_reset = struct_reset.rename(columns={struct_reset.columns[0]: "Datetime"}).sort_values("Datetime")

        bos_reset = bos_15m.reset_index()
        bos_reset = bos_reset.rename(columns={bos_reset.columns[0]: "Datetime"}).sort_values("Datetime")

        merged = pd.merge_asof(base, struct_reset, on="Datetime", direction="backward")
        merged = pd.merge_asof(merged, bos_reset, on="Datetime", direction="backward")
        merged = merged.set_index("Datetime")

        bool_cols = ["BullishBias", "BearishBias", "InDemandZone", "InSupplyZone",
                     "BullishBOS", "BearishBOS", "EmaTouchLong", "EmaTouchShort"]
        for c in bool_cols:
            merged[c] = merged[c].fillna(False).astype(bool)

        merged["DayId"] = merged.index.normalize()
        hhmm = merged.index.hour * 100 + merged.index.minute
        merged["InWindow"] = (hhmm >= 930) & (hhmm <= 1500)
        merged = merged.dropna(subset=["Close", "EMA"])

        # ---- current live status panel ----
        st.markdown("### 📟 Current Status (as of the latest available 5-minute bar)")
        last = merged.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        bias_txt = "BULLISH 📈" if last["BullishBias"] else ("BEARISH 📉" if last["BearishBias"] else "NEUTRAL ⏳")
        c1.metric("1H Bias", bias_txt)
        c2.metric("9 EMA (5m)", "Above ✅" if last["Close"] > last["EMA"] else "Below ❌")
        c3.metric("Demand Zone", "Inside 🟢" if last["InDemandZone"] else "Outside")
        c4.metric("Supply Zone", "Inside 🔴" if last["InSupplyZone"] else "Outside")
        st.caption(f"Last bar: {merged.index[-1]} IST · Nifty 50 close: {last['Close']:.2f}")

        # ---- run the same engine on this merged intraday frame ----
        trades_raw = run_confluence_backtest(merged, shared_params)

        if not trades_raw:
            st.info("No completed trade signals in the available intraday window "
                     "(last ~60 days). This is expected — the confluence rules are strict "
                     "by design (all of bias + zone + BOS + EMA must align).")
        else:
            idx = merged.index
            tdf = pd.DataFrame(trades_raw)
            tdf["entry_time"] = tdf["entry_idx"].map(lambda i: idx[i])
            tdf["exit_time"] = tdf["exit_idx"].map(lambda i: idx[i])

            daily_vol_lookup = build_vol_lookup(df_daily_for_vol["Close"], vol_window) if not df_daily_for_vol.empty \
                else (lambda ts: 0.13)
            opt_trades = attach_option_pnl(tdf, daily_vol_lookup, option_params)

            # ---- Live NSE option chain for the most recent signal ----
            st.markdown("### 🔗 Live NSE Option Chain — most recent signal")
            last_sig = opt_trades.iloc[-1]
            with st.spinner("Fetching live option chain from NSE India..."):
                nse_chain = fetch_nse_option_chain(nse_symbol)

            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Direction", last_sig["direction"].upper())
            lc2.metric("Suggested ITM Strike", f"{last_sig['Strike']:.0f} {last_sig['OptionType']}")
            lc3.metric("Simulated Premium (Black-Scholes)", f"₹{last_sig['EntryPremium']:.2f}")

            if nse_chain:
                option_type = "call" if last_sig["direction"] == "long" else "put"
                ltp, iv = lookup_nse_premium(nse_chain, last_sig["Strike"], option_type)
                lc4.metric("Live NSE Premium", f"₹{ltp:.2f}" if ltp is not None else "Not listed",
                           f"IV {iv:.1f}%" if iv else None)
                st.success(f"✅ Live NSE chain fetched — spot {nse_chain['spot']}, "
                           f"nearest expiry {nse_chain['nearest_expiry']}.")
                with st.expander("View live NSE option chain (nearest expiry, around ATM)"):
                    atm = get_itm_strike(nse_chain["spot"], "call", strike_step, 0)
                    band = strike_step * 8
                    rows = nse_chain["rows"]
                    view = rows[(rows["strike"] >= atm - band) & (rows["strike"] <= atm + band)]
                    st.dataframe(view, use_container_width=True, hide_index=True)
            else:
                lc4.metric("Live NSE Premium", "Unavailable")
                st.warning(
                    "⚠️ Could not reach NSE's option-chain API right now — NSE frequently blocks "
                    "automated/cloud requests (this is common on hosted apps, including Streamlit "
                    "Cloud), or markets may be closed. Showing the Black-Scholes simulated premium "
                    "above instead."
                )

            st.markdown(f"### 📋 Recent Signal Log — last ~60 days ({len(opt_trades)} trades)")
            show_cols = ["entry_time", "direction", "OptionType", "Strike", "entry_price",
                         "EntryPremium", "exit_time", "exit_price", "ExitPremium",
                         "exit_reason", "entry_type", "Quantity", "PnL"]
            st.dataframe(opt_trades[show_cols].round(2), use_container_width=True, hide_index=True)

            perf = compute_performance(opt_trades, initial_capital)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Trades", perf["total_trades"])
            p2.metric("Win Rate", f"{perf['win_rate']:.1f}%")
            p3.metric("Net P&L", f"₹{perf['net_pnl']:,.0f}")
            p4.metric("Max Drawdown", f"₹{perf['max_drawdown']:,.0f}")
            st.caption("This is a short recent-history sample (~60 days), not the 3-year backtest "
                       "— see the next tab for that.")

        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=merged.index[-500:], y=merged["Close"].iloc[-500:], name="Close"))
        price_fig.add_trace(go.Scatter(x=merged.index[-500:], y=merged["EMA"].iloc[-500:], name=f"EMA {ema_len}"))
        price_fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                                 title="Nifty 50 — 5-minute close & EMA (last 500 bars)")
        st.plotly_chart(price_fig, use_container_width=True)

# ========================================================================================
# TAB 2 — 3-YEAR BACKTEST (DAILY-ADAPTED)
# ========================================================================================
with tab_backtest:
    st.markdown("Same confluence rules, adapted to **daily bars** so a genuine multi-year "
                 "backtest is possible with free Yahoo Finance data.")

    years_back = st.slider("Backtest lookback (years)", 1, 10, 3, key="years_back")

    with st.spinner("Fetching daily Nifty 50 history from Yahoo Finance..."):
        df_daily = fetch_history("1d", f"{years_back + 1}y")  # +1y warm-up for pivots/EMA

    if df_daily.empty:
        st.error("Could not fetch daily data for ^NSEI from Yahoo Finance right now. Try again shortly.")
    else:
        structure_d = build_structure_state(df_daily, slow_lb, slow_lb, zone_tol_pct)
        bos_d = build_bos_state(df_daily, fast_lb, fast_lb)
        ema_d = add_ema_and_touch(df_daily, ema_len)

        merged_d = df_daily.join(structure_d).join(bos_d).join(ema_d)
        merged_d["DayId"] = merged_d.index  # every bar is its own "day"
        merged_d["InWindow"] = True

        cutoff = merged_d.index.max() - pd.DateOffset(years=years_back)
        backtest_window = merged_d[merged_d.index >= cutoff].copy()

        trades_raw = run_confluence_backtest(merged_d, shared_params)

        idx = merged_d.index
        tdf_all = pd.DataFrame(trades_raw)
        if not tdf_all.empty:
            tdf_all["entry_time"] = tdf_all["entry_idx"].map(lambda i: idx[i])
            tdf_all["exit_time"] = tdf_all["exit_idx"].map(lambda i: idx[i])
            # keep only trades whose ENTRY falls inside the requested backtest window
            tdf = tdf_all[tdf_all["entry_time"] >= cutoff].reset_index(drop=True)
        else:
            tdf = tdf_all

        if tdf.empty:
            st.warning(
                "No trades were generated in this window with the current parameter settings. "
                "The confluence rules require bias + zone + BOS + EMA to all align — try loosening "
                "the zone tolerance, SL buffer, or RR ratio in the sidebar, or extend the lookback."
            )
        else:
            daily_vol_lookup = build_vol_lookup(df_daily["Close"], vol_window)
            opt_trades = attach_option_pnl(tdf, daily_vol_lookup, option_params)
            perf = compute_performance(opt_trades, initial_capital)

            st.markdown(f"### 📈 Backtest Results — last {years_back} year(s)")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Trades", perf["total_trades"])
            m2.metric("Win Rate", f"{perf['win_rate']:.1f}%")
            m3.metric("Net P&L", f"₹{perf['net_pnl']:,.0f}")
            m4.metric("Profit Factor", f"{perf['profit_factor']:.2f}" if np.isfinite(perf["profit_factor"]) else "∞")
            m5.metric("Max Drawdown", f"₹{perf['max_drawdown']:,.0f}")

            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=opt_trades["exit_time"], y=perf["equity_curve"], mode="lines+markers",
                name="Equity", line=dict(color="#1f77b4"),
            ))
            eq_fig.add_hline(y=initial_capital, line_dash="dash", line_color="grey",
                              annotation_text="Starting reference capital")
            eq_fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                                  title="Simulated Equity Curve (2 lots per trade, option premium P&L)")
            st.plotly_chart(eq_fig, use_container_width=True)

            win_loss_fig = go.Figure()
            colors = ["#2ca02c" if p > 0 else "#d62728" for p in opt_trades["PnL"]]
            win_loss_fig.add_trace(go.Bar(x=opt_trades["entry_time"], y=opt_trades["PnL"], marker_color=colors))
            win_loss_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                                        title="Per-Trade P&L (₹)")
            st.plotly_chart(win_loss_fig, use_container_width=True)

            st.markdown("### 📋 Trade Log")
            show_cols = ["entry_time", "direction", "OptionType", "Strike", "entry_price",
                         "EntryPremium", "ImpliedVolUsed", "exit_time", "exit_price",
                         "ExitPremium", "exit_reason", "entry_type", "Quantity", "PnL"]
            display_df = opt_trades[show_cols].round(3)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download trade log as CSV", csv_bytes,
                                file_name=f"nifty_confluence_backtest_{years_back}y.csv", mime="text/csv")

            with st.expander("More performance detail"):
                d1, d2, d3 = st.columns(3)
                d1.metric("Wins / Losses", f"{perf['wins']} / {perf['losses']}")
                d2.metric("Avg Win", f"₹{perf['avg_win']:,.0f}")
                d3.metric("Avg Loss", f"₹{perf['avg_loss']:,.0f}")
                st.metric("Final Simulated Equity", f"₹{perf['final_equity']:,.0f}")

        price_fig_d = go.Figure()
        price_fig_d.add_trace(go.Scatter(x=merged_d.index, y=merged_d["Close"], name="Nifty 50 Close"))
        price_fig_d.add_trace(go.Scatter(x=merged_d.index, y=merged_d["EMA"], name=f"EMA {ema_len}"))
        if not tdf.empty:
            longs = tdf[tdf["direction"] == "long"]
            shorts = tdf[tdf["direction"] == "short"]
            if not longs.empty:
                price_fig_d.add_trace(go.Scatter(
                    x=longs["entry_time"], y=longs["entry_price"], mode="markers", name="Long Entry (CE)",
                    marker=dict(symbol="triangle-up", size=11, color="green"),
                ))
            if not shorts.empty:
                price_fig_d.add_trace(go.Scatter(
                    x=shorts["entry_time"], y=shorts["entry_price"], mode="markers", name="Short Entry (PE)",
                    marker=dict(symbol="triangle-down", size=11, color="red"),
                ))
        price_fig_d.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                                   title="Nifty 50 Daily Close with Trade Entries")
        st.plotly_chart(price_fig_d, use_container_width=True)

st.markdown("---")
st.caption(
    "Data: Yahoo Finance via `yfinance` (free, unofficial). Strategy logic adapted from a "
    "user-supplied TradingView Pine Script. Option premiums are simulated with Black-Scholes — "
    "not sourced from a live NSE options chain. Educational tool only, not investment advice."
)
