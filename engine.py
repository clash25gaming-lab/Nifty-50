"""
engine.py
=========
Core, UI-agnostic logic for the Nifty 50 "Confluence: BOS + Retest + 9EMA"
strategy adapted from a TradingView Pine Script v5 strategy.

This module is deliberately separate from app.py so it can be unit-tested
without Streamlit or network access.

Pine Script -> Python mapping
-----------------------------
- ta.pivothigh(src, L, R) / ta.pivotlow(src, L, R)  -> detect_pivots()
- 1H bias (HH/HL, LH/LL) + Supply/Demand zones      -> build_structure_state()
- 15m BOS + SL reference                            -> build_bos_state()
- 9 EMA + retest touch                              -> add_ema_and_touch()
- Full state machine (BOS confirm / retest / entry) -> run_confluence_backtest()
- Option leg (ITM strike + Black-Scholes premium)    -> attach_option_pnl()
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================================
# 1) PIVOT DETECTION  (ta.pivothigh / ta.pivotlow equivalent, non-repainting)
# ============================================================================

def detect_pivots(high: pd.Series, low: pd.Series, left: int, right: int):
    """
    Replicates TradingView's ta.pivothigh/ta.pivotlow(src, left, right).

    A pivot high at bar i requires high[i] to be the strict maximum over the
    window [i-left, i+right]. It is only *known* (confirmed / non-repainting)
    `right` bars later, at bar i+right - exactly like Pine's lookahead_off.

    Returns four pandas Series aligned to the same index:
        confirmed_high_value : pivot HIGH value, placed at its confirmation bar
        confirmed_low_value  : pivot LOW value,  placed at its confirmation bar
        confirmed_high_barhigh / confirmed_high_barlow : the ORIGINAL pivot
            bar's high/low (used to build the Supply zone box)
        confirmed_low_barhigh / confirmed_low_barlow : the ORIGINAL pivot
            bar's high/low (used to build the Demand zone box)
    """
    n = len(high)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)

    raw_pivot_high = np.full(n, np.nan)
    raw_pivot_low = np.full(n, np.nan)

    for i in range(left, n - right):
        h_win = h[i - left:i + right + 1]
        if h[i] == h_win.max() and np.sum(h_win == h[i]) == 1:
            raw_pivot_high[i] = h[i]
        l_win = l[i - left:i + right + 1]
        if l[i] == l_win.min() and np.sum(l_win == l[i]) == 1:
            raw_pivot_low[i] = l[i]

    idx = high.index

    # The pivot VALUE, known `right` bars after it forms:
    confirmed_high_value = pd.Series(raw_pivot_high, index=idx).shift(right)
    confirmed_low_value = pd.Series(raw_pivot_low, index=idx).shift(right)

    # The original pivot bar's own high/low (for zone boxes), also shifted
    # forward to the confirmation bar so there is no lookahead:
    high_at_pivhigh = pd.Series(np.where(~np.isnan(raw_pivot_high), h, np.nan), index=idx).shift(right)
    low_at_pivhigh = pd.Series(np.where(~np.isnan(raw_pivot_high), l, np.nan), index=idx).shift(right)
    high_at_pivlow = pd.Series(np.where(~np.isnan(raw_pivot_low), h, np.nan), index=idx).shift(right)
    low_at_pivlow = pd.Series(np.where(~np.isnan(raw_pivot_low), l, np.nan), index=idx).shift(right)

    return {
        "confirmed_high_value": confirmed_high_value,
        "confirmed_low_value": confirmed_low_value,
        "high_at_pivhigh": high_at_pivhigh,
        "low_at_pivhigh": low_at_pivhigh,
        "high_at_pivlow": high_at_pivlow,
        "low_at_pivlow": low_at_pivlow,
    }


# ============================================================================
# 2) STRUCTURE STATE  (bias + supply/demand zones - the "1H" role)
# ============================================================================

def build_structure_state(df: pd.DataFrame, left: int, right: int, zone_tol: float) -> pd.DataFrame:
    """
    Replicates the 1H bias + Supply/Demand zone block of the Pine script,
    computed on whatever timeframe `df` represents (an actual 1H frame, or a
    daily frame used as a stand-in "slow" structure timeframe).

    df must have columns: High, Low, Close (and be sorted ascending by time).
    """
    piv = detect_pivots(df["High"], df["Low"], left, right)

    sh1 = piv["confirmed_high_value"].ffill()          # htfSH1
    sl1 = piv["confirmed_low_value"].ffill()           # htfSL1

    # second-most-recent confirmed pivot (htfSH2 / htfSL2)
    sh_events = piv["confirmed_high_value"].dropna()
    sh2_at_events = sh_events.shift(1)
    sh2 = sh2_at_events.reindex(df.index).ffill()

    sl_events = piv["confirmed_low_value"].dropna()
    sl2_at_events = sl_events.shift(1)
    sl2 = sl2_at_events.reindex(df.index).ffill()

    bullish_bias = sh1.notna() & sh2.notna() & sl1.notna() & sl2.notna() & (sh1 > sh2) & (sl1 > sl2)
    bearish_bias = sh1.notna() & sh2.notna() & sl1.notna() & sl2.notna() & (sh1 < sh2) & (sl1 < sl2)

    demand_high = piv["high_at_pivlow"].ffill()
    demand_low = piv["low_at_pivlow"].ffill()
    supply_high = piv["high_at_pivhigh"].ffill()
    supply_low = piv["low_at_pivhigh"].ffill()

    in_demand_zone = (
        demand_high.notna() & demand_low.notna()
        & (df["Low"] <= demand_high * (1 + zone_tol))
        & (df["High"] >= demand_low * (1 - zone_tol))
    )
    in_supply_zone = (
        supply_high.notna() & supply_low.notna()
        & (df["Low"] <= supply_high * (1 + zone_tol))
        & (df["High"] >= supply_low * (1 - zone_tol))
    )

    out = pd.DataFrame(index=df.index)
    out["BullishBias"] = bullish_bias.fillna(False)
    out["BearishBias"] = bearish_bias.fillna(False)
    out["DemandHigh"] = demand_high
    out["DemandLow"] = demand_low
    out["SupplyHigh"] = supply_high
    out["SupplyLow"] = supply_low
    out["InDemandZone"] = in_demand_zone.fillna(False)
    out["InSupplyZone"] = in_supply_zone.fillna(False)
    return out


# ============================================================================
# 3) BOS STATE  (break-of-structure + SL reference - the "15m" role)
# ============================================================================

def build_bos_state(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """
    Replicates the 15m BOS + SL-reference block. df must have High, Low, Close.
    """
    piv = detect_pivots(df["High"], df["Low"], left, right)
    bos_level_high = piv["confirmed_high_value"].ffill()   # bosLevelHigh / slSwingHigh
    bos_level_low = piv["confirmed_low_value"].ffill()     # bosLevelLow  / slSwingLow

    bullish_bos = bos_level_high.notna() & (df["Close"] > bos_level_high)
    bearish_bos = bos_level_low.notna() & (df["Close"] < bos_level_low)

    out = pd.DataFrame(index=df.index)
    out["SLSwingHigh"] = bos_level_high
    out["SLSwingLow"] = bos_level_low
    out["BullishBOS"] = bullish_bos.fillna(False)
    out["BearishBOS"] = bearish_bos.fillna(False)
    return out


# ============================================================================
# 4) EMA + retest touch  (the "5m entry trigger" role)
# ============================================================================

def add_ema_and_touch(df: pd.DataFrame, length: int) -> pd.DataFrame:
    ema = df["Close"].ewm(span=length, adjust=False).mean()
    ema_touch_long = (df["Low"] <= ema * 1.001) & (df["Close"] > ema)
    ema_touch_short = (df["High"] >= ema * 0.999) & (df["Close"] < ema)
    out = pd.DataFrame(index=df.index)
    out["EMA"] = ema
    out["EmaTouchLong"] = ema_touch_long
    out["EmaTouchShort"] = ema_touch_short
    return out


# ============================================================================
# 5) THE STATE MACHINE / BACKTEST LOOP
#    (mirrors the Pine "if longCondition / shortCondition" execution block)
# ============================================================================

def run_confluence_backtest(merged: pd.DataFrame, params: dict) -> list:
    """
    merged must contain (all aligned on the entry timeframe's index):
        Close, High, Low, EMA, EmaTouchLong, EmaTouchShort,
        BullishBias, BearishBias, InDemandZone, InSupplyZone,
        BullishBOS, BearishBOS, SLSwingHigh, SLSwingLow,
        DayId (grouping key that resets the daily trade counter / BOS state),
        InWindow (bool - True if this bar is inside the tradeable time window)

    params:
        sl_buffer, rr_ratio, max_trades_per_day, max_holding_bars

    Returns a list of trade dicts (index-based; caller maps back to timestamps).
    """
    close = merged["Close"].to_numpy(dtype=float)
    high = merged["High"].to_numpy(dtype=float)
    low = merged["Low"].to_numpy(dtype=float)
    ema = merged["EMA"].to_numpy(dtype=float)
    ema_touch_long = merged["EmaTouchLong"].to_numpy(dtype=bool)
    ema_touch_short = merged["EmaTouchShort"].to_numpy(dtype=bool)
    bullish_bias = merged["BullishBias"].to_numpy(dtype=bool)
    bearish_bias = merged["BearishBias"].to_numpy(dtype=bool)
    in_demand = merged["InDemandZone"].to_numpy(dtype=bool)
    in_supply = merged["InSupplyZone"].to_numpy(dtype=bool)
    bullish_bos = merged["BullishBOS"].to_numpy(dtype=bool)
    bearish_bos = merged["BearishBOS"].to_numpy(dtype=bool)
    sl_high_ref = merged["SLSwingHigh"].to_numpy(dtype=float)
    sl_low_ref = merged["SLSwingLow"].to_numpy(dtype=float)
    day_id = merged["DayId"].to_numpy()
    in_window = merged["InWindow"].to_numpy(dtype=bool) if "InWindow" in merged.columns else np.ones(len(merged), dtype=bool)

    n = len(merged)
    sl_buffer = params["sl_buffer"]
    rr_ratio = params["rr_ratio"]
    max_trades_per_day = params["max_trades_per_day"]
    max_holding_bars = params["max_holding_bars"]

    bos_bull_confirmed = False
    bos_bear_confirmed = False
    waiting_retest_l = False
    waiting_retest_s = False
    trades_today = 0
    last_day = None

    open_trade = None
    trades = []

    for i in range(n):
        d = day_id[i]
        if d != last_day:
            trades_today = 0
            last_day = d
            bos_bull_confirmed = False
            bos_bear_confirmed = False
            waiting_retest_l = False
            waiting_retest_s = False

        # ---- manage an already-open trade: check for exit first ----
        if open_trade is not None:
            direction = open_trade["direction"]
            exit_price = None
            exit_reason = None
            if direction == "long":
                if low[i] <= open_trade["sl"]:
                    exit_price, exit_reason = open_trade["sl"], "Stop Loss"
                elif high[i] >= open_trade["target"]:
                    exit_price, exit_reason = open_trade["target"], "Target"
            else:
                if high[i] >= open_trade["sl"]:
                    exit_price, exit_reason = open_trade["sl"], "Stop Loss"
                elif low[i] <= open_trade["target"]:
                    exit_price, exit_reason = open_trade["target"], "Target"

            if exit_price is None and (i - open_trade["entry_idx"]) >= max_holding_bars:
                exit_price, exit_reason = close[i], "Time Exit"

            if exit_price is not None:
                open_trade["exit_idx"] = i
                open_trade["exit_price"] = exit_price
                open_trade["exit_reason"] = exit_reason
                trades.append(open_trade)
                open_trade = None
            continue  # no new entry on the same bar we're managing/just closed

        # ---- flat: evaluate entry conditions ----
        can_trade = (trades_today < max_trades_per_day) and in_window[i]

        if not np.isnan(sl_low_ref[i]):
            long_sl = sl_low_ref[i] - sl_buffer
        else:
            long_sl = low[i] - sl_buffer
        if not np.isnan(sl_high_ref[i]):
            short_sl = sl_high_ref[i] + sl_buffer
        else:
            short_sl = high[i] + sl_buffer

        long_target = close[i] + (close[i] - long_sl) * rr_ratio
        short_target = close[i] - (short_sl - close[i]) * rr_ratio
        long_risk = close[i] - long_sl
        short_risk = short_sl - close[i]
        long_rr = (long_target - close[i]) / long_risk if long_risk > 0 else 0.0
        short_rr = (close[i] - short_target) / short_risk if short_risk > 0 else 0.0

        long_bos_direct = (
            can_trade and bullish_bias[i] and in_demand[i] and bullish_bos[i]
            and close[i] > ema[i] and long_rr >= rr_ratio
        )
        long_retest = (
            can_trade and bos_bull_confirmed and waiting_retest_l and in_demand[i]
            and ema_touch_long[i] and long_rr >= rr_ratio
        )
        long_condition = long_bos_direct or long_retest

        short_bos_direct = (
            can_trade and bearish_bias[i] and in_supply[i] and bearish_bos[i]
            and close[i] < ema[i] and short_rr >= rr_ratio
        )
        short_retest = (
            can_trade and bos_bear_confirmed and waiting_retest_s and in_supply[i]
            and ema_touch_short[i] and short_rr >= rr_ratio
        )
        short_condition = short_bos_direct or short_retest

        # update the BOS-confirmed / waiting-retest state machine
        if bullish_bos[i] and bullish_bias[i] and in_demand[i]:
            bos_bull_confirmed, bos_bear_confirmed = True, False
            waiting_retest_l, waiting_retest_s = True, False
        if bearish_bos[i] and bearish_bias[i] and in_supply[i]:
            bos_bear_confirmed, bos_bull_confirmed = True, False
            waiting_retest_s, waiting_retest_l = True, False

        if long_condition:
            open_trade = {
                "direction": "long",
                "entry_idx": i,
                "entry_price": close[i],
                "sl": long_sl,
                "target": long_target,
                "entry_type": "BOS Direct" if long_bos_direct else "Retest EMA",
            }
            trades_today += 1
            bos_bull_confirmed, waiting_retest_l = False, False
        elif short_condition:
            open_trade = {
                "direction": "short",
                "entry_idx": i,
                "entry_price": close[i],
                "sl": short_sl,
                "target": short_target,
                "entry_type": "BOS Direct" if short_bos_direct else "Retest EMA",
            }
            trades_today += 1
            bos_bear_confirmed, waiting_retest_s = False, False

    if open_trade is not None:
        open_trade["exit_idx"] = n - 1
        open_trade["exit_price"] = close[n - 1]
        open_trade["exit_reason"] = "End of Data"
        trades.append(open_trade)

    return trades


# ============================================================================
# 6) OPTION LEG - ITM strike selection + Black-Scholes premium simulation
# ============================================================================

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Standard Black-Scholes European option price. T in years."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_itm_strike(spot: float, option_type: str, strike_step: float = 50.0, itm_depth: int = 1) -> float:
    """
    Nearest ATM strike, then move `itm_depth` steps into-the-money.
    Call ITM strikes sit BELOW spot; Put ITM strikes sit ABOVE spot.
    """
    atm = round(spot / strike_step) * strike_step
    if option_type == "call":
        return atm - strike_step * itm_depth
    else:
        return atm + strike_step * itm_depth


def attach_option_pnl(trades_df: pd.DataFrame, realized_vol_lookup, params: dict) -> pd.DataFrame:
    """
    For each row in trades_df (needs: direction, entry_time, exit_time,
    entry_price [spot], exit_price [spot]) simulate the option leg with
    Black-Scholes, since free historical NSE options-chain data isn't
    available. Adds: Strike, EntryPremium, ExitPremium, PnL, PnLPerLot.

    realized_vol_lookup(timestamp) -> annualized sigma (float) as of that time.
    """
    strike_step = params["strike_step"]
    itm_depth = params["itm_depth"]
    r = params["risk_free_rate"]
    days_to_expiry = params["days_to_expiry"]
    lot_size = params["lot_size"]
    n_lots = params["n_lots"]

    rows = []
    for _, tr in trades_df.iterrows():
        option_type = "call" if tr["direction"] == "long" else "put"
        strike = get_itm_strike(tr["entry_price"], option_type, strike_step, itm_depth)
        sigma = realized_vol_lookup(tr["entry_time"])
        if sigma is None or np.isnan(sigma) or sigma <= 0:
            sigma = 0.12  # sane fallback annualized vol if history is too short

        T_entry = max(days_to_expiry, 0.5) / 365.0
        holding_days = max((tr["exit_time"] - tr["entry_time"]).total_seconds() / 86400.0, 0.0)
        T_exit = max(days_to_expiry - holding_days, 0.25) / 365.0

        entry_premium = black_scholes_price(tr["entry_price"], strike, T_entry, r, sigma, option_type)
        exit_premium = black_scholes_price(tr["exit_price"], strike, T_exit, r, sigma, option_type)

        qty = lot_size * n_lots
        pnl = (exit_premium - entry_premium) * qty

        row = tr.to_dict()
        row.update({
            "OptionType": "CE" if option_type == "call" else "PE",
            "Strike": strike,
            "ImpliedVolUsed": sigma,
            "EntryPremium": entry_premium,
            "ExitPremium": exit_premium,
            "Quantity": qty,
            "PnL": pnl,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def compute_performance(trades_df: pd.DataFrame, initial_capital: float) -> dict:
    if trades_df.empty:
        return {}
    pnl = trades_df["PnL"]
    equity = initial_capital + pnl.cumsum()
    running_max = equity.cummax()
    drawdown = running_max - equity
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    return {
        "total_trades": len(trades_df),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl <= 0).sum()),
        "win_rate": (pnl > 0).mean() * 100,
        "net_pnl": pnl.sum(),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "avg_win": pnl[pnl > 0].mean() if (pnl > 0).any() else 0.0,
        "avg_loss": pnl[pnl <= 0].mean() if (pnl <= 0).any() else 0.0,
        "max_drawdown": drawdown.max(),
        "final_equity": equity.iloc[-1],
        "equity_curve": equity,
    }
