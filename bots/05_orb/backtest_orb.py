"""
================================================================================
 BACKTESTER — Multi-Timeframe 30m ORB Bot  (mirrors orb_bot_live.py)
================================================================================
 Replicates the live bot's rules EXACTLY:

   30m : opening range = 08:00-08:30 UK candle (high/low incl. wicks)
   10m : breakout candle opens inside range, closes outside,
         body >= 70% of range, breakout-side wick <= 20%
   2m  : pullback to 20 EMA, enter on the reclaim close
   SL  : 5-candle 2m swing +/- buffer   TP : fixed 2R
   Risk: 1% per trade, one trade per direction per day
   Time: breakouts AND entries only between 08:30 and 10:00 UK
         (pending setups die at 10:00, same as the live cancel_setup)

 Data strategy
 -------------
 Fetches M2 bars from MT5 and RESAMPLES them to 10m and 30m. This guarantees
 all three timeframes are built from identical underlying data (no feed
 misalignment) and only needs one history request per symbol.

 CAVEAT: brokers keep limited M2 history — often only 2-6 months. The report
 header prints the ACTUAL coverage obtained; judge sample size accordingly.

 Conservative fill assumptions
 -----------------------------
 * Entry at the 2m trigger close + half-spread (long) / - half-spread (short)
 * If SL and TP are both touched inside the same 2m bar -> SL is assumed hit
 * Spread applied on exit too (full round-trip cost = 1x spread)
 * Positions left open at end of data are closed at the last close ("EOD")

 IMPORTANT: if you change strategy rules in orb_bot_live.py, change them here
 too — the two files must stay in sync or the backtest is testing a different
 system than the one you're running.
================================================================================
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG — keep these numbers identical to orb_bot_live.py
# ------------------------------------------------------------------------------
LONDON_TZ = ZoneInfo("Europe/London")

SESSION_OPEN      = dtime(8, 0)
ORB_CLOSE         = dtime(8, 30)
BREAKOUT_DEADLINE = dtime(10, 0)

RISK_PER_TRADE_PCT = 0.01
REWARD_MULTIPLE    = 2.0
BODY_MIN_RATIO     = 0.70
WICK_MAX_RATIO     = 0.20
SWING_LOOKBACK     = 5
EMA_PERIOD         = 20

START_BALANCE  = 10_000.0
LOOKBACK_DAYS  = 365          # requested history; actual coverage is reported

# buffer / typical spread in PRICE units per symbol
SYMBOLS: dict[str, dict] = {
    "GBPUSD": {"buffer": 0.0002, "pip_size": 0.0001, "spread": 0.00012},
    "USDJPY": {"buffer": 0.02,   "pip_size": 0.01,   "spread": 0.012},
    "GBPJPY": {"buffer": 0.02,   "pip_size": 0.01,   "spread": 0.020},
    "XAUUSD": {"buffer": 0.20,   "pip_size": 0.01,   "spread": 0.25},
    "DE40":   {"buffer": 2.0,    "pip_size": 0.1,    "spread": 1.5},
}


# ==============================================================================
# DATA — fetch M2 once, resample up (guarantees timeframe alignment)
# ==============================================================================
def fetch_m2_history(symbol: str, days: int) -> pd.DataFrame:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize() failed: {mt5.last_error()}")
    mt5.symbol_select(symbol, True)

    utc_to = datetime.utcnow()
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M2, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        raise SystemExit(f"No M2 data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").tz_convert(LONDON_TZ)
    return df.rename(columns={"tick_volume": "volume"})[
        ["open", "high", "low", "close", "volume"]]


def resample(df_2m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M2 -> 10m/30m. label='left' so the index is the candle's OPEN
    time, matching how MT5 (and the live bot) timestamp candles."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df_2m.resample(rule, label="left", closed="left").agg(agg).dropna()


# ==============================================================================
# STRATEGY RULES — verbatim ports from the live bot
# ==============================================================================
def is_valid_breakout_candle(candle: pd.Series, orb_high: float,
                             orb_low: float) -> str | None:
    o, h, l, c = (float(candle[k]) for k in ("open", "high", "low", "close"))
    rng = h - l
    if rng <= 0:
        return None
    body_ratio = abs(c - o) / rng
    upper_wick = (h - max(o, c)) / rng
    lower_wick = (min(o, c) - l) / rng
    if not (orb_low <= o <= orb_high) or body_ratio < BODY_MIN_RATIO:
        return None
    if c > orb_high and upper_wick <= WICK_MAX_RATIO:
        return "long"
    if c < orb_low and lower_wick <= WICK_MAX_RATIO:
        return "short"
    return None


def compute_ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ==============================================================================
# TRADE RECORD
# ==============================================================================
@dataclass
class Trade:
    symbol: str
    day: object
    direction: str
    entry_time: object
    entry: float
    sl: float
    tp: float
    exit_time: object = None
    exit_price: float = np.nan
    outcome: str = ""          # "TP" | "SL" | "EOD"
    r_result: float = np.nan   # result in R multiples (after spread)
    pnl: float = np.nan        # currency P&L at 1% risk sizing


# ==============================================================================
# CORE SIMULATION — one symbol
# ==============================================================================
def simulate_symbol(symbol: str, meta: dict, df_2m: pd.DataFrame) -> list[Trade]:
    df_10m = resample(df_2m, "10min")
    df_30m = resample(df_2m, "30min")

    # Precompute the 2m EMA over the full series once (identical to the live
    # bot, which computes EMA over its rolling 120-candle fetch — values match
    # to within float noise once >100 bars of warm-up exist).
    ema = compute_ema(df_2m["close"])

    spread = meta["spread"]
    buffer = meta["buffer"]
    half_spread = spread / 2

    trades: list[Trade] = []

    for day, day_30m in df_30m.groupby(df_30m.index.date):
        if pd.Timestamp(day).weekday() >= 5:
            continue

        # ---------------- STAGE 1: opening range ----------------
        orb_rows = day_30m[day_30m.index.time == SESSION_OPEN]
        if orb_rows.empty:
            continue
        orb_high = float(orb_rows.iloc[0]["high"])
        orb_low = float(orb_rows.iloc[0]["low"])

        long_taken = False
        short_taken = False

        # 10m candles whose OPEN is in [08:30, 10:00) — the last one that can
        # confirm opens at 09:50 and closes exactly at the 10:00 deadline.
        day_10m = df_10m[(df_10m.index.date == day)
                         & (df_10m.index.time >= ORB_CLOSE)
                         & (df_10m.index.time < BREAKOUT_DEADLINE)]

        # ---------------- STAGE 2: breakout hunt ----------------
        for ts10, candle in day_10m.iterrows():
            direction = is_valid_breakout_candle(candle, orb_high, orb_low)
            if direction is None:
                continue
            if (direction == "long" and long_taken) or \
               (direction == "short" and short_taken):
                continue

            breakout_close_time = ts10 + pd.Timedelta(minutes=10)

            # ---------------- STAGE 3: 2m pullback & reclaim ----------------
            deadline = pd.Timestamp.combine(
                pd.Timestamp(day), BREAKOUT_DEADLINE).tz_localize(LONDON_TZ)
            # 2m candles from breakout close up to (not incl.) the deadline;
            # the candle OPENING at 09:58 closes at 10:00 and still counts —
            # the live bot evaluates it because cancel_setup fires at >=10:00
            # on the NEXT poll after that close.
            window = df_2m[(df_2m.index >= breakout_close_time)
                           & (df_2m.index.time < BREAKOUT_DEADLINE)
                           & (df_2m.index.date == day)]

            pullback_seen = False
            trade: Trade | None = None

            for ts2, bar in window.iterrows():
                e = float(ema.loc[ts2])
                if direction == "long":
                    if not pullback_seen:
                        if float(bar["low"]) <= e:
                            pullback_seen = True
                        continue
                    if float(bar["close"]) > e:
                        # swing low of the most recent 5 bars INCLUDING this one
                        pos = df_2m.index.get_loc(ts2)
                        swing = float(df_2m["low"].iloc[max(0, pos - SWING_LOOKBACK + 1): pos + 1].min())
                        entry = float(bar["close"]) + half_spread   # buy at ask
                        sl = swing - buffer
                        risk = entry - sl
                        if risk <= 0:
                            break
                        tp = entry + risk * REWARD_MULTIPLE
                        trade = Trade(symbol, day, "long", ts2, entry, sl, tp)
                        break
                else:
                    if not pullback_seen:
                        if float(bar["high"]) >= e:
                            pullback_seen = True
                        continue
                    if float(bar["close"]) < e:
                        pos = df_2m.index.get_loc(ts2)
                        swing = float(df_2m["high"].iloc[max(0, pos - SWING_LOOKBACK + 1): pos + 1].max())
                        entry = float(bar["close"]) - half_spread   # sell at bid
                        sl = swing + buffer
                        risk = sl - entry
                        if risk <= 0:
                            break
                        tp = entry - risk * REWARD_MULTIPLE
                        trade = Trade(symbol, day, "short", ts2, entry, sl, tp)
                        break

            if trade is None:
                continue    # setup armed but never triggered before 10:00

            # ---------------- Resolve the trade bar-by-bar ----------------
            _resolve(trade, df_2m, half_spread)
            trades.append(trade)
            if trade.direction == "long":
                long_taken = True
            else:
                short_taken = True
            # loop continues: opposite-direction breakout may still arm today

    return trades


def _resolve(trade: Trade, df_2m: pd.DataFrame, half_spread: float) -> None:
    """Walk forward on 2m bars until SL or TP is touched.
    Both touched in one bar -> SL assumed (conservative).
    Exit prices include the half-spread on the way out."""
    start = df_2m.index.get_loc(trade.entry_time) + 1
    risk = abs(trade.entry - trade.sl)

    for ts, bar in df_2m.iloc[start:].iterrows():
        h, l = float(bar["high"]), float(bar["low"])
        if trade.direction == "long":
            hit_sl = (l - half_spread) <= trade.sl      # closed at bid
            hit_tp = (h - half_spread) >= trade.tp
        else:
            hit_sl = (h + half_spread) >= trade.sl      # closed at ask
            hit_tp = (l + half_spread) <= trade.tp

        if hit_sl:                                       # SL wins ties
            trade.exit_time, trade.exit_price, trade.outcome = ts, trade.sl, "SL"
            trade.r_result = -1.0
            break
        if hit_tp:
            trade.exit_time, trade.exit_price, trade.outcome = ts, trade.tp, "TP"
            trade.r_result = REWARD_MULTIPLE
            break
    else:
        # Data ended with the position open — mark to last close
        last = df_2m.iloc[-1]
        trade.exit_time = df_2m.index[-1]
        trade.exit_price = float(last["close"])
        trade.outcome = "EOD"
        move = (trade.exit_price - trade.entry) if trade.direction == "long" \
               else (trade.entry - trade.exit_price)
        trade.r_result = move / risk if risk > 0 else 0.0

    trade.pnl = trade.r_result * START_BALANCE * RISK_PER_TRADE_PCT


# ==============================================================================
# REPORTING
# ==============================================================================
def report(symbol: str, trades: list[Trade], coverage: str) -> dict:
    print("=" * 62)
    print(f"  ORB BACKTEST — {symbol}   ({coverage})")
    print("=" * 62)
    if not trades:
        print("  No trades generated. The filters may be too strict for this")
        print("  symbol/period, or M2 history is too short.")
        return {"symbol": symbol, "trades": 0}

    r = np.array([t.r_result for t in trades])
    pnl = np.array([t.pnl for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    win_rate = len(wins) / len(r)
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    expectancy_r = r.mean()
    t_stat = expectancy_r / (r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else 0.0

    equity = START_BALANCE + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[START_BALANCE], equity]))
    dd = ((np.concatenate([[START_BALANCE], equity]) - peak) / peak).min()

    # longest losing streak
    streak = worst = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)

    n_long = sum(1 for t in trades if t.direction == "long")
    n_eod = sum(1 for t in trades if t.outcome == "EOD")

    print(f"  Trades taken .......... {len(trades)}  "
          f"({n_long} long / {len(trades)-n_long} short"
          f"{f', {n_eod} EOD-marked' if n_eod else ''})")
    print(f"  Win rate .............. {win_rate:.1%}  "
          f"({len(wins)}W / {len(losses)}L)")
    if len(wins) and len(losses):
        print(f"  Payoff ratio .......... {wins.mean() / abs(losses.mean()):.2f} : 1")
    print(f"  Expectancy per trade .. {expectancy_r:+.2f} R   "
          f"(t-stat {t_stat:.2f}{'  — significant' if abs(t_stat) >= 2 else '  — NOT significant yet'})")
    print(f"  Profit factor ......... {pf:.2f}")
    print("  " + "-" * 56)
    print(f"  Net P&L ............... {pnl.sum():+.2f}  "
          f"({pnl.sum()/START_BALANCE:+.2%} on {START_BALANCE:,.0f})")
    print(f"  Max drawdown .......... {dd:.2%}")
    print(f"  Longest losing streak . {worst} trades")
    verdict = ("PROMISING — forward-test on demo" if pf >= 1.2 and len(r) >= 30
               else "INCONCLUSIVE — needs more data" if pf >= 1.0
               else "NEGATIVE EDGE as configured")
    print(f"  Verdict: {verdict}")
    print("=" * 62 + "\n")
    return {"symbol": symbol, "trades": len(trades), "pf": pf,
            "exp_r": expectancy_r, "pnl": pnl.sum(), "r": r}


def fleet_report(results: list[dict]) -> None:
    live = [x for x in results if x.get("trades", 0) > 0]
    if len(live) < 2:
        return
    all_r = np.concatenate([x["r"] for x in live])
    total = sum(x["pnl"] for x in live)
    exp = all_r.mean()
    t = exp / (all_r.std(ddof=1) / math.sqrt(len(all_r))) if len(all_r) > 1 else 0
    print("=" * 62)
    print("  FLEET COMBINED")
    print("=" * 62)
    print(f"  Total trades .......... {len(all_r)}")
    print(f"  Expectancy ............ {exp:+.2f} R  (t-stat {t:.2f})")
    print(f"  Combined P&L .......... {total:+.2f}")
    for x in sorted(live, key=lambda v: v["exp_r"], reverse=True):
        print(f"    {x['symbol']:<8} {x['trades']:>4} trades   "
              f"PF {x['pf']:.2f}   {x['exp_r']:+.2f} R/trade")
    print("=" * 62)


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    symbols = sys.argv[1:] or list(SYMBOLS)
    results = []
    for sym in symbols:
        meta = SYMBOLS.get(sym)
        if meta is None:
            print(f"Unknown symbol '{sym}' — add it to the SYMBOLS dict.")
            continue
        print(f"Fetching M2 history for {sym} ...")
        df = fetch_m2_history(sym, LOOKBACK_DAYS)
        coverage = (f"{df.index[0].date()} → {df.index[-1].date()}, "
                    f"{len(df):,} M2 bars, "
                    f"{(df.index[-1] - df.index[0]).days} days")
        print(f"  Coverage: {coverage}")
        trades = simulate_symbol(sym, meta, df)
        results.append(report(sym, trades, coverage))
    fleet_report(results)


if __name__ == "__main__":
    main()