"""
BACKTESTER for GOLD SESSION PRO v1.0
====================================================================
Replays the EXACT decision logic of gold_session_pro.py against
historical XAUUSD data pulled from your own MT5 terminal, and reports
the numbers that actually matter: expectancy, payoff ratio, profit
factor, and max drawdown.

WHY A CUSTOM BACKTESTER (and not the MT5 Strategy Tester):
  The Strategy Tester only runs compiled MQL5 EAs. Your bot is external
  Python over the MT5 API, so the Strategy Tester can't run it. This
  script tests the real logic you'll actually deploy.

HOW TO RUN:
  1. Open MT5, log in, make sure XAUUSD is in Market Watch.
  2. pip install MetaTrader5 pandas numpy matplotlib
  3. python backtest_gold_session_pro.py
  Outputs: printed summary + backtest_trades.csv + backtest_equity.png

KEEP IN SYNC: the STRATEGY PARAMS below must match gold_session_pro.py.
If you tune the live bot, change them here too, or the backtest lies.

HONEST LIMITATIONS (read these):
  - Bar-resolution fills. When a single M15 bar touches BOTH stop and
    target, this assumes the STOP hit first (conservative/pessimistic).
  - Spread modelled as one full round-turn charged at entry. Real
    slippage on breakouts can be worse; don't treat results as a floor.
  - Drawdown is measured on closed-trade equity; true intra-trade
    drawdown is larger.
  - Past performance is not future performance. A good backtest is a
    filter that removes obviously-broken systems, not a profit promise.
    Forward-test on demo before risking anything real.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# MT5 is only needed for the data pull; guard it so the engine stays testable.
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# ==================== BACKTEST SETTINGS ====================
SYMBOL          = "XAUUSD"
LOOKBACK_DAYS   = 180            # how far back to test
START_BALANCE   = 100_000.0      # match your account
SPREAD_USD      = 0.25           # round-turn spread cost in price, charged at entry
COMMISSION_PER_LOT = 0.0         # $ per lot round-turn, if your broker charges it
# ==========================================================

# ============ STRATEGY PARAMS (mirror gold_session_pro.py) ============
RISK_PER_TRADE_PCT  = 0.75
MAX_TRADES_PER_DAY  = 3
MIN_LOT             = 0.01
MAX_LOT             = 2.0
COOLDOWN_MIN        = 30

ASIA_START_H        = 0
ASIA_END_H          = 7
LONDON_WINDOW       = (7, 11)
NY_WINDOW           = (12, 16)   # NY entries from 12:30 (see in_window)

ATR_PERIOD          = 14
EMA_BIAS_PERIOD     = 50
BREAK_BUFFER_ATR    = 0.25
SL_ATR_MULT         = 1.2
TP_R_MULT           = 2.0
BE_TRIGGER_R        = 1.0
TRAIL_ATR_MULT      = 2.0
MIN_RANGE_USD       = 4.0
MAX_RANGE_USD       = 45.0
# ======================================================================


# ----------------------- indicators -----------------------

def add_atr(df, period=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


def h1_bias_series(dfh1, period=EMA_BIAS_PERIOD):
    """Return a frame with 'avail' (epoch when the H1 bar becomes closed) and 'bias'."""
    ema = dfh1["close"].ewm(span=period, adjust=False).mean()
    bias = np.where(dfh1["close"] > ema, 1, np.where(dfh1["close"] < ema, -1, 0))
    out = pd.DataFrame({"avail": dfh1["time"].astype(np.int64) + 3600, "bias": bias})
    # first `period` bars have an unreliable EMA -> treat as no-trade
    out.loc[: period - 1, "bias"] = 0
    return out


# Session toggles. NY is OFF: the Asian range is ~12h stale by the NY session
# (London has usually already broken it), so a NY breakout of the ASIAN high/low
# is a late, extended signal. The backtest confirmed a negative edge there
# (PF 0.28). Only flip TRADE_NY back on if you re-validate it on more data.
TRADE_LONDON = True
TRADE_NY = False

def in_window(hour_float):
    if TRADE_LONDON and LONDON_WINDOW[0] <= hour_float < LONDON_WINDOW[1]:
        return "LONDON"
    if TRADE_NY and NY_WINDOW[0] + 0.5 <= hour_float < NY_WINDOW[1]:
        return "NY"
    return None


def lot_for_risk(sl_dist, balance, cfg):
    tick_val, tick_size = cfg["tick_value"], cfg["tick_size"]
    if not tick_val or not tick_size or sl_dist <= 0:
        return MIN_LOT
    loss_per_lot = (sl_dist / tick_size) * tick_val
    risk_amount = balance * RISK_PER_TRADE_PCT / 100.0
    lots = risk_amount / loss_per_lot
    step = cfg.get("volume_step", 0.01) or 0.01
    lots = max(cfg.get("volume_min", MIN_LOT), min(lots, cfg.get("volume_max", MAX_LOT), MAX_LOT))
    lots = round(lots / step) * step
    return round(max(lots, MIN_LOT), 2)


# ----------------------- the engine -----------------------

def run_backtest(df15, dfh1, cfg):
    """Pure function: takes M15 + H1 frames, returns dict(trades, equity, stats)."""
    df = df15.copy().reset_index(drop=True)
    df = add_atr(df)

    dt = pd.to_datetime(df["time"], unit="s", utc=True)
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
    df["date"] = dt.dt.date
    df["h_only"] = dt.dt.hour

    # merge H1 bias (last CLOSED H1 bar available at each M15 bar)
    bias_tbl = h1_bias_series(dfh1).sort_values("avail")
    df = pd.merge_asof(
        df.sort_values("time"),
        bias_tbl.rename(columns={"avail": "time"}),
        on="time", direction="backward",
    ).reset_index(drop=True)
    df["bias"] = df["bias"].fillna(0)

    # per-day Asian range (max high / min low over 00:00-07:00 UTC).
    # NOTE: max over M15 highs in a window == max over M5 highs, so this
    # is mathematically identical to the live bot's M5-based range.
    asia = df[df["h_only"] < ASIA_END_H].groupby("date").agg(hi=("high", "max"), lo=("low", "min"))
    asia_hi = asia["hi"].to_dict()
    asia_lo = asia["lo"].to_dict()

    tick_size = cfg["tick_size"]
    tick_value = cfg["tick_value"]

    balance = cfg["start_balance"]
    n = len(df)

    in_pos = False
    side = None
    entry_px = sl = tp = 0.0
    init_risk = 0.0
    lots = 0.0
    entry_time = 0
    past_be = False

    cur_date = None
    long_taken = short_taken = False
    trades_today = 0
    last_loss_time = None

    equity = [balance]
    trades = []

    o = df["open"].values
    h = df["high"].values
    lo_ = df["low"].values
    c = df["close"].values
    atr = df["atr"].values
    bias = df["bias"].values
    tsec = df["time"].values.astype(np.int64)

    for k in range(n):
        row_date = df["date"].iloc[k]
        if row_date != cur_date:
            cur_date = row_date
            long_taken = short_taken = False
            trades_today = 0

        # ---------------- manage an open position ----------------
        if in_pos:
            bh, bl, bc, ba = h[k], lo_[k], c[k], atr[k]
            exit_px = None
            if side == "BUY":
                hit_sl = bl <= sl
                hit_tp = bh >= tp
                if hit_sl and hit_tp:
                    exit_px = sl                 # conservative: stop first
                elif hit_sl:
                    exit_px = sl
                elif hit_tp:
                    exit_px = tp
            else:
                hit_sl = bh >= sl
                hit_tp = bl <= tp
                if hit_sl and hit_tp:
                    exit_px = sl
                elif hit_sl:
                    exit_px = sl
                elif hit_tp:
                    exit_px = tp

            if exit_px is not None:
                move = (exit_px - entry_px) if side == "BUY" else (entry_px - exit_px)
                pnl = (move / tick_size) * tick_value * lots
                pnl -= COMMISSION_PER_LOT * lots
                balance += pnl
                r_mult = move / init_risk if init_risk else 0.0
                trades.append({
                    "entry_time": datetime.fromtimestamp(entry_time, tz=timezone.utc),
                    "exit_time": datetime.fromtimestamp(int(tsec[k]), tz=timezone.utc),
                    "side": side, "entry": round(entry_px, 2), "exit": round(exit_px, 2),
                    "lots": lots, "R": round(r_mult, 2), "pnl": round(pnl, 2),
                    "balance": round(balance, 2),
                })
                equity.append(balance)
                if pnl < 0:
                    last_loss_time = int(tsec[k])
                in_pos = False
                side = None
                past_be = False
                continue

            # no exit -> update stop (breakeven, then chandelier trail)
            if not np.isnan(ba):
                if side == "BUY":
                    profit = bc - entry_px
                    if not past_be:
                        if profit >= BE_TRIGGER_R * init_risk:
                            sl = max(sl, entry_px)
                            past_be = True
                    else:
                        sl = max(sl, bc - TRAIL_ATR_MULT * ba)
                else:
                    profit = entry_px - bc
                    if not past_be:
                        if profit >= BE_TRIGGER_R * init_risk:
                            sl = min(sl, entry_px)
                            past_be = True
                    else:
                        sl = min(sl, bc + TRAIL_ATR_MULT * ba)
            continue

        # ---------------- look for an entry ----------------
        session = in_window(df["hour"].iloc[k])
        if session is None:
            continue
        if trades_today >= MAX_TRADES_PER_DAY:
            continue
        if last_loss_time is not None and (int(tsec[k]) - last_loss_time) < COOLDOWN_MIN * 60:
            continue
        d = row_date
        if d not in asia_hi:
            continue
        hi, low = asia_hi[d], asia_lo[d]
        rng = hi - low
        if rng < MIN_RANGE_USD or rng > MAX_RANGE_USD:
            continue
        a = atr[k]
        if np.isnan(a) or a <= 0:
            continue
        buf = BREAK_BUFFER_ATR * a
        close_k = c[k]
        b = bias[k]

        want = None
        if (not long_taken) and b == 1 and close_k > hi + buf:
            want = "BUY"
        elif (not short_taken) and b == -1 and close_k < low - buf:
            want = "SELL"
        if want is None or k + 1 >= n:
            continue

        # enter at NEXT bar open (mirrors live: signal on closed bar -> market fill)
        fill = o[k + 1]
        if want == "BUY":
            fill = fill + SPREAD_USD                       # adverse: pay spread
            stop = fill - SL_ATR_MULT * a
            stop = min(stop, hi - buf)
            risk = fill - stop
            target = fill + TP_R_MULT * risk
        else:
            fill = fill - SPREAD_USD
            stop = fill + SL_ATR_MULT * a
            stop = max(stop, low + buf)
            risk = stop - fill
            target = fill - TP_R_MULT * risk
        if risk <= 0:
            continue

        lots = lot_for_risk(risk, balance, cfg)
        in_pos = True
        side = want
        entry_px = fill
        sl = stop
        tp = target
        init_risk = risk
        entry_time = int(tsec[k + 1])
        past_be = False
        if want == "BUY":
            long_taken = True
        else:
            short_taken = True
        trades_today += 1

    # close any position still open at the end, at last close
    if in_pos:
        last_c = c[-1]
        move = (last_c - entry_px) if side == "BUY" else (entry_px - last_c)
        pnl = (move / tick_size) * tick_value * lots - COMMISSION_PER_LOT * lots
        balance += pnl
        trades.append({
            "entry_time": datetime.fromtimestamp(entry_time, tz=timezone.utc),
            "exit_time": datetime.fromtimestamp(int(tsec[-1]), tz=timezone.utc),
            "side": side, "entry": round(entry_px, 2), "exit": round(last_c, 2),
            "lots": lots, "R": round(move / init_risk, 2) if init_risk else 0,
            "pnl": round(pnl, 2), "balance": round(balance, 2), "note": "open_end",
        })
        equity.append(balance)

    trades_df = pd.DataFrame(trades)
    return {"trades": trades_df, "equity": equity, "stats": compute_stats(trades_df, equity, cfg)}


def compute_stats(trades_df, equity, cfg):
    start = cfg["start_balance"]
    if trades_df.empty:
        return {"trades": 0}
    wins = trades_df[trades_df["pnl"] > 0]["pnl"]
    losses = trades_df[trades_df["pnl"] <= 0]["pnl"]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    n = len(trades_df)
    # longest losing streak
    streak = worst = 0
    for p in trades_df["pnl"]:
        streak = streak + 1 if p <= 0 else 0
        worst = max(worst, streak)
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n * 100,
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "payoff": (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan"),
        "expectancy_$": trades_df["pnl"].mean(),
        "expectancy_R": trades_df["R"].mean(),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "net_pnl": trades_df["pnl"].sum(),
        "total_return_pct": (eq[-1] - start) / start * 100,
        "max_drawdown_pct": dd.max() * 100,
        "longest_losing_streak": worst,
        "end_balance": eq[-1],
    }


def print_report(res, cfg):
    s = res["stats"]
    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT — {cfg['symbol']}")
    print("=" * 60)
    if s.get("trades", 0) == 0:
        print("  No trades generated in this period.")
        print("  Widen LOOKBACK_DAYS or loosen the range filters to check.")
        print("=" * 60)
        return
    print(f"  Trades taken .......... {s['trades']}")
    print(f"  Win rate .............. {s['win_rate']:.1f}%  ({s['wins']}W / {s['losses']}L)")
    print(f"  Avg win / avg loss .... ${s['avg_win']:.2f} / ${s['avg_loss']:.2f}")
    print(f"  Payoff ratio .......... {s['payoff']:.2f} : 1")
    print(f"  Expectancy per trade .. ${s['expectancy_$']:.2f}   ({s['expectancy_R']:.2f} R)")
    print(f"  Profit factor ......... {s['profit_factor']:.2f}")
    print("  " + "-" * 56)
    print(f"  Net P&L ............... ${s['net_pnl']:,.2f}")
    print(f"  Total return .......... {s['total_return_pct']:.2f}%")
    print(f"  Max drawdown .......... {s['max_drawdown_pct']:.2f}%")
    print(f"  Longest losing streak . {s['longest_losing_streak']} trades")
    print(f"  End balance ........... ${s['end_balance']:,.2f}")
    print("=" * 60)
    pf = s["profit_factor"]
    exp = s["expectancy_R"]
    verdict = ("PROMISING — forward-test on demo" if (pf > 1.3 and exp > 0.1)
               else "MARGINAL — needs work before demo" if pf > 1.0
               else "NEGATIVE EDGE — do not deploy")
    print(f"  Verdict: {verdict}")
    print("  (One backtest is not proof. Forward-test on demo regardless.)")
    print("=" * 60 + "\n")


# ----------------------- MT5 data pull -----------------------

def load_from_mt5(cfg):
    if mt5 is None:
        print("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        sys.exit(1)
    if mt5.terminal_info() is None and not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}  (open the terminal and log in first)")
        sys.exit(1)

    info = mt5.symbol_info(SYMBOL)
    if info is None or not info.visible:
        mt5.symbol_select(SYMBOL, True)
        info = mt5.symbol_info(SYMBOL)
    if info is None:
        print(f"Symbol {SYMBOL} not available on this account.")
        sys.exit(1)

    cfg["tick_value"] = info.trade_tick_value
    cfg["tick_size"] = info.trade_tick_size
    cfg["volume_step"] = info.volume_step
    cfg["volume_min"] = info.volume_min
    cfg["volume_max"] = info.volume_max

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    r15 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, start, end)
    rh1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, start, end)
    if r15 is None or rh1 is None or len(r15) == 0:
        print("No history returned. Scroll the XAUUSD chart back once to load history, then retry.")
        sys.exit(1)
    return pd.DataFrame(r15), pd.DataFrame(rh1)


def main():
    cfg = {"symbol": SYMBOL, "start_balance": START_BALANCE}
    df15, dfh1 = load_from_mt5(cfg)
    print(f"Loaded {len(df15)} M15 bars, {len(dfh1)} H1 bars over {LOOKBACK_DAYS} days.")
    res = run_backtest(df15, dfh1, cfg)
    print_report(res, cfg)

    if not res["trades"].empty:
        res["trades"].to_csv("backtest_trades.csv", index=False)
        print("Saved trade-by-trade log -> backtest_trades.csv")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(11, 5))
            plt.plot(res["equity"], linewidth=1.4)
            plt.title(f"{SYMBOL} — Equity curve ({LOOKBACK_DAYS}d backtest)")
            plt.xlabel("Trade #"); plt.ylabel("Balance ($)")
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig("backtest_equity.png", dpi=120)
            print("Saved equity curve -> backtest_equity.png")
        except Exception as e:
            print(f"(Skipped chart: {e})")


if __name__ == "__main__":
    main()
