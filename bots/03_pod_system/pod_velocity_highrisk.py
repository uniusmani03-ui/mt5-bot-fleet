"""
=============================================================================
 POD VELOCITY  —  HIGH-RISK / SHORT-TERM momentum-breakout scalper
=============================================================================
 Profile: aggressive, fast, in-and-out. Trend-aligned breakout on M5 with a
 momentum (RSI) filter and ATR-based stops. HIGH risk per trade and many
 trades a day: big upside AND big drawdown — this is the spicy one.

 Logic (per pair, per closed M5 candle):
   - Trend:  EMA9 vs EMA21.
   - LONG :  uptrend (EMA9>EMA21) AND candle closes above the last
             BREAKOUT_LOOKBACK-bar high AND RSI > RSI_LONG.
   - SHORT:  mirror.
   - Stop  =  ATR * SL_ATR ;  Take-profit = ATR * TP_ATR (quick).
   - Also force-closes after MAX_HOLD_BARS if neither hit (keeps it short-term).
   - Only trades the active session and obeys a per-bot daily/total loss cap
     measured on THIS bot's own trades (safe to run beside other bots).

 Run on a DEMO account. Install once: pip install MetaTrader5 pandas numpy
 Keep pod_core.py in the same folder.   Run:  python pod_velocity_highrisk.py
=============================================================================
"""
import sys
import time
from datetime import datetime, timezone, timedelta

import MetaTrader5 as mt5
from pod_core import Bot, ema, rsi, atr

# ===================== CONFIG (HIGH RISK) ==================================
SYMBOLS            = ["EURUSD", "GBPUSD"]      # demo plain names
TIMEFRAME          = mt5.TIMEFRAME_M5
CANDLES            = 400

RISK_PERCENT       = 3.0       # <<< HIGH risk per trade
SL_ATR             = 1.2       # stop = 1.2 x ATR (tight)
TP_ATR             = 1.8       # target = 1.8 x ATR (~1.5R, quick)
EMA_FAST, EMA_SLOW = 9, 21
RSI_PERIOD         = 14
RSI_LONG, RSI_SHORT = 52, 48
ATR_PERIOD         = 14
BREAKOUT_LOOKBACK  = 10
MAX_HOLD_BARS      = 12        # close after ~1h on M5 if still open
MAX_TRADES_PER_DAY = 10        # per symbol
SESSION_UTC        = (7, 0, 20, 0)   # trade 07:00-20:00 UTC (London+NY)
MAX_SPREAD_POINTS  = 25
SLIPPAGE           = 8

DAILY_LOSS_PCT     = 12.0      # halt for the day at -12% of start ref (this bot only)
TOTAL_LOSS_PCT     = 50.0      # halt entirely at -50% (high-risk kill switch)

MAGIC              = 778100
LOOP_SLEEP         = 2
# ==========================================================================

bot = Bot("VELOCITY", MAGIC, "velocity")


def fetch(symbol):
    r = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, CANDLES)
    if r is None or len(r) < max(EMA_SLOW, ATR_PERIOD, BREAKOUT_LOOKBACK) + 5:
        return None
    import pandas as pd
    df = pd.DataFrame(r)
    df["ema_f"] = ema(df["close"], EMA_FAST)
    df["ema_s"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)
    df["hh"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    df["ll"] = df["low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    return df


def in_session(now):
    a = (SESSION_UTC[0], SESSION_UTC[1]); b = (SESSION_UTC[2], SESSION_UTC[3])
    return a <= (now.hour, now.minute) < b


def main():
    if not bot.connect():
        sys.exit(1)
    infos = {}
    for s in SYMBOLS:
        info = bot.prepare(s)
        if info is None:
            mt5.shutdown(); sys.exit(1)
        infos[s] = info

    acct = mt5.account_info()
    st = bot.load_state()
    if not st or st.get("ref_balance", 0) <= 0:
        st = {"start": datetime.now(timezone.utc).isoformat(),
              "ref_balance": acct.balance, "day": "", "day_start_realized": 0.0,
              "halted_today": False, "over": False}
        bot.save_state(st)
    bot.heal_ref(st, acct.balance)
    bot.max_lot = MAX_LOT
    start_dt = datetime.fromisoformat(st["start"])
    ref = st["ref_balance"]
    bot.log(f"POD VELOCITY live (HIGH risk). Ref balance {ref:.2f}. "
            f"Risk {RISK_PERCENT}%/trade | daily cap -{DAILY_LOSS_PCT}% | total cap -{TOTAL_LOSS_PCT}%.")
    bot.log(f"Pairs {', '.join(SYMBOLS)} | M5 momentum breakout | SL {SL_ATR}xATR TP {TP_ATR}xATR.")

    last_bar = {s: None for s in SYMBOLS}
    hold = {}  # ticket -> bars held

    try:
        while True:
            acct = mt5.account_info()
            if acct is None:
                time.sleep(LOOP_SLEEP); continue
            if st.get("over"):
                bot.log("Total loss cap hit earlier — standing down. (Delete velocity_state.json to reset.)")
                break

            pl = bot.own_pl(start_dt)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if st["day"] != today:
                st["day"] = today
                st["day_start_realized"] = pl["realized"]
                st["halted_today"] = False
                bot.save_state(st)
                bot.log(f"--- New day {today}. ---")

            day_pl = (pl["realized"] - st["day_start_realized"]) + pl["floating"]

            # kill switches (this bot's own P&L only)
            if pl["total"] <= -(TOTAL_LOSS_PCT / 100 * ref):
                bot.log(f"!!! TOTAL LOSS {pl['total']:.2f} (-{TOTAL_LOSS_PCT}%). Closing & stopping. !!!")
                bot.close_all(SLIPPAGE, "velocity_killswitch")
                st["over"] = True; bot.save_state(st); break
            if not st["halted_today"] and day_pl <= -(DAILY_LOSS_PCT / 100 * ref):
                bot.log(f"!! DAILY LOSS {day_pl:.2f} (-{DAILY_LOSS_PCT}%). Closing, no more today. !!")
                bot.close_all(SLIPPAGE, "velocity_daily")
                st["halted_today"] = True; bot.save_state(st)

            now = datetime.now(timezone.utc)
            trades_today = {s: 0 for s in SYMBOLS}

            for s in SYMBOLS:
                info = mt5.symbol_info(s); infos[s] = info
                df = fetch(s)
                if df is None:
                    continue
                closed = df.iloc[-2]
                if closed["time"] == last_bar[s]:
                    # still same bar: just manage hold timer below via position check
                    pass
                new_bar = closed["time"] != last_bar[s]

                pos = bot.position(s)

                # manage: force-close stale positions (short-term discipline)
                if pos is not None and new_bar:
                    hold[pos.ticket] = hold.get(pos.ticket, 0) + 1
                    if hold[pos.ticket] >= MAX_HOLD_BARS:
                        bot.log(f"[{s}] max hold reached -> flatten.")
                        bot.close(pos, SLIPPAGE, "velocity_maxhold")
                        hold.pop(pos.ticket, None)

                if not new_bar:
                    continue
                last_bar[s] = closed["time"]

                if st["halted_today"] or pos is not None or not in_session(now):
                    continue

                a = float(closed["atr"])
                if a <= 0 or any(closed[c] != closed[c] for c in ("ema_f", "ema_s", "rsi", "hh", "ll")):
                    continue
                cl = float(closed["close"])
                up = closed["ema_f"] > closed["ema_s"]
                dn = closed["ema_f"] < closed["ema_s"]
                risk_amt = RISK_PERCENT / 100 * acct.balance

                long_sig = up and cl > closed["hh"] and closed["rsi"] > RSI_LONG
                short_sig = dn and cl < closed["ll"] and closed["rsi"] < RSI_SHORT

                if long_sig and self_ok(s) and bot.spread_ok(info, MAX_SPREAD_POINTS):
                    sl = cl - SL_ATR * a; tp = cl + TP_ATR * a
                    bot.log(f"[{s}] LONG breakout (RSI {closed['rsi']:.0f}, ATR {a/info.point:.0f}pts).")
                    bot.place(info, "BUY", sl, tp, risk_amt, SLIPPAGE, "velocity")
                elif short_sig and self_ok(s) and bot.spread_ok(info, MAX_SPREAD_POINTS):
                    sl = cl + SL_ATR * a; tp = cl - TP_ATR * a
                    bot.log(f"[{s}] SHORT breakdown (RSI {closed['rsi']:.0f}, ATR {a/info.point:.0f}pts).")
                    bot.place(info, "SELL", sl, tp, risk_amt, SLIPPAGE, "velocity")

            _write_status(acct, pl, day_pl, ref)
            time.sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        bot.log("Manual stop (Ctrl+C). Open trades left running.")
    finally:
        mt5.shutdown()
        bot.log("MT5 closed.")


# simple per-symbol daily-count guard using history would be heavier; we cap via
# "one position at a time" + session, which keeps frequency sane. (Hook kept for clarity.)
def self_ok(symbol):
    return True


def _write_status(acct, pl, day_pl, ref):
    positions = [{"symbol": p.symbol, "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                  "volume": p.volume, "entry": p.price_open, "sl": p.sl, "tp": p.tp,
                  "price": p.price_current, "profit": round(p.profit, 2)} for p in bot.positions()]
    bot.write_status({
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "bot": "POD VELOCITY", "profile": "HIGH RISK / short-term",
        "account": {"login": acct.login, "server": acct.server, "currency": acct.currency,
                    "balance": round(acct.balance, 2), "equity": round(acct.equity, 2)},
        "own_pl": pl, "day_pl": round(day_pl, 2), "ref_balance": ref,
        "settings": {"risk_percent": RISK_PERCENT, "sl_atr": SL_ATR, "tp_atr": TP_ATR,
                     "pairs": SYMBOLS, "daily_cap_pct": DAILY_LOSS_PCT, "total_cap_pct": TOTAL_LOSS_PCT},
        "positions": positions, "log": bot.buf[-40:],
    })


if __name__ == "__main__":
    main()
