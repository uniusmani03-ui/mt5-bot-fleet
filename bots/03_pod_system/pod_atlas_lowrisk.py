"""
=============================================================================
 POD ATLAS  —  LOW-RISK / LONG-TERM Donchian trend-follower
=============================================================================
 Profile: patient, defensive, lets winners run. The classic turtle-style
 breakout system on the H4 timeframe with a long-term trend filter. LOW risk
 per trade, wide stops, NO fixed take-profit — it stays in a trend until the
 trend breaks, aiming to catch the occasional big multi-day move.

 Logic (per pair, per closed H4 candle):
   - Trend filter: EMA(TREND_EMA) on H4.
   - LONG entry : close breaks above the prior ENTRY_N-bar high AND close>EMA.
   - SHORT entry: close breaks below the prior ENTRY_N-bar low  AND close<EMA.
   - Stop  =  ATR * SL_ATR (wide, catastrophe protection).
   - EXIT  =  opposite EXIT_N-bar channel (trail the trend out); no profit cap.
   - Per-bot daily/total loss caps measured on THIS bot's own trades only.

 Most breakouts fail and get stopped for a small loss; the few that trend pay
 for all of them. That is the whole point — keep losers tiny, let winners run.

 Run on a DEMO account. Install once: pip install MetaTrader5 pandas numpy
 Keep pod_core.py in the same folder.   Run:  python pod_atlas_lowrisk.py
=============================================================================
"""
import sys
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
from pod_core import Bot, ema, atr, donchian

# ===================== CONFIG (LOW RISK · £200 demo) ======================
SYMBOLS            = ["EURUSD", "AUDUSD"]   # 2 cheaper majors (smaller min-lot risk than JPY/GBP)
TIMEFRAME          = mt5.TIMEFRAME_H4
CANDLES            = 400

RISK_PERCENT       = 0.5       # target risk per trade (only fully achievable on a cent account)
# --- Small-account guards ---
# On a £200 STANDARD account the broker's minimum lot (0.01) risks far more than
# 0.5% on H4 stops. This ceiling makes the bot SKIP any trade that would risk
# more than MAX_RISK_PER_TRADE_PCT of the account, so it can never over-risk you.
# Result on a standard £200 demo: it only takes the cheaper (tighter-stop) setups
# and skips the rest. On a CENT account, 0.5% works properly and nothing is skipped.
MAX_RISK_PER_TRADE_PCT  = 2.0
MAX_CONCURRENT_POSITIONS = 2

SL_ATR             = 2.5       # wide stop (room to breathe)
TREND_EMA          = 50        # long-term trend filter on H4
ENTRY_N            = 20        # breakout channel (~3.3 days of H4)
EXIT_N             = 10        # trailing exit channel
ATR_PERIOD         = 14
MAX_SPREAD_POINTS  = 40
SLIPPAGE           = 10

DAILY_LOSS_PCT     = 4.0       # halt for the day at -4% of start ref (this bot only)
TOTAL_LOSS_PCT     = 12.0      # halt entirely at -12%

MAGIC              = 778200
LOOP_SLEEP         = 5
# ==========================================================================

bot = Bot("ATLAS", MAGIC, "atlas")


def fetch(symbol):
    r = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, CANDLES)
    if r is None or len(r) < max(TREND_EMA, ENTRY_N, ATR_PERIOD) + 5:
        return None
    import pandas as pd
    df = pd.DataFrame(r)
    df["ema"] = ema(df["close"], TREND_EMA)
    df["atr"] = atr(df, ATR_PERIOD)
    df["ent_hi"], df["ent_lo"] = donchian(df, ENTRY_N)
    df["ex_hi"], df["ex_lo"] = donchian(df, EXIT_N)
    return df


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
    bot.log(f"POD ATLAS live (LOW risk). Ref balance {ref:.2f}. "
            f"Risk {RISK_PERCENT}%/trade | daily cap -{DAILY_LOSS_PCT}% | total cap -{TOTAL_LOSS_PCT}%.")
    bot.log(f"Pairs {', '.join(SYMBOLS)} | H4 Donchian({ENTRY_N}/{EXIT_N}) + EMA{TREND_EMA} filter | "
            f"SL {SL_ATR}xATR | winners run (no TP).")
    bot.log(f"Small-account mode: skipping any trade that would risk > {MAX_RISK_PER_TRADE_PCT}% "
            f"(~{MAX_RISK_PER_TRADE_PCT/100*ref:.2f}), max {MAX_CONCURRENT_POSITIONS} open at once.")

    last_bar = {s: None for s in SYMBOLS}

    try:
        while True:
            acct = mt5.account_info()
            if acct is None:
                time.sleep(LOOP_SLEEP); continue
            if st.get("over"):
                bot.log("Total loss cap hit earlier — standing down. (Delete atlas_state.json to reset.)")
                break

            pl = bot.own_pl(start_dt)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if st["day"] != today:
                st["day"] = today
                st["day_start_realized"] = pl["realized"]
                st["halted_today"] = False
                bot.save_state(st)

            day_pl = (pl["realized"] - st["day_start_realized"]) + pl["floating"]

            if pl["total"] <= -(TOTAL_LOSS_PCT / 100 * ref):
                bot.log(f"!!! TOTAL LOSS {pl['total']:.2f} (-{TOTAL_LOSS_PCT}%). Closing & stopping. !!!")
                bot.close_all(SLIPPAGE, "atlas_killswitch")
                st["over"] = True; bot.save_state(st); break
            if not st["halted_today"] and day_pl <= -(DAILY_LOSS_PCT / 100 * ref):
                bot.log(f"!! DAILY LOSS {day_pl:.2f} (-{DAILY_LOSS_PCT}%). Closing, no more today. !!")
                bot.close_all(SLIPPAGE, "atlas_daily")
                st["halted_today"] = True; bot.save_state(st)

            for s in SYMBOLS:
                info = mt5.symbol_info(s); infos[s] = info
                df = fetch(s)
                if df is None:
                    continue
                closed = df.iloc[-2]
                if closed["time"] == last_bar[s]:
                    continue
                last_bar[s] = closed["time"]

                if any(closed[c] != closed[c] for c in ("ema", "atr", "ent_hi", "ent_lo", "ex_hi", "ex_lo")):
                    continue
                cl = float(closed["close"]); a = float(closed["atr"])
                pos = bot.position(s)

                # ---- manage open trade: trail out on the opposite channel ----
                if pos is not None:
                    if pos.type == mt5.POSITION_TYPE_BUY and cl < closed["ex_lo"]:
                        bot.log(f"[{s}] close below {EXIT_N}-bar low -> trend exit (long).")
                        bot.close(pos, SLIPPAGE, "atlas_trail_exit")
                    elif pos.type == mt5.POSITION_TYPE_SELL and cl > closed["ex_hi"]:
                        bot.log(f"[{s}] close above {EXIT_N}-bar high -> trend exit (short).")
                        bot.close(pos, SLIPPAGE, "atlas_trail_exit")
                    continue

                if st["halted_today"] or a <= 0:
                    continue
                if len(bot.positions()) >= MAX_CONCURRENT_POSITIONS:
                    continue  # keep total exposure bounded on a small account

                risk_amt = RISK_PERCENT / 100 * acct.balance
                ceiling = MAX_RISK_PER_TRADE_PCT / 100 * acct.balance
                long_sig = cl > closed["ent_hi"] and cl > closed["ema"]
                short_sig = cl < closed["ent_lo"] and cl < closed["ema"]

                if long_sig and bot.spread_ok(info, MAX_SPREAD_POINTS):
                    sl = cl - SL_ATR * a
                    bot.log(f"[{s}] LONG breakout (>{ENTRY_N}-bar high, above EMA{TREND_EMA}).")
                    bot.place(info, "BUY", sl, 0.0, risk_amt, SLIPPAGE, "atlas", max_risk_amount=ceiling)
                elif short_sig and bot.spread_ok(info, MAX_SPREAD_POINTS):
                    sl = cl + SL_ATR * a
                    bot.log(f"[{s}] SHORT breakdown (<{ENTRY_N}-bar low, below EMA{TREND_EMA}).")
                    bot.place(info, "SELL", sl, 0.0, risk_amt, SLIPPAGE, "atlas", max_risk_amount=ceiling)

            _write_status(acct, pl, day_pl, ref)
            time.sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        bot.log("Manual stop (Ctrl+C). Open trades left running.")
    finally:
        mt5.shutdown()
        bot.log("MT5 closed.")


def _write_status(acct, pl, day_pl, ref):
    positions = [{"symbol": p.symbol, "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                  "volume": p.volume, "entry": p.price_open, "sl": p.sl, "tp": p.tp,
                  "price": p.price_current, "profit": round(p.profit, 2)} for p in bot.positions()]
    bot.write_status({
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "bot": "POD ATLAS", "profile": "LOW RISK / long-term",
        "account": {"login": acct.login, "server": acct.server, "currency": acct.currency,
                    "balance": round(acct.balance, 2), "equity": round(acct.equity, 2)},
        "own_pl": pl, "day_pl": round(day_pl, 2), "ref_balance": ref,
        "settings": {"risk_percent": RISK_PERCENT, "sl_atr": SL_ATR, "entry_n": ENTRY_N,
                     "exit_n": EXIT_N, "trend_ema": TREND_EMA, "pairs": SYMBOLS,
                     "daily_cap_pct": DAILY_LOSS_PCT, "total_cap_pct": TOTAL_LOSS_PCT},
        "positions": positions, "log": bot.buf[-40:],
    })


if __name__ == "__main__":
    main()
