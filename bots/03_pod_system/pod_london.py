"""
=============================================================================
 POD LONDON  —  automated core of the FTR "London Playbook" (GBP/USD)
=============================================================================
 An automation of the MECHANICAL parts of Paul Magee's London Playbook. It is
 NOT the full discretionary course — it codes the rules that can be coded:

   1. ASIAN RANGE: high/low of GBP/USD from 00:00-07:00 UK time (BST/GMT handled
      automatically). The spread hours (22:00-00:00) are excluded by design.
   2. DAILY ATR TARGET: 14-period ATR on the daily chart, ROUNDED DOWN to whole
      pips — the day's expected move and the take-profit anchor.
   3. ENTRIES (07:00-11:00 UK only — the "11am cutoff"), gated so they never
      contradict each other on the same candle:
        * FAKEOUT  -> CBIR: a candle wicks beyond the range but its body closes
          back inside  -> market fade. SL beyond the wick, TP = opposite range end.
        * BREAKOUT -> a candle's BODY closes beyond the range:
            - ORDER-BLOCK retracement (limit) into a valid 3-candle imbalance,
              SL beyond the block, TP = ATR projection. Needs >= MIN_RR (1:4).
            - if no valid block, IMPULSE (stop) just beyond the high/low to chase
              the continuation, SL behind the pullback (capped).
   4. SET & FORGET: broker-side SL/TP; the FRIDAY RULE flattens everything by
      ~21:30 UK so nothing is held over the weekend gap.

 NOT automated (these are discretionary in the course): scaling-in, manual
 structural trailing, and momentum-vs-exhaustion reads. Set & forget only.

 Run on a DEMO. Install once: pip install MetaTrader5 pandas numpy
 Keep pod_core.py in the same folder.   Run:  python pod_london.py
=============================================================================
"""
import sys
import time
import math
from datetime import datetime, timezone, timedelta

import MetaTrader5 as mt5
from pod_core import Bot, atr

# ===================== CONFIG =============================================
SYMBOL             = "GBPUSD"          # the Playbook is GBP/USD specific
TIMEFRAME          = mt5.TIMEFRAME_M5
CANDLES            = 360               # ~30h of M5 (covers the Asian session + London)

ASIA_START_UK      = (0, 0)            # 00:00 UK
ASIA_END_UK        = (7, 0)            # 07:00 UK (range closes)
ENTRY_CUTOFF_UK    = (11, 0)           # no NEW entries after 11:00 UK
PENDING_EXPIRE_UK  = (13, 0)           # cancel any unfilled pendings after this
FRIDAY_CLOSE_UK    = (21, 30)          # flatten everything by 21:30 UK Friday

RISK_PERCENT       = 1.0
MAX_RISK_PER_TRADE_PCT = 2.0           # small-account safety ceiling (skip oversized)
MIN_RR             = 4.0               # order-block entry must offer >= 1:4
CBIR_MIN_RR        = 1.5               # skip silly CBIR fades
MAX_IMPULSE_STOP_PIPS = 10            # impulse SL cap (GBP/USD M5)
BUFFER_PIPS        = 0.5               # buffer beyond wicks / blocks
ATR_PERIOD         = 14
MAX_TRADES_PER_DAY = 2
MAX_LOT            = 5.0     # hard cap: never open more than this many lots
MIN_STOP_PIPS      = 8       # skip trades whose stop is tighter than this (avoids
                             # tiny-stop, huge-lot, spread-murdered scalps)

USE_CBIR           = True
USE_ORDER_BLOCK    = True
USE_IMPULSE        = True

MAX_SPREAD_POINTS  = 30
SLIPPAGE           = 8
DAILY_LOSS_PCT     = 4.0
TOTAL_LOSS_PCT     = 12.0

MAGIC              = 778300
LOOP_SLEEP         = 2
# ==========================================================================

bot = Bot("LONDON", MAGIC, "london")


# ---------- UK time (manual BST/GMT, no tzdata dependency) ----------------
def _last_sunday(year, month):
    nxt = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
    d = nxt - timedelta(days=1)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def uk_offset_hours(dt_utc):
    start = _last_sunday(dt_utc.year, 3).replace(hour=1)
    end = _last_sunday(dt_utc.year, 10).replace(hour=1)
    return 1 if start <= dt_utc < end else 0


def to_uk(dt_utc):
    return dt_utc + timedelta(hours=uk_offset_hours(dt_utc))


def pip_size(info):
    return 10 * info.point if info.digits in (3, 5) else info.point


def fetch(offset_hours):
    import pandas as pd
    r = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, CANDLES)
    if r is None or len(r) < 60:
        return None
    df = pd.DataFrame(r)
    df["utc"] = pd.to_datetime(df["time"], unit="s") - pd.Timedelta(hours=offset_hours)
    return df


def daily_atr_pips(info):
    import pandas as pd
    r = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, ATR_PERIOD + 5)
    if r is None or len(r) < ATR_PERIOD + 1:
        return None
    df = pd.DataFrame(r)
    a = atr(df, ATR_PERIOD).iloc[-1]
    if a != a:  # NaN
        return None
    return math.floor(a / pip_size(info))   # round DOWN, in pips


def build_range(df, uk_date):
    rows = []
    for _, row in df.iterrows():
        u = to_uk(row["utc"].to_pydatetime().replace(tzinfo=timezone.utc))
        if u.date() == uk_date and ASIA_START_UK <= (u.hour, u.minute) < ASIA_END_UK:
            rows.append(row)
    if not rows:
        return None, None
    return max(r["high"] for r in rows), min(r["low"] for r in rows)


def find_order_block(df, bullish, pip):
    """Scan the last ~8 closed candles for a 3-candle imbalance order block.
    Bullish: a down candle C1 then a gap where high(C1) < low(C3). Returns
    (entry, sl) using the block edge, or None."""
    closed = df.iloc[:-1]            # exclude the forming candle
    n = len(closed)
    for i in range(n - 3, max(n - 11, 0) - 1, -1):
        c1, c3 = closed.iloc[i], closed.iloc[i + 2]
        if bullish:
            if c1["close"] < c1["open"] and c1["high"] < c3["low"]:   # bearish C1 + bullish gap
                return float(c1["high"]), float(c1["low"]) - BUFFER_PIPS * pip
        else:
            if c1["close"] > c1["open"] and c1["low"] > c3["high"]:   # bullish C1 + bearish gap
                return float(c1["low"]), float(c1["high"]) + BUFFER_PIPS * pip
    return None


def main():
    if not bot.connect():
        sys.exit(1)
    info = bot.prepare(SYMBOL)
    if info is None:
        mt5.shutdown(); sys.exit(1)
    pip = pip_size(info)
    offset = bot.broker_utc_offset(SYMBOL)

    acct = mt5.account_info()
    st = bot.load_state()
    if not st or st.get("ref_balance", 0) <= 0:
        st = {"start": datetime.now(timezone.utc).isoformat(), "ref_balance": acct.balance,
              "day": "", "day_start_realized": 0.0, "halted_today": False, "over": False}
        bot.save_state(st)
    bot.heal_ref(st, acct.balance)            # re-anchor a stale ref balance
    bot.max_lot = MAX_LOT                      # hard cap on position size
    start_dt = datetime.fromisoformat(st["start"]); ref = st["ref_balance"]

    bot.log(f"POD LONDON live. Ref {ref:.2f} {acct.currency}. Broker UTC{offset:+d}h.")
    bot.log(f"Asian range {ASIA_START_UK[0]:02d}:00-{ASIA_END_UK[0]:02d}:00 UK | "
            f"entries until {ENTRY_CUTOFF_UK[0]:02d}:00 UK | risk {RISK_PERCENT}% | "
            f"OB min {MIN_RR}:1 | Friday close {FRIDAY_CLOSE_UK[0]}:{FRIDAY_CLOSE_UK[1]:02d} UK.")
    ents = [n for n, on in [("CBIR", USE_CBIR), ("OrderBlock", USE_ORDER_BLOCK), ("Impulse", USE_IMPULSE)] if on]
    bot.log(f"Entries enabled: {', '.join(ents)}. Set & forget (no scaling/trailing).")

    day = {"date": None, "rh": None, "rl": None, "atr": None, "trades": 0, "last_bar": None}

    try:
        while True:
            acct = mt5.account_info()
            if acct is None:
                time.sleep(LOOP_SLEEP); continue
            if st.get("over"):
                bot.log("Total loss cap hit earlier — standing down. (Delete london_state.json to reset.)")
                break

            now_utc = datetime.now(timezone.utc)
            now_uk = to_uk(now_utc)
            today_uk = now_uk.date()

            # --- per-bot risk accounting + caps ---
            pl = bot.own_pl(start_dt)
            iso_day = today_uk.isoformat()
            if st["day"] != iso_day:
                st["day"] = iso_day; st["day_start_realized"] = pl["realized"]
                st["halted_today"] = False; bot.save_state(st)
            day_pl = (pl["realized"] - st["day_start_realized"]) + pl["floating"]
            if pl["total"] <= -(TOTAL_LOSS_PCT / 100 * ref):
                bot.log(f"!!! TOTAL LOSS {pl['total']:.2f}. Closing & stopping. !!!")
                bot.cancel_all_pending(); bot.close_all(SLIPPAGE, "london_kill")
                st["over"] = True; bot.save_state(st); break
            if not st["halted_today"] and day_pl <= -(DAILY_LOSS_PCT / 100 * ref):
                bot.log(f"!! DAILY LOSS {day_pl:.2f}. Closing, done for today. !!")
                bot.cancel_all_pending(); bot.close_all(SLIPPAGE, "london_daily")
                st["halted_today"] = True; bot.save_state(st)

            # --- Friday close rule ---
            if now_uk.weekday() == 4 and (now_uk.hour, now_uk.minute) >= FRIDAY_CLOSE_UK:
                if bot.positions() or bot.pending_orders():
                    bot.log("Friday close — flattening everything before the weekend gap.")
                    bot.cancel_all_pending(); bot.close_all(SLIPPAGE, "london_friday")

            # --- new UK day: reset ---
            if day["date"] != today_uk:
                if day["date"] is not None:
                    bot.cancel_all_pending()
                day.update(date=today_uk, rh=None, rl=None, atr=None, trades=0, last_bar=None)

            df = fetch(offset)
            if df is None:
                time.sleep(LOOP_SLEEP); continue

            past_asia = (now_uk.hour, now_uk.minute) >= ASIA_END_UK
            before_cutoff = (now_uk.hour, now_uk.minute) < ENTRY_CUTOFF_UK

            # --- build the Asian range once it has closed ---
            if day["rh"] is None and past_asia:
                rh, rl = build_range(df, today_uk)
                if rh is not None and rl > 0 and rh > rl:
                    day["rh"], day["rl"] = rh, rl
                    day["atr"] = daily_atr_pips(info)
                    width = (rh - rl) / pip
                    bot.log(f"Asian range set: HIGH={rh:.{info.digits}f} LOW={rl:.{info.digits}f} "
                            f"(width {width:.0f} pips). Daily ATR target {day['atr']} pips.")

            # --- cancel stale pendings after the expire time ---
            if (now_uk.hour, now_uk.minute) >= PENDING_EXPIRE_UK and bot.pending_orders():
                bot.log("Past pending-expire time — cancelling unfilled orders.")
                bot.cancel_all_pending()

            # --- work only on a newly CLOSED M5 candle ---
            closed = df.iloc[-2]
            if day["rh"] is not None and closed["time"] != day["last_bar"]:
                day["last_bar"] = closed["time"]
                _evaluate(info, pip, day, acct, closed)

            _write_status(acct, pl, day_pl, ref, day, info)
            time.sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        bot.log("Manual stop (Ctrl+C). Open trades/pendings left as-is.")
    finally:
        mt5.shutdown(); bot.log("MT5 closed.")


def _evaluate(info, pip, day, acct, closed):
    """Decide entries on a freshly closed M5 candle."""
    if st_halted() or day["trades"] >= MAX_TRADES_PER_DAY:
        return
    if bot.positions() or bot.pending_orders():
        return  # one setup at a time (set & forget)
    now_uk = to_uk(datetime.now(timezone.utc))
    if (now_uk.hour, now_uk.minute) >= ENTRY_CUTOFF_UK:
        return
    if not bot.spread_ok(info, MAX_SPREAD_POINTS):
        return

    rh, rl, atr_pips = day["rh"], day["rl"], day["atr"]
    if atr_pips is None or atr_pips <= 0:
        return
    o, h, l, c = (float(closed[k]) for k in ("open", "high", "low", "close"))
    buf = BUFFER_PIPS * pip
    atr_dist = atr_pips * pip
    risk_amt = RISK_PERCENT / 100 * acct.balance
    ceil = MAX_RISK_PER_TRADE_PCT / 100 * acct.balance

    # ---- FAKEOUT: CBIR (body closes back inside after a wick beyond) ----
    if USE_CBIR:
        if h > rh and rl < c < rh:                      # faked the high
            sl = h + buf; tp = rl                       # fade short, target opposite end
            if _rr(c, sl, tp) >= CBIR_MIN_RR and _stop_ok(c, sl, pip):
                bot.log(f"CBIR fake-high -> SHORT @mkt, SL {sl:.{info.digits}f} TP {tp:.{info.digits}f}.")
                if bot.place(info, "SELL", sl, tp, risk_amt, SLIPPAGE, "london_cbir", ceil):
                    day["trades"] += 1
            return
        if l < rl and rl < c < rh:                      # faked the low
            sl = l - buf; tp = rh
            if _rr(c, sl, tp) >= CBIR_MIN_RR and _stop_ok(c, sl, pip):
                bot.log(f"CBIR fake-low -> LONG @mkt, SL {sl:.{info.digits}f} TP {tp:.{info.digits}f}.")
                if bot.place(info, "BUY", sl, tp, risk_amt, SLIPPAGE, "london_cbir", ceil):
                    day["trades"] += 1
            return

    # ---- BREAKOUT: body close beyond the range ----
    bull = min(o, c) > rh
    bear = max(o, c) < rl
    if not (bull or bear):
        return

    df = fetch(bot.broker_utc_offset(SYMBOL))
    if bull:
        tp = rh + atr_dist
        if USE_ORDER_BLOCK:
            ob = find_order_block(df, True, pip)
            if ob:
                entry, sl = ob
                if _rr(entry, sl, tp) >= MIN_RR:
                    bot.log(f"Body break UP -> ORDER-BLOCK long limit @ {entry:.{info.digits}f}.")
                    if bot.place_pending(info, "BUY_LIMIT", entry, sl, tp, risk_amt, SLIPPAGE, "london_ob", ceil):
                        day["trades"] += 1
                    return
        if USE_IMPULSE:
            sl = _impulse_stop(df, True, pip)
            entry = h + buf
            if sl is not None and (entry - sl) <= MAX_IMPULSE_STOP_PIPS * pip and _rr(entry, sl, tp) >= MIN_RR:
                bot.log(f"Body break UP -> IMPULSE buy-stop @ {entry:.{info.digits}f}.")
                if bot.place_pending(info, "BUY_STOP", entry, sl, tp, risk_amt, SLIPPAGE, "london_impulse", ceil):
                    day["trades"] += 1
    else:
        tp = rl - atr_dist
        if USE_ORDER_BLOCK:
            ob = find_order_block(df, False, pip)
            if ob:
                entry, sl = ob
                if _rr(entry, sl, tp) >= MIN_RR:
                    bot.log(f"Body break DOWN -> ORDER-BLOCK short limit @ {entry:.{info.digits}f}.")
                    if bot.place_pending(info, "SELL_LIMIT", entry, sl, tp, risk_amt, SLIPPAGE, "london_ob", ceil):
                        day["trades"] += 1
                    return
        if USE_IMPULSE:
            sl = _impulse_stop(df, False, pip)
            entry = l - buf
            if sl is not None and (sl - entry) <= MAX_IMPULSE_STOP_PIPS * pip and _rr(entry, sl, tp) >= MIN_RR:
                bot.log(f"Body break DOWN -> IMPULSE sell-stop @ {entry:.{info.digits}f}.")
                if bot.place_pending(info, "SELL_STOP", entry, sl, tp, risk_amt, SLIPPAGE, "london_impulse", ceil):
                    day["trades"] += 1


def _impulse_stop(df, bullish, pip):
    """SL behind the recent pullback (last 3 closed candles' extreme) + buffer."""
    closed = df.iloc[:-1]
    last3 = closed.iloc[-3:]
    if bullish:
        return float(last3["low"].min()) - BUFFER_PIPS * pip
    return float(last3["high"].max()) + BUFFER_PIPS * pip


def _rr(entry, sl, tp):
    risk = abs(entry - sl)
    return abs(tp - entry) / risk if risk > 0 else 0.0


def _stop_ok(entry, sl, pip):
    return abs(entry - sl) / pip >= MIN_STOP_PIPS


def st_halted():
    s = bot.load_state()
    return s.get("halted_today", False)


def _write_status(acct, pl, day_pl, ref, day, info):
    positions = [{"symbol": p.symbol, "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                  "volume": p.volume, "entry": p.price_open, "sl": p.sl, "tp": p.tp,
                  "price": p.price_current, "profit": round(p.profit, 2)} for p in bot.positions()]
    pend = [{"ticket": o.ticket, "type": o.type, "price": o.price_open,
             "sl": o.sl, "tp": o.tp, "volume": o.volume_current} for o in bot.pending_orders()]
    bot.write_status({
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "bot": "POD LONDON", "profile": "FTR London Playbook (auto core)",
        "account": {"login": acct.login, "server": acct.server, "currency": acct.currency,
                    "balance": round(acct.balance, 2), "equity": round(acct.equity, 2)},
        "own_pl": pl, "day_pl": round(day_pl, 2), "ref_balance": ref,
        "range": {"high": day["rh"], "low": day["rl"], "atr_pips": day["atr"], "trades_today": day["trades"]},
        "settings": {"risk_percent": RISK_PERCENT, "min_rr": MIN_RR, "pairs": [SYMBOL],
                     "daily_cap_pct": DAILY_LOSS_PCT, "total_cap_pct": TOTAL_LOSS_PCT},
        "positions": positions, "pending": pend, "log": bot.buf[-40:],
    })


if __name__ == "__main__":
    main()
