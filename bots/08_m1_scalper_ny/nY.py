"""
NY Breakout PRO – Full version with Account Selector, CSV Logging & Pattern Filters
- Trades NY high/low breakout during Asian session
- Fixed 5 lots, 1:1.5 R:R, BE at 1R
- Optional: ATR filter, Body-break confirmation, Volume spike
- Logs every trade to trades_log.csv
- Explicit MT5 account selection
"""

import sys
import time
import csv
import os
import math
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
import pandas as pd
from pod_core import Bot

# ========== CHOOSE YOUR MT5 ACCOUNT ==========
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0")) or None          # <-- YOUR LOGIN
MT5_PASSWORD = os.getenv("MT5_PASSWORD")   # <-- YOUR PASSWORD
MT5_SERVER   = os.getenv("MT5_SERVER", "MetaQuotes-Demo") # <-- YOUR SERVER
# =============================================

# ============ STRATEGY CONFIG ============
SYMBOL = "GBPUSD"
TIMEFRAME = mt5.TIMEFRAME_M5

NY_START_HOUR = 13      # 13:00 UTC
NY_END_HOUR = 22        # 22:00 UTC
ASIA_START_HOUR = 22    # 22:00 UTC
ASIA_END_HOUR = 8       # 08:00 UTC

FIXED_LOT = 5.0
RR_RATIO = 1.5
BUFFER_PIPS = 0.5

# -------- PATTERN FILTERS (toggle) --------
USE_ATR_FILTER = True           # Skip if NY range > 65% of daily ATR
USE_BODY_BREAK = True           # Wait for a candle to CLOSE beyond the range
USE_VOLUME_SPIKE = True         # Breakout candle volume must be > 1.3x average
ATR_PERIOD = 14
MAX_RANGE_PCT_OF_ATR = 0.65     # 65%
VOLUME_MULTIPLIER = 1.3         # 1.3x average volume
# -----------------------------------------

MAGIC = 778302                  # Change this if running multiple bots
LOOP_SLEEP = 2
MAX_SPREAD_POINTS = 30
SLIPPAGE = 8
# ==========================================

bot = Bot("NY_PRO", MAGIC, "ny_pro")

def pip_size(info):
    return 10 * info.point if info.digits in (3, 5) else info.point

def get_ny_range(info):
    """Return (high, low) of the most recent NY session (13:00-22:00 UTC)."""
    now = datetime.now(timezone.utc)
    if now.hour >= NY_END_HOUR:
        session_date = now.date()
    else:
        session_date = (now - timedelta(days=1)).date()

    start_dt = datetime(session_date.year, session_date.month, session_date.day,
                        NY_START_HOUR, 0, tzinfo=timezone.utc)
    end_dt = datetime(session_date.year, session_date.month, session_date.day,
                      NY_END_HOUR, 0, tzinfo=timezone.utc)

    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return None, None
    df = pd.DataFrame(rates)
    return float(df['high'].max()), float(df['low'].min())

def get_daily_atr(info):
    """Return 14-period ATR in price units."""
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, ATR_PERIOD + 5)
    if rates is None or len(rates) < ATR_PERIOD:
        return None
    df = pd.DataFrame(rates)
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else None

def get_avg_volume():
    """Average tick volume over last 20 M5 candles."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 20)
    if rates is None or len(rates) < 20:
        return None
    return sum(r[5] for r in rates) / len(rates)  # tick volume is index 5

def log_trade_to_csv(data):
    """Append a trade record to trades_log.csv."""
    file_exists = os.path.isfile("trades_log.csv")
    with open("trades_log.csv", mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Date", "Symbol", "Direction", "Entry", "SL", "TP",
                "Exit_Date", "Exit_Price", "Profit_Pips", "Profit_USD",
                "R_Multiple", "Reason"
            ])
        writer.writerow([
            data['date'], data['symbol'], data['direction'],
            data['entry'], data['sl'], data['tp'],
            data['exit_date'], data['exit_price'],
            round(data['profit_pips'], 1), round(data['profit_usd'], 2),
            round(data['r_multiple'], 2), data['reason']
        ])

def main():
    # --- Explicitly connect to MT5 with YOUR account ---
    if not mt5.initialize():
        print("MT5 initialization failed")
        sys.exit(1)
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        print(f"Login failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    # --- Attach the bot (it uses the already-logged-in MT5) ---
    if not bot.connect():
        mt5.shutdown()
        sys.exit(1)

    info = bot.prepare(SYMBOL)
    if info is None:
        mt5.shutdown()
        sys.exit(1)
    pip = pip_size(info)

    bot.log(f"NY PRO started. Filters: ATR={USE_ATR_FILTER}, BodyBreak={USE_BODY_BREAK}, Volume={USE_VOLUME_SPIKE}")
    bot.log(f"Account: {MT5_LOGIN} @ {MT5_SERVER}")

    # Load persistent state
    state = bot.load_state()
    if not state:
        state = {
            "date": "",
            "ny_high": None,
            "ny_low": None,
            "traded_today": False,
            "logged_exit": False,
            "open_ticket": None,
            "entry_price": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0
        }
        bot.save_state(state)

    while True:
        now = datetime.now(timezone.utc)
        current_date = now.date().isoformat()

        # ----- Update NY range at the start of each new day (after 22:00 UTC) -----
        if state["date"] != current_date and now.hour >= NY_END_HOUR:
            h, l = get_ny_range(info)
            if h is not None and l is not None:
                # ---- ATR filter check ----
                if USE_ATR_FILTER:
                    atr = get_daily_atr(info)
                    if atr and (h - l) > (MAX_RANGE_PCT_OF_ATR * atr):
                        bot.log(f"SKIP: NY range width {(h-l)/pip:.0f} pips > {MAX_RANGE_PCT_OF_ATR*100:.0f}% of ATR ({atr/pip:.0f} pips)")
                        # Mark as traded today so we don't try again
                        state["date"] = current_date
                        state["ny_high"] = h
                        state["ny_low"] = l
                        state["traded_today"] = True
                        bot.save_state(state)
                        time.sleep(LOOP_SLEEP)
                        continue

                state["date"] = current_date
                state["ny_high"] = h
                state["ny_low"] = l
                state["traded_today"] = False
                state["logged_exit"] = False
                state["open_ticket"] = None
                bot.save_state(state)
                bot.log(f"NY range set: HIGH={h:.{info.digits}f} LOW={l:.{info.digits}f} width={(h-l)/pip:.1f} pips")
            else:
                bot.log("Failed to fetch NY range, retrying...")

        # ----- Manage Break-Even & Detect closed trades for logging -----
        positions = bot.positions()
        if positions:
            pos = positions[0]  # only one trade at a time
            state["open_ticket"] = pos.ticket
            bot.save_state(state)

            # BE logic: move SL to entry when profit >= 1R
            if pos.type == mt5.POSITION_TYPE_BUY:
                profit_pips = (pos.price_current - pos.price_open) / pip
                risk_pips = (pos.price_open - pos.sl) / pip if pos.sl else None
            else:
                profit_pips = (pos.price_open - pos.price_current) / pip
                risk_pips = (pos.sl - pos.price_open) / pip if pos.sl else None

            if risk_pips and profit_pips >= risk_pips and pos.sl != pos.price_open:
                new_sl = pos.price_open
                # Add a tiny buffer to avoid immediate stop-out
                if pos.type == mt5.POSITION_TYPE_BUY:
                    new_sl += 0.5 * pip
                else:
                    new_sl -= 0.5 * pip

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    bot.log(f"BE: SL moved to entry for pos {pos.ticket}")
                else:
                    bot.log(f"BE move failed: {result.comment}")

        else:
            # No position. If we just closed one, log it.
            if state["open_ticket"] is not None and not state["logged_exit"]:
                deals = mt5.history_deals_get(position=state["open_ticket"])
                if deals and len(deals) > 0:
                    deal = deals[-1]  # closing deal
                    profit_usd = deal.profit
                    entry = state.get("entry_price", 0.0)
                    sl = state.get("sl_price", 0.0)
                    tp = state.get("tp_price", 0.0)
                    if entry and sl:
                        risk = abs(entry - sl)
                        if deal.type == mt5.DEAL_TYPE_BUY:
                            profit_pips = (deal.price - entry) / pip
                            direction = "LONG"
                        else:
                            profit_pips = (entry - deal.price) / pip
                            direction = "SHORT"
                        r_mult = profit_pips / (risk / pip) if risk > 0 else 0
                        reason = "TP_HIT" if profit_pips > 0 else "SL_HIT"
                        # If profit is tiny (BE trade), mark it as BE
                        if abs(profit_pips) < 0.5:
                            reason = "BREAK_EVEN"

                        log_trade_to_csv({
                            "date": datetime.fromtimestamp(deal.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            "symbol": SYMBOL,
                            "direction": direction,
                            "entry": round(entry, info.digits),
                            "sl": round(sl, info.digits),
                            "tp": round(tp, info.digits),
                            "exit_date": datetime.fromtimestamp(deal.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            "exit_price": round(deal.price, info.digits),
                            "profit_pips": profit_pips,
                            "profit_usd": profit_usd,
                            "r_multiple": r_mult,
                            "reason": reason
                        })
                        bot.log(f"LOGGED trade: {reason}, R={r_mult:.2f}, P/L=${profit_usd:.2f}")
                state["logged_exit"] = True
                state["open_ticket"] = None
                bot.save_state(state)

        # ----- Cancel pending orders if outside Asian session -----
        if not (ASIA_START_HOUR <= now.hour or now.hour < ASIA_END_HOUR):
            if bot.pending_orders():
                bot.cancel_all_pending()
                bot.log("Cancelled pending orders (outside Asian session).")

        # ----- Place new pending orders (with pattern filters) -----
        if not state["traded_today"] and state["ny_high"] is not None and state["ny_low"] is not None:
            if (ASIA_START_HOUR <= now.hour or now.hour < ASIA_END_HOUR):
                if not bot.pending_orders() and not bot.positions():
                    if not bot.spread_ok(info, MAX_SPREAD_POINTS):
                        time.sleep(LOOP_SLEEP)
                        continue

                    h = state["ny_high"]
                    l = state["ny_low"]
                    buf = BUFFER_PIPS * pip

                    # ---- BODY BREAK CONFIRMATION (if enabled) ----
                    if USE_BODY_BREAK:
                        # Get last 3 closed M5 candles
                        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 3)
                        if rates is None or len(rates) < 2:
                            time.sleep(LOOP_SLEEP)
                            continue
                        last_candle = rates[-2]  # most recently closed candle
                        close = float(last_candle[4])
                        high = float(last_candle[2])
                        low = float(last_candle[3])
                        vol = float(last_candle[5])

                        # Check if it closed OUTSIDE the range
                        if close > h + buf:   # Bullish body break
                            # ---- VOLUME SPIKE FILTER ----
                            if USE_VOLUME_SPIKE:
                                avg_vol = get_avg_volume()
                                if avg_vol and vol < (VOLUME_MULTIPLIER * avg_vol):
                                    bot.log(f"SKIP: Volume {vol:.0f} < {VOLUME_MULTIPLIER:.1f}x avg ({avg_vol:.0f})")
                                    state["traded_today"] = True
                                    bot.save_state(state)
                                    time.sleep(LOOP_SLEEP)
                                    continue

                            entry_price = high + buf
                            sl_price = l - buf
                            tp_price = entry_price + RR_RATIO * (entry_price - sl_price)
                            order_type = mt5.ORDER_TYPE_BUY_STOP
                            comment = "NY_BodyLong"
                            state["entry_price"] = entry_price
                            state["sl_price"] = sl_price
                            state["tp_price"] = tp_price

                        elif close < l - buf:  # Bearish body break
                            if USE_VOLUME_SPIKE:
                                avg_vol = get_avg_volume()
                                if avg_vol and vol < (VOLUME_MULTIPLIER * avg_vol):
                                    bot.log(f"SKIP: Volume {vol:.0f} < {VOLUME_MULTIPLIER:.1f}x avg ({avg_vol:.0f})")
                                    state["traded_today"] = True
                                    bot.save_state(state)
                                    time.sleep(LOOP_SLEEP)
                                    continue

                            entry_price = low - buf
                            sl_price = h + buf
                            tp_price = entry_price - RR_RATIO * (sl_price - entry_price)
                            order_type = mt5.ORDER_TYPE_SELL_STOP
                            comment = "NY_BodyShort"
                            state["entry_price"] = entry_price
                            state["sl_price"] = sl_price
                            state["tp_price"] = tp_price

                        else:
                            # No body break; wait for next candle
                            time.sleep(LOOP_SLEEP)
                            continue

                        # Place the single pending order
                        req = {
                            "action": mt5.TRADE_ACTION_PENDING,
                            "symbol": SYMBOL,
                            "volume": FIXED_LOT,
                            "type": order_type,
                            "price": entry_price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "deviation": SLIPPAGE,
                            "magic": MAGIC,
                            "comment": comment,
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_FOK,
                        }
                        res = mt5.order_send(req)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            bot.save_state(state)
                            bot.log(f"Placed {comment} pending at {entry_price:.{info.digits}f}")
                        else:
                            bot.log(f"Pending failed: {res.comment}")

                    else:
                        # ---- LEGACY: Dual pending orders (both directions) ----
                        long_entry = h + buf
                        long_sl = l - buf
                        long_tp = long_entry + RR_RATIO * (long_entry - long_sl)

                        short_entry = l - buf
                        short_sl = h + buf
                        short_tp = short_entry - RR_RATIO * (short_sl - short_entry)

                        buy_req = {
                            "action": mt5.TRADE_ACTION_PENDING,
                            "symbol": SYMBOL,
                            "volume": FIXED_LOT,
                            "type": mt5.ORDER_TYPE_BUY_STOP,
                            "price": long_entry,
                            "sl": long_sl,
                            "tp": long_tp,
                            "deviation": SLIPPAGE,
                            "magic": MAGIC,
                            "comment": "NY_BreakLong",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_FOK,
                        }
                        sell_req = {
                            "action": mt5.TRADE_ACTION_PENDING,
                            "symbol": SYMBOL,
                            "volume": FIXED_LOT,
                            "type": mt5.ORDER_TYPE_SELL_STOP,
                            "price": short_entry,
                            "sl": short_sl,
                            "tp": short_tp,
                            "deviation": SLIPPAGE,
                            "magic": MAGIC,
                            "comment": "NY_BreakShort",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_FOK,
                        }
                        res1 = mt5.order_send(buy_req)
                        res2 = mt5.order_send(sell_req)
                        if res1.retcode == mt5.TRADE_RETCODE_DONE or res2.retcode == mt5.TRADE_RETCODE_DONE:
                            bot.log("Placed dual pending orders (BUY_STOP & SELL_STOP).")
                            # Store entry/sl for logging
                            state["entry_price"] = long_entry
                            state["sl_price"] = long_sl
                            state["tp_price"] = long_tp
                            bot.save_state(state)
                        else:
                            bot.log(f"Pending orders failed: {res1.comment} / {res2.comment}")

        # ----- If a position exists, mark as traded and cancel opposite pending -----
        if bot.positions() and not state["traded_today"]:
            bot.cancel_all_pending()
            state["traded_today"] = True
            bot.save_state(state)
            bot.log("Trade filled, cancelled opposite pending order.")

        # ----- Status output -----
        acct = mt5.account_info()
        if acct:
            bot.write_status({
                "updated": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "bot": "NY PRO",
                "filters": {"ATR": USE_ATR_FILTER, "BodyBreak": USE_BODY_BREAK, "Volume": USE_VOLUME_SPIKE},
                "account": {"login": acct.login, "balance": round(acct.balance,2), "equity": round(acct.equity,2)},
                "state": state,
                "positions": [{"ticket": p.ticket, "profit": round(p.profit,2)} for p in bot.positions()],
                "pending": [{"ticket": o.ticket, "price": o.price_open} for o in bot.pending_orders()],
                "log": bot.buf[-20:],
            })

        time.sleep(LOOP_SLEEP)

if __name__ == "__main__":
    main()