"""
GOLD SESSION PRO v1.0  —  Asian Range Breakout (London/NY sessions)
====================================================================
Replaces GOLD v3. Philosophy: FEW trades, WIDE stops, LET WINNERS RUN.

What changed vs v3 and why:
  - ONE strategy (session breakout + H1 trend filter), not 5 fighting signals
  - Evaluates once per closed M15 bar, not every 2 seconds  -> no re-firing
  - Correct lot sizing from broker tick_value/tick_size     -> v3 was off 10x
  - ATR-based stop (1.2x ATR M15), broker stops_level aware -> no $0.12 trails
  - Breakeven at +1R, then 2x-ATR chandelier trail          -> winners can run
  - Max 3 trades/day, 1 per direction, 30-min cooldown after a loss
  - Fixed Asian-range window bug (00:00-07:00 UTC, correct day handling)
  - Daily loss halt (2%), Friday flatten, spread filter, filling-mode detect
  - No credentials in source: uses env vars or an already-running terminal

Run: keep MT5 terminal open and logged in, then  python gold_session_pro.py
State file:  gold_pro2_state.json   Status file: gold_pro2_status.json
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone, date

import MetaTrader5 as mt5
import pandas as pd

# ========================= CONFIG =========================
SYMBOL              = "XAUUSD"
MAGIC               = 778700
TAG                 = "gold_pro2"

# --- Risk ---
RISK_PER_TRADE_PCT  = 0.75      # % of balance risked per trade
MAX_DAILY_LOSS_PCT  = 2.0       # halt for the day beyond this equity drawdown
MAX_TRADES_PER_DAY  = 3
MAX_CONCURRENT      = 1         # one position at a time. Breakout bots do not stack.
MIN_LOT             = 0.01
MAX_LOT             = 2.0       # hard cap regardless of risk math
COOLDOWN_MIN        = 30        # minutes to stand down after a losing trade

# --- Strategy ---
ASIA_START_H        = 0         # Asian range window, UTC
ASIA_END_H          = 7
LONDON_WINDOW       = (7, 11)   # UTC hours during which London breakouts may trigger
NY_WINDOW           = (12, 16)  # UTC hours for NY breakouts
FRIDAY_FLAT_UTC     = (20, 45)  # flatten & halt Friday at 20:45 UTC

TF_ENTRY            = mt5.TIMEFRAME_M15
TF_BIAS             = mt5.TIMEFRAME_H1
ATR_PERIOD          = 14
EMA_BIAS_PERIOD     = 50        # H1 trend filter
BREAK_BUFFER_ATR    = 0.25      # close must exceed range by this * ATR
SL_ATR_MULT         = 1.2       # initial stop = 1.2 x ATR(M15)
TP_R_MULT           = 2.0       # take profit at 2R
BE_TRIGGER_R        = 1.0       # move stop to breakeven at +1R
TRAIL_ATR_MULT      = 2.0       # chandelier trail distance after breakeven
MIN_RANGE_USD       = 4.0       # skip days with a degenerate Asian range
MAX_RANGE_USD       = 45.0      # skip days where the range already IS the move

# --- Execution ---
MAX_SPREAD_POINTS   = 60        # $0.60 on 2-digit gold; tighten if broker allows
SLIPPAGE_POINTS     = 30
LOOP_SLEEP          = 5
# ==========================================================

log_buf = []

def log(msg):
    line = f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}"
    print(line)
    log_buf.append(line)
    if len(log_buf) > 200:
        log_buf.pop(0)

# ------------------------- MT5 plumbing -------------------------

def init_mt5():
    if mt5.terminal_info() is not None:
        log("MT5 already connected")
        return True
    if not mt5.initialize():
        log(f"MT5 init failed: {mt5.last_error()}")
        return False
    login = os.environ.get("MT5_LOGIN")
    pwd = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    if login and pwd and server:
        if not mt5.login(int(login), pwd, server):
            log(f"Login failed: {mt5.last_error()}")
            return False
    acct = mt5.account_info()
    if acct is None:
        log("No account attached. Log into the terminal manually or set MT5_LOGIN/MT5_PASSWORD/MT5_SERVER env vars.")
        return False
    log(f"Connected | {acct.login} | Balance ${acct.balance:,.2f}")
    return True

def sym():
    info = mt5.symbol_info(SYMBOL)
    if info is None or not info.visible:
        mt5.symbol_select(SYMBOL, True)
        info = mt5.symbol_info(SYMBOL)
    return info

def detect_filling(info):
    """Pick a filling mode the broker actually supports."""
    fm = info.filling_mode
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def bars(tf, count):
    r = mt5.copy_rates_from_pos(SYMBOL, tf, 0, count)
    if r is None or len(r) < count:
        return None
    return pd.DataFrame(r)

def atr(tf, period=ATR_PERIOD):
    df = bars(tf, period + 10)
    if df is None:
        return None
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-2]   # use last CLOSED bar
    return float(v) if not pd.isna(v) else None

def h1_bias():
    """+1 bullish, -1 bearish, 0 no trade. Price vs EMA50 on H1, using closed bars."""
    df = bars(TF_BIAS, EMA_BIAS_PERIOD + 20)
    if df is None:
        return 0
    ema = df["close"].ewm(span=EMA_BIAS_PERIOD, adjust=False).mean()
    close, e = float(df["close"].iloc[-2]), float(ema.iloc[-2])
    if close > e:
        return 1
    if close < e:
        return -1
    return 0

def asian_range(now):
    """High/low of today's 00:00-07:00 UTC window. Returns (high, low) or (None, None)."""
    start = datetime(now.year, now.month, now.day, ASIA_START_H, tzinfo=timezone.utc)
    end = datetime(now.year, now.month, now.day, ASIA_END_H, tzinfo=timezone.utc)
    if now < end:                      # session not finished yet
        return None, None
    r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, start, end)
    if r is None or len(r) == 0:
        return None, None
    df = pd.DataFrame(r)
    return float(df["high"].max()), float(df["low"].min())

# ------------------------- Sizing (correct math) -------------------------

def lot_for_risk(info, sl_distance_price, balance):
    """
    Lots such that hitting SL loses RISK_PER_TRADE_PCT of balance.
    Uses the broker's own tick economics -- no hardcoded pip values.
    """
    tick_val = info.trade_tick_value
    tick_size = info.trade_tick_size
    if not tick_val or not tick_size or sl_distance_price <= 0:
        return MIN_LOT
    loss_per_lot = (sl_distance_price / tick_size) * tick_val
    risk_amount = balance * RISK_PER_TRADE_PCT / 100.0
    lots = risk_amount / loss_per_lot
    step = info.volume_step or 0.01
    lots = max(info.volume_min or MIN_LOT, min(lots, info.volume_max or MAX_LOT, MAX_LOT))
    lots = round(lots / step) * step
    return round(max(lots, MIN_LOT), 2)

def min_stop_distance(info):
    """Broker-enforced minimum stop distance in price units."""
    return max(info.trade_stops_level, 0) * info.point

# ------------------------- Orders -------------------------

def send_market(info, direction, sl, tp, comment):
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        return False
    price = tick.ask if direction == "BUY" else tick.bid
    sl_dist = abs(price - sl)
    if sl_dist < min_stop_distance(info):
        log(f"Skip: SL {sl_dist:.2f} inside broker stops_level")
        return False
    balance = mt5.account_info().balance
    lots = lot_for_risk(info, sl_dist, balance)
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": round(sl, info.digits),
        "tp": round(tp, info.digits),
        "deviation": SLIPPAGE_POINTS,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": detect_filling(info),
    }
    res = mt5.order_send(req)
    if res is None:
        log(f"order_send returned None: {mt5.last_error()}")
        return False
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"{direction} {lots} @ {price:.2f} | SL {sl:.2f} | TP {tp:.2f} | risk {RISK_PER_TRADE_PCT}% | {comment}")
        return True
    log(f"Order failed [{res.retcode}] {res.comment}")
    return False

def modify_sl(info, pos, new_sl):
    new_sl = round(new_sl, info.digits)
    if abs(new_sl - pos.sl) < info.point * 5:   # don't spam micro-modifications
        return
    req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket,
           "symbol": SYMBOL, "sl": new_sl, "tp": pos.tp}
    mt5.order_send(req)

def close_position(info, pos, comment="halt"):
    tick = mt5.symbol_info_tick(SYMBOL)
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "position": pos.ticket,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if is_buy else tick.ask,
        "deviation": SLIPPAGE_POINTS, "magic": MAGIC, "comment": comment,
        "type_filling": detect_filling(info),
    }
    mt5.order_send(req)

def my_positions():
    return [p for p in (mt5.positions_get(symbol=SYMBOL) or []) if p.magic == MAGIC]

def last_closed_deal_today(day_start):
    """(close_time, profit) of my most recent closed deal today, else (None, 0)."""
    deals = mt5.history_deals_get(day_start, datetime.now(timezone.utc) + timedelta(minutes=1))
    if not deals:
        return None, 0.0
    mine = [d for d in deals if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT]
    if not mine:
        return None, 0.0
    d = max(mine, key=lambda x: x.time)
    return datetime.fromtimestamp(d.time, tz=timezone.utc), d.profit + d.commission + d.swap

# ------------------------- Position management -------------------------

def manage(info, pos, atr_val):
    """Breakeven at +1R, then chandelier trail at TRAIL_ATR_MULT * ATR."""
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    entry, cur = pos.price_open, pos.price_current
    # reconstruct initial risk from current SL if pre-BE, else from comment-free heuristic:
    # we store initial risk implicitly: before BE, |entry - sl| IS the risk.
    risk = abs(entry - pos.sl) if pos.sl else None
    profit_price = (cur - entry) if is_buy else (entry - cur)

    at_or_past_be = (is_buy and pos.sl >= entry) or ((not is_buy) and pos.sl and pos.sl <= entry)

    if not at_or_past_be:
        if risk and profit_price >= BE_TRIGGER_R * risk:
            spread = mt5.symbol_info_tick(SYMBOL).ask - mt5.symbol_info_tick(SYMBOL).bid
            be = entry + spread if is_buy else entry - spread
            modify_sl(info, pos, be)
            log(f"Breakeven set on #{pos.ticket}")
        return

    # past breakeven: chandelier trail
    if atr_val:
        trail = cur - TRAIL_ATR_MULT * atr_val if is_buy else cur + TRAIL_ATR_MULT * atr_val
        if (is_buy and trail > pos.sl) or ((not is_buy) and trail < pos.sl):
            modify_sl(info, pos, trail)

# ------------------------- State -------------------------

def load_state():
    try:
        with open(f"{TAG}_state.json") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(s):
    with open(f"{TAG}_state.json", "w") as f:
        json.dump(s, f, indent=2)

def fresh_day(balance):
    return {
        "day": date.today().isoformat(),
        "start_balance": balance,
        "halted": False,
        "trades_today": 0,
        "long_taken": False,
        "short_taken": False,
        "last_bar": 0,
    }

# ------------------------- Main -------------------------

# Session toggles. NY is OFF: by the NY session the Asian range is ~12h stale
# (London has usually already broken it), so a NY breakout of the ASIAN high/low
# is a late, extended signal. Backtest confirmed a negative edge there (PF 0.28).
# Only flip TRADE_NY back on if you re-validate it on more data.
TRADE_LONDON = True
TRADE_NY = False

def in_window(now):
    h = now.hour + now.minute / 60
    if TRADE_LONDON and LONDON_WINDOW[0] <= h < LONDON_WINDOW[1]:
        return "LONDON"
    if TRADE_NY and NY_WINDOW[0] + 0.5 <= h < NY_WINDOW[1]:   # NY from 12:30 UTC
        return "NY"
    return None

def main():
    if not init_mt5():
        sys.exit(1)

    info = sym()
    if info is None:
        log(f"Symbol {SYMBOL} unavailable")
        sys.exit(1)

    log("=" * 66)
    log("GOLD SESSION PRO v1.0 — Asian range breakout, London/NY")
    log(f"Risk {RISK_PER_TRADE_PCT}%/trade | Daily halt {MAX_DAILY_LOSS_PCT}% | Max {MAX_TRADES_PER_DAY} trades/day")
    log("=" * 66)

    state = load_state()
    acct = mt5.account_info()
    if state.get("day") != date.today().isoformat():
        state = fresh_day(acct.balance)
        save_state(state)

    while True:
        try:
            now = datetime.now(timezone.utc)
            acct = mt5.account_info()
            info = sym()
            if acct is None or info is None:
                time.sleep(LOOP_SLEEP)
                continue

            # --- daily rollover ---
            if state["day"] != date.today().isoformat():
                state = fresh_day(acct.balance)
                save_state(state)
                log(f"New day. Start balance ${acct.balance:,.2f}")

            positions = my_positions()

            # --- Friday flatten ---
            if now.weekday() == 4 and (now.hour, now.minute) >= FRIDAY_FLAT_UTC:
                if positions:
                    log("Friday flatten")
                    for p in positions:
                        close_position(info, p, "friday_flat")
                state["halted"] = True
                save_state(state)

            # --- daily loss guard (equity-based, includes floating) ---
            dd = (state["start_balance"] - acct.equity) / state["start_balance"] * 100 if state["start_balance"] else 0
            if dd >= MAX_DAILY_LOSS_PCT and not state["halted"]:
                log(f"DAILY LOSS LIMIT {dd:.1f}% — halting and flattening")
                for p in positions:
                    close_position(info, p, "daily_halt")
                state["halted"] = True
                save_state(state)

            # --- manage open positions (every loop; modifications are throttled) ---
            atr_m15 = atr(TF_ENTRY)
            for p in positions:
                manage(info, p, atr_m15)

            # --- status file for the fleet console ---
            with open(f"{TAG}_status.json", "w") as f:
                json.dump({
                    "time": now.strftime("%H:%M:%S"), "bot": "GOLD SESSION PRO",
                    "balance": round(acct.balance, 2), "equity": round(acct.equity, 2),
                    "positions": len(positions), "trades_today": state["trades_today"],
                    "daily_dd": f"{dd:.2f}%", "halted": state["halted"],
                    "strategy": "ASIA_RANGE_BREAKOUT",
                }, f, indent=2)

            if state["halted"] or positions or state["trades_today"] >= MAX_TRADES_PER_DAY:
                time.sleep(LOOP_SLEEP)
                continue

            session = in_window(now)
            if session is None:
                time.sleep(LOOP_SLEEP)
                continue

            # --- once per closed M15 bar ---
            df = bars(TF_ENTRY, 3)
            if df is None:
                time.sleep(LOOP_SLEEP)
                continue
            closed_bar_time = int(df["time"].iloc[-2])
            if closed_bar_time == state.get("last_bar", 0):
                time.sleep(LOOP_SLEEP)
                continue
            state["last_bar"] = closed_bar_time
            save_state(state)

            # --- cooldown after a loss ---
            day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            last_close_t, last_pnl = last_closed_deal_today(day_start)
            if last_close_t and last_pnl < 0 and (now - last_close_t) < timedelta(minutes=COOLDOWN_MIN):
                time.sleep(LOOP_SLEEP)
                continue

            # --- spread filter ---
            if info.spread > MAX_SPREAD_POINTS:
                time.sleep(LOOP_SLEEP)
                continue

            # --- levels & filters ---
            hi, lo = asian_range(now)
            if hi is None or atr_m15 is None:
                time.sleep(LOOP_SLEEP)
                continue
            rng = hi - lo
            if rng < MIN_RANGE_USD or rng > MAX_RANGE_USD:
                time.sleep(LOOP_SLEEP)
                continue

            bias = h1_bias()
            close = float(df["close"].iloc[-2])
            buf = BREAK_BUFFER_ATR * atr_m15
            tick = mt5.symbol_info_tick(SYMBOL)

            # --- LONG: M15 close above range + buffer, H1 bullish, not yet taken ---
            if (not state["long_taken"]) and bias == 1 and close > hi + buf:
                sl = tick.ask - SL_ATR_MULT * atr_m15
                sl = min(sl, hi - buf)                      # at least below the broken level
                risk = tick.ask - sl
                tp = tick.ask + TP_R_MULT * risk
                if send_market(info, "BUY", sl, tp, f"{session}_BREAK_LONG"):
                    state["long_taken"] = True
                    state["trades_today"] += 1
                    save_state(state)

            # --- SHORT: mirror ---
            elif (not state["short_taken"]) and bias == -1 and close < lo - buf:
                sl = tick.bid + SL_ATR_MULT * atr_m15
                sl = max(sl, lo + buf)
                risk = sl - tick.bid
                tp = tick.bid - TP_R_MULT * risk
                if send_market(info, "SELL", sl, tp, f"{session}_BREAK_SHORT"):
                    state["short_taken"] = True
                    state["trades_today"] += 1
                    save_state(state)

            time.sleep(LOOP_SLEEP)

        except KeyboardInterrupt:
            log("Stopped by user")
            break
        except Exception as e:
            log(f"Loop error: {e}")
            time.sleep(LOOP_SLEEP)

if __name__ == "__main__":
    main()
