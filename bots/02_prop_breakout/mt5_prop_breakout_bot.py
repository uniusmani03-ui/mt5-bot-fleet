"""
=============================================================================
 POD 2 - MT5 PROP-FIRM LIQUIDITY-SWEEP BOT  (range / false-breakout fade)
=============================================================================
 Strategy (per pair, per day):
   1. Build a RANGE from the high/low of a fixed time window (default
      05:00-09:00 UTC = 01:00-05:00 New York / 06:00-10:00 London-summer).
   2. After the window closes, drop to M5 and watch for a liquidity sweep:
        - Price sweeps ABOVE the range high, then a candle CLOSES back inside
          the range  -> SHORT (fading the false breakout).
        - Price sweeps BELOW the range low, then a candle CLOSES back inside
          -> LONG.
   3. EMA confluence (9 vs 21) must agree with the trade direction.
   4. Stop loss beyond the sweep extreme; take profit at RR_RATIO x risk.

 Prop guardrails (hard rules):
   - Risk 1% of balance per trade.
   - Daily loss limit (stop trading for the day).
   - Total challenge loss limit (stop the challenge).
   - Profit target (stop once passed).
   These use EQUITY (incl. floating P/L), the way prop firms measure them, and
   are persisted to disk so restarting the bot does NOT reset your limits.

 -----------------------------------------------------------------------------
 INSTALL (run once):
     pip install MetaTrader5 pandas numpy
 Windows only, with MT5 open and logged into the CHALLENGE (or a 10k demo).
 Run this on its OWN account - do not run it on the same account as pod 1,
 because the daily/total loss checks measure the whole account's equity.
=============================================================================
"""

import os
import csv
import sys
import json
import time
import math
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

try:
    import winsound  # Windows-only; used for audible alerts
except ImportError:
    winsound = None

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# --- MODE -------------------------------------------------------------------
# ALERT_ONLY = True  -> the bot NEVER places, modifies, or closes any order.
# It only SIGNALS you (log + sound + dashboard banner) the exact trade to place
# by hand, and tells you when to take the partial / move to breakeven / exit.
# Use this for prop firms that ban EAs/bots (e.g. Maven Trading).
# ALERT_ONLY = False -> the bot trades automatically (only for EA-allowed
# accounts / your own broker).
ALERT_ONLY          = False   # DEMO: auto-trade so you can test & build a trade log.
                              # Set True for live/Maven so it only SIGNALS (you place by hand).
ALERT_SOUND         = True    # beep on a new signal (Windows)

SYMBOLS             = ["GBPUSD"]            # MetaQuotes-Demo uses plain names (no suffix)
# ^ Add more if you like, e.g. ["GBPUSD", "GBPJPY"]. On VT Markets use the
#   suffixed names instead (e.g. "GBPUSD-ECN") — check Market Watch (Ctrl+U).
TIMEFRAME           = mt5.TIMEFRAME_M5       # entry timeframe (step 9: "drop to 5 min")
CANDLES_TO_FETCH    = 500

# --- Risk / reward ---
RISK_PERCENT        = 1.0       # % of balance risked per trade (step 5)
# On a TINY account (e.g. £100) 1% may be smaller than the broker's minimum
# lot (0.01). If so and this is True, the bot uses the 0.01 minimum and logs
# the REAL risk (which will be MORE than 1%). Set False to skip such trades.
ALLOW_MIN_LOT_FALLBACK = True
RR_RATIO            = 1.5       # reward : risk when partials are OFF (steps 10-12)
SL_BUFFER_POINTS    = 20        # extra points beyond the sweep extreme for the stop

# --- Trade management: scale out + breakeven ---
# When ON: at +PARTIAL_AT_RR, close PARTIAL_PCT% of the position and move the
# stop on the remainder to entry (breakeven). The runner then targets
# FINAL_TP_RR. (When partials are ON, FINAL_TP_RR is used for the order's TP
# instead of RR_RATIO, and FINAL_TP_RR must be > PARTIAL_AT_RR.)
USE_PARTIALS        = True
PARTIAL_AT_RR       = 1.5       # take the partial when open profit hits this multiple of risk
PARTIAL_PCT         = 50        # % of the position to close at the partial
MOVE_SL_TO_BE       = True      # move stop to entry after taking the partial
FINAL_TP_RR         = 3.0       # broker-side TP. With the 9EMA trail ON this is a FAR SAFETY CAP only.

# --- Runner exit: trail on the fast (9) EMA ---
# When ON, the (remaining) position is CLOSED when a CLOSED M5 candle closes
# back across the fast EMA against the trade:
#   SHORT -> exit on a close ABOVE the 9 EMA ;  LONG -> exit on a close BELOW.
# This becomes the primary take-profit; FINAL_TP_RR stays as a far safety cap
# for disconnects. The 21 EMA is no longer used anywhere.
USE_EMA_TRAIL_EXIT  = True

# --- Range window (UTC). 05:00-09:00 UTC = 01:00-05:00 EDT = 06:00-10:00 London (summer).
#     In winter (EST/GMT) shift these back one hour, e.g. (4,0)-(8,0). ---
RANGE_START_UTC     = (5, 0)    # (hour, minute) inclusive
RANGE_END_UTC       = (9, 0)    # (hour, minute) exclusive
TRADE_UNTIL_UTC     = (20, 0)   # stop opening NEW trades after this UTC time each day

# --- Entry confirmation (consecutive M5 closes vs the fast EMA, steps 14-15) ---
# After a sweep + reclaim, require this many consecutive CLOSED M5 candles on
# the correct side of the fast EMA before entering:
#   SHORT -> closes BELOW the 9 EMA ;  LONG -> closes ABOVE the 9 EMA.
# Set 1 for a faster (noisier) entry, 2+ for stronger (later) confirmation.
# This replaces the old 9/21 cross, which lagged too much for this entry.
CONFIRM_CLOSES      = 2
EMA_FAST            = 9
EMA_SLOW            = 21
# Optional EXTRA trend gate using 9/21 alignment (off by default). If True:
#   SHORT also requires 9 EMA < 21 EMA ;  LONG requires 9 EMA > 21 EMA.
USE_TREND_GATE      = False

# --- Trade controls ---
MAX_TRADES_PER_DAY_PER_SYMBOL = 2
SLIPPAGE_POINTS     = 5
MAGIC_NUMBER        = 778001    # pod 2's tag (different from pod 1!)
MAX_SPREAD_POINTS   = 35        # GBPJPY/GBPUSD spreads are wider than EURUSD

# --- Risk limits as PERCENT of balance (scales to ANY account size: £100 or 10k) ---
# These replace the old fixed £ amounts, which were calibrated for a 10k
# challenge and gave NO protection on a small account.
USE_PROFIT_TARGET    = False   # personal account: no challenge to "pass". True = auto-stop in profit.
PROFIT_TARGET_PCT    = 50.0    # if enabled, stop once equity is up this % of the start balance
MAX_DAILY_LOSS_PCT   = 5.0     # stop trading for the day at -this% of the day's starting equity
MAX_TOTAL_LOSS_PCT   = 25.0    # stop the bot entirely at -this% of the starting balance
DAILY_RESET_HOUR_UTC = 0       # when the "trading day" resets for the daily-loss rule

# --- Files ---
TRADE_LOG_FILE      = "pod2_trade_log.csv"
STATE_FILE          = "pod2_challenge_state.json"
STATUS_FILE         = "pod2_status.json"     # live snapshot for the dashboard

# --- News guard (Forex Factory) — for Maven's no-trade-around-news rule ---
# Pulls the FF calendar, keeps red-folder (High) events for your pairs'
# currencies, and shows a dashboard warning. Inside the BLOCK window it also
# holds the bot's signals so it never tells you to open/close around news.
USE_NEWS_GUARD      = True
NEWS_FEED_URL       = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_CACHE_FILE     = "pod2_news_cache.json"
NEWS_IMPACTS        = ["High"]
NEWS_BLOCK_BEFORE   = 2          # minutes before a red event: DO NOT trade (Maven = 2)
NEWS_BLOCK_AFTER    = 2          # minutes after
NEWS_WARN_BEFORE    = 15         # minutes before: amber heads-up on the dashboard
NEWS_REFRESH_HOURS  = 6
LOOP_SLEEP_SECONDS  = 2

# Optional explicit login (leave None to attach to the running terminal).
MT5_LOGIN    = None
MT5_PASSWORD = None
MT5_SERVER   = None
MT5_PATH     = None


# =============================================================================
# 2. LOGGING + CSV
# =============================================================================
_log_buffer = []  # recent log lines, surfaced to the dashboard


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)
    _log_buffer.append(line)
    if len(_log_buffer) > 60:
        del _log_buffer[:-60]


_CSV_FIELDS = ["timestamp_utc", "event", "symbol", "direction", "volume",
               "price", "sl", "tp", "retcode", "ticket", "deal",
               "balance", "equity", "comment"]


def log_trade_csv(event, symbol, direction="", volume=0.0, price=0.0, sl=0.0,
                  tp=0.0, retcode=None, ticket=None, deal=None, comment=""):
    acct = mt5.account_info()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "event": event, "symbol": symbol, "direction": direction, "volume": volume,
        "price": price, "sl": sl, "tp": tp,
        "retcode": retcode if retcode is not None else "",
        "ticket": ticket if ticket is not None else "",
        "deal": deal if deal is not None else "",
        "balance": round(acct.balance, 2) if acct else "",
        "equity": round(acct.equity, 2) if acct else "",
        "comment": comment,
    }
    try:
        write_header = not os.path.exists(TRADE_LOG_FILE)
        with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as exc:
        log(f"WARNING: failed to write trade CSV -> {exc}")


# =============================================================================
# 2b. ALERT HELPERS (manual / ALERT_ONLY mode)
# =============================================================================
_active_alert = None  # latest signal, surfaced to the dashboard banner


def _beep(kind="entry"):
    if not ALERT_SOUND or winsound is None:
        return
    try:
        if kind == "entry":
            for _ in range(2):
                winsound.Beep(1000, 220)
        elif kind == "exit":
            winsound.Beep(760, 380)
        else:  # danger / manage
            winsound.Beep(520, 550)
    except Exception:
        pass


def _set_alert(text, level="entry"):
    global _active_alert
    _active_alert = {"text": text, "level": level,
                     "time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC")}


def find_any_position(symbol):
    """First open position on a symbol regardless of magic (catches manual trades)."""
    ps = mt5.positions_get(symbol=symbol)
    return ps[0] if ps else None


def position_for(symbol):
    """In ALERT mode watch ANY position (yours, placed by hand); else only ours."""
    return find_any_position(symbol) if ALERT_ONLY else open_position(symbol)


def signal_entry(info, direction, sl_price, risk_amount):
    """ALERT_ONLY: compute the exact trade and tell the user to place it by hand."""
    tick = mt5.symbol_info_tick(info.name)
    if tick is None:
        return False
    point, digits = info.point, info.digits
    min_dist = max(info.trade_stops_level, 1) * point
    tp_rr = FINAL_TP_RR if USE_PARTIALS else RR_RATIO

    if direction == "SELL":
        entry = tick.bid
        if sl_price <= entry + min_dist:
            sl_price = entry + min_dist
        risk_dist = sl_price - entry
        tp_price = entry - tp_rr * risk_dist
        partial_lvl = entry - PARTIAL_AT_RR * risk_dist
    else:
        entry = tick.ask
        if sl_price >= entry - min_dist:
            sl_price = entry - min_dist
        risk_dist = entry - sl_price
        tp_price = entry + tp_rr * risk_dist
        partial_lvl = entry + PARTIAL_AT_RR * risk_dist

    if risk_dist <= 0:
        return False
    volume = calculate_lot_size(info, risk_dist, risk_amount)
    if volume is None:
        return False

    f = lambda x: f"{x:.{digits}f}"
    log("============== SIGNAL: %s %s ==============" % (direction, info.name))
    log(f"   ENTER ~{f(entry)} (market)   SIZE {volume} lots   (~{risk_amount:.0f} risk)")
    log(f"   STOP  {f(sl_price)}     TARGET {f(tp_price)}")
    if USE_PARTIALS:
        log(f"   PLAN  take {PARTIAL_PCT}% + move SL to entry at {f(partial_lvl)} (={PARTIAL_AT_RR}R),"
            f" then exit the rest on the 9 EMA")
    log("=" * 46)
    _set_alert(f"{direction} {info.name}  @~{f(entry)}  |  SL {f(sl_price)}  |  "
               f"TP {f(tp_price)}  |  {volume} lots", "entry")
    _beep("entry")
    log_trade_csv("signal", info.name, direction, volume, entry, sl_price, tp_price,
                  comment="manual_signal")
    return True


# =============================================================================
# 3. CONNECTION, SYMBOLS, TIME
# =============================================================================
def initialize_mt5() -> bool:
    kwargs = {}
    if MT5_PATH:
        kwargs["path"] = MT5_PATH
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        kwargs.update(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)
    if not mt5.initialize(**kwargs):
        log(f"ERROR: mt5.initialize() failed -> {mt5.last_error()}")
        return False
    term, acct = mt5.terminal_info(), mt5.account_info()
    if term is None or acct is None:
        log(f"ERROR: could not read terminal/account -> {mt5.last_error()}")
        return False
    if not term.trade_allowed:
        log("ERROR: Algo trading disabled in terminal. Enable the 'Algo Trading' button.")
        return False
    log(f"Connected. Account #{acct.login} | {acct.server} | "
        f"Balance {acct.balance:.2f} {acct.currency} | Leverage 1:{acct.leverage}")
    return True


def prepare_symbol(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        log(f"ERROR: symbol '{symbol}' not found -> {mt5.last_error()}")
        return None
    if not info.visible and not mt5.symbol_select(symbol, True):
        log(f"ERROR: could not select '{symbol}' -> {mt5.last_error()}")
        return None
    return mt5.symbol_info(symbol)


def broker_utc_offset_hours() -> int:
    """
    MT5 candle/tick times are in BROKER SERVER time, not UTC. Estimate the
    offset (server - UTC) in whole hours so we can convert candle times to UTC.
    Retries until a live tick is available, so the offset isn't misread as 0
    before prices start flowing.
    """
    for _ in range(20):
        tick = mt5.symbol_info_tick(SYMBOLS[0])
        if tick is not None and tick.time != 0:
            return int(round((tick.time - time.time()) / 3600.0))
        time.sleep(0.5)
    return 0


# =============================================================================
# 4. DATA + INDICATORS
# =============================================================================
def fetch_data(symbol, offset_hours):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, CANDLES_TO_FETCH)
    if rates is None or len(rates) < EMA_SLOW + 5:
        return None
    df = pd.DataFrame(rates)
    # Convert broker-server time -> true UTC
    df["time_utc"] = pd.to_datetime(df["time"], unit="s") - pd.Timedelta(hours=offset_hours)
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return df


def compute_range(df, day_utc):
    """
    High/low of candles whose UTC time falls inside the range window on day_utc.
    Returns (range_high, range_low) or (None, None) if the window has no data yet.
    """
    start = datetime.combine(day_utc, datetime.min.time(),
                             tzinfo=timezone.utc).replace(hour=RANGE_START_UTC[0],
                                                          minute=RANGE_START_UTC[1])
    end = datetime.combine(day_utc, datetime.min.time(),
                           tzinfo=timezone.utc).replace(hour=RANGE_END_UTC[0],
                                                        minute=RANGE_END_UTC[1])
    t = df["time_utc"].dt.tz_localize("UTC")
    mask = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))
    window = df[mask]
    if window.empty:
        return None, None
    return float(window["high"].max()), float(window["low"].min())


def closes_confirm(df, direction, n):
    """
    True if the last n CLOSED M5 candles are all on the correct side of the
    fast EMA. Closed candles are iloc[-2] (latest closed), iloc[-3], ...
      SELL -> every close BELOW its ema_fast ; BUY -> every close ABOVE.
    """
    for k in range(2, n + 2):
        if k > len(df):
            return False
        row = df.iloc[-k]
        if direction == "SELL" and not (row["close"] < row["ema_fast"]):
            return False
        if direction == "BUY" and not (row["close"] > row["ema_fast"]):
            return False
    return True


# =============================================================================
# 5. EXECUTION MATH
# =============================================================================
def _volume_decimals(step):
    return max(0, int(round(-math.log10(step)))) if step > 0 else 2


def calculate_lot_size(info, sl_distance_price, risk_amount):
    tv, ts = info.trade_tick_value, info.trade_tick_size
    if ts <= 0 or tv <= 0 or sl_distance_price <= 0:
        return None
    loss_per_lot = (sl_distance_price / ts) * tv
    if loss_per_lot <= 0:
        return None
    step = info.volume_step
    vol = math.floor((risk_amount / loss_per_lot) / step) * step
    vol = round(vol, _volume_decimals(step))
    if vol < info.volume_min:
        if ALLOW_MIN_LOT_FALLBACK:
            min_risk = loss_per_lot * info.volume_min
            log(f"   {info.name}: {RISK_PERCENT}% (~{risk_amount:.2f}) is below the min "
                f"lot {info.volume_min}. Using min lot = ~{min_risk:.2f} risk "
                f"({min_risk / risk_amount:.1f}x your target — bigger than {RISK_PERCENT}%).")
            return info.volume_min
        log(f"   {info.name}: required lot {vol} < min {info.volume_min}; skipping.")
        return None
    return min(vol, info.volume_max)


def get_filling_mode(info):
    fm = info.filling_mode
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def spread_ok(info):
    tick = mt5.symbol_info_tick(info.name)
    if tick is None:
        return False
    pts = (tick.ask - tick.bid) / info.point
    if pts > MAX_SPREAD_POINTS:
        log(f"   {info.name}: spread {pts:.0f} pts > {MAX_SPREAD_POINTS}; skipping.")
        return False
    return True


def open_position(symbol):
    """Return this bot's open position object for a symbol, or None."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if p.magic == MAGIC_NUMBER:
            return p
    return None


def place_trade(info, direction, sl_price, risk_amount):
    tick = mt5.symbol_info_tick(info.name)
    if tick is None:
        log(f"   {info.name}: no tick; abort.")
        return False
    point, digits = info.point, info.digits
    min_dist = max(info.trade_stops_level, 1) * point
    tp_rr = FINAL_TP_RR if USE_PARTIALS else RR_RATIO

    if direction == "SELL":
        entry = tick.bid
        if sl_price <= entry + min_dist:
            sl_price = entry + min_dist
        risk_dist = sl_price - entry
        tp_price = entry - tp_rr * risk_dist
        order_type = mt5.ORDER_TYPE_SELL
    else:  # BUY
        entry = tick.ask
        if sl_price >= entry - min_dist:
            sl_price = entry - min_dist
        risk_dist = entry - sl_price
        tp_price = entry + tp_rr * risk_dist
        order_type = mt5.ORDER_TYPE_BUY

    if risk_dist <= 0:
        log(f"   {info.name}: invalid stop distance; abort.")
        return False

    volume = calculate_lot_size(info, risk_dist, risk_amount)
    if volume is None:
        return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": info.name, "volume": volume,
        "type": order_type, "price": round(entry, digits),
        "sl": round(sl_price, digits), "tp": round(tp_price, digits),
        "deviation": SLIPPAGE_POINTS, "magic": MAGIC_NUMBER,
        "comment": "pod2_sweep", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(info),
    }
    log(f">>> {direction} {info.name}: vol={volume} entry={request['price']} "
        f"SL={request['sl']} TP={request['tp']} (risk ~{risk_amount:.2f})")
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        rc = result.retcode if result else mt5.last_error()
        cm = result.comment if result else ""
        log(f"   ORDER REJECTED ({rc}) {cm}")
        log_trade_csv("rejected", info.name, direction, volume, request["price"],
                      request["sl"], request["tp"], retcode=rc, comment=str(cm))
        return False
    log(f"   FILLED ticket={result.order} price={result.price}")
    log_trade_csv("filled", info.name, direction, result.volume, result.price,
                  request["sl"], request["tp"], retcode=result.retcode,
                  ticket=result.order, deal=result.deal, comment="pod2_sweep")
    return True


def close_my_positions(reason="risk limit", level="danger"):
    if ALERT_ONLY:
        log(f"   >>> ACTION NEEDED: {reason} — CLOSE ALL OPEN TRADES MANUALLY NOW.")
        _set_alert(reason + " — close all trades now", level)
        _beep("danger")
        return
    for symbol in SYMBOLS:
        pos = open_position(symbol)
        while pos is not None:
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if info is None or tick is None:
                break
            if pos.type == mt5.POSITION_TYPE_BUY:
                otype, price = mt5.ORDER_TYPE_SELL, tick.bid
            else:
                otype, price = mt5.ORDER_TYPE_BUY, tick.ask
            req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                   "volume": pos.volume, "type": otype, "position": pos.ticket,
                   "price": round(price, info.digits), "deviation": SLIPPAGE_POINTS,
                   "magic": MAGIC_NUMBER, "comment": "pod2_close",
                   "type_time": mt5.ORDER_TIME_GTC, "type_filling": get_filling_mode(info)}
            res = mt5.order_send(req)
            ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
            log(f"Closing {symbol} #{pos.ticket}: {'OK' if ok else 'FAILED'}")
            log_trade_csv("closed" if ok else "close_failed", symbol,
                          volume=pos.volume, price=price, ticket=pos.ticket,
                          comment="pod2_close")
            if not ok:
                break
            pos = open_position(symbol)


# Tracks per-ticket management state (in-memory). Reconstructed on restart by
# detecting whether a position's stop is already at breakeven.
_managed = {}


def _modify_sl_to_breakeven(info, pos):
    req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol,
           "position": pos.ticket, "sl": round(pos.price_open, info.digits),
           "tp": pos.tp}
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log(f"   {pos.symbol} #{pos.ticket}: move SL to breakeven {'OK' if ok else 'FAILED'}"
        + ("" if ok else f" ({mt5.last_error()})"))
    return ok


def _close_partial(info, pos, fraction):
    """Close `fraction` of a position's volume. Returns True if a partial was sent."""
    step = info.volume_step
    part = math.floor((pos.volume * fraction) / step) * step
    part = round(part, _volume_decimals(step))
    remaining = round(pos.volume - part, _volume_decimals(step))
    if part < info.volume_min or remaining < info.volume_min:
        log(f"   {pos.symbol} #{pos.ticket}: volume {pos.volume} can't be split "
            f"(min {info.volume_min}); skipping partial, moving to breakeven only.")
        return False
    tick = mt5.symbol_info_tick(pos.symbol)
    if pos.type == mt5.POSITION_TYPE_BUY:
        otype, price = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        otype, price = mt5.ORDER_TYPE_BUY, tick.ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": part,
           "type": otype, "position": pos.ticket, "price": round(price, info.digits),
           "deviation": SLIPPAGE_POINTS, "magic": MAGIC_NUMBER,
           "comment": "pod2_partial", "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": get_filling_mode(info)}
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log(f"   {pos.symbol} #{pos.ticket}: partial close {part} lots "
        f"({PARTIAL_PCT}%) {'OK' if ok else 'FAILED'}.")
    if ok:
        log_trade_csv("partial", pos.symbol, volume=part, price=price,
                      ticket=pos.ticket, comment=f"{PARTIAL_PCT}%@{PARTIAL_AT_RR}R")
    return ok


def manage_open_positions(news_blocked=False):
    """At +PARTIAL_AT_RR: scale out (or, in ALERT mode, tell the user to)."""
    if not USE_PARTIALS:
        return
    open_tickets = set()
    for symbol in SYMBOLS:
        pos = position_for(symbol)
        if pos is None:
            continue
        open_tickets.add(pos.ticket)
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None or pos.sl == 0:
            continue

        entry = pos.price_open
        rec = _managed.get(pos.ticket)
        if rec is None:
            # First time we see this ticket: reconstruct state. A stop sitting
            # at entry means the partial was already taken (restart-safe).
            at_be = abs(pos.sl - entry) <= info.point * 2
            rec = {"risk": abs(entry - pos.sl) if not at_be else None,
                   "done": at_be}
            _managed[pos.ticket] = rec

        if rec["done"] or not rec["risk"]:
            continue

        target = (entry + PARTIAL_AT_RR * rec["risk"] if pos.type == mt5.POSITION_TYPE_BUY
                  else entry - PARTIAL_AT_RR * rec["risk"])
        reached = (tick.bid >= target if pos.type == mt5.POSITION_TYPE_BUY
                   else tick.ask <= target)
        if reached:
            if ALERT_ONLY and news_blocked:
                continue  # hold the alert; Maven forbids acting around news
            if ALERT_ONLY:
                log(f"[{symbol}] >>> ACTION: +{PARTIAL_AT_RR}R hit — take {PARTIAL_PCT}% "
                    f"partial and move your stop to entry now.")
                _set_alert(f"{symbol}: +{PARTIAL_AT_RR}R hit — take {PARTIAL_PCT}% partial "
                           f"& move SL to breakeven", "manage")
                _beep("exit")
            else:
                log(f"[{symbol}] #{pos.ticket} reached +{PARTIAL_AT_RR}R -> scaling out.")
                _close_partial(info, pos, PARTIAL_PCT / 100.0)
                if MOVE_SL_TO_BE:
                    pos = open_position(symbol) or pos  # refresh after partial
                    _modify_sl_to_breakeven(info, pos)
            rec["done"] = True

    # prune closed tickets
    for t in list(_managed.keys()):
        if t not in open_tickets:
            _managed.pop(t, None)


def close_one(symbol, comment="pod2_exit"):
    """Market-close the full open position for one symbol (e.g. 9EMA trail exit)."""
    pos = open_position(symbol)
    if pos is None:
        return
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return
    if pos.type == mt5.POSITION_TYPE_BUY:
        otype, price = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        otype, price = mt5.ORDER_TYPE_BUY, tick.ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": pos.volume,
           "type": otype, "position": pos.ticket, "price": round(price, info.digits),
           "deviation": SLIPPAGE_POINTS, "magic": MAGIC_NUMBER, "comment": comment,
           "type_time": mt5.ORDER_TIME_GTC, "type_filling": get_filling_mode(info)}
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log(f"   {symbol} #{pos.ticket}: trail exit {'OK' if ok else 'FAILED'} @ {price}.")
    log_trade_csv("closed" if ok else "close_failed", symbol, volume=pos.volume,
                  price=price, ticket=pos.ticket, comment=comment)


# =============================================================================
# 6. CHALLENGE STATE (persisted across restarts)
# =============================================================================
def load_state(current_equity):
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    state = {
        "challenge_start_balance": current_equity,
        "current_day": "",
        "day_start_equity": current_equity,
        "challenge_over": False,
    }
    save_state(state)
    return state


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as exc:
        log(f"WARNING: could not save state -> {exc}")


def trading_day_str():
    """The current 'trading day' key, offset by DAILY_RESET_HOUR_UTC."""
    now = datetime.now(timezone.utc) - timedelta(hours=DAILY_RESET_HOUR_UTC)
    return now.strftime("%Y-%m-%d")


# =============================================================================
# 6b. NEWS GUARD (Forex Factory calendar)
# =============================================================================
_news_events = []
_news_last_fetch = None


def _pair_currencies():
    cur = set()
    for s in SYMBOLS:
        core = "".join(c for c in s if c.isalpha()).upper()
        if len(core) >= 6:
            cur.add(core[:3])
            cur.add(core[3:6])
    return cur


def _parse_news_payload(text):
    wanted = _pair_currencies()
    data = json.loads(text)  # raises on a 'Request Denied' HTML page
    events = []
    for item in data:
        try:
            if item.get("country") not in wanted:
                continue
            if item.get("impact") not in NEWS_IMPACTS:
                continue
            dt = datetime.fromisoformat(item["date"]).astimezone(timezone.utc)
            events.append({"time_utc": dt, "title": item.get("title", "?"),
                           "currency": item.get("country", "?")})
        except (KeyError, ValueError):
            continue
    events.sort(key=lambda e: e["time_utc"])
    return events


def _save_news_cache(events):
    try:
        payload = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                   "events": [{"time_utc": e["time_utc"].isoformat(), "title": e["title"],
                               "currency": e["currency"]} for e in events]}
        with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _load_news_cache():
    try:
        with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        events = [{"time_utc": datetime.fromisoformat(e["time_utc"]), "title": e["title"],
                   "currency": e["currency"]} for e in payload["events"]]
        return events, fetched_at
    except Exception:
        return [], None


def refresh_news(force=False):
    """Refresh the calendar, heavily cached to respect the feed's rate limit."""
    global _news_events, _news_last_fetch
    if not USE_NEWS_GUARD:
        return
    now = datetime.now(timezone.utc)
    if _news_last_fetch is None and not force:
        cached, cached_at = _load_news_cache()
        if cached_at is not None and (now - cached_at) < timedelta(hours=NEWS_REFRESH_HOURS):
            _news_events, _news_last_fetch = cached, cached_at
            log(f"News: loaded {len(cached)} cached red-folder events.")
            return
    if (not force) and _news_last_fetch is not None and \
            (now - _news_last_fetch) < timedelta(hours=NEWS_REFRESH_HOURS):
        return
    try:
        req = urllib.request.Request(NEWS_FEED_URL,
                                     headers={"User-Agent": "Mozilla/5.0 (pod2-newsguard)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        _news_events = _parse_news_payload(text)
        _news_last_fetch = now
        _save_news_cache(_news_events)
        log(f"News: downloaded calendar. {len(_news_events)} red-folder "
            f"{'/'.join(sorted(_pair_currencies()))} events this week.")
    except (urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
        log(f"WARNING: news refresh failed ({exc}). Keeping {len(_news_events)} cached.")
        if _news_last_fetch is None:
            _news_last_fetch = now - timedelta(hours=NEWS_REFRESH_HOURS) + timedelta(minutes=20)


def news_state():
    """Return the current news guard state for gating + the dashboard."""
    if not USE_NEWS_GUARD:
        return {"state": "off", "blocked": False, "text": ""}
    now = datetime.now(timezone.utc)
    for e in _news_events:
        if (e["time_utc"] - timedelta(minutes=NEWS_BLOCK_BEFORE)) <= now \
                <= (e["time_utc"] + timedelta(minutes=NEWS_BLOCK_AFTER)):
            return {"state": "blocked", "blocked": True,
                    "text": f"{e['currency']} {e['title']}",
                    "time": e["time_utc"].strftime("%H:%M UTC")}
    nxt = next((e for e in _news_events if e["time_utc"] > now), None)
    if nxt:
        mins = int((nxt["time_utc"] - now).total_seconds() // 60)
        state = "warn" if mins <= NEWS_WARN_BEFORE else "clear"
        return {"state": state, "blocked": False,
                "text": f"{nxt['currency']} {nxt['title']} in {mins} min",
                "time": nxt["time_utc"].strftime("%H:%M UTC")}
    return {"state": "clear", "blocked": False, "text": "no red-folder events queued"}


def write_status(state, sym_state, halted_today):
    """Write a live JSON snapshot of everything the dashboard shows."""
    acct = mt5.account_info()
    if acct is None:
        return
    start_bal = state["challenge_start_balance"]
    day_start = state["day_start_equity"]

    positions = []
    for s in SYMBOLS:
        for p in (mt5.positions_get(symbol=s) or []):
            if not ALERT_ONLY and p.magic != MAGIC_NUMBER:
                continue  # in ALERT mode show manual trades too (magic 0)
            positions.append({
                "symbol": s,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": p.volume, "entry": p.price_open,
                "sl": p.sl, "tp": p.tp, "price": p.price_current,
                "profit": round(p.profit, 2),
            })

    symbols = {}
    for s in SYMBOLS:
        st = sym_state[s]
        in_trade = any(pos["symbol"] == s for pos in positions)
        if halted_today:
            note = "Daily loss limit hit - halted until tomorrow"
        elif st["range_high"] is None:
            note = "Waiting for the range window to close"
        elif in_trade:
            note = "In a trade - trailing the 9 EMA"
        elif st["trades_today"] >= MAX_TRADES_PER_DAY_PER_SYMBOL:
            note = "Max trades reached for today"
        elif st["swept_high"]:
            note = f"Swept the HIGH - awaiting {CONFIRM_CLOSES} close(s) below 9 EMA"
        elif st["swept_low"]:
            note = f"Swept the LOW - awaiting {CONFIRM_CLOSES} close(s) above 9 EMA"
        else:
            note = "Range set - watching for a liquidity sweep"
        symbols[s] = {
            "range_high": st["range_high"], "range_low": st["range_low"],
            "swept_high": st["swept_high"], "swept_low": st["swept_low"],
            "trades_today": st["trades_today"], "in_trade": in_trade, "note": note,
        }

    snapshot = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "mode": "ALERT" if ALERT_ONLY else "AUTO",
        "alert": _active_alert,
        "news": news_state(),
        "account": {
            "login": acct.login, "server": acct.server, "currency": acct.currency,
            "balance": round(acct.balance, 2), "equity": round(acct.equity, 2),
            "floating": round(acct.profit, 2),
        },
        "challenge": {
            "start_balance": round(start_bal, 2),
            "target": round(PROFIT_TARGET_PCT / 100.0 * start_bal, 2) if USE_PROFIT_TARGET else 0,
            "daily_limit": round(MAX_DAILY_LOSS_PCT / 100.0 * day_start, 2),
            "total_limit": round(MAX_TOTAL_LOSS_PCT / 100.0 * start_bal, 2),
            "total_pl": round(acct.equity - start_bal, 2),
            "day_pl": round(acct.equity - day_start, 2),
            "halted_today": halted_today, "over": state.get("challenge_over", False),
        },
        "settings": {
            "pairs": SYMBOLS, "risk_percent": RISK_PERCENT, "rr": RR_RATIO,
            "confirm_closes": CONFIRM_CLOSES,
            "partial": f"{PARTIAL_PCT}% @ {PARTIAL_AT_RR}R" if USE_PARTIALS else "off",
            "runner_exit": f"{EMA_FAST}EMA trail" if USE_EMA_TRAIL_EXIT else f"{FINAL_TP_RR}R",
            "window": f"{RANGE_START_UTC[0]:02d}:{RANGE_START_UTC[1]:02d}-"
                      f"{RANGE_END_UTC[0]:02d}:{RANGE_END_UTC[1]:02d} UTC",
        },
        "symbols": symbols,
        "positions": positions,
        "log": _log_buffer[-40:],
    }
    try:
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        os.replace(tmp, STATUS_FILE)  # atomic, so the dashboard never reads a half-written file
    except Exception as exc:
        log(f"WARNING: could not write status -> {exc}")


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    if not initialize_mt5():
        sys.exit(1)

    infos = {}
    for s in SYMBOLS:
        info = prepare_symbol(s)
        if info is None:
            mt5.shutdown()
            sys.exit(1)
        infos[s] = info

    offset = broker_utc_offset_hours()
    log(f"Broker server time is UTC{offset:+d}h. Range window {RANGE_START_UTC[0]:02d}:"
        f"{RANGE_START_UTC[1]:02d}-{RANGE_END_UTC[0]:02d}:{RANGE_END_UTC[1]:02d} UTC.")

    acct = mt5.account_info()
    state = load_state(acct.equity)
    if ALERT_ONLY:
        log("################################################################")
        log("#  ALERT-ONLY MODE: this bot will NOT place or close orders.   #")
        log("#  It only SIGNALS trades for you to place by hand. Safe for   #")
        log("#  prop firms that ban EAs (e.g. Maven Trading).               #")
        log("################################################################")
    else:
        log("*** AUTO MODE: this bot WILL place and manage live orders. ***")
    sb = state['challenge_start_balance']
    log(f"Start balance: {sb:.2f}. "
        f"{'Target +' + format(PROFIT_TARGET_PCT,'.0f') + '%' if USE_PROFIT_TARGET else 'No profit target'} | "
        f"Daily loss -{MAX_DAILY_LOSS_PCT}% (-{MAX_DAILY_LOSS_PCT/100*sb:.2f}) | "
        f"Total loss -{MAX_TOTAL_LOSS_PCT}% (-{MAX_TOTAL_LOSS_PCT/100*sb:.2f}).")
    log(f"RR {RR_RATIO}:1 | risk {RISK_PERCENT}% | pairs {', '.join(SYMBOLS)}.")
    log(f"Entry confirm: {CONFIRM_CLOSES} consecutive M5 close(s) vs {EMA_FAST}EMA"
        f"{' + 9/21 trend gate' if USE_TREND_GATE else ''}.")
    if USE_PARTIALS:
        if FINAL_TP_RR <= PARTIAL_AT_RR:
            log(f"WARNING: FINAL_TP_RR ({FINAL_TP_RR}) must exceed PARTIAL_AT_RR "
                f"({PARTIAL_AT_RR}); bumping runner target to {PARTIAL_AT_RR + 1.0}.")
            globals()["FINAL_TP_RR"] = PARTIAL_AT_RR + 1.0
        log(f"Partials ON: scale {PARTIAL_PCT}% at +{PARTIAL_AT_RR}R, "
            f"{'move stop to breakeven' if MOVE_SL_TO_BE else 'keep stop'}, "
            f"runner TP at {FINAL_TP_RR}R.")
    else:
        log(f"Partials OFF: single TP at {RR_RATIO}R.")
    if USE_EMA_TRAIL_EXIT:
        log(f"Runner exit: trail on {EMA_FAST}EMA (close back across it). "
            f"{FINAL_TP_RR}R kept as a far safety cap. 21EMA unused.")
    log(f"Trade log -> {os.path.abspath(TRADE_LOG_FILE)}")

    if USE_NEWS_GUARD:
        ccy = "/".join(sorted(_pair_currencies()))
        log(f"News guard ON: red-folder {ccy} events; block {NEWS_BLOCK_BEFORE}min before "
            f"to {NEWS_BLOCK_AFTER}min after (Maven rule), warn {NEWS_WARN_BEFORE}min ahead.")
        refresh_news(force=True)
    else:
        log("News guard OFF.")

    # Per-symbol daily state
    sym_state = {s: {"day": "", "range_high": None, "range_low": None,
                     "swept_high": False, "swept_low": False,
                     "sweep_high_px": None, "sweep_low_px": None,
                     "trades_today": 0, "last_bar": None} for s in SYMBOLS}
    halted_today = False

    log("Pod 2 live. Watching for the range...")
    try:
        while True:
            acct = mt5.account_info()
            if acct is None:
                time.sleep(LOOP_SLEEP_SECONDS)
                continue
            equity = acct.equity
            start_bal = state["challenge_start_balance"]

            # --- Challenge already finished? ---
            if state.get("challenge_over"):
                log("Challenge is marked complete. Standing down. (Delete pod2_challenge_state.json to reset.)")
                break

            # --- Day rollover: reset daily figures + per-symbol range state ---
            today = trading_day_str()
            if state["current_day"] != today:
                state["current_day"] = today
                state["day_start_equity"] = equity
                save_state(state)
                halted_today = False
                for s in SYMBOLS:
                    sym_state[s].update(day="", range_high=None, range_low=None,
                                        swept_high=False, swept_low=False,
                                        sweep_high_px=None, sweep_low_px=None,
                                        trades_today=0)
                log(f"--- New trading day {today}. Daily start equity {equity:.2f}. ---")

            day_start = state["day_start_equity"]
            target_amt = PROFIT_TARGET_PCT / 100.0 * start_bal
            total_amt = MAX_TOTAL_LOSS_PCT / 100.0 * start_bal
            daily_amt = MAX_DAILY_LOSS_PCT / 100.0 * day_start

            # --- PROFIT TARGET (optional) ---
            if USE_PROFIT_TARGET and equity - start_bal >= target_amt:
                log(f"*** TARGET HIT: equity {equity:.2f} (+{equity - start_bal:.2f}, "
                    f"+{PROFIT_TARGET_PCT}%). Closing trades and stopping. ***")
                close_my_positions("PROFIT TARGET HIT — stopping", "good")
                state["challenge_over"] = True
                save_state(state)
                break

            # --- TOTAL LOSS KILL SWITCH ---
            if start_bal - equity >= total_amt:
                log(f"!!! TOTAL LOSS LIMIT: down {start_bal - equity:.2f} "
                    f"(-{MAX_TOTAL_LOSS_PCT}% of {start_bal:.2f}). Closing & stopping. !!!")
                close_my_positions("TOTAL LOSS LIMIT — stopped", "danger")
                state["challenge_over"] = True
                save_state(state)
                break

            # --- DAILY LOSS LIMIT ---
            if not halted_today and (day_start - equity) >= daily_amt:
                log(f"!! DAILY LOSS LIMIT: down {day_start - equity:.2f} today "
                    f"(-{MAX_DAILY_LOSS_PCT}%). No more entries until tomorrow. !!")
                close_my_positions("DAILY LOSS LIMIT — stop for today", "danger")
                halted_today = True

            # --- Manage open trades (partial scale-out + breakeven) ---
            refresh_news()                      # self-throttling
            news = news_state()
            news_blocked = news.get("blocked", False)

            if not halted_today:
                manage_open_positions(news_blocked)

            now_utc = datetime.now(timezone.utc)
            day_utc = now_utc.date()
            past_window = (now_utc.hour, now_utc.minute) >= RANGE_END_UTC
            before_cutoff = (now_utc.hour, now_utc.minute) < TRADE_UNTIL_UTC

            # --- Per symbol work ---
            for s in SYMBOLS:
                info = mt5.symbol_info(s)
                if info is None:
                    continue
                infos[s] = info
                st = sym_state[s]

                df = fetch_data(s, offset)
                if df is None:
                    continue

                # Build today's range once the window has closed.
                if st["range_high"] is None and past_window:
                    rh, rl = compute_range(df, day_utc)
                    if rh is not None and rl is not None and rh > rl:
                        st["range_high"], st["range_low"] = rh, rl
                        st["day"] = today
                        log(f"[{s}] Range set: HIGH={rh:.{info.digits}f} LOW={rl:.{info.digits}f} "
                            f"(width {(rh - rl) / info.point:.0f} pts).")

                if st["range_high"] is None:
                    continue  # range not ready yet today

                # Work only on newly CLOSED candles (iloc[-2]); ignore forming candle.
                closed = df.iloc[-2]
                bar_time = closed["time_utc"]
                if bar_time == st["last_bar"]:
                    continue
                st["last_bar"] = bar_time

                rh, rl = st["range_high"], st["range_low"]
                hi, lo, cl = float(closed["high"]), float(closed["low"]), float(closed["close"])

                # --- If already in a trade: 9EMA trail exit (alert or auto). ---
                pos = position_for(s)
                if pos is not None:
                    if USE_EMA_TRAIL_EXIT:
                        ef = float(closed["ema_fast"])
                        exit_short = pos.type == mt5.POSITION_TYPE_SELL and cl > ef
                        exit_long = pos.type == mt5.POSITION_TYPE_BUY and cl < ef
                        if exit_short or exit_long:
                            if ALERT_ONLY and news_blocked:
                                log(f"[{s}] 9EMA exit signalled but inside news window — "
                                    f"holding (Maven rule).")
                            elif ALERT_ONLY:
                                log(f"[{s}] >>> ACTION: price closed back across the 9 EMA "
                                    f"-> CLOSE {s} now.")
                                _set_alert(f"{s}: 9 EMA exit — close the trade now", "exit")
                                _beep("exit")
                            else:
                                log(f"[{s}] close {cl:.{info.digits}f} crossed back through "
                                    f"9EMA {ef:.{info.digits}f} -> trail exit.")
                                close_one(s, comment="pod2_9ema_exit")
                    continue

                # Track sweeps (wicks beyond the range take liquidity).
                if hi > rh:
                    st["swept_high"] = True
                    st["sweep_high_px"] = max(st["sweep_high_px"] or hi, hi)
                if lo < rl:
                    st["swept_low"] = True
                    st["sweep_low_px"] = min(st["sweep_low_px"] or lo, lo)

                # Gate conditions for a NEW entry.
                can_trade = (not halted_today and past_window and before_cutoff
                             and st["trades_today"] < MAX_TRADES_PER_DAY_PER_SYMBOL
                             and position_for(s) is None)

                trend_ok_short = (not USE_TREND_GATE) or (closed["ema_fast"] < closed["ema_slow"])
                trend_ok_long  = (not USE_TREND_GATE) or (closed["ema_fast"] > closed["ema_slow"])

                acted = False

                # SHORT: swept the high, price reclaimed back inside the range,
                # and CONFIRM_CLOSES consecutive closes are below the 9 EMA.
                if can_trade and st["swept_high"] and cl < rh:
                    confirmed = closes_confirm(df, "SELL", CONFIRM_CLOSES)
                    if confirmed and trend_ok_short:
                        if news_blocked:
                            log(f"[{s}] SHORT confirmed but NEWS WINDOW — holding (no trade around news).")
                            acted = True
                        elif spread_ok(info):
                            risk_amt = RISK_PERCENT / 100.0 * acct.balance
                            sl = st["sweep_high_px"] + SL_BUFFER_POINTS * info.point
                            log(f"[{s}] SHORT setup confirmed ({CONFIRM_CLOSES} closes < 9EMA, "
                                f"sweep {st['sweep_high_px']:.{info.digits}f}).")
                            placed = (signal_entry(info, "SELL", sl, risk_amt) if ALERT_ONLY
                                      else place_trade(info, "SELL", sl, risk_amt))
                            if placed:
                                st["trades_today"] += 1
                                st["swept_high"] = False  # disarm after signal/fill
                        acted = True  # stay armed if not filled (spread/reject) and retry
                    else:
                        why = "EMA not bearish" if not trend_ok_short else \
                              f"awaiting {CONFIRM_CLOSES} closes < 9EMA"
                        log(f"[{s}] swept high & reclaimed, holding ({why}).")
                        acted = True

                # LONG: swept the low, reclaimed into range, CONFIRM_CLOSES closes above 9 EMA.
                elif can_trade and st["swept_low"] and cl > rl:
                    confirmed = closes_confirm(df, "BUY", CONFIRM_CLOSES)
                    if confirmed and trend_ok_long:
                        if news_blocked:
                            log(f"[{s}] LONG confirmed but NEWS WINDOW — holding (no trade around news).")
                            acted = True
                        elif spread_ok(info):
                            risk_amt = RISK_PERCENT / 100.0 * acct.balance
                            sl = st["sweep_low_px"] - SL_BUFFER_POINTS * info.point
                            log(f"[{s}] LONG setup confirmed ({CONFIRM_CLOSES} closes > 9EMA, "
                                f"sweep {st['sweep_low_px']:.{info.digits}f}).")
                            placed = (signal_entry(info, "BUY", sl, risk_amt) if ALERT_ONLY
                                      else place_trade(info, "BUY", sl, risk_amt))
                            if placed:
                                st["trades_today"] += 1
                                st["swept_low"] = False
                        acted = True
                    else:
                        why = "EMA not bullish" if not trend_ok_long else \
                              f"awaiting {CONFIRM_CLOSES} closes > 9EMA"
                        log(f"[{s}] swept low & reclaimed, holding ({why}).")
                        acted = True

                if not acted:
                    # light heartbeat
                    log(f"[{s}] {bar_time.strftime('%H:%M')}UTC close={cl:.{info.digits}f} "
                        f"range[{rl:.{info.digits}f}-{rh:.{info.digits}f}] "
                        f"sweptH={st['swept_high']} sweptL={st['swept_low']} "
                        f"trades={st['trades_today']} | eq={equity:.0f}")

            # --- Broadcast live state for the dashboard ---
            write_status(state, sym_state, halted_today)

            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        log("Manual stop (Ctrl+C). Shutting down cleanly. (Open trades are left running.)")
    finally:
        mt5.shutdown()
        log("MT5 connection closed.")


if __name__ == "__main__":
    main()