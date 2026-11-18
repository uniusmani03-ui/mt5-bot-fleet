"""
=============================================================================
 MT5 FOREX MICRO-SCALPING BOT  (Demo-account ready)
=============================================================================
 Strategy : 9/21 EMA crossover + VWAP filter + 4-period RSI filter
 Timeframe: M1
 Risk     : Fixed cash risk per trade with dynamic lot sizing, 1:1.5 RR
 Safety   : Spread filter, single-position concurrency, daily drawdown kill
            switch, strict NON-REPAINTING (closed-candle) signal logic.

 -----------------------------------------------------------------------------
 INSTALL DEPENDENCIES (run once in your terminal):

     pip install MetaTrader5 pandas numpy

 NOTE: 'pandas_ta' is intentionally NOT used. Its releases have repeatedly
 broken across pandas/numpy versions. All indicators here are computed with
 plain pandas/numpy so the bot has zero fragile dependencies. The MetaTrader5
 package only works on Windows (or Wine) with an installed, logged-in MT5
 terminal.
 -----------------------------------------------------------------------------

 BEFORE RUNNING:
   1. Open your MT5 terminal and log into a DEMO account.
   2. Enable "Algo Trading" (the button in the toolbar must be green).
   3. Make sure EURUSD is visible in Market Watch.
=============================================================================
"""

import os
import csv
import sys
import time
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
SYMBOL              = "EURUSD"
TIMEFRAME           = mt5.TIMEFRAME_M1   # 1-minute chart
RISK_PER_TRADE      = 2.0                # cash risk per trade in account currency (e.g. £2)
MAX_SPREAD_POINTS   = 10                 # max allowed spread (points) to avoid fee traps
SLIPPAGE_POINTS     = 3                  # max execution deviation (points)
MAX_DAILY_DRAWDOWN  = 50.0               # kill switch: stop if equity drops this much from start

# --- Strategy / execution parameters ---
STOP_LOSS_POINTS    = 50                 # SL distance from entry (50 points = 5 pips on a 5-digit pair)
RR_RATIO            = 1.5                # take-profit = SL distance * 1.5  (1 : 1.5)
EMA_FAST            = 9
EMA_SLOW            = 21
RSI_PERIOD          = 4
RSI_OVERBOUGHT      = 70
RSI_OVERSOLD        = 30
CANDLES_TO_FETCH    = 200
MAGIC_NUMBER        = 234000             # tags this bot's trades
LOOP_SLEEP_SECONDS  = 1                  # rate-limit the local API

# --- CSV trade logging ---
TRADE_LOG_FILE      = "mt5_trade_log.csv"

# --- Session filter (London/NY overlap) ---
# Times are in UTC. The classic London/NY overlap is ~13:00-16:00 UTC.
# NOTE: this uses your machine's real UTC clock, NOT the broker's server
# time (server time is often UTC+2/+3 and shifts with DST). Adjust the
# window if your liquidity profile differs, or set USE_SESSION_FILTER=False
# to trade around the clock.
USE_SESSION_FILTER  = True
SESSION_START_UTC   = (13, 0)            # (hour, minute) inclusive
SESSION_END_UTC     = (16, 0)            # (hour, minute) exclusive

# Optional explicit login. Leave as None to attach to the already-running,
# already-logged-in terminal (recommended for demo).
MT5_LOGIN    = None       # e.g. 12345678
MT5_PASSWORD = None       # e.g. "your_password"
MT5_SERVER   = None       # e.g. "MetaQuotes-Demo"
MT5_PATH     = None       # e.g. r"C:\Program Files\MetaTrader 5\terminal64.exe"


# =============================================================================
# 2. LOGGING HELPER
# =============================================================================
def log(message: str) -> None:
    """Timestamped heartbeat logging."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


_CSV_FIELDS = [
    "timestamp_utc", "event", "symbol", "direction", "volume",
    "price", "sl", "tp", "retcode", "ticket", "deal",
    "balance", "equity", "comment",
]


def log_trade_csv(event: str, symbol: str, direction: str = "",
                  volume: float = 0.0, price: float = 0.0,
                  sl: float = 0.0, tp: float = 0.0,
                  retcode=None, ticket=None, deal=None,
                  comment: str = "") -> None:
    """Append one row per order event (sent / filled / rejected / closed) to CSV."""
    acct = mt5.account_info()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "event":     event,
        "symbol":    symbol,
        "direction": direction,
        "volume":    volume,
        "price":     price,
        "sl":        sl,
        "tp":        tp,
        "retcode":   retcode if retcode is not None else "",
        "ticket":    ticket if ticket is not None else "",
        "deal":      deal if deal is not None else "",
        "balance":   round(acct.balance, 2) if acct else "",
        "equity":    round(acct.equity, 2) if acct else "",
        "comment":   comment,
    }
    try:
        write_header = not os.path.exists(TRADE_LOG_FILE)
        with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:  # logging must never crash the bot
        log(f"WARNING: failed to write trade CSV -> {exc}")


def in_trading_session() -> bool:
    """True if current UTC time is inside the configured session window."""
    if not USE_SESSION_FILTER:
        return True
    now = datetime.now(timezone.utc)
    start = now.replace(hour=SESSION_START_UTC[0], minute=SESSION_START_UTC[1],
                        second=0, microsecond=0)
    end = now.replace(hour=SESSION_END_UTC[0], minute=SESSION_END_UTC[1],
                      second=0, microsecond=0)
    if start <= end:                      # same-day window (e.g. 13:00-16:00)
        return start <= now < end
    return now >= start or now < end      # window wraps past midnight


# =============================================================================
# 3. CONNECTION & VALIDATION
# =============================================================================
def initialize_mt5() -> bool:
    """Connect to the MT5 terminal. Returns True on success."""
    kwargs = {}
    if MT5_PATH:
        kwargs["path"] = MT5_PATH
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        kwargs.update(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)

    if not mt5.initialize(**kwargs):
        log(f"ERROR: mt5.initialize() failed -> {mt5.last_error()}")
        return False

    term = mt5.terminal_info()
    acct = mt5.account_info()
    if term is None or acct is None:
        log(f"ERROR: Could not read terminal/account info -> {mt5.last_error()}")
        return False

    if not term.trade_allowed:
        log("ERROR: Algo trading is DISABLED in the terminal. Enable the 'Algo Trading' button and restart.")
        return False

    log(f"Connected. Account #{acct.login} | Server: {acct.server} | "
        f"Balance: {acct.balance:.2f} {acct.currency} | Leverage 1:{acct.leverage}")
    return True


def validate_symbol(symbol: str):
    """Ensure symbol exists and is selected in Market Watch. Returns symbol_info or None."""
    info = mt5.symbol_info(symbol)
    if info is None:
        log(f"ERROR: Symbol '{symbol}' not found on this server -> {mt5.last_error()}")
        return None
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            log(f"ERROR: Failed to select '{symbol}' in Market Watch -> {mt5.last_error()}")
            return None
        info = mt5.symbol_info(symbol)  # refresh after enabling
    log(f"Symbol ready: {symbol} | digits={info.digits} | point={info.point} | "
        f"vol step={info.volume_step} | min={info.volume_min} | max={info.volume_max}")
    return info


# =============================================================================
# 4. DATA HANDLING  (STRICT NON-REPAINTING)
# =============================================================================
def fetch_data(symbol: str, timeframe: int, count: int):
    """Pull candles and return a time-ascending DataFrame, or None on failure."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < EMA_SLOW + 5:
        log(f"WARNING: Not enough candle data returned -> {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df  # ascending: iloc[-1] is the CURRENT, still-forming candle


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA(9), EMA(21), RSI(4) and daily-anchored VWAP."""
    df = df.copy()

    # --- EMAs ---
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # --- RSI (Wilder's smoothing) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = df["rsi"].fillna(100.0)  # zero average loss => RSI 100

    # --- Daily session VWAP (anchored to broker-server day) ---
    df["session"] = df["time"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["_tpv"] = typical * df["tick_volume"]
    cum_tpv = df.groupby("session")["_tpv"].cumsum()
    cum_vol = df.groupby("session")["tick_volume"].cumsum()
    df["vwap"] = cum_tpv / cum_vol.replace(0, np.nan)

    return df


def check_signal(df: pd.DataFrame):
    """
    Evaluate signals ONLY on closed candles.
      df.iloc[-1] = current forming candle  -> IGNORED (anti-repaint)
      df.iloc[-2] = last CLOSED candle      -> the candle we trade on
      df.iloc[-3] = previous closed candle  -> used for crossover detection
    Returns ("BUY"|"SELL"|None, last_closed_candle_time).
    """
    curr = df.iloc[-2]   # last fully closed candle
    prev = df.iloc[-3]   # the one before it

    # EMA crossover (closed-candle confirmed)
    cross_up   = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
    cross_down = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

    above_vwap = curr["close"] > curr["vwap"]
    below_vwap = curr["close"] < curr["vwap"]

    if cross_up and above_vwap and curr["rsi"] < RSI_OVERBOUGHT:
        return "BUY", curr["time"]
    if cross_down and below_vwap and curr["rsi"] > RSI_OVERSOLD:
        return "SELL", curr["time"]
    return None, curr["time"]


# =============================================================================
# 5. FILTERS
# =============================================================================
def open_positions_count(symbol: str) -> int:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len(positions)


def spread_ok(symbol_info) -> bool:
    """Live spread (Ask - Bid) in points must be <= MAX_SPREAD_POINTS."""
    tick = mt5.symbol_info_tick(symbol_info.name)
    if tick is None:
        log("WARNING: Could not read live tick for spread check.")
        return False
    spread_points = (tick.ask - tick.bid) / symbol_info.point
    if spread_points > MAX_SPREAD_POINTS:
        log(f"SKIP: spread {spread_points:.1f} pts > limit {MAX_SPREAD_POINTS} pts.")
        return False
    return True


# =============================================================================
# 6. RISK MANAGEMENT & EXECUTION MATH
# =============================================================================
def _volume_decimals(step: float) -> int:
    if step <= 0:
        return 2
    return max(0, int(round(-math.log10(step))))


def calculate_lot_size(symbol_info, sl_distance_price: float, risk_amount: float):
    """
    Dynamic lot sizing so that hitting the SL loses exactly `risk_amount`.
        loss_per_lot = (sl_distance / tick_size) * tick_value
        volume       = risk_amount / loss_per_lot
    Rounds DOWN to the broker's volume_step (conservative) and clamps to limits.
    Returns a valid volume, or None if it cannot be sized safely.
    """
    tick_value = symbol_info.trade_tick_value
    tick_size  = symbol_info.trade_tick_size
    if tick_size <= 0 or tick_value <= 0 or sl_distance_price <= 0:
        log("ERROR: Invalid tick_value/tick_size/SL distance for lot sizing.")
        return None

    loss_per_lot = (sl_distance_price / tick_size) * tick_value
    if loss_per_lot <= 0:
        log("ERROR: loss_per_lot computed as <= 0.")
        return None

    raw_volume = risk_amount / loss_per_lot

    step = symbol_info.volume_step
    decimals = _volume_decimals(step)
    volume = math.floor(raw_volume / step) * step
    volume = round(volume, decimals)

    if volume < symbol_info.volume_min:
        log(f"SKIP: required lot {volume} < broker min {symbol_info.volume_min}. "
            f"Risk of {risk_amount} is too small for this SL distance.")
        return None
    if volume > symbol_info.volume_max:
        volume = symbol_info.volume_max
        log(f"NOTE: capped lot to broker max {volume}.")

    est_loss = (sl_distance_price / tick_size) * tick_value * volume
    log(f"Lot sizing: volume={volume} | est. loss at SL ~= {est_loss:.2f} (target {risk_amount}).")
    return volume


def get_filling_mode(symbol_info) -> int:
    """Pick a filling mode the broker actually supports for this symbol."""
    fm = symbol_info.filling_mode
    if fm & 2:  # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    if fm & 1:  # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


# =============================================================================
# 7. ORDER PLACEMENT
# =============================================================================
def place_order(direction: str, symbol_info) -> None:
    """Build and send a market order with dynamic volume, SL, TP."""
    tick = mt5.symbol_info_tick(symbol_info.name)
    if tick is None:
        log("ERROR: No tick available; aborting order.")
        return

    point  = symbol_info.point
    digits = symbol_info.digits

    # Respect broker's minimum stop distance.
    sl_points = max(STOP_LOSS_POINTS, symbol_info.trade_stops_level)
    if sl_points > STOP_LOSS_POINTS:
        log(f"NOTE: broker min stop level is {symbol_info.trade_stops_level} pts; "
            f"using {sl_points} pts instead of {STOP_LOSS_POINTS}.")
    sl_distance = sl_points * point
    tp_distance = sl_distance * RR_RATIO

    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - sl_distance
        tp = price + tp_distance
    else:  # SELL
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + sl_distance
        tp = price - tp_distance

    volume = calculate_lot_size(symbol_info, sl_distance, RISK_PER_TRADE)
    if volume is None:
        return

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol_info.name,
        "volume":       volume,
        "type":         order_type,
        "price":        round(price, digits),
        "sl":           round(sl, digits),
        "tp":           round(tp, digits),
        "deviation":    SLIPPAGE_POINTS,
        "magic":        MAGIC_NUMBER,
        "comment":      "ema_vwap_scalper",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(symbol_info),
    }

    log(f"SENDING {direction}: vol={volume} entry={request['price']} "
        f"SL={request['sl']} TP={request['tp']}")
    result = mt5.order_send(request)

    if result is None:
        log(f"ERROR: order_send returned None -> {mt5.last_error()}")
        log_trade_csv("send_failed", symbol_info.name, direction, volume,
                      request["price"], request["sl"], request["tp"],
                      comment=str(mt5.last_error()))
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"ORDER REJECTED: retcode={result.retcode} ({result.comment})")
        log_trade_csv("rejected", symbol_info.name, direction, volume,
                      request["price"], request["sl"], request["tp"],
                      retcode=result.retcode, comment=result.comment)
        return

    log(f"FILLED: ticket={result.order} deal={result.deal} "
        f"price={result.price} volume={result.volume}")
    log_trade_csv("filled", symbol_info.name, direction, result.volume,
                  result.price, request["sl"], request["tp"],
                  retcode=result.retcode, ticket=result.order, deal=result.deal,
                  comment="ema_vwap_scalper")


# =============================================================================
# 8. SAFETY: CLOSE POSITIONS & DRAWDOWN
# =============================================================================
def close_position(pos, symbol_info) -> None:
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        log(f"ERROR: cannot close {pos.ticket}; no tick.")
        return
    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         order_type,
        "position":     pos.ticket,
        "price":        round(price, symbol_info.digits),
        "deviation":    SLIPPAGE_POINTS,
        "magic":        MAGIC_NUMBER,
        "comment":      "kill_switch_close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(symbol_info),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"ERROR closing {pos.ticket}: {mt5.last_error()} / result={result}")
        log_trade_csv("close_failed", pos.symbol, volume=pos.volume,
                      price=request["price"], ticket=pos.ticket,
                      comment="kill_switch_close")
    else:
        log(f"Closed position {pos.ticket}.")
        log_trade_csv("closed", pos.symbol, volume=pos.volume,
                      price=result.price, retcode=result.retcode,
                      ticket=pos.ticket, deal=result.deal,
                      comment="kill_switch_close")


def close_all_positions(symbol: str, symbol_info) -> None:
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for pos in positions:
        close_position(pos, symbol_info)


def check_drawdown(initial_balance: float, symbol: str, symbol_info) -> None:
    """Terminate the bot if equity has fallen by MAX_DAILY_DRAWDOWN from start."""
    acct = mt5.account_info()
    if acct is None:
        return
    drawdown = initial_balance - acct.equity
    if drawdown >= MAX_DAILY_DRAWDOWN:
        log(f"!!! KILL SWITCH: drawdown {drawdown:.2f} >= {MAX_DAILY_DRAWDOWN}. "
            f"Closing all positions and terminating.")
        close_all_positions(symbol, symbol_info)
        mt5.shutdown()
        sys.exit(0)


# =============================================================================
# 9. MAIN LOOP
# =============================================================================
def main() -> None:
    if not initialize_mt5():
        sys.exit(1)

    symbol_info = validate_symbol(SYMBOL)
    if symbol_info is None:
        mt5.shutdown()
        sys.exit(1)

    acct = mt5.account_info()
    initial_balance = acct.balance
    log(f"Initial balance recorded: {initial_balance:.2f} {acct.currency}. "
        f"Drawdown kill switch at -{MAX_DAILY_DRAWDOWN}.")
    if USE_SESSION_FILTER:
        log(f"Session filter ON: entries only {SESSION_START_UTC[0]:02d}:{SESSION_START_UTC[1]:02d}"
            f"-{SESSION_END_UTC[0]:02d}:{SESSION_END_UTC[1]:02d} UTC. "
            f"Currently {'INSIDE' if in_trading_session() else 'OUTSIDE'} the window.")
    else:
        log("Session filter OFF: trading 24h.")
    log(f"Trade log -> {os.path.abspath(TRADE_LOG_FILE)}")

    last_processed_bar = None  # avoids re-evaluating the same closed candle

    log("Bot is live. Watching the market...")
    try:
        while True:
            # 1) Safety first.
            check_drawdown(initial_balance, SYMBOL, symbol_info)

            # 2) Refresh symbol info (spreads/levels can change).
            symbol_info = mt5.symbol_info(SYMBOL)
            if symbol_info is None:
                log("WARNING: lost symbol info; retrying.")
                time.sleep(LOOP_SLEEP_SECONDS)
                continue

            # 3) Data + indicators.
            df = fetch_data(SYMBOL, TIMEFRAME, CANDLES_TO_FETCH)
            if df is None:
                time.sleep(LOOP_SLEEP_SECONDS)
                continue
            df = compute_indicators(df)

            signal, bar_time = check_signal(df)

            # 4) Only act once per newly-closed candle.
            if bar_time != last_processed_bar:
                last_processed_bar = bar_time

                if signal is not None:
                    log(f"SIGNAL: {signal} on closed candle {bar_time} "
                        f"(EMA{EMA_FAST}/{EMA_SLOW} cross + VWAP + RSI{RSI_PERIOD}).")

                    if not in_trading_session():
                        log(f"SKIP: outside trading session "
                            f"({SESSION_START_UTC[0]:02d}:{SESSION_START_UTC[1]:02d}-"
                            f"{SESSION_END_UTC[0]:02d}:{SESSION_END_UTC[1]:02d} UTC).")
                    elif open_positions_count(SYMBOL) > 0:
                        log("SKIP: a position is already open (one trade at a time).")
                    elif not spread_ok(symbol_info):
                        pass  # spread_ok already logged the reason
                    else:
                        place_order(signal, symbol_info)
                else:
                    # quiet heartbeat on each new candle
                    last = df.iloc[-2]
                    log(f"Heartbeat | closed {bar_time} | close={last['close']:.{symbol_info.digits}f} "
                        f"VWAP={last['vwap']:.{symbol_info.digits}f} RSI={last['rsi']:.1f} "
                        f"| positions={open_positions_count(SYMBOL)}")

            time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        log("Manual stop requested (Ctrl+C). Shutting down cleanly.")
    finally:
        mt5.shutdown()
        log("MT5 connection closed.")


if __name__ == "__main__":
    main()
