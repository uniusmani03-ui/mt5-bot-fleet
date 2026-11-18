"""
M1 Scalper – Bollinger Bands + RSI
- Entries on M1 when price touches outer band and RSI is extreme (>70 or <30)
- TP = 6 pips, SL = 12 pips
- Max 3 concurrent trades, BE at +4 pips
- Daily loss limit: 2%
"""

import os
import sys
import time
import math
from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from pod_core import Bot

# ========== CHOOSE YOUR MT5 ACCOUNT ==========
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0")) or None         # <-- YOUR LOGIN
MT5_PASSWORD = os.getenv("MT5_PASSWORD")   # <-- YOUR PASSWORD
MT5_SERVER   = os.getenv("MT5_SERVER", "MetaQuotes-Demo") # <-- YOUR SERVER
# =============================================

# ============ CONFIG ============
TIMEFRAME_ENTRY = mt5.TIMEFRAME_M1
TIMEFRAME_TREND = mt5.TIMEFRAME_M5

# --- Strategy ---
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 25
RSI_OVERBOUGHT = 75
VOLUME_MULTIPLIER = 1.5

# --- Risk Management ---
RISK_PERCENT = 1.0
MAX_LOT = 0.5
MIN_STOP_PIPS = 10
MAX_STOP_PIPS = 50
TP_SL_RATIO = 1.5
MAX_SPREAD_POINTS = 30
MAX_CONCURRENT_TRADES = 2
DAILY_LOSS_LIMIT_PCT = 2.0
MAX_CONSECUTIVE_LOSSES = 3  # Reserved for future closed-trade monitoring.
MAGIC_BASE = 778500
LOOP_SLEEP = 2
SLIPPAGE = 5

# --- Optional manual symbol filter (empty = auto-filter) ---
MANUAL_SYMBOLS = []  # e.g. ["GBPUSD", "EURUSD", "XAUUSD"]
# =============================================


def get_all_tradeable_symbols():
    """Return visible, fully tradeable broker symbols matching the configured filter."""
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return []

    selected = []
    common = [
        "USD", "JPY", "GBP", "EUR", "CHF", "AUD", "CAD", "NZD",
        "XAG", "XAU", "BTC", "ETH", "US30", "GER30", "UK100",
        "NAS100", "SPX500", "DAX", "FTSE", "JP225",
    ]

    for symbol_info in all_symbols:
        if (
            not symbol_info.visible
            or symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL
            or symbol_info.trade_contract_size <= 0
        ):
            continue

        name = symbol_info.name

        if MANUAL_SYMBOLS:
            if name in MANUAL_SYMBOLS:
                selected.append(name)
            continue

        description = symbol_info.description or ""
        if (
            any(token in name for token in common)
            or "Index" in description
            or "Metal" in description
        ):
            selected.append(name)

    # Preserve broker order while removing duplicates.
    return list(dict.fromkeys(selected))


def pip_size(info):
    """Return an approximate FX-style pip size from MT5 symbol precision."""
    if info.digits in (3, 5):
        return 10 * info.point
    return info.point


def normalize_price(info, price):
    """Round a price to the symbol's configured precision."""
    return round(float(price), int(info.digits))


def calculate_indicators(df):
    close = df["close"]

    ma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std()
    upper = ma + (BB_STD * std)
    lower = ma - (BB_STD * std)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    # If there were gains but no losses, RSI is conventionally 100.
    no_loss = loss == 0
    rsi = rsi.mask(no_loss & (gain > 0), 100.0)
    # If there were losses but no gains, RSI is conventionally 0.
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)

    return upper.iloc[-1], lower.iloc[-1], rsi.iloc[-1]


def get_volume_spike(symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_ENTRY, 0, 25)
    if rates is None or len(rates) < 20:
        return False

    previous = rates[:-1]
    if len(previous) == 0:
        return False

    avg = sum(float(r[5]) for r in previous) / len(previous)
    current = float(rates[-1][5])

    if avg <= 0:
        return False

    return current > VOLUME_MULTIPLIER * avg


def get_trend_filter(symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_TREND, 0, 210)
    if rates is None or len(rates) < 200:
        return 0

    df = pd.DataFrame(rates)
    ema = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    current = df["close"].iloc[-1]

    if current > ema:
        return 1
    if current < ema:
        return -1
    return 0


def get_atr(symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_TREND, 0, 20)
    if rates is None or len(rates) < 14:
        return None

    df = pd.DataFrame(rates)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    return float(atr) if not pd.isna(atr) else None


def volume_decimals(step):
    """Return a sensible decimal precision for an MT5 volume step."""
    if step <= 0:
        return 2
    text = f"{step:.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def get_lot_size(symbol, entry, sl, risk_pct=RISK_PERCENT, max_lot=MAX_LOT):
    """Calculate risk-based lot size using MT5 profit calculation where possible."""
    info = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if info is None or account is None:
        return 0.0

    entry = float(entry)
    sl = float(sl)
    if entry <= 0 or sl <= 0 or entry == sl:
        return 0.0

    risk_amount = (risk_pct / 100.0) * float(account.balance)
    if risk_amount <= 0:
        return 0.0

    order_type = mt5.ORDER_TYPE_BUY if sl < entry else mt5.ORDER_TYPE_SELL

    # Most reliable cross-asset calculation: ask MT5 what 1.0 lot would lose at SL.
    one_lot_profit = mt5.order_calc_profit(order_type, symbol, 1.0, entry, sl)
    if one_lot_profit is not None and one_lot_profit != 0:
        risk_per_lot = abs(float(one_lot_profit))
    else:
        # Fallback to tick value / tick size when order_calc_profit is unavailable.
        tick_value = float(info.trade_tick_value or 0.0)
        tick_size = float(info.trade_tick_size or 0.0)
        if tick_value <= 0 or tick_size <= 0:
            return 0.0
        risk_per_lot = (abs(entry - sl) / tick_size) * tick_value

    if risk_per_lot <= 0:
        return 0.0

    raw_lot = risk_amount / risk_per_lot
    step = float(info.volume_step or 0.01)
    volume_min = float(info.volume_min or step)
    volume_max = float(info.volume_max or max_lot)
    cap = min(float(max_lot), volume_max)

    # Do not force the broker minimum if it would exceed the intended risk.
    if raw_lot < volume_min:
        return 0.0

    lot = min(raw_lot, cap)
    lot = math.floor((lot + 1e-12) / step) * step

    if lot < volume_min:
        return 0.0

    return round(lot, volume_decimals(step))


def place_market_order(symbol, direction, signal_entry, sl, tp, comment, magic):
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        print(f"{symbol}: no live symbol/tick data")
        return False

    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        actual_entry = float(tick.ask)
        sl_distance = max(float(signal_entry) - float(sl), 0.0)
        tp_distance = max(float(tp) - float(signal_entry), 0.0)
        sl = actual_entry - sl_distance
        tp = actual_entry + tp_distance
    else:
        order_type = mt5.ORDER_TYPE_SELL
        actual_entry = float(tick.bid)
        sl_distance = max(float(sl) - float(signal_entry), 0.0)
        tp_distance = max(float(signal_entry) - float(tp), 0.0)
        sl = actual_entry + sl_distance
        tp = actual_entry - tp_distance

    # Respect broker minimum stop distance when one is configured.
    broker_min_stop = float(info.trade_stops_level or 0) * float(info.point)
    if broker_min_stop > 0:
        if direction == "BUY":
            sl = min(sl, actual_entry - broker_min_stop)
            tp = max(tp, actual_entry + broker_min_stop)
        else:
            sl = max(sl, actual_entry + broker_min_stop)
            tp = min(tp, actual_entry - broker_min_stop)

    actual_entry = normalize_price(info, actual_entry)
    sl = normalize_price(info, sl)
    tp = normalize_price(info, tp)

    lot = get_lot_size(symbol, actual_entry, sl)
    if lot <= 0:
        print(f"{symbol}: lot size is below broker/risk minimum")
        return False

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": actual_entry,
        "sl": sl,
        "tp": tp,
        "deviation": SLIPPAGE,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = mt5.order_send(req)
    if result is None:
        print(f"{symbol} order_send returned None: {mt5.last_error()}")
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(
            f"{symbol} {direction}: {lot} lots @ {actual_entry:.{info.digits}f} "
            f"SL={sl:.{info.digits}f} TP={tp:.{info.digits}f}"
        )
        return True

    print(
        f"{symbol} order failed: retcode={result.retcode}, "
        f"comment={getattr(result, 'comment', 'unknown')}"
    )
    return False


def log_trade(symbol, direction, entry, sl, tp, result, profit_pips, profit_usd):
    """Append a completed trade to the CSV log."""
    filename = "multi_scalper_trades.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Date", "Symbol", "Direction", "Entry", "SL", "TP",
                    "Result", "Profit_Pips", "Profit_USD",
                ]
            )
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                symbol,
                direction,
                entry,
                sl,
                tp,
                result,
                profit_pips,
                profit_usd,
            ]
        )


def close_bot_position(symbol, position):
    """Close one position belonging to this bot."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"{symbol}: cannot close position {position.ticket}; no tick data")
        return False

    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "position": position.ticket,
        "volume": position.volume,
        "type": order_type,
        "price": price,
        "deviation": SLIPPAGE,
        "magic": position.magic,
        "comment": "DailyLoss_Close",
    }

    result = mt5.order_send(req)
    if result is None:
        print(f"{symbol}: close failed for {position.ticket}: {mt5.last_error()}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(
            f"{symbol}: close failed for {position.ticket}: "
            f"retcode={result.retcode}, comment={getattr(result, 'comment', 'unknown')}"
        )
        return False

    return True


def main():
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    try:
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            print(f"Login failed: {mt5.last_error()}")
            sys.exit(1)

        symbols = get_all_tradeable_symbols()
        if not symbols:
            print("No tradeable symbols found with the current filter.")
            sys.exit(1)

        print(f"Trading on {len(symbols)} symbols: {', '.join(symbols[:10])}...")

        states = {
            symbol: {
                "day": "",
                "start_balance": 0.0,
                "halted": False,
                "consecutive_losses": 0,
                "last_candle_time": None,
                "trades_today": 0,
            }
            for symbol in symbols
        }
        magics = {symbol: MAGIC_BASE + i for i, symbol in enumerate(symbols)}

        global_day = None
        global_start_balance = 0.0
        global_halted = False

        while True:
            now = datetime.now(timezone.utc)
            today = now.date().isoformat()

            account = mt5.account_info()
            if account is None:
                print(f"account_info failed: {mt5.last_error()}")
                time.sleep(LOOP_SLEEP)
                continue

            # Reset daily state once per new UTC date.
            if global_day != today:
                global_day = today
                global_start_balance = float(account.balance)
                global_halted = False

                for symbol in symbols:
                    states[symbol]["day"] = today
                    states[symbol]["start_balance"] = float(account.balance)
                    states[symbol]["halted"] = False
                    states[symbol]["consecutive_losses"] = 0
                    states[symbol]["trades_today"] = 0

                print(
                    f"New trading day {today}; starting balance="
                    f"{global_start_balance:.2f}"
                )

            if global_halted:
                time.sleep(LOOP_SLEEP)
                continue

            # Global daily equity-loss check.
            if global_start_balance > 0:
                equity_loss = (
                    (global_start_balance - float(account.equity))
                    / global_start_balance
                    * 100.0
                )
            else:
                equity_loss = 0.0

            if equity_loss >= DAILY_LOSS_LIMIT_PCT:
                print(
                    f"GLOBAL DAILY LOSS LIMIT HIT ({equity_loss:.2f}%). "
                    "Closing bot positions and halting trading for today."
                )
                global_halted = True

                for symbol in symbols:
                    positions = mt5.positions_get(symbol=symbol) or ()
                    for position in positions:
                        if position.magic == magics[symbol]:
                            close_bot_position(symbol, position)

                time.sleep(LOOP_SLEEP)
                continue

            for symbol in symbols:
                if states[symbol]["halted"]:
                    continue

                magic = magics[symbol]
                info = mt5.symbol_info(symbol)
                if info is None:
                    continue

                # Spread check.
                if info.spread > MAX_SPREAD_POINTS:
                    continue

                # Existing positions for this symbol.
                positions = mt5.positions_get(symbol=symbol) or ()
                my_positions = [p for p in positions if p.magic == magic]
                if len(my_positions) >= MAX_CONCURRENT_TRADES:
                    continue

                # Fetch M1 data.
                rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_ENTRY, 0, 50)
                if rates is None or len(rates) < max(BB_PERIOD, RSI_PERIOD) + 2:
                    continue

                df = pd.DataFrame(rates)
                upper, lower, rsi = calculate_indicators(df)
                if pd.isna(upper) or pd.isna(lower) or pd.isna(rsi):
                    continue

                last = df.iloc[-1]
                price_close = float(last["close"])
                price_low = float(last["low"])
                price_high = float(last["high"])
                candle_time = int(last["time"])

                if states[symbol]["last_candle_time"] == candle_time:
                    continue
                states[symbol]["last_candle_time"] = candle_time

                trend = get_trend_filter(symbol)
                if trend == 0:
                    continue

                # Volume spike filter, only if the tick exposes a positive volume.
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                tick_volume = getattr(tick, "volume", 0) or 0
                if tick_volume > 0 and not get_volume_spike(symbol):
                    continue

                # Compute dynamic stop distance.
                pip = pip_size(info)
                atr = get_atr(symbol)
                if atr is not None and atr > 0:
                    sl_distance = max(atr * 0.5, MIN_STOP_PIPS * pip)
                else:
                    sl_distance = MIN_STOP_PIPS * pip

                sl_distance = min(sl_distance, MAX_STOP_PIPS * pip)

                # Respect broker minimum stop distance.
                broker_min_stop = float(info.trade_stops_level or 0) * float(info.point)
                if broker_min_stop > sl_distance:
                    sl_distance = broker_min_stop

                # Entry signals.
                if trend == 1 and price_low <= lower and rsi < RSI_OVERSOLD:
                    entry = price_close
                    sl = entry - sl_distance
                    tp = entry + (sl_distance * TP_SL_RATIO)
                    if place_market_order(
                        symbol,
                        "BUY",
                        entry,
                        sl,
                        tp,
                        "Scalp_BB_RSI",
                        magic,
                    ):
                        states[symbol]["trades_today"] += 1

                elif trend == -1 and price_high >= upper and rsi > RSI_OVERBOUGHT:
                    entry = price_close
                    sl = entry + sl_distance
                    tp = entry - (sl_distance * TP_SL_RATIO)
                    if place_market_order(
                        symbol,
                        "SELL",
                        entry,
                        sl,
                        tp,
                        "Scalp_BB_RSI",
                        magic,
                    ):
                        states[symbol]["trades_today"] += 1

            time.sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()