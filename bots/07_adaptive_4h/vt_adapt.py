#!/usr/bin/env python3
"""
VT Markets / MetaTrader 5 adaptive multi-strategy FX bot (DEMO by default, LIVE opt-in)

Implements:
  1) Donchian Trend Portfolio (20/10)
  2) Donchian Ensemble (20/10, 40/20, 80/40; 1/3 risk each)
  3) H4 Liquidity Sweep Reversal

A transparent regime router chooses ONE strategy family per FX symbol, refreshing daily and on each new completed H4 candle.
Existing positions are always managed by the strategy that opened them.

IMPORTANT SAFETY DESIGN:
  - DEMO by default. REAL-money accounts require an explicit three-part opt-in (see below).
  - DRY-RUN by default. Set BOT_EXECUTE_ORDERS=true to allow orders.
  - Uses only completed candles for signals.
  - Persists kill switches / baselines / routing state to JSON.
  - No martingale, grid, averaging down, recovery sizing, or stop widening.
  - Default risk is 0.10% of equity per normal trade, with a fixed 2:1 reward:risk target.

Install on the same Windows machine as MetaTrader 5:
    py -m pip install MetaTrader5 pandas numpy

Typical use (MT5 already open and logged into a VT Markets demo account):
    py vt_markets_adaptive_2R_0.1pct_h4_scored_demo_bot.py

Enable DEMO order execution only after dry-run review:
    set BOT_EXECUTE_ORDERS=true
    py vt_markets_adaptive_2R_0.1pct_h4_scored_demo_bot.py

LIVE (REAL-MONEY) TRADING -- three opt-ins are required to connect to a real account:
    set BOT_ALLOW_LIVE=true
    set BOT_LIVE_ACK=I_ACCEPT_REAL_MONEY_RISK
    py vt_markets_adaptive_2R_0.1pct_h4_scored_demo_bot.py --live      # dry run on live prices

Add the execution switch to actually send orders with real capital:
    set BOT_EXECUTE_ORDERS=true
    py vt_markets_adaptive_2R_0.1pct_h4_scored_demo_bot.py --live

Any of the three opt-ins missing and a REAL account is refused outright. --live is
not persisted anywhere; it must be passed on every launch.

Optional live-only guardrails:
    BOT_LIVE_MIN_EQUITY=500        # sticky halt if live equity drops below this absolute floor
    BOT_LIVE_RISK_PER_TRADE=0.0005 # overrides BOT_RISK_PER_TRADE on real accounts only
    BOT_LIVE_MAX_LOTS=0.50         # tighter than the 5.00 global ceiling, live only

Optional risk overrides:
    BOT_RISK_PER_TRADE=0.001   # 0.10%
    BOT_REWARD_RISK=2.0        # 2:1 reward:risk

Optional login environment variables:
    MT5_PATH=C:\\Program Files\\MetaTrader 5\\terminal64.exe
    MT5_LOGIN=12345678
    MT5_PASSWORD=...
    MT5_SERVER=...

Manual-review reset after a sticky monthly/full halt:
    set BOT_MANUAL_REVIEW_ACK=YES_I_REVIEWED
    py vt_markets_adaptive_2R_0.1pct_h4_scored_demo_bot.py --review-reset

This software is execution tooling, not a claim of profitability. Forward-test on demo
until the edge is statistically established; live mode risks real capital and can lose it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import MetaTrader5 as mt5
except Exception as exc:  # pragma: no cover - lets --help / source inspection still work
    mt5 = None
    MT5_IMPORT_ERROR = exc
else:
    MT5_IMPORT_ERROR = None

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MAGIC_DTP20 = 26081001
MAGIC_EN20 = 26081011
MAGIC_EN40 = 26081012
MAGIC_EN80 = 26081013
MAGIC_LSR = 26081021
BOT_MAGICS = {MAGIC_DTP20, MAGIC_EN20, MAGIC_EN40, MAGIC_EN80, MAGIC_LSR}

MAGIC_META = {
    MAGIC_DTP20: ("DTP20", 20, 10),
    MAGIC_EN20: ("ENS20", 20, 10),
    MAGIC_EN40: ("ENS40", 40, 20),
    MAGIC_EN80: ("ENS80", 80, 40),
    MAGIC_LSR: ("LSR", None, None),
}

FIAT_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
    "CNH", "CNY", "HKD", "SGD", "NOK", "SEK", "DKK", "PLN",
    "CZK", "HUF", "TRY", "ZAR", "MXN", "ILS", "THB", "RON",
    "AED", "SAR", "RUB",
}

CURRENCY_BLOCK = {
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "HAVEN", "CHF": "HAVEN",
    "AUD": "COMMODITY", "NZD": "COMMODITY", "CAD": "COMMODITY",
}

# User-specified core list is retained as a preference/tie-breaker, but by default
# discovery scans every tradable FX pair exposed by the connected MT5 account.
CORE_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD",
}


@dataclass
class Config:
    state_path: Path = Path(os.getenv("BOT_STATE_PATH", "vt_adaptive_state.json"))
    log_path: Path = Path(os.getenv("BOT_LOG_PATH", "vt_adaptive_bot.log"))
    execute_orders: bool = os.getenv("BOT_EXECUTE_ORDERS", "false").lower() == "true"

    # --- REAL-MONEY OPT-IN -------------------------------------------------
    # A real account is only accepted when allow_live is true, live_ack matches
    # the exact phrase, AND --live was passed on the command line (which sets
    # live_flag at runtime). Defaults keep the bot demo-only.
    allow_live: bool = os.getenv("BOT_ALLOW_LIVE", "false").lower() == "true"
    live_ack: str = os.getenv("BOT_LIVE_ACK", "")
    live_flag: bool = False  # set by --live in main(); never read from disk/env
    live_ack_phrase: str = "I_ACCEPT_REAL_MONEY_RISK"

    # Live-only guardrails. None/0 means "fall back to the demo setting".
    live_min_equity: float = float(os.getenv("BOT_LIVE_MIN_EQUITY", "0") or 0)
    live_risk_per_trade: float = float(os.getenv("BOT_LIVE_RISK_PER_TRADE", "0") or 0)
    live_max_lots: float = float(os.getenv("BOT_LIVE_MAX_LOTS", "0") or 0)
    # -----------------------------------------------------------------------

    loop_seconds: int = int(os.getenv("BOT_LOOP_SECONDS", "30"))
    h4_entry_grace_seconds: int = int(os.getenv("BOT_H4_ENTRY_GRACE_SECONDS", "120"))

    # 0.10% of equity per normal trade. The ensemble splits this across 3 sleeves.
    risk_per_trade: float = float(os.getenv("BOT_RISK_PER_TRADE", "0.001"))
    # Fixed reward:risk target. 2.0 means TP is twice the initial stop distance.
    reward_risk: float = float(os.getenv("BOT_REWARD_RISK", "2.0"))
    max_lots: float = 5.0
    min_stop_pips: float = 8.0
    max_open_positions: int = 6
    max_total_risk: float = 0.03
    max_same_direction_block: int = 2
    max_currency_r: float = 2.0
    max_position_corr: float = 0.70

    daily_loss_limit: float = 0.02
    weekly_loss_limit: float = 0.04
    monthly_loss_limit: float = 0.07
    half_risk_peak_dd: float = 0.12
    full_stop_peak_dd: float = 0.18

    rollover_before_min: int = 10
    rollover_after_min: int = 15
    broker_offset_check_hours: int = 12
    allowed_broker_offsets: Tuple[int, ...] = field(default_factory=lambda: tuple(
        int(x.strip()) for x in os.getenv("BOT_ALLOWED_BROKER_UTC_OFFSETS", "2,3").split(",") if x.strip()
    ))

    # Spread caps are deliberately configurable because VT symbol groups/account types
    # can have different spreads. Defaults are conservative fallbacks, not promises.
    major_spread_cap_pips: float = float(os.getenv("BOT_MAJOR_SPREAD_CAP_PIPS", "2.0"))
    cross_spread_cap_pips: float = float(os.getenv("BOT_CROSS_SPREAD_CAP_PIPS", "3.5"))
    exotic_spread_cap_pips: float = float(os.getenv("BOT_EXOTIC_SPREAD_CAP_PIPS", "8.0"))

    deviation_points: int = int(os.getenv("BOT_DEVIATION_POINTS", "20"))

    # H4 liquidity-sweep quality score. A sweep + trend alignment are mandatory;
    # engulfing/outside-bar structure, strong trend, healthy volatility and low spread
    # add confidence. 4/7 keeps the base setup tradable without forcing engulfing.
    lsr_min_quality_points: int = int(os.getenv("BOT_LSR_MIN_QUALITY_POINTS", "4"))

    # To strictly prevent >6 positions if several stops trigger together, pending orders
    # consume future position slots too. This is more conservative than the raw rule.
    reserve_slots_for_pending: bool = True


CFG = Config()
STOP_REQUESTED = False
IS_LIVE = False          # True once a REAL account has passed the opt-in gate
LIVE_BANNER_SHOWN = False


# -----------------------------------------------------------------------------
# Logging / state
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(CFG.log_path, encoding="utf-8"))
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def default_state() -> dict:
    return {
        "version": 1,
        "day_key": None,
        "week_key": None,
        "month_key": None,
        "day_start_equity": None,
        "week_start_equity": None,
        "month_start_equity": None,
        "peak_equity": None,
        "daily_halt_key": None,
        "weekly_halt_key": None,
        "manual_halt": False,
        "manual_halt_reason": None,
        "broker_offset": None,
        "broker_offset_verified_at": None,
        "last_d1_open": {},
        "last_h4_open": {},
        "route": {},
        "route_scores": {},
    }


def load_state() -> dict:
    state = default_state()
    if CFG.state_path.exists():
        try:
            loaded = json.loads(CFG.state_path.read_text(encoding="utf-8"))
            state.update(loaded)
        except Exception as exc:
            logging.error("Could not read state file %s: %s", CFG.state_path, exc)
            raise
    return state


def save_state(state: dict) -> None:
    tmp = CFG.state_path.with_suffix(CFG.state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(CFG.state_path)


# -----------------------------------------------------------------------------
# Basic MT5 helpers
# -----------------------------------------------------------------------------

def require_mt5() -> None:
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 Python package is not available. On the Windows machine running MT5, "
            "install it with: py -m pip install MetaTrader5 pandas numpy"
        ) from MT5_IMPORT_ERROR


def initialize_mt5() -> None:
    require_mt5()
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    kwargs = {}
    if login:
        kwargs["login"] = int(login)
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server

    ok = mt5.initialize(path, **kwargs) if path else mt5.initialize(**kwargs)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def live_opt_in_missing() -> List[str]:
    """Requirements for *connecting* to a real account that are not satisfied.

    BOT_EXECUTE_ORDERS is deliberately not part of this list: with the three
    opt-ins present but execute_orders false, the bot connects to the real
    account and runs a full DRY-RUN so signals can be inspected against live
    prices without a single order being sent.
    """
    missing = []
    if not CFG.allow_live:
        missing.append("BOT_ALLOW_LIVE=true")
    if CFG.live_ack != CFG.live_ack_phrase:
        missing.append(f"BOT_LIVE_ACK={CFG.live_ack_phrase}")
    if not CFG.live_flag:
        missing.append("--live command-line flag")
    return missing


def effective_risk_per_trade() -> float:
    if IS_LIVE and CFG.live_risk_per_trade > 0:
        return CFG.live_risk_per_trade
    return CFG.risk_per_trade


def effective_max_lots() -> float:
    if IS_LIVE and CFG.live_max_lots > 0:
        return min(CFG.live_max_lots, CFG.max_lots)
    return CFG.max_lots


def assert_account_and_trading_ready() -> Tuple[object, object]:
    global IS_LIVE, LIVE_BANNER_SHOWN

    acct = mt5.account_info()
    term = mt5.terminal_info()
    if acct is None or term is None:
        raise RuntimeError(f"Cannot read account/terminal info: {mt5.last_error()}")

    real_const = getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
    demo_const = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
    contest_const = getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1)

    if acct.trade_mode == real_const:
        missing = live_opt_in_missing()
        if missing:
            raise RuntimeError(
                "REAL-MONEY ACCOUNT DETECTED and live trading is not fully enabled. "
                "Missing: " + ", ".join(missing) + ". "
                "Connect a demo account, or supply every requirement to trade real capital. "
                "(Tip: the three opt-ins without BOT_EXECUTE_ORDERS gives a real-account dry run.)"
            )
        IS_LIVE = True
        if not LIVE_BANNER_SHOWN:
            LIVE_BANNER_SHOWN = True
            logging.warning("=" * 78)
            logging.warning("LIVE MODE ACTIVE - REAL MONEY IS AT RISK ON ACCOUNT %s @ %s", acct.login, acct.server)
            logging.warning(
                "equity=%.2f | risk/trade=%.4f%% | max lots=%.2f | equity floor=%s",
                float(acct.equity), effective_risk_per_trade() * 100.0,
                effective_max_lots(),
                f"{CFG.live_min_equity:.2f}" if CFG.live_min_equity > 0 else "none",
            )
            if CFG.execute_orders:
                logging.warning("BOT_EXECUTE_ORDERS=true -> ORDERS WILL BE SENT TO THE MARKET.")
            else:
                logging.warning("BOT_EXECUTE_ORDERS is not true -> DRY-RUN: no orders will be sent.")
            logging.warning("Kill the process (Ctrl+C) or disable Algo Trading in MT5 to stop new entries.")
            logging.warning("=" * 78)
    elif acct.trade_mode in (demo_const, contest_const):
        IS_LIVE = False
    else:
        raise RuntimeError(f"Unrecognised account trade_mode={acct.trade_mode}; refusing to trade.")

    if IS_LIVE and CFG.live_min_equity > 0 and float(acct.equity) < CFG.live_min_equity:
        raise RuntimeError(
            f"LIVE equity {float(acct.equity):.2f} is below the configured floor "
            f"{CFG.live_min_equity:.2f}. Halting."
        )

    if not bool(getattr(term, "connected", False)):
        raise RuntimeError("MT5 terminal is not connected.")
    if not bool(getattr(term, "trade_allowed", False)):
        raise RuntimeError("MT5 terminal trading / Algo Trading is not allowed.")
    if hasattr(acct, "trade_allowed") and not bool(acct.trade_allowed):
        raise RuntimeError("Trading is not allowed on this account.")
    if hasattr(acct, "trade_expert") and not bool(acct.trade_expert):
        raise RuntimeError("Expert/algorithmic trading is not allowed on this account.")
    return acct, term


# Backwards-compatible alias: the gate is no longer demo-only, but every existing
# call site still routes through the same checks.
assert_demo_and_trading_ready = assert_account_and_trading_ready


def is_hedging_account(acct: object) -> bool:
    hedging_const = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
    return int(getattr(acct, "margin_mode", -999)) == hedging_const


def rates(symbol: str, timeframe: int, count: int, completed_only: bool = True) -> Optional[pd.DataFrame]:
    start = 1 if completed_only else 0
    arr = mt5.copy_rates_from_pos(symbol, timeframe, start, count)
    if arr is None or len(arr) < count * 0.75:
        return None
    df = pd.DataFrame(arr)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def current_bar_open(symbol: str, timeframe: int) -> Optional[datetime]:
    arr = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if arr is None or len(arr) != 1:
        return None
    return datetime.fromtimestamp(int(arr[0]["time"]), tz=timezone.utc)


def pip_size_from_info(info: object) -> float:
    quote = str(getattr(info, "currency_profit", ""))
    return 0.01 if quote == "JPY" else 0.0001


def spread_pips(symbol: str) -> Optional[float]:
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None or tick.ask <= 0 or tick.bid <= 0:
        return None
    return (tick.ask - tick.bid) / pip_size_from_info(info)


def spread_cap_pips(info: object) -> float:
    base = str(getattr(info, "currency_base", ""))
    quote = str(getattr(info, "currency_profit", ""))
    majors = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
    pair = f"{base}{quote}"
    if pair in {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"}:
        return CFG.major_spread_cap_pips
    if base in majors and quote in majors:
        return CFG.cross_spread_cap_pips
    return CFG.exotic_spread_cap_pips


def normalize_price(info: object, price: float) -> float:
    digits = int(getattr(info, "digits", 5))
    return round(float(price), digits)


def floor_volume(raw: float, info: object) -> Optional[float]:
    vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
    hard_cap = effective_max_lots()
    vmax = min(float(getattr(info, "volume_max", hard_cap) or hard_cap), hard_cap)
    step = float(getattr(info, "volume_step", vmin) or vmin)
    capped = min(raw, vmax)
    floored = math.floor((capped + 1e-12) / step) * step
    decimals = max(0, int(round(-math.log10(step))) if step < 1 else 0)
    floored = round(floored, decimals + 2)
    if floored + 1e-12 < vmin:
        return None
    return floored


def monetary_loss_for_one_lot(symbol: str, direction: int, entry: float, stop: float) -> Optional[float]:
    order_type = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    p = mt5.order_calc_profit(order_type, symbol, 1.0, entry, stop)
    if p is None:
        return None
    # If a proposed "stop" is profitable relative to entry, there is no loss-at-stop.
    return max(0.0, -float(p))


def size_for_risk(symbol: str, direction: int, entry: float, stop: float, equity: float, risk_fraction: float) -> Optional[float]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    pip = pip_size_from_info(info)
    stop_pips = abs(entry - stop) / pip
    if stop_pips < CFG.min_stop_pips:
        logging.info("%s refused: stop %.2f pips < %.2f minimum", symbol, stop_pips, CFG.min_stop_pips)
        return None
    one_lot_loss = monetary_loss_for_one_lot(symbol, direction, entry, stop)
    if not one_lot_loss or one_lot_loss <= 0:
        return None
    raw = (equity * risk_fraction) / one_lot_loss
    return floor_volume(raw, info)


def is_bot_magic(magic: int) -> bool:
    return int(magic) in BOT_MAGICS


def position_direction(pos: object) -> int:
    return 1 if int(pos.type) == mt5.POSITION_TYPE_BUY else -1


def pending_direction(order: object) -> Optional[int]:
    buy_types = {
        getattr(mt5, "ORDER_TYPE_BUY_LIMIT", -1),
        getattr(mt5, "ORDER_TYPE_BUY_STOP", -1),
        getattr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", -1),
    }
    sell_types = {
        getattr(mt5, "ORDER_TYPE_SELL_LIMIT", -1),
        getattr(mt5, "ORDER_TYPE_SELL_STOP", -1),
        getattr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", -1),
    }
    if int(order.type) in buy_types:
        return 1
    if int(order.type) in sell_types:
        return -1
    return None


def fx_legs(symbol: str) -> Optional[Tuple[str, str]]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    base = str(getattr(info, "currency_base", ""))
    quote = str(getattr(info, "currency_profit", ""))
    if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
        return base, quote
    return None


def discover_fx_symbols() -> List[str]:
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        raise RuntimeError(f"No MT5 symbols returned: {mt5.last_error()}")

    grouped: Dict[Tuple[str, str], List[object]] = {}
    disabled = getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0)
    for s in all_symbols:
        base = str(getattr(s, "currency_base", ""))
        quote = str(getattr(s, "currency_profit", ""))
        if base not in FIAT_CURRENCIES or quote not in FIAT_CURRENCIES or base == quote:
            continue
        if int(getattr(s, "trade_mode", disabled)) == disabled:
            continue
        if float(getattr(s, "trade_contract_size", 0) or 0) <= 0:
            continue
        grouped.setdefault((base, quote), []).append(s)

    chosen = []
    for (base, quote), candidates in grouped.items():
        # Prefer visible, then exact six-character name, then shortest name.
        candidates.sort(key=lambda s: (
            0 if bool(getattr(s, "visible", False)) else 1,
            0 if str(s.name).upper() == f"{base}{quote}" else 1,
            len(str(s.name)),
        ))
        s = candidates[0]
        if not bool(getattr(s, "visible", False)):
            mt5.symbol_select(s.name, True)
        chosen.append(s.name)

    chosen.sort(key=lambda name: (0 if canonical_pair(name) in CORE_PAIRS else 1, canonical_pair(name), name))
    return chosen


def canonical_pair(symbol: str) -> str:
    info = mt5.symbol_info(symbol) if mt5 is not None else None
    if info is not None:
        b = str(getattr(info, "currency_base", ""))
        q = str(getattr(info, "currency_profit", ""))
        if b and q:
            return b + q
    return "".join(ch for ch in symbol.upper() if ch.isalpha())[:6]


# -----------------------------------------------------------------------------
# Indicators / signals
# -----------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    # Wilder-style ATR.
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def efficiency_ratio(close: pd.Series, n: int) -> float:
    if len(close) < n + 1:
        return 0.0
    c = close.iloc[-(n + 1):].astype(float)
    net = abs(c.iloc[-1] - c.iloc[0])
    gross = c.diff().abs().sum()
    return float(net / gross) if gross > 0 else 0.0


def volatility_status(d1: pd.DataFrame) -> Tuple[bool, float, float]:
    a = atr(d1, 20).dropna()
    if len(a) < 251:
        return False, float("nan"), 0.0
    current = float(a.iloc[-1])
    history = a.iloc[-251:-1]
    p10 = float(np.percentile(history, 10))
    pct = float((history <= current).mean())
    return current > p10, current, pct


def dtp_trend_direction(d1: pd.DataFrame) -> int:
    if len(d1) < 200:
        return 0
    sma200 = float(d1["close"].rolling(200).mean().iloc[-1])
    close = float(d1["close"].iloc[-1])
    return 1 if close > sma200 else (-1 if close < sma200 else 0)


def lsr_candle_features(h4: pd.DataFrame) -> Dict[str, object]:
    """Describe the last two COMPLETED H4 candles. A sweep is the mandatory pattern."""
    out: Dict[str, object] = {
        "direction": 0,
        "bull_sweep": False,
        "bear_sweep": False,
        "bull_body_engulf": False,
        "bear_body_engulf": False,
        "outside_bar": False,
    }
    if len(h4) < 2:
        return out

    prev = h4.iloc[-2]
    cur = h4.iloc[-1]

    bull_sweep = float(cur.low) < float(prev.low) and float(cur.close) > float(prev.close)
    bear_sweep = float(cur.high) > float(prev.high) and float(cur.close) < float(prev.close)

    # Conventional body engulf: prior candle is opposite-coloured and the current
    # real body fully covers the prior real body.
    bull_body_engulf = (
        float(prev.close) < float(prev.open)
        and float(cur.close) > float(cur.open)
        and float(cur.open) <= float(prev.close)
        and float(cur.close) >= float(prev.open)
    )
    bear_body_engulf = (
        float(prev.close) > float(prev.open)
        and float(cur.close) < float(cur.open)
        and float(cur.open) >= float(prev.close)
        and float(cur.close) <= float(prev.open)
    )
    outside_bar = float(cur.low) < float(prev.low) and float(cur.high) > float(prev.high)

    direction = 1 if bull_sweep and not bear_sweep else (-1 if bear_sweep and not bull_sweep else 0)
    out.update({
        "direction": direction,
        "bull_sweep": bool(bull_sweep),
        "bear_sweep": bool(bear_sweep),
        "bull_body_engulf": bool(bull_body_engulf),
        "bear_body_engulf": bool(bear_body_engulf),
        "outside_bar": bool(outside_bar),
    })
    return out


def lsr_pattern(h4: pd.DataFrame) -> int:
    """Return +1 bullish sweep, -1 bearish sweep, 0 none."""
    return int(lsr_candle_features(h4)["direction"])

def lsr_trend_direction(h4: pd.DataFrame) -> int:
    if len(h4) < 300:
        return 0
    sma300 = float(h4["close"].rolling(300).mean().iloc[-1])
    close = float(h4["close"].iloc[-1])
    return 1 if close > sma300 else (-1 if close < sma300 else 0)


def sweep_rate_in_trend(h4: pd.DataFrame, lookback: int = 60) -> float:
    if len(h4) < max(302, lookback + 2):
        return 0.0
    c = h4["close"]
    sma = c.rolling(300).mean()
    start = max(1, len(h4) - lookback)
    hits = 0
    valid = 0
    for i in range(start, len(h4)):
        if pd.isna(sma.iloc[i]):
            continue
        direction = 1 if c.iloc[i] > sma.iloc[i] else -1
        prev, cur = h4.iloc[i - 1], h4.iloc[i]
        bull = cur.low < prev.low and cur.close > prev.close
        bear = cur.high > prev.high and cur.close < prev.close
        valid += 1
        if (direction > 0 and bull) or (direction < 0 and bear):
            hits += 1
    return hits / valid if valid else 0.0


def lsr_quality(symbol: str, d1: pd.DataFrame, h4: pd.DataFrame) -> Dict[str, object]:
    """Score the live H4 reversal setup from 0-7. Sweep + trend alignment are mandatory."""
    f = lsr_candle_features(h4)
    direction = int(f["direction"])
    trend = lsr_trend_direction(h4)

    points = 0
    labels: List[str] = []
    if direction != 0:
        points += 1
        labels.append("sweep")
    aligned = direction != 0 and direction == trend
    if aligned:
        points += 1
        labels.append("trend")

    body_engulf = bool(f["bull_body_engulf"] if direction > 0 else f["bear_body_engulf"] if direction < 0 else False)
    if body_engulf:
        points += 1
        labels.append("body-engulf")
    if direction != 0 and bool(f["outside_bar"]):
        points += 1
        labels.append("outside-bar")

    strong_trend = False
    if len(h4) >= 300:
        hclose = h4["close"].astype(float)
        h_atr = atr(h4, 14).iloc[-1]
        h_sma = hclose.rolling(300).mean().iloc[-1]
        if np.isfinite(h_atr) and h_atr > 0 and np.isfinite(h_sma):
            strength = abs(float(hclose.iloc[-1]) - float(h_sma)) / (4.0 * float(h_atr))
            strong_trend = strength >= 0.25
    if strong_trend:
        points += 1
        labels.append("strong-H4-trend")

    vol_ok, _, _ = volatility_status(d1)
    if vol_ok:
        points += 1
        labels.append("volatility")

    low_spread = False
    info = mt5.symbol_info(symbol)
    sp = spread_pips(symbol)
    if info is not None and sp is not None:
        cap = spread_cap_pips(info)
        # Full point when spread is comfortably below the hard gate. The hard cap
        # is still enforced again immediately before order submission.
        low_spread = sp <= 0.75 * cap
    if low_spread:
        points += 1
        labels.append("low-spread")

    eligible = aligned and points >= CFG.lsr_min_quality_points
    return {
        "direction": direction,
        "trend": trend,
        "points": points,
        "max_points": 7,
        "eligible": eligible,
        "body_engulf": body_engulf,
        "outside_bar": bool(f["outside_bar"]),
        "labels": labels,
    }


def route_scores(symbol: str, d1: pd.DataFrame, h4: pd.DataFrame, hedging: bool) -> Dict[str, float]:
    vol_ok, atr20, vol_pct = volatility_status(d1)
    ddir = dtp_trend_direction(d1)
    hdir = lsr_trend_direction(h4)

    dclose = d1["close"].astype(float)
    hclose = h4["close"].astype(float)
    er20 = efficiency_ratio(dclose, 20)
    er40 = efficiency_ratio(dclose, 40)
    er80 = efficiency_ratio(dclose, 80)
    her30 = efficiency_ratio(hclose, 30)

    sma200 = dclose.rolling(200).mean()
    trend_strength = 0.0
    if len(sma200.dropna()) and atr20 and np.isfinite(atr20) and atr20 > 0:
        trend_strength = min(abs(float(dclose.iloc[-1]) - float(sma200.iloc[-1])) / (4.0 * atr20), 1.0)

    alignment_bits = []
    for n in (20, 40, 80):
        if len(dclose) >= n + 1:
            delta = float(dclose.iloc[-1] - dclose.iloc[-(n + 1)])
            alignment_bits.append(1.0 if np.sign(delta) == ddir and ddir != 0 else 0.0)
    alignment = float(np.mean(alignment_bits)) if alignment_bits else 0.0

    prox = 0.0
    if len(d1) >= 20 and atr20 and np.isfinite(atr20) and atr20 > 0 and ddir != 0:
        if ddir > 0:
            level = float(d1["high"].iloc[-20:].max())
            dist = max(0.0, level - float(dclose.iloc[-1]))
        else:
            level = float(d1["low"].iloc[-20:].min())
            dist = max(0.0, float(dclose.iloc[-1]) - level)
        prox = 1.0 - min(dist / atr20, 1.0)

    h_atr = atr(h4, 14).iloc[-1] if len(h4) >= 14 else np.nan
    h_sma300 = hclose.rolling(300).mean().iloc[-1] if len(h4) >= 300 else np.nan
    htrend_strength = 0.0
    if np.isfinite(h_atr) and h_atr > 0 and np.isfinite(h_sma300):
        htrend_strength = min(abs(float(hclose.iloc[-1]) - float(h_sma300)) / (4.0 * float(h_atr)), 1.0)
    sweep_rate = min(sweep_rate_in_trend(h4) / 0.12, 1.0)  # 12% of bars is treated as "frequent".

    scores = {"DTP20": -1.0, "ENSEMBLE": -1.0, "LSR": -1.0}
    if vol_ok and ddir != 0:
        scores["DTP20"] = 0.35 * er20 + 0.25 * trend_strength + 0.25 * prox + 0.15 * vol_pct
        if hedging:
            scores["ENSEMBLE"] = (
                0.30 * float(np.mean([er20, er40, er80]))
                + 0.30 * alignment
                + 0.25 * trend_strength
                + 0.15 * vol_pct
            )
    quality = lsr_quality(symbol, d1, h4)
    if bool(quality["eligible"]):
        regime = (
            0.35 * (1.0 - her30)
            + 0.25 * sweep_rate
            + 0.25 * htrend_strength
            + 0.15 * vol_pct
        )
        setup = float(quality["points"]) / float(quality["max_points"])
        # Current candle structure matters most; the regime score prevents one
        # attractive candle from completely overriding hostile market behaviour.
        scores["LSR"] = 0.65 * setup + 0.35 * regime
    return scores


def choose_route(scores: Dict[str, float]) -> Optional[str]:
    valid = {k: v for k, v in scores.items() if np.isfinite(v) and v >= 0}
    if not valid:
        return None
    # No artificial "must trade" threshold: choosing a route does not itself create a trade.
    return max(valid, key=valid.get)


# -----------------------------------------------------------------------------
# Broker time / kill switches
# -----------------------------------------------------------------------------

def verify_broker_offset(state: dict, reference_symbol: str) -> bool:
    now = datetime.now(timezone.utc)
    last = state.get("broker_offset_verified_at")
    if last and state.get("broker_offset") in CFG.allowed_broker_offsets:
        try:
            last_dt = datetime.fromisoformat(last)
            if now - last_dt < timedelta(hours=CFG.broker_offset_check_hours):
                return True
        except Exception:
            pass

    d1_open = current_bar_open(reference_symbol, mt5.TIMEFRAME_D1)
    if d1_open is None:
        return False
    # If broker midnight corresponds to 21:00 UTC => UTC+3; 22:00 => UTC+2.
    inferred = (24 - d1_open.hour) % 24
    if inferred > 12:
        inferred -= 24
    ok = inferred in CFG.allowed_broker_offsets
    state["broker_offset"] = inferred if ok else None
    state["broker_offset_verified_at"] = now.isoformat()
    logging.info("Broker offset check: D1 opens %02d:00 UTC -> inferred UTC%+d (%s)", d1_open.hour, inferred, "OK" if ok else "BLOCKED")
    return ok


def broker_now(state: dict) -> Optional[datetime]:
    off = state.get("broker_offset")
    if off is None:
        return None
    return datetime.now(timezone.utc) + timedelta(hours=int(off))


def in_rollover_window(state: dict) -> bool:
    now = broker_now(state)
    if now is None:
        return True
    minute_of_day = now.hour * 60 + now.minute
    before_midnight = 24 * 60 - CFG.rollover_before_min
    after_midnight = CFG.rollover_after_min
    return minute_of_day >= before_midnight or minute_of_day <= after_midnight


def period_keys(state: dict) -> Optional[Tuple[str, str, str]]:
    now = broker_now(state)
    if now is None:
        return None
    day = now.date().isoformat()
    iso = now.isocalendar()
    week = f"{iso.year}-W{iso.week:02d}"
    month = f"{now.year:04d}-{now.month:02d}"
    return day, week, month


def update_kill_switches(state: dict, equity: float) -> Tuple[bool, float, str]:
    keys = period_keys(state)
    if keys is None:
        return True, 0.0, "broker offset unavailable"
    day, week, month = keys

    if state.get("day_key") != day:
        state["day_key"] = day
        state["day_start_equity"] = equity
        state["daily_halt_key"] = None
    if state.get("week_key") != week:
        state["week_key"] = week
        state["week_start_equity"] = equity
        state["weekly_halt_key"] = None
    if state.get("month_key") != month:
        state["month_key"] = month
        state["month_start_equity"] = equity
        # Monthly halt is deliberately NOT auto-cleared: manual review is required.

    for k in ("day_start_equity", "week_start_equity", "month_start_equity", "peak_equity"):
        if not state.get(k):
            state[k] = equity
    state["peak_equity"] = max(float(state["peak_equity"]), equity)

    day_dd = 1.0 - equity / float(state["day_start_equity"])
    week_dd = 1.0 - equity / float(state["week_start_equity"])
    month_dd = 1.0 - equity / float(state["month_start_equity"])
    peak_dd = 1.0 - equity / float(state["peak_equity"])

    if day_dd >= CFG.daily_loss_limit:
        state["daily_halt_key"] = day
    if week_dd >= CFG.weekly_loss_limit:
        state["weekly_halt_key"] = week
    if month_dd >= CFG.monthly_loss_limit and not state.get("manual_halt"):
        state["manual_halt"] = True
        state["manual_halt_reason"] = f"Monthly equity drawdown reached {month_dd:.2%}"
    if peak_dd >= CFG.full_stop_peak_dd and not state.get("manual_halt"):
        state["manual_halt"] = True
        state["manual_halt_reason"] = f"Peak-to-current drawdown reached {peak_dd:.2%}"

    risk_multiplier = 0.5 if peak_dd >= CFG.half_risk_peak_dd else 1.0

    if state.get("manual_halt"):
        return True, risk_multiplier, str(state.get("manual_halt_reason") or "manual halt")
    if state.get("daily_halt_key") == day:
        return True, risk_multiplier, f"daily loss halt ({day_dd:.2%})"
    if state.get("weekly_halt_key") == week:
        return True, risk_multiplier, f"weekly loss halt ({week_dd:.2%})"
    return False, risk_multiplier, "ok"


# -----------------------------------------------------------------------------
# Exposure / correlation / risk gate
# -----------------------------------------------------------------------------

def open_positions() -> List[object]:
    return list(mt5.positions_get() or [])


def pending_orders() -> List[object]:
    return list(mt5.orders_get() or [])


def position_risk_amount(pos: object) -> Optional[float]:
    if float(pos.sl) <= 0:
        return None
    direction = position_direction(pos)
    one = monetary_loss_for_one_lot(pos.symbol, direction, float(pos.price_open), float(pos.sl))
    if one is None:
        return None
    return one * float(pos.volume)


def order_risk_amount(order: object) -> Optional[float]:
    direction = pending_direction(order)
    if direction is None or float(getattr(order, "sl", 0.0)) <= 0:
        return None
    entry = float(getattr(order, "price_open", 0.0))
    one = monetary_loss_for_one_lot(order.symbol, direction, entry, float(order.sl))
    if one is None:
        return None
    return one * float(order.volume_current)


def committed_risk_amount() -> Optional[float]:
    total = 0.0
    for p in open_positions():
        r = position_risk_amount(p)
        if r is None:
            logging.warning("Open position %s ticket=%s has no measurable stop risk; new entries blocked.", p.symbol, p.ticket)
            return None
        total += r
    for o in pending_orders():
        # Count pending orders conservatively too; unknown/manual stopless pendings block entries.
        r = order_risk_amount(o)
        if r is None:
            logging.warning("Pending order %s ticket=%s has no measurable stop risk; new entries blocked.", o.symbol, o.ticket)
            return None
        total += r
    return total


def risk_units_from_amount(risk_amount: float, equity: float) -> float:
    denom = equity * effective_risk_per_trade()
    return risk_amount / denom if denom > 0 else 999.0


def exposure_items(equity: float) -> List[Tuple[str, int, float, bool]]:
    """(symbol, direction, risk_units, is_ensemble_sleeve)."""
    items = []
    for p in open_positions():
        legs = fx_legs(p.symbol)
        if not legs:
            continue
        r = position_risk_amount(p)
        if r is None:
            continue
        items.append((p.symbol, position_direction(p), risk_units_from_amount(r, equity), int(p.magic) in {MAGIC_EN20, MAGIC_EN40, MAGIC_EN80}))
    for o in pending_orders():
        d = pending_direction(o)
        if d is None or not fx_legs(o.symbol):
            continue
        r = order_risk_amount(o)
        if r is None:
            continue
        items.append((o.symbol, d, risk_units_from_amount(r, equity), int(o.magic) in {MAGIC_EN20, MAGIC_EN40, MAGIC_EN80}))
    return items


def currency_net_r(items: Iterable[Tuple[str, int, float, bool]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for symbol, direction, r, _ in items:
        legs = fx_legs(symbol)
        if not legs:
            continue
        base, quote = legs
        out[base] = out.get(base, 0.0) + direction * r
        out[quote] = out.get(quote, 0.0) - direction * r
    return out


def block_direction_counts(items: Iterable[Tuple[str, int, float, bool]]) -> Dict[Tuple[str, int], int]:
    counts: Dict[Tuple[str, int], int] = {}
    # The three 1/3-risk ensemble sleeves on the same symbol are one composite bet
    # for currency-block counting. Otherwise sleeve #3 would be prohibited by the
    # global "two same-direction positions per block" rule despite total risk = 1R.
    seen_ensemble_clusters = set()
    for symbol, direction, _, is_ensemble in items:
        if is_ensemble:
            cluster = (symbol, direction)
            if cluster in seen_ensemble_clusters:
                continue
            seen_ensemble_clusters.add(cluster)
        legs = fx_legs(symbol)
        if not legs:
            continue
        base, quote = legs
        contribution: Dict[str, int] = {}
        for cur, sign in ((base, direction), (quote, -direction)):
            block = CURRENCY_BLOCK.get(cur, cur)
            contribution[block] = contribution.get(block, 0) + sign
        for block, net_sign in contribution.items():
            if net_sign == 0:
                continue
            s = 1 if net_sign > 0 else -1
            counts[(block, s)] = counts.get((block, s), 0) + 1
    return counts


def return_series(symbol: str, bars: int = 90) -> Optional[pd.Series]:
    df = rates(symbol, mt5.TIMEFRAME_D1, bars + 1, completed_only=True)
    if df is None or len(df) < 50:
        return None
    s = pd.Series(df["close"].pct_change().values, index=df["time"]).dropna()
    return s


def pnl_correlation(symbol_a: str, dir_a: int, symbol_b: str, dir_b: int) -> Optional[float]:
    if symbol_a == symbol_b:
        return float(dir_a * dir_b)
    a = return_series(symbol_a)
    b = return_series(symbol_b)
    if a is None or b is None:
        return None
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 40:
        return None
    raw = float(joined["a"].corr(joined["b"]))
    if not np.isfinite(raw):
        return None
    return raw * dir_a * dir_b


def pretrade_gate(
    state: dict,
    symbol: str,
    direction: int,
    entry: float,
    stop: float,
    volume: float,
    risk_fraction: float,
    candidate_is_ensemble: bool,
) -> Tuple[bool, str]:
    try:
        acct, _ = assert_demo_and_trading_ready()
    except Exception as exc:
        return False, str(exc)

    if not verify_broker_offset(state, symbol):
        return False, "broker time offset could not be verified"
    if in_rollover_window(state):
        return False, "daily rollover window"

    halted, _, reason = update_kill_switches(state, float(acct.equity))
    if halted:
        return False, reason

    info = mt5.symbol_info(symbol)
    sp = spread_pips(symbol)
    if info is None or sp is None:
        return False, "missing symbol/tick data"
    cap = spread_cap_pips(info)
    if sp > cap:
        return False, f"spread {sp:.2f} > cap {cap:.2f} pips"

    pip = pip_size_from_info(info)
    if abs(entry - stop) / pip < CFG.min_stop_pips:
        return False, "stop is under 8 pips"
    lot_cap = effective_max_lots()
    if volume > lot_cap + 1e-12:
        return False, f"volume above {lot_cap:.2f}-lot hard cap"

    positions = open_positions()
    orders = pending_orders()
    committed_slots = len(positions) + (len(orders) if CFG.reserve_slots_for_pending else 0)
    if committed_slots >= CFG.max_open_positions:
        return False, f"position-slot cap reached ({committed_slots}/{CFG.max_open_positions})"

    committed = committed_risk_amount()
    if committed is None:
        return False, "existing account risk cannot be measured"
    candidate_one = monetary_loss_for_one_lot(symbol, direction, entry, stop)
    if candidate_one is None:
        return False, "candidate risk cannot be calculated"
    candidate_risk = candidate_one * volume
    if committed + candidate_risk > float(acct.equity) * CFG.max_total_risk + 1e-9:
        return False, "3% total committed-risk cap"

    items = exposure_items(float(acct.equity))
    candidate_r = risk_units_from_amount(candidate_risk, float(acct.equity))

    # Net exposure to each exact currency <= 2R.
    net = currency_net_r(items)
    legs = fx_legs(symbol)
    if legs:
        base, quote = legs
        projected = dict(net)
        projected[base] = projected.get(base, 0.0) + direction * candidate_r
        projected[quote] = projected.get(quote, 0.0) - direction * candidate_r
        for cur, val in projected.items():
            if abs(val) > CFG.max_currency_r + 1e-9:
                return False, f"{cur} projected net exposure {val:.2f}R exceeds 2R"

        # Same-direction block <= 2. Treat each ensemble sleeve as a position for this cap.
        counts = block_direction_counts(items)
        contribution: Dict[str, int] = {}
        for cur, sign in ((base, direction), (quote, -direction)):
            block = CURRENCY_BLOCK.get(cur, cur)
            contribution[block] = contribution.get(block, 0) + sign
        for block, net_sign in contribution.items():
            if net_sign == 0:
                continue
            s = 1 if net_sign > 0 else -1
            if counts.get((block, s), 0) >= CFG.max_same_direction_block:
                return False, f"same-direction {block} block cap reached"

    # Correlation gate. Ensemble sleeves on THE SAME symbol are treated as one composite
    # strategy cluster; otherwise the ensemble could never legally hold its 3 sleeves.
    checked = set()
    for existing_symbol, existing_dir, _, existing_is_ensemble in items:
        key = (existing_symbol, existing_dir)
        if key in checked:
            continue
        checked.add(key)
        if candidate_is_ensemble and existing_is_ensemble and existing_symbol == symbol and existing_dir == direction:
            continue
        corr = pnl_correlation(symbol, direction, existing_symbol, existing_dir)
        if corr is not None and corr > CFG.max_position_corr:
            return False, f"PnL correlation {corr:.2f} with {existing_symbol} exceeds 0.70"

    return True, "ok"


# -----------------------------------------------------------------------------
# Order execution
# -----------------------------------------------------------------------------

def filling_modes() -> List[int]:
    vals = []
    for name in ("ORDER_FILLING_IOC", "ORDER_FILLING_FOK", "ORDER_FILLING_RETURN"):
        if hasattr(mt5, name):
            val = getattr(mt5, name)
            if val not in vals:
                vals.append(val)
    return vals


def send_request(request: dict) -> Optional[object]:
    if not CFG.execute_orders:
        logging.info("DRY-RUN order: %s", request)
        return None

    last_result = None
    for fill in filling_modes() or [None]:
        req = dict(request)
        if fill is not None:
            req["type_filling"] = fill
        check = mt5.order_check(req)
        if check is not None and int(check.retcode) != 0:
            # order_check retcode=0 is success in MqlTradeCheckResult.
            last_result = check
            continue
        result = mt5.order_send(req)
        last_result = result
        if result is not None and int(result.retcode) in {
            getattr(mt5, "TRADE_RETCODE_DONE", 10009),
            getattr(mt5, "TRADE_RETCODE_PLACED", 10008),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
        }:
            logging.info("Order accepted: retcode=%s order=%s deal=%s", result.retcode, getattr(result, "order", None), getattr(result, "deal", None))
            return result
    logging.error("Order failed: %s | result=%s | last_error=%s", request, last_result, mt5.last_error())
    return last_result


def place_pending_stop(state: dict, symbol: str, direction: int, entry: float, stop: float, risk_fraction: float, magic: int, comment: str) -> None:
    acct = mt5.account_info()
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if acct is None or info is None or tick is None:
        return
    entry = normalize_price(info, entry)
    stop = normalize_price(info, stop)
    r_distance = abs(entry - stop)
    tp = normalize_price(info, entry + direction * CFG.reward_risk * r_distance)

    # Never convert a missed stop entry into a market entry.
    min_broker_dist = max(float(getattr(info, "trade_stops_level", 0) or 0) * float(info.point), float(getattr(info, "trade_tick_size", info.point) or info.point))
    if direction > 0 and entry <= tick.ask + min_broker_dist:
        logging.info("%s %s skipped: buy-stop level already reached/too close", symbol, comment)
        return
    if direction < 0 and entry >= tick.bid - min_broker_dist:
        logging.info("%s %s skipped: sell-stop level already reached/too close", symbol, comment)
        return

    volume = size_for_risk(symbol, direction, entry, stop, float(acct.equity), risk_fraction)
    if volume is None:
        return
    ok, reason = pretrade_gate(state, symbol, direction, entry, stop, volume, risk_fraction, magic in {MAGIC_EN20, MAGIC_EN40, MAGIC_EN80})
    if not ok:
        logging.info("%s %s blocked: %s", symbol, comment, reason)
        return

    d1_open = current_bar_open(symbol, mt5.TIMEFRAME_D1)
    if d1_open is None:
        logging.info("%s %s blocked: cannot establish one-day expiration", symbol, comment)
        return
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY_STOP if direction > 0 else mt5.ORDER_TYPE_SELL_STOP,
        "price": entry,
        "sl": stop,
        "tp": tp,
        "deviation": CFG.deviation_points,
        "magic": magic,
        "comment": comment[:31],
        # Server-side expiry protects the one-day rule even if the Python process
        # is stopped. The next daily-open timestamp is used as the expiry boundary.
        "type_time": getattr(mt5, "ORDER_TIME_SPECIFIED", mt5.ORDER_TIME_GTC),
        "expiration": int((d1_open + timedelta(days=1)).timestamp()),
    }
    send_request(request)


def place_market_lsr(state: dict, symbol: str, direction: int, stop: float, risk_fraction: float) -> None:
    acct = mt5.account_info()
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if acct is None or info is None or tick is None:
        return
    entry = float(tick.ask if direction > 0 else tick.bid)
    stop = normalize_price(info, stop)
    r_distance = abs(entry - stop)
    tp = normalize_price(info, entry + direction * CFG.reward_risk * r_distance)
    volume = size_for_risk(symbol, direction, entry, stop, float(acct.equity), risk_fraction)
    if volume is None:
        return
    ok, reason = pretrade_gate(state, symbol, direction, entry, stop, volume, risk_fraction, False)
    if not ok:
        logging.info("%s LSR blocked: %s", symbol, reason)
        return
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
        "price": entry,
        "sl": stop,
        "tp": tp,
        "deviation": CFG.deviation_points,
        "magic": MAGIC_LSR,
        "comment": "LSR-H4",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    send_request(request)


def cancel_order(order: object) -> None:
    if not CFG.execute_orders:
        logging.info("DRY-RUN cancel order %s %s", order.ticket, order.symbol)
        return
    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(order.ticket)}
    result = mt5.order_send(req)
    if result is None or int(result.retcode) != getattr(mt5, "TRADE_RETCODE_DONE", 10009):
        logging.error("Cancel failed ticket=%s result=%s", order.ticket, result)


def cancel_bot_pending(symbol: Optional[str] = None) -> None:
    for o in pending_orders():
        if not is_bot_magic(int(o.magic)):
            continue
        if symbol is not None and o.symbol != symbol:
            continue
        cancel_order(o)


def close_position(pos: object, reason: str) -> None:
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return
    direction = position_direction(pos)
    close_type = mt5.ORDER_TYPE_SELL if direction > 0 else mt5.ORDER_TYPE_BUY
    price = float(tick.bid if direction > 0 else tick.ask)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "position": int(pos.ticket),
        "volume": float(pos.volume),
        "type": close_type,
        "price": price,
        "deviation": CFG.deviation_points,
        "magic": int(pos.magic),
        "comment": ("EXIT-" + reason)[:31],
        "type_time": mt5.ORDER_TIME_GTC,
    }
    logging.info("Closing %s ticket=%s: %s", pos.symbol, pos.ticket, reason)
    send_request(request)


# -----------------------------------------------------------------------------
# Strategy lifecycle
# -----------------------------------------------------------------------------

def bot_positions_for_symbol(symbol: str) -> List[object]:
    return [p for p in open_positions() if p.symbol == symbol and is_bot_magic(int(p.magic))]


def any_account_position_on_symbol(symbol: str) -> bool:
    return any(p.symbol == symbol for p in open_positions())


def has_magic_position_or_order(symbol: str, magic: int) -> bool:
    if any(p.symbol == symbol and int(p.magic) == magic for p in open_positions()):
        return True
    if any(o.symbol == symbol and int(o.magic) == magic for o in pending_orders()):
        return True
    return False


def manage_donchian_exits(symbol: str, d1: pd.DataFrame) -> None:
    for pos in bot_positions_for_symbol(symbol):
        if int(pos.magic) not in {MAGIC_DTP20, MAGIC_EN20, MAGIC_EN40, MAGIC_EN80}:
            continue
        _, _, exit_n = MAGIC_META[int(pos.magic)]
        if exit_n is None or len(d1) < exit_n + 1:
            continue
        signal_bar = d1.iloc[-1]
        prior = d1.iloc[-(exit_n + 1):-1]  # excludes signal bar, so the close can cross the prior channel
        direction = position_direction(pos)
        if direction > 0:
            exit_level = float(prior["low"].min())
            if float(signal_bar.close) < exit_level:
                close_position(pos, f"D{exit_n}-close")
        else:
            exit_level = float(prior["high"].max())
            if float(signal_bar.close) > exit_level:
                close_position(pos, f"D{exit_n}-close")


def h4_bars_since_position(symbol: str, pos_time_epoch: int) -> Optional[int]:
    df = rates(symbol, mt5.TIMEFRAME_H4, 80, completed_only=True)
    if df is None:
        return None
    opened = datetime.fromtimestamp(int(pos_time_epoch), tz=timezone.utc)
    return int((df["time"] >= opened).sum())


def manage_lsr_time_exits(symbol: str) -> None:
    for pos in bot_positions_for_symbol(symbol):
        if int(pos.magic) != MAGIC_LSR:
            continue
        bars_open = h4_bars_since_position(symbol, int(pos.time))
        if bars_open is not None and bars_open >= 30:
            close_position(pos, "LSR-30bars")


def donchian_entry(symbol: str, d1: pd.DataFrame, entry_n: int, direction: int) -> Optional[Tuple[float, float]]:
    info = mt5.symbol_info(symbol)
    if info is None or len(d1) < max(entry_n, 20):
        return None
    a20 = atr(d1, 20).iloc[-1]
    if not np.isfinite(a20) or a20 <= 0:
        return None
    tick_size = float(getattr(info, "trade_tick_size", 0) or getattr(info, "point", 0) or 0)
    if tick_size <= 0:
        return None
    if direction > 0:
        entry = float(d1["high"].iloc[-entry_n:].max()) + tick_size
        stop = entry - 2.0 * float(a20)
    else:
        entry = float(d1["low"].iloc[-entry_n:].min()) - tick_size
        stop = entry + 2.0 * float(a20)
    return normalize_price(info, entry), normalize_price(info, stop)


def place_dtp20_for_day(state: dict, symbol: str, d1: pd.DataFrame, risk_fraction: float) -> None:
    if any_account_position_on_symbol(symbol):
        return
    vol_ok, _, _ = volatility_status(d1)
    direction = dtp_trend_direction(d1)
    if not vol_ok or direction == 0:
        return
    setup = donchian_entry(symbol, d1, 20, direction)
    if setup:
        place_pending_stop(state, symbol, direction, setup[0], setup[1], risk_fraction, MAGIC_DTP20, "DTP20-D1")


def place_ensemble_for_day(state: dict, symbol: str, d1: pd.DataFrame, risk_fraction: float, hedging: bool) -> None:
    if not hedging:
        logging.info("%s ensemble skipped: account is not MT5 hedging mode", symbol)
        return
    # If a non-ensemble position exists on the symbol, do not stack a new strategy family on it.
    existing = bot_positions_for_symbol(symbol)
    if any(int(p.magic) not in {MAGIC_EN20, MAGIC_EN40, MAGIC_EN80} for p in existing):
        return
    vol_ok, _, _ = volatility_status(d1)
    direction = dtp_trend_direction(d1)
    if not vol_ok or direction == 0:
        return
    sleeve_risk = risk_fraction / 3.0
    for magic, entry_n in ((MAGIC_EN20, 20), (MAGIC_EN40, 40), (MAGIC_EN80, 80)):
        if has_magic_position_or_order(symbol, magic):
            continue
        setup = donchian_entry(symbol, d1, entry_n, direction)
        if setup:
            place_pending_stop(state, symbol, direction, setup[0], setup[1], sleeve_risk, magic, f"ENS{entry_n}-D1")


def process_lsr_signal(
    state: dict, symbol: str, d1: pd.DataFrame, h4: pd.DataFrame, risk_fraction: float, current_h4_open: datetime
) -> bool:
    now = datetime.now(timezone.utc)
    age = (now - current_h4_open).total_seconds()
    if age < -5 or age > CFG.h4_entry_grace_seconds:
        logging.info("%s LSR signal window skipped: current H4 bar is %.0fs old", symbol, age)
        return False
    if any_account_position_on_symbol(symbol):
        return False

    q = lsr_quality(symbol, d1, h4)
    direction = int(q["direction"])
    if not bool(q["eligible"]):
        if direction != 0:
            logging.info(
                "%s H4 sweep rejected: quality=%d/%d required=%d features=%s",
                symbol, int(q["points"]), int(q["max_points"]), CFG.lsr_min_quality_points,
                ",".join(q["labels"]) or "none",
            )
        return False

    a14 = atr(h4, 14).iloc[-1]
    if not np.isfinite(a14) or a14 <= 0 or len(h4) < 10:
        return False
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    buffer = 0.10 * float(a14)
    if direction > 0:
        stop = float(h4["low"].iloc[-10:].min()) - buffer
    else:
        stop = float(h4["high"].iloc[-10:].max()) + buffer

    logging.info(
        "%s H4 LSR ACCEPTED %s quality=%d/%d | engulf=%s outside=%s | %s",
        symbol, "BUY" if direction > 0 else "SELL", int(q["points"]), int(q["max_points"]),
        bool(q["body_engulf"]), bool(q["outside_bar"]), ",".join(q["labels"]),
    )
    place_market_lsr(state, symbol, direction, normalize_price(info, stop), risk_fraction)
    return True


# -----------------------------------------------------------------------------
# Main cycle
# -----------------------------------------------------------------------------

def choose_reference_symbol(symbols: List[str]) -> Optional[str]:
    for target in ("EURUSD", "GBPUSD", "USDJPY"):
        for s in symbols:
            if canonical_pair(s) == target:
                return s
    return symbols[0] if symbols else None


def run_cycle(state: dict, symbols: List[str]) -> None:
    acct, _ = assert_demo_and_trading_ready()
    ref = choose_reference_symbol(symbols)
    if not ref or not verify_broker_offset(state, ref):
        cancel_bot_pending()
        logging.error("No entries: broker UTC offset is unverified. Allowed offsets=%s", CFG.allowed_broker_offsets)
        save_state(state)
        return

    halted, risk_mult, halt_reason = update_kill_switches(state, float(acct.equity))
    if halted or in_rollover_window(state):
        cancel_bot_pending()
        if halted:
            logging.warning("ENTRY HALT: %s", halt_reason)
        else:
            logging.info("ENTRY HALT: daily rollover window")

    hedging = is_hedging_account(acct)
    normal_risk = effective_risk_per_trade() * risk_mult

    for symbol in symbols:
        if STOP_REQUESTED:
            break
        try:
            d1 = rates(symbol, mt5.TIMEFRAME_D1, 330, completed_only=True)
            h4 = rates(symbol, mt5.TIMEFRAME_H4, 380, completed_only=True)
            d1_open = current_bar_open(symbol, mt5.TIMEFRAME_D1)
            h4_open = current_bar_open(symbol, mt5.TIMEFRAME_H4)
            if d1 is None or h4 is None or d1_open is None or h4_open is None:
                logging.debug("%s skipped: insufficient history", symbol)
                continue

            d1_key = d1_open.isoformat()
            h4_key = h4_open.isoformat()
            last_d1 = state["last_d1_open"].get(symbol)
            last_h4 = state["last_h4_open"].get(symbol)
            new_d1 = d1_key != last_d1
            new_h4 = h4_key != last_h4

            # Exits are always managed, even while entry kill switches are active.
            if new_d1:
                manage_donchian_exits(symbol, d1)
            if new_h4:
                manage_lsr_time_exits(symbol)

            if new_d1:
                # DTP pending orders live for one broker day only.
                cancel_bot_pending(symbol)
                scores = route_scores(symbol, d1, h4, hedging)
                route = choose_route(scores)
                state["route"][symbol] = route
                state["route_scores"][symbol] = {k: round(float(v), 4) for k, v in scores.items()}
                logging.info(
                    "%s route=%s | DTP20=%.3f ENS=%.3f LSR=%.3f",
                    symbol, route, scores["DTP20"], scores["ENSEMBLE"], scores["LSR"]
                )

                if not halted and not in_rollover_window(state):
                    if route == "DTP20":
                        place_dtp20_for_day(state, symbol, d1, normal_risk)
                    elif route == "ENSEMBLE":
                        place_ensemble_for_day(state, symbol, d1, normal_risk, hedging)

                state["last_d1_open"][symbol] = d1_key

            route = state["route"].get(symbol)
            if new_h4:
                # Re-score on every completed H4 candle. This lets a fresh, high-quality
                # sweep/engulfing setup compete with the daily Donchian route immediately.
                h4_scores = route_scores(symbol, d1, h4, hedging)
                h4_route = choose_route(h4_scores)
                q = lsr_quality(symbol, d1, h4)
                if int(q["direction"]) != 0:
                    logging.info(
                        "%s H4 candidate quality=%d/%d features=%s | DTP20=%.3f ENS=%.3f LSR=%.3f best=%s",
                        symbol, int(q["points"]), int(q["max_points"]),
                        ",".join(q["labels"]) or "none",
                        h4_scores["DTP20"], h4_scores["ENSEMBLE"], h4_scores["LSR"], h4_route,
                    )

                age = (datetime.now(timezone.utc) - h4_open).total_seconds()
                timely = -5 <= age <= CFG.h4_entry_grace_seconds
                if (
                    not halted
                    and not in_rollover_window(state)
                    and timely
                    and h4_route == "LSR"
                    and bool(q["eligible"])
                ):
                    # One strategy family per symbol: if the H4 reversal wins the router,
                    # remove that symbol's unfilled Donchian orders before adding H4 risk.
                    cancel_bot_pending(symbol)
                    state["route"][symbol] = "LSR"
                    state["route_scores"][symbol] = {k: round(float(v), 4) for k, v in h4_scores.items()}
                    process_lsr_signal(state, symbol, d1, h4, normal_risk, h4_open)

                state["last_h4_open"][symbol] = h4_key

        except Exception:
            logging.exception("Error processing %s", symbol)

    save_state(state)


def review_reset(state: dict) -> None:
    if os.getenv("BOT_MANUAL_REVIEW_ACK") != "YES_I_REVIEWED":
        raise RuntimeError(
            "Sticky halt reset requires BOT_MANUAL_REVIEW_ACK=YES_I_REVIEWED. "
            "Review logs/strategy first; this does not bypass the live opt-in requirements."
        )
    acct, _ = assert_demo_and_trading_ready()
    eq = float(acct.equity)
    state["manual_halt"] = False
    state["manual_halt_reason"] = None
    state["peak_equity"] = eq
    state["day_start_equity"] = eq
    state["week_start_equity"] = eq
    state["month_start_equity"] = eq
    state["daily_halt_key"] = None
    state["weekly_halt_key"] = None
    save_state(state)
    logging.warning("Manual-review halt reset acknowledged. New equity baselines set to %.2f", eq)


def status(state: dict, symbols: List[str]) -> None:
    acct, term = assert_demo_and_trading_ready()
    print(json.dumps({
        "login": acct.login,
        "server": acct.server,
        "equity": acct.equity,
        "balance": acct.balance,
        "connected": term.connected,
        "trade_allowed": term.trade_allowed,
        "execute_orders": CFG.execute_orders,
        "account_mode": "LIVE" if IS_LIVE else "DEMO",
        "live_opt_in_missing": live_opt_in_missing(),
        "risk_per_trade": effective_risk_per_trade(),
        "max_lots": effective_max_lots(),
        "hedging_account": is_hedging_account(acct),
        "fx_symbols_discovered": len(symbols),
        "positions": len(open_positions()),
        "pending_orders": len(pending_orders()),
        "state": state,
    }, indent=2, default=str))


def handle_signal(signum, frame) -> None:  # noqa: ARG001
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.info("Shutdown requested; finishing current operation.")


def main() -> int:
    parser = argparse.ArgumentParser(description="VT Markets adaptive MT5 FX bot (demo by default, live opt-in)")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--status", action="store_true", help="Show account/bot state then exit")
    parser.add_argument("--review-reset", action="store_true", help="Reset sticky monthly/full halt after manual review acknowledgement")
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Permit trading on a REAL-MONEY account. Also requires BOT_ALLOW_LIVE=true, "
            "BOT_LIVE_ACK=I_ACCEPT_REAL_MONEY_RISK and BOT_EXECUTE_ORDERS=true."
        ),
    )
    args = parser.parse_args()

    CFG.live_flag = bool(args.live)

    setup_logging()
    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    initialize_mt5()
    try:
        acct, _ = assert_demo_and_trading_ready()
        state = load_state()
        symbols = discover_fx_symbols()
        logging.info(
            "Connected %s account %s @ %s | equity=%.2f | discovered %d FX pairs | mode=%s",
            "LIVE" if IS_LIVE else "DEMO",
            acct.login, acct.server, acct.equity, len(symbols),
            "EXECUTE" if CFG.execute_orders else "DRY-RUN",
        )
        if args.live and not IS_LIVE:
            logging.info("--live was passed but the connected account is not a real account; running in demo mode.")
        if not symbols:
            raise RuntimeError("No tradable FX pairs discovered from MT5 symbol metadata.")
        if not is_hedging_account(acct):
            logging.warning("Account is not hedging mode: Donchian Ensemble routing is disabled; DTP20/LSR remain available.")

        if args.review_reset:
            review_reset(state)
            return 0
        if args.status:
            status(state, symbols)
            return 0

        while not STOP_REQUESTED:
            started = time.time()
            try:
                run_cycle(state, symbols)
            except Exception:
                logging.exception("Cycle failed")
                try:
                    save_state(state)
                except Exception:
                    pass
            if args.once:
                break
            elapsed = time.time() - started
            sleep_for = max(1.0, CFG.loop_seconds - elapsed)
            end = time.time() + sleep_for
            while not STOP_REQUESTED and time.time() < end:
                time.sleep(min(1.0, end - time.time()))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())