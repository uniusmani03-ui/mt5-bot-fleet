"""
================================================================================
 MULTI-TIMEFRAME 30-MINUTE OPENING RANGE BREAKOUT (ORB) BOT
================================================================================
 Strategy summary
 ----------------
 1. 30m  : Capture the 08:00-08:30 UK-session opening range (high/low incl. wicks)
 2. 10m  : Confirm a structural breakout candle (open inside -> close outside,
           body >= 70% of range, breakout-side wick <= 20% of range)
 3. 2m   : Wait for a pullback to the 20 EMA, enter on the EMA reclaim close
 Risk    : SL at 5-candle 2m swing +/- buffer, TP at 2R, 1% account risk/trade
 Windows : Breakouts only valid 08:30-10:00 UK time; setups cancelled after
 State   : Max ONE trade per breakout direction per day

 Timezone note
 -------------
 All session logic runs in the "Europe/London" zone (zoneinfo), which
 automatically flips between GMT and BST — no manual daylight-saving offsets.

 Broker integration
 ------------------
 All broker touch-points are isolated in the BrokerAPI class. Swap the
 placeholder bodies for MetaTrader5 / Alpaca / your broker of choice.
================================================================================
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ORB_BOT")

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
LONDON_TZ = ZoneInfo("Europe/London")          # Handles GMT <-> BST automatically

SESSION_OPEN      = dtime(8, 0)                # ORB candle open   (UK local)
ORB_CLOSE         = dtime(8, 30)               # ORB candle close  (UK local)
BREAKOUT_DEADLINE = dtime(10, 0)               # No new 10m breakouts after this

RISK_PER_TRADE_PCT = 0.01                      # 1% of account balance per trade
REWARD_MULTIPLE    = 2.0                       # 2:1 reward-to-risk

# --- Hard safety guardrails (ported from pod fleet post-incident fixes) ---
MAX_LOT            = 5.0                       # absolute lot ceiling per order
MIN_STOP_PIPS      = 8                         # reject setups with stops tighter
                                               # than this (in instrument pips)

BODY_MIN_RATIO   = 0.70                        # 10m body >= 70% of High-Low range
WICK_MAX_RATIO   = 0.20                        # breakout-side wick <= 20% of range
SWING_LOOKBACK   = 5                           # 2m candles used for the swing SL
EMA_PERIOD       = 20                          # 2m EMA period

# Per-instrument settings.
#   buffer      : spread-protection buffer in PRICE units
#                 (FX ~0.0002, JPY pairs 0.02, gold 0.20, indices 2.0 points)
#   pip_size    : smallest quoted move used only for logging/readability
#   asset_class : used by the position-sizing helper
SYMBOLS: dict[str, dict] = {
    "GBPUSD":  {"buffer": 0.0002, "pip_size": 0.0001, "asset_class": "fx"},
    "USDJPY":  {"buffer": 0.02,   "pip_size": 0.01,   "asset_class": "fx_jpy"},
    "GBPJPY-ECN":  {"buffer": 0.02,   "pip_size": 0.01,   "asset_class": "fx_jpy"},
    "XAUUSD-ECN":  {"buffer": 0.20,   "pip_size": 0.01,   "asset_class": "metal"},
    "GER40.":   {"buffer": 2.0,    "pip_size": 0.1,    "asset_class": "index"},
}

POLL_SECONDS = 5                               # main-loop heartbeat

# Broker server-time offset from true UTC, in hours.
#   None  = auto-detect at startup (recommended)
#   0     = server runs genuine UTC (MetaQuotes demo)
#   3     = typical UTC+3 broker, etc.
# WHY THIS EXISTS: the MT5 Python API returns candle epochs stamped in SERVER
# time, not UTC. On a UTC+3 broker, naively treating them as UTC shifts every
# candle +2h vs UK time — the bot then reads the 06:00 UK candle as its
# "08:00" opening range and trades breakouts of the wrong box.
SERVER_UTC_OFFSET_HOURS: int | None = None


# ==============================================================================
# BROKER / DATA LAYER  —  swap these bodies for your live API
# ==============================================================================
class BrokerAPI:
    """
    Live MetaTrader5 implementation.

    Requirements:
      * pip install MetaTrader5
      * The MT5 desktop terminal must be RUNNING and logged in (your demo
        account). mt5.initialize() attaches to that running terminal.
    """

    def __init__(self):
        self.mt5 = None
        self._server_offset: pd.Timedelta | None = None

    def connect(self) -> bool:
        import MetaTrader5 as mt5
        self.mt5 = mt5
        if not mt5.initialize():
            log.error(f"MT5 initialize() failed: {mt5.last_error()} — "
                      f"is the MT5 terminal open and logged in?")
            return False
        acct = mt5.account_info()
        log.info(f"Connected to MT5 | account={acct.login} server={acct.server} "
                 f"balance={acct.balance} {acct.currency}")
        return True

    def get_balance(self) -> float:
        return float(self.mt5.account_info().balance)

    def _detect_server_offset(self, symbol: str) -> pd.Timedelta:
        """
        Measure how far the broker's clock runs ahead of true UTC by comparing
        the last tick's server timestamp against actual UTC now, rounded to
        the nearest whole hour. Cached after first detection.

        Manual override via SERVER_UTC_OFFSET_HOURS skips detection entirely.
        """
        if SERVER_UTC_OFFSET_HOURS is not None:
            off = pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)
            log.info(f"Server offset manually set: UTC{'+' if SERVER_UTC_OFFSET_HOURS>=0 else ''}{SERVER_UTC_OFFSET_HOURS}")
            return off

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None or tick.time == 0:
            raise RuntimeError(f"Cannot read tick time for {symbol} to detect "
                               f"server offset — set SERVER_UTC_OFFSET_HOURS manually.")
        server_now = pd.Timestamp(tick.time, unit="s")     # naive, server clock
        utc_now = pd.Timestamp.utcnow().tz_localize(None)
        diff = server_now - utc_now
        hours = round(diff.total_seconds() / 3600)
        residual = abs(diff - pd.Timedelta(hours=hours))

        # Sanity: during market hours the last tick is seconds old, so the
        # residual after removing whole hours must be tiny. A large residual
        # means a stale tick (weekend/holiday) — refuse to guess.
        if abs(hours) > 12 or residual > pd.Timedelta(minutes=10):
            raise RuntimeError(
                f"Server offset detection unreliable (raw diff {diff}). Market "
                f"may be closed. Set SERVER_UTC_OFFSET_HOURS manually and restart.")
        log.info(f"Detected broker server time = UTC{'+' if hours>=0 else ''}{hours} "
                 f"(tick clock vs real UTC). All candle times will be corrected.")
        return pd.Timedelta(hours=hours)

    def _mt5_tf(self, timeframe: str):
        return {"30m": self.mt5.TIMEFRAME_M30,
                "10m": self.mt5.TIMEFRAME_M10,
                "2m":  self.mt5.TIMEFRAME_M2}[timeframe]

    def validate_symbol(self, symbol: str) -> bool:
        """Make sure the symbol exists on this broker and is visible in
        Market Watch (data calls fail for hidden symbols)."""
        info = self.mt5.symbol_info(symbol)
        if info is None:
            return False
        if not info.visible:
            return self.mt5.symbol_select(symbol, True)
        return True

    def fetch_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """
        Most recent `count` CLOSED candles, index converted to Europe/London.
        copy_rates_from_pos(..., 1, count) starts at position 1, which skips
        the currently forming candle at position 0 — closed candles only.
        """
        rates = self.mt5.copy_rates_from_pos(
            symbol, self._mt5_tf(timeframe), 1, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"No {timeframe} data for {symbol}: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)

        # --- SERVER-TIME CORRECTION ---------------------------------------
        # Candle epochs are stamped in SERVER time. Subtract the detected
        # offset to get true UTC, THEN convert to London. Without this, a
        # UTC+3 broker shifts every candle +2h vs UK and the bot captures
        # the wrong opening range.
        if self._server_offset is None:
            self._server_offset = self._detect_server_offset(symbol)
        df["time"] = (pd.to_datetime(df["time"], unit="s")
                      - self._server_offset).dt.tz_localize("UTC")
        df = df.set_index("time").tz_convert(LONDON_TZ)
        return df.rename(columns={"tick_volume": "volume"})[
            ["open", "high", "low", "close", "volume"]]

    def symbol_info(self, symbol: str) -> dict:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) returned None")
        return {
            "tick_size":   info.trade_tick_size,
            "tick_value":  info.trade_tick_value,
            "volume_min":  info.volume_min,
            "volume_max":  info.volume_max,
            "volume_step": info.volume_step,
            "digits":      info.digits,
            "filling":     info.filling_mode,
        }

    # ------------------------- ORDER EXECUTION -------------------------
    def _filling_type(self, symbol: str):
        """Pick a filling mode the broker actually supports (retcode-10030 fix)."""
        mt5 = self.mt5
        modes = mt5.symbol_info(symbol).filling_mode
        if modes & 1:            # FOK supported
            return mt5.ORDER_FILLING_FOK
        if modes & 2:            # IOC supported
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _send(self, symbol: str, order_type, price: float,
              lots: float, sl: float, tp: float, tag: str) -> bool:
        mt5 = self.mt5
        digits = mt5.symbol_info(symbol).digits
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(lots),
            "type":         order_type,
            "price":        round(price, digits),
            "sl":           round(sl, digits),
            "tp":           round(tp, digits),
            "deviation":    10,
            "magic":        30301,
            "comment":      tag,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_type(symbol),
        }
        result = mt5.order_send(request)
        if result is None:
            log.error(f"[{symbol}] order_send returned None: {mt5.last_error()}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"[{symbol}] Order rejected: retcode={result.retcode} "
                      f"comment={result.comment}")
            return False
        log.info(f"[{symbol}] FILLED {tag} ticket={result.order} "
                 f"price={result.price} lots={lots}")
        return True

    def send_buy_order(self, symbol: str, lots: float, sl: float, tp: float) -> bool:
        tick = self.mt5.symbol_info_tick(symbol)
        return self._send(symbol, self.mt5.ORDER_TYPE_BUY, tick.ask,
                          lots, sl, tp, "ORB_MTF_LONG")

    def send_sell_order(self, symbol: str, lots: float, sl: float, tp: float) -> bool:
        tick = self.mt5.symbol_info_tick(symbol)
        return self._send(symbol, self.mt5.ORDER_TYPE_SELL, tick.bid,
                          lots, sl, tp, "ORB_MTF_SHORT")


# ==============================================================================
# STATE MANAGEMENT — one instance per symbol, reset each trading day
# ==============================================================================
@dataclass
class DailyState:
    trading_day: date | None = None

    # 30m opening range
    orb_captured: bool = False
    orb_high: float = np.nan
    orb_low: float = np.nan

    # 10m breakout lock
    breakout_dir: str | None = None            # "long" | "short" | None
    breakout_time: pd.Timestamp | None = None

    # 2m pullback tracking
    pullback_seen: bool = False

    # one-trade-per-direction-per-day flags
    long_taken: bool = False
    short_taken: bool = False

    # de-duplication of processed candle timestamps
    last_10m_ts: pd.Timestamp | None = None
    last_2m_ts: pd.Timestamp | None = None

    def reset_for(self, day: date) -> None:
        """Full wipe at the start of every new trading day."""
        self.__init__()
        self.trading_day = day

    def cancel_setup(self) -> None:
        """Drop any armed-but-untriggered breakout (used at the 10:00 cutoff)."""
        self.breakout_dir = None
        self.breakout_time = None
        self.pullback_seen = False


# ==============================================================================
# PURE STRATEGY LOGIC (pandas / numpy — fully unit-testable, no broker calls)
# ==============================================================================
def extract_orb(df_30m: pd.DataFrame, today: date) -> tuple[float, float] | None:
    """
    Find TODAY's 08:00 (UK local) 30m candle and return (orb_high, orb_low)
    including wicks. Returns None if the candle hasn't closed / isn't present.

    The 30m feed index is already Europe/London, so 08:00 here is 08:00 BST in
    summer and 08:00 GMT in winter with zero extra handling.
    """
    day_candles = df_30m[df_30m.index.date == today]
    orb = day_candles[day_candles.index.time == SESSION_OPEN]
    if orb.empty:
        return None
    row = orb.iloc[0]
    return float(row["high"]), float(row["low"])


def is_valid_breakout_candle(candle: pd.Series, orb_high: float,
                             orb_low: float) -> str | None:
    """
    Validate a CLOSED 10m candle against the structural breakout rules.
    Returns "long", "short", or None.

    Rules:
      * open must be INSIDE the ORB range (structural break, not a gap/runaway)
      * close must be OUTSIDE the range in the breakout direction
      * real body >= 70% of total High-Low range
      * wick on the breakout side <= 20% of total range (no heavy rejection)
    """
    o, h, l, c = (float(candle[k]) for k in ("open", "high", "low", "close"))
    rng = h - l
    if rng <= 0:
        return None

    body_ratio = abs(c - o) / rng
    upper_wick_ratio = (h - max(o, c)) / rng
    lower_wick_ratio = (min(o, c) - l) / rng
    opened_inside = orb_low <= o <= orb_high

    if not opened_inside or body_ratio < BODY_MIN_RATIO:
        return None

    # Bullish structural break
    if c > orb_high and upper_wick_ratio <= WICK_MAX_RATIO:
        return "long"
    # Bearish structural break
    if c < orb_low and lower_wick_ratio <= WICK_MAX_RATIO:
        return "short"
    return None


def compute_ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    """Standard EMA (adjust=False matches charting-platform EMAs)."""
    return series.ewm(span=period, adjust=False).mean()


def check_2m_entry(df_2m: pd.DataFrame, state: DailyState,
                   buffer: float) -> dict | None:
    """
    Pullback-and-reclaim entry logic on the 2m chart.

    LONG : after the 10m breakout, wait for price to tag/dip below the 20 EMA
           (pullback), then trigger on the first 2m candle CLOSING back above it.
    SHORT: mirror — pullback to/above the EMA, trigger on a close back below it.

    Only 2m candles that formed AFTER the confirming 10m candle closed are
    considered (multi-timeframe sync: the 10m close timestamp is the hand-off
    point between the two frames).

    Returns a trade spec dict {direction, entry, sl, tp} or None.
    """
    if state.breakout_dir is None:
        return None

    df = df_2m.copy()
    df["ema"] = compute_ema(df["close"])

    # ---- MTF SYNC: ignore 2m candles from before the breakout confirmation --
    post_break = df[df.index > state.breakout_time]
    if post_break.empty:
        return None

    last = post_break.iloc[-1]          # most recent CLOSED 2m candle
    ts = post_break.index[-1]

    # De-dup: never evaluate the same closed candle twice
    if state.last_2m_ts is not None and ts <= state.last_2m_ts:
        return None
    state.last_2m_ts = ts

    ema_now = float(last["ema"])

    if state.breakout_dir == "long":
        # Phase 1 — pullback: price touches or dips below the 20 EMA
        if not state.pullback_seen:
            if float(last["low"]) <= ema_now:
                state.pullback_seen = True
                log.info(f"    2m pullback to EMA registered @ {ts}")
            return None
        # Phase 2 — reclaim: candle closes back ABOVE the EMA *and* is
        # itself a bullish candle (close > open). A bearish candle drifting
        # shut above the EMA is not confirmation -> keep waiting.
        if float(last["close"]) > ema_now and float(last["close"]) > float(last["open"]):
            entry = float(last["close"])
            # SL: lowest low of the most recent 5 2m candles, minus buffer.
            swing_low = float(df["low"].iloc[-SWING_LOOKBACK:].min())
            sl = swing_low - buffer
            risk = entry - sl
            if risk <= 0:
                return None
            tp = entry + risk * REWARD_MULTIPLE
            return {"direction": "long", "entry": entry, "sl": sl, "tp": tp}

    elif state.breakout_dir == "short":
        # Phase 1 — pullback: price touches or pokes above the 20 EMA
        if not state.pullback_seen:
            if float(last["high"]) >= ema_now:
                state.pullback_seen = True
                log.info(f"    2m pullback to EMA registered @ {ts}")
            return None
        # Phase 2 — rejection: candle closes back BELOW the EMA *and* is
        # itself a bearish candle (close < open). Mirror of the long rule.
        if float(last["close"]) < ema_now and float(last["close"]) < float(last["open"]):
            entry = float(last["close"])
            # SL: highest high of the most recent 5 2m candles, plus buffer.
            swing_high = float(df["high"].iloc[-SWING_LOOKBACK:].max())
            sl = swing_high + buffer
            risk = sl - entry
            if risk <= 0:
                return None
            tp = entry - risk * REWARD_MULTIPLE
            return {"direction": "short", "entry": entry, "sl": sl, "tp": tp}

    return None


# ==============================================================================
# POSITION SIZING — exactly 1% of balance at risk between entry and SL
# ==============================================================================
def calculate_position_size(balance: float, entry: float, sl: float,
                            sym_meta: dict) -> float:
    """
    lots = (balance * 1%) / (stop_distance_in_ticks * tick_value_per_lot)

    tick_size / tick_value come from the broker (symbol_info) so the same
    formula works for FX pips, JPY quotes, gold cents and index points alike.
    The result is floored to the broker's volume_step and clamped to
    [volume_min, volume_max].
    """
    risk_amount = balance * RISK_PER_TRADE_PCT
    stop_distance = abs(entry - sl)
    if stop_distance <= 0:
        return 0.0

    tick_size = sym_meta["tick_size"]
    tick_value = sym_meta["tick_value"]
    risk_per_lot = (stop_distance / tick_size) * tick_value
    if risk_per_lot <= 0:
        return 0.0

    lots = risk_amount / risk_per_lot

    # Snap DOWN to the broker volume step (never risk more than 1%).
    # The tiny epsilon prevents float error shaving an exact multiple
    # of the step down a notch (e.g. 5.00 lots -> 4.99).
    step = sym_meta.get("volume_step", 0.01)
    lots = np.floor(lots / step + 1e-9) * step
    lots = float(np.clip(lots, sym_meta.get("volume_min", 0.01),
                         sym_meta.get("volume_max", 100.0)))
    # Hard safety ceiling — independent of anything the broker allows
    lots = min(lots, MAX_LOT)
    return round(lots, 2)


# ==============================================================================
# ORCHESTRATOR — multi-timeframe synchronization loop
# ==============================================================================
class ORBBot:
    """
    Per-symbol driver. Timeframe hand-off order on every tick of the loop:

        (1) 30m  -> capture ORB once the 08:00 candle has closed (>= 08:30)
        (2) 10m  -> scan closed candles for a structural breakout (08:30-10:00)
        (3) 2m   -> once a breakout is locked, run pullback/reclaim entry logic

    Higher timeframes always gate lower ones: no 10m scan without an ORB,
    no 2m entries without a confirmed 10m breakout.
    """

    def __init__(self, broker: BrokerAPI, symbol: str, meta: dict):
        self.broker = broker
        self.symbol = symbol
        self.meta = meta
        self.state = DailyState()

    # ------------------------------------------------------------------ utils
    def _now(self) -> datetime:
        return datetime.now(LONDON_TZ)

    # ------------------------------------------------------------- main entry
    def step(self) -> None:
        now = self._now()
        today = now.date()

        # ---------- daily reset ----------
        if self.state.trading_day != today:
            self.state.reset_for(today)
            log.info(f"[{self.symbol}] New trading day {today} — state reset.")

        # Skip weekends entirely
        if now.weekday() >= 5:
            return

        # ---------- 10:00 cutoff: cancel pending, untriggered setups ----------
        if now.time() >= BREAKOUT_DEADLINE and self.state.breakout_dir is not None:
            log.info(f"[{self.symbol}] 10:00 deadline — cancelling pending "
                     f"{self.state.breakout_dir} setup.")
            self.state.cancel_setup()

        # =====================================================================
        # STAGE 1 — 30m: capture the opening range (only after 08:30 UK)
        # =====================================================================
        if not self.state.orb_captured:
            if now.time() < ORB_CLOSE:
                return                                   # ORB candle still forming
            df_30m = self.broker.fetch_candles(self.symbol, "30m", count=20)
            orb = extract_orb(df_30m, today)
            if orb is None:
                return                                   # feed lag — retry next tick
            self.state.orb_high, self.state.orb_low = orb
            self.state.orb_captured = True
            log.info(f"[{self.symbol}] ORB locked: high={orb[0]} low={orb[1]}")

        # =====================================================================
        # STAGE 2 — 10m: hunt for the structural breakout (08:30 → 10:00 only)
        # =====================================================================
        in_breakout_window = ORB_CLOSE <= now.time() < BREAKOUT_DEADLINE
        if self.state.breakout_dir is None and in_breakout_window:
            df_10m = self.broker.fetch_candles(self.symbol, "10m", count=12)
            # Only today's candles that closed after the ORB window
            candidates = df_10m[
                (df_10m.index.date == today)
                & (df_10m.index.time >= ORB_CLOSE)
            ]
            for ts, candle in candidates.iterrows():
                # De-dup: skip 10m candles already evaluated
                if self.state.last_10m_ts is not None and ts <= self.state.last_10m_ts:
                    continue
                self.state.last_10m_ts = ts

                direction = is_valid_breakout_candle(
                    candle, self.state.orb_high, self.state.orb_low
                )
                if direction is None:
                    continue
                # Enforce one trade per direction per day at the SETUP level too
                if direction == "long" and self.state.long_taken:
                    continue
                if direction == "short" and self.state.short_taken:
                    continue

                self.state.breakout_dir = direction
                # 10m candle index = open time; the 2m hand-off point is its CLOSE
                self.state.breakout_time = ts + pd.Timedelta(minutes=10)
                self.state.pullback_seen = False
                log.info(f"[{self.symbol}] 10m {direction.upper()} breakout "
                         f"confirmed @ {ts} — switching to 2m entry logic.")
                break

        # =====================================================================
        # STAGE 3 — 2m: pullback-and-reclaim entry off the 20 EMA
        # =====================================================================
        if self.state.breakout_dir is not None:
            df_2m = self.broker.fetch_candles(self.symbol, "2m", count=120)
            trade = check_2m_entry(df_2m, self.state, self.meta["buffer"])
            if trade:
                self._execute(trade)

    # ------------------------------------------------------------- execution
    def _execute(self, trade: dict) -> None:
        balance = self.broker.get_balance()
        sym_meta = self.broker.symbol_info(self.symbol)

        # Guardrail: reject setups with stops tighter than MIN_STOP_PIPS.
        # Tiny stops -> huge computed lot sizes; this is the failure mode
        # behind runaway position sizing.
        stop_pips = abs(trade["entry"] - trade["sl"]) / self.meta["pip_size"]
        if stop_pips < MIN_STOP_PIPS:
            log.warning(f"[{self.symbol}] Stop of {stop_pips:.1f} pips < "
                        f"MIN_STOP_PIPS={MIN_STOP_PIPS} — setup rejected.")
            self.state.cancel_setup()
            return

        lots = calculate_position_size(balance, trade["entry"], trade["sl"], sym_meta)
        if lots <= 0:
            log.warning(f"[{self.symbol}] Position size computed as 0 — skipping.")
            self.state.cancel_setup()
            return

        log.info(
            f"[{self.symbol}] {trade['direction'].upper()} | entry={trade['entry']:.5f} "
            f"SL={trade['sl']:.5f} TP={trade['tp']:.5f} lots={lots} "
            f"(risk={RISK_PER_TRADE_PCT:.0%} of {balance:.2f})"
        )

        if trade["direction"] == "long":
            ok = self.broker.send_buy_order(self.symbol, lots, trade["sl"], trade["tp"])
            if ok:
                self.state.long_taken = True             # one long max today
        else:
            ok = self.broker.send_sell_order(self.symbol, lots, trade["sl"], trade["tp"])
            if ok:
                self.state.short_taken = True            # one short max today

        # Setup consumed whether filled or rejected; a fresh 10m breakout in the
        # OPPOSITE direction can still arm before 10:00 (flags block repeats).
        self.state.cancel_setup()


# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main() -> None:
    broker = BrokerAPI()
    if not broker.connect():
        raise SystemExit("Broker connection failed.")

    # Validate symbols against this broker; skip any it doesn't offer
    # (e.g. MetaQuotes demo has no GER40 — broker names also vary: DE40,
    # GER40.cash, XAUUSD.a etc. Adjust the SYMBOLS dict keys to match.)
    active = {}
    for sym, meta in SYMBOLS.items():
        if broker.validate_symbol(sym):
            active[sym] = meta
        else:
            log.warning(f"Symbol '{sym}' not available on this broker — skipping. "
                        f"Check the exact name in Market Watch.")
    if not active:
        raise SystemExit("No valid symbols — fix names in the SYMBOLS dict.")

    bots = [ORBBot(broker, sym, meta) for sym, meta in active.items()]
    log.info(f"ORB multi-timeframe bot running on: {', '.join(active)}")

    while True:
        for bot in bots:
            try:
                bot.step()
            except Exception:
                log.exception(f"[{bot.symbol}] step() error — continuing.")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()