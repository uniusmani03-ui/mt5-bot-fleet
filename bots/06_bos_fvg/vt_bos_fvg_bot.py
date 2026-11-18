"""
vt_bos_fvg_bot.py — 5M BOS -> FVG pullback bot (MT5 / VT Markets)

STRATEGY
--------
1. Track confirmed swing highs / lows on the M5 closed candles (fractal pivots).
2. BOS = a candle CLOSE beyond the most recent unbroken swing.
      close < last swing low  -> bearish BOS (new LL) -> bearish bias
      close > last swing high -> bullish BOS (new HH) -> bullish bias
3. Inside / after the BOS impulse leg, find a 3-candle FVG in the BOS direction.
      bearish FVG: low[1] > high[3]   -> gap = [high[3] (bottom), low[1] (top)]
      bullish FVG: high[1] < low[3]   -> gap = [high[1] (bottom), low[3] (top)]
4. Place a PENDING limit order at the FVG entry edge (proximal by default), so the
   trade only fires if price actually pulls back into the gap.
      short -> SELL_LIMIT at gap bottom (price is below the gap, rises into it)
      long  -> BUY_LIMIT  at gap top    (price is above the gap, falls into it)
5. Structural SL = far side of the FVG + buffer. This defines R.
      short -> above gap top ;  long -> below gap bottom
6. TP = entry +/- 1.5R  (RR_TARGET).
7. H1 trend filter: M5 BOS/FVG setups are accepted only when the closed H1
   market is trending in the same direction (EMA50/EMA200 alignment + slope).
8. IFVG exit: the FVG is *inverted* when a candle CLOSES through its far side.
   That close (not a wick) is the real stop. In SL_MODE="ifvg" the broker SL is
   parked wider as a disaster stop and the close-based logic does the exiting.

SAFETY (fleet standard)
-----------------------
- Broker server-time offset detected dynamically at runtime (never hardcoded).
- Default position sizing risks at most 1% of equity to the actual broker SL.
- MAX_LOT hard ceiling, min stop distance filter, broker stops-level respected.
- Daily loss halt + peak-equity drawdown circuit breaker, with self-healing refs.
- One live position + one pending order per symbol, per magic.
- Pending orders expire if the pullback never comes.

*** DEMO FIRST. Forward-test to t-stat >= 2.0 and PF >= 1.3 before live capital. ***
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    print("MetaTrader5 package not installed.  pip install MetaTrader5")
    raise

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # ---- connection -------------------------------------------------------
    login: Optional[int] = None            # or set MT5_LOGIN
    password: Optional[str] = None          # or set MT5_PASSWORD
    server: Optional[str] = None            # e.g. "VTMarkets-Demo" / MT5_SERVER
    terminal_path: Optional[str] = None     # optional terminal64.exe path

    # ---- universe ---------------------------------------------------------
    # True = discover every tradable FX pair exposed by the connected MT5
    # account, including VT Markets suffixes such as EURUSD+, GBPUSD.a, etc.
    trade_all_forex: bool = True
    # Used only when trade_all_forex=False (or when --symbols is supplied).
    symbols: Tuple[str, ...] = ("GBPUSD", "EURUSD", "USDJPY")
    # Optional exclusions can be either exact broker symbols (EURUSD.a) or
    # six-letter pair roots (EURUSD), which also exclude suffixed variants.
    forex_exclude: Tuple[str, ...] = ()
    timeframe: int = mt5.TIMEFRAME_M5
    magic: int = 770501

    # ---- structure --------------------------------------------------------
    pivot_left: int = 2                 # bars either side for a fractal pivot
    pivot_right: int = 2
    bars_history: int = 600             # M5 bars pulled each cycle
    bos_lookback_swings: int = 6        # how many recent swings to keep

    # ---- FVG --------------------------------------------------------------
    min_fvg_points: float = 0.0         # 0 = auto (0.15 * ATR14 in points)
    fvg_auto_atr_frac: float = 0.15
    max_bars_wait_for_fvg: int = 8      # after BOS, how long to wait for a gap
    entry_edge: str = "proximal"        # proximal | mid | distal
    require_unfilled: bool = True       # reject a gap already closed through
    require_virgin: bool = True         # reject a gap price has already traded into

    # ---- risk / exits -----------------------------------------------------
    lot_mode: str = "risk"              # fixed | risk
    fixed_lot: float = 0.01             # used only when lot_mode == "fixed"
    risk_pct: float = 1.0               # % equity risked to the actual broker SL
    rr_target: float = 1.5
    sl_mode: str = "ifvg"               # ifvg | structural
    sl_buffer_atr_frac: float = 0.10    # buffer beyond FVG far side
    disaster_pad_r: float = 0.75        # extra R beyond structural SL in ifvg mode
    min_stop_points: int = 80           # hard floor on stop distance, in points
    max_lot: float = 5.0                # ABSOLUTE ceiling (fleet standard)
    max_spread_points: int = 40         # skip if spread wider than this
    pending_expiry_bars: int = 12       # cancel un-triggered limit after N bars
    max_concurrent_positions: int = 3
    breakeven_at_r: float = 0.0         # 0 = off; e.g. 1.0 = move SL to BE at 1R

    # ---- higher-timeframe quality filter ----------------------------------
    use_htf_trend_filter: bool = True
    htf_timeframe: int = mt5.TIMEFRAME_H1
    htf_bars_history: int = 320          # extra warm-up for EMA200
    htf_fast_ema: int = 50
    htf_slow_ema: int = 200
    htf_slope_lookback: int = 3          # closed H1 bars
    htf_min_separation_atr_frac: float = 0.05  # reject flat EMA regimes

    # ---- session filter (UTC; offset-safe) --------------------------------
    use_session_filter: bool = False
    session_windows_utc: Tuple[Tuple[str, str], ...] = (("07:00", "11:00"),
                                                        ("12:30", "16:00"))
    trade_days: Tuple[int, ...] = (0, 1, 2, 3, 4)   # Mon..Fri

    # ---- circuit breakers -------------------------------------------------
    daily_loss_halt_pct: float = 2.0    # % of day-start equity
    max_drawdown_pct: float = 8.0       # % from peak equity -> full stop
    equity_sanity_ratio: float = 5.0    # heal refs if off by more than this

    # ---- plumbing ---------------------------------------------------------
    poll_seconds: int = 5
    state_file: str = "vt_bos_fvg_state.json"
    log_file: str = "vt_bos_fvg.log"
    dry_run: bool = False               # log intent, place nothing


CFG = Config()

log = logging.getLogger("vt_bos_fvg")


def setup_logging(cfg: Config, verbose: bool = False) -> None:
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(cfg.log_file)
    fh.setFormatter(fmt)
    log.addHandler(fh)


# =============================================================================
# BROKER TIME  — detected, never assumed
# =============================================================================

class BrokerClock:
    """
    Resolves the broker's server-time offset from UTC at runtime.

    The whole class exists because a hardcoded offset is how the ORB bot lost
    money. Nothing in this file ever writes a literal offset.
    """

    def __init__(self) -> None:
        self.offset_hours: float = 0.0
        self._resolved_at: Optional[datetime] = None

    def resolve(self, symbols: List[str]) -> float:
        best = None
        for sym in symbols:
            tick = mt5.symbol_info_tick(sym)
            if tick is None or tick.time == 0:
                continue
            server_dt = datetime.fromtimestamp(tick.time, tz=timezone.utc)
            true_utc = datetime.now(timezone.utc)
            delta_h = (server_dt - true_utc).total_seconds() / 3600.0
            # snap to nearest 0.5h; brokers use whole/half-hour offsets
            snapped = round(delta_h * 2) / 2
            if best is None or abs(snapped) > abs(best):
                best = snapped
        self.offset_hours = best if best is not None else 0.0
        self._resolved_at = datetime.now(timezone.utc)
        log.info("Broker server-time offset detected: UTC%+g", self.offset_hours)
        return self.offset_hours

    def refresh_if_stale(self, symbols: List[str], hours: int = 6) -> None:
        if (self._resolved_at is None
                or datetime.now(timezone.utc) - self._resolved_at > timedelta(hours=hours)):
            self.resolve(symbols)

    def server_to_utc(self, server_epoch: int) -> datetime:
        """MT5 bar/tick timestamps are server-local stamped as epoch."""
        naive = datetime.fromtimestamp(server_epoch, tz=timezone.utc)
        return naive - timedelta(hours=self.offset_hours)

    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def server_now_epoch(self) -> int:
        return int((self.utc_now() + timedelta(hours=self.offset_hours)).timestamp())


CLOCK = BrokerClock()


def in_session(cfg: Config, utc_dt: datetime) -> bool:
    if not cfg.use_session_filter:
        return utc_dt.weekday() in cfg.trade_days
    if utc_dt.weekday() not in cfg.trade_days:
        return False
    minutes = utc_dt.hour * 60 + utc_dt.minute
    for start, end in cfg.session_windows_utc:
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
        if sh * 60 + sm <= minutes < eh * 60 + em:
            return True
    return False


# =============================================================================
# MARKET DATA
# =============================================================================

@dataclass
class Bar:
    idx: int
    time: int          # server epoch
    open: float
    high: float
    low: float
    close: float


def latest_closed_bar_time(symbol: str, cfg: Config) -> Optional[int]:
    """Cheap new-candle probe used before downloading the full history."""
    rates = mt5.copy_rates_from_pos(symbol, cfg.timeframe, 1, 1)
    if rates is None or len(rates) != 1:
        return None
    return int(rates[0]["time"])


def fetch_bars(symbol: str, cfg: Config) -> List[Bar]:
    """Closed bars only. rates[-1] from MT5 is the forming bar — dropped."""
    rates = mt5.copy_rates_from_pos(symbol, cfg.timeframe, 0, cfg.bars_history)
    if rates is None or len(rates) < 60:
        log.warning("%s: insufficient bars (%s)", symbol, None if rates is None else len(rates))
        return []
    bars = [Bar(i, int(r["time"]), float(r["open"]), float(r["high"]),
                float(r["low"]), float(r["close"])) for i, r in enumerate(rates)]
    return bars[:-1]


def atr(bars: List[Bar], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(len(bars) - period, len(bars)):
        prev_close = bars[i - 1].close
        trs.append(max(bars[i].high - bars[i].low,
                       abs(bars[i].high - prev_close),
                       abs(bars[i].low - prev_close)))
    return sum(trs) / len(trs)


def ema_last(values: List[float], period: int) -> Optional[float]:
    """Return the final EMA, seeded with an SMA to reduce warm-up distortion."""
    if period <= 0 or len(values) < period:
        return None
    value = sum(values[:period]) / period
    alpha = 2.0 / (period + 1.0)
    for price in values[period:]:
        value = alpha * price + (1.0 - alpha) * value
    return value


# =============================================================================
# STRUCTURE:  swings -> BOS -> bias
# =============================================================================

@dataclass
class Swing:
    idx: int
    time: int
    price: float
    kind: str          # "high" | "low"
    broken: bool = False


class StructureTracker:
    """Fractal pivots, then a CLOSE beyond an unbroken pivot = BOS."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.swings: List[Swing] = []
        self.bias: str = "none"          # bullish | bearish | none
        self.last_bos_idx: Optional[int] = None
        self.last_bos_time: Optional[int] = None
        self.last_bos_ref_idx: Optional[int] = None   # index of the broken swing

    # -- pivots ------------------------------------------------------------
    def _find_pivots(self, bars: List[Bar]) -> List[Swing]:
        L, R = self.cfg.pivot_left, self.cfg.pivot_right
        out: List[Swing] = []
        for i in range(L, len(bars) - R):
            hi = bars[i].high
            lo = bars[i].low
            if (all(hi > bars[j].high for j in range(i - L, i)) and
                    all(hi >= bars[j].high for j in range(i + 1, i + R + 1))):
                out.append(Swing(i, bars[i].time, hi, "high"))
            if (all(lo < bars[j].low for j in range(i - L, i)) and
                    all(lo <= bars[j].low for j in range(i + 1, i + R + 1))):
                out.append(Swing(i, bars[i].time, lo, "low"))
        return out

    def update(self, bars: List[Bar]) -> Optional[Dict]:
        """Recompute structure. Returns a BOS event dict if the latest closed
        bar produced a fresh break, else None."""
        pivots = self._find_pivots(bars)
        if not pivots:
            return None
        self.swings = pivots[-(self.cfg.bos_lookback_swings * 2):]

        last = bars[-1]
        event = None

        # most recent unbroken swing of each kind that sits BEFORE this bar
        recent_high = next((s for s in reversed(self.swings)
                            if s.kind == "high" and s.idx < last.idx), None)
        recent_low = next((s for s in reversed(self.swings)
                           if s.kind == "low" and s.idx < last.idx), None)

        # bearish BOS: close below prior swing low -> new LL
        if recent_low and last.close < recent_low.price:
            if self.last_bos_ref_idx != recent_low.idx or self.bias != "bearish":
                self.bias = "bearish"
                self.last_bos_idx = last.idx
                self.last_bos_time = last.time
                self.last_bos_ref_idx = recent_low.idx
                event = {"dir": "sell", "bos_idx": last.idx, "bos_time": last.time,
                         "leg_start_idx": recent_low.idx, "broken_level": recent_low.price}

        # bullish BOS: close above prior swing high -> new HH
        if recent_high and last.close > recent_high.price:
            if self.last_bos_ref_idx != recent_high.idx or self.bias != "bullish":
                self.bias = "bullish"
                self.last_bos_idx = last.idx
                self.last_bos_time = last.time
                self.last_bos_ref_idx = recent_high.idx
                event = {"dir": "buy", "bos_idx": last.idx, "bos_time": last.time,
                         "leg_start_idx": recent_high.idx, "broken_level": recent_high.price}

        return event


# =============================================================================
# FVG
# =============================================================================

@dataclass
class FVG:
    direction: str      # "sell" (bearish gap) | "buy" (bullish gap)
    bottom: float
    top: float
    formed_idx: int     # index of the 3rd candle
    formed_time: int

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def entry_price(self, edge: str) -> float:
        # proximal = the side price reaches FIRST on the pullback
        if self.direction == "sell":
            return {"proximal": self.bottom, "mid": self.mid, "distal": self.top}[edge]
        return {"proximal": self.top, "mid": self.mid, "distal": self.bottom}[edge]

    def far_side(self) -> float:
        """Where invalidation lives: gap top for shorts, gap bottom for longs."""
        return self.top if self.direction == "sell" else self.bottom


def find_fvg(bars: List[Bar], direction: str, start_idx: int, end_idx: int,
             min_size: float, require_unfilled: bool,
             require_virgin: bool = True, entry_edge: str = "proximal") -> Optional[FVG]:
    """
    Scan [start_idx, end_idx] for the most recent qualifying 3-candle gap.
    Indices are positions in `bars` (which is 0..n-1 for closed bars).
    """
    lo = max(2, start_idx)
    hi = min(end_idx, len(bars) - 1)
    candidates: List[FVG] = []

    for i in range(lo, hi + 1):
        c1, c3 = bars[i - 2], bars[i]
        if direction == "sell" and c1.low > c3.high:
            gap = FVG("sell", c3.high, c1.low, i, c3.time)
        elif direction == "buy" and c1.high < c3.low:
            gap = FVG("buy", c1.high, c3.low, i, c3.time)
        else:
            continue
        if gap.size < min_size:
            continue
        if require_unfilled and _violated_after(bars, gap):
            continue
        if require_virgin and _already_tapped(bars, gap, entry_edge):
            continue
        # price must still be on the correct side for a limit entry to make sense
        entry = gap.entry_price(entry_edge)
        last_close = bars[-1].close
        if direction == "sell" and last_close >= entry:
            continue
        if direction == "buy" and last_close <= entry:
            continue
        candidates.append(gap)

    return candidates[-1] if candidates else None


def _violated_after(bars: List[Bar], gap: FVG) -> bool:
    """True if a later candle already CLOSED through the far side (already an IFVG)."""
    for b in bars[gap.formed_idx + 1:]:
        if gap.direction == "sell" and b.close > gap.top:
            return True
        if gap.direction == "buy" and b.close < gap.bottom:
            return True
    return False


def _already_tapped(bars: List[Bar], gap: FVG, entry_edge: str) -> bool:
    """
    True if price has already traded to the entry edge since the gap formed.
    A tapped gap is a spent setup — the pullback happened without us.
    """
    entry = gap.entry_price(entry_edge)
    for b in bars[gap.formed_idx + 1:]:
        if gap.direction == "sell" and b.high >= entry:
            return True
        if gap.direction == "buy" and b.low <= entry:
            return True
    return False


# =============================================================================
# RISK GUARD
# =============================================================================

class RiskGuard:
    def __init__(self, cfg: Config, state: Dict) -> None:
        self.cfg = cfg
        self.state = state

    def _today_key(self) -> str:
        return CLOCK.utc_now().strftime("%Y-%m-%d")

    def heal_ref(self, equity: float) -> None:
        """
        Self-healing equity references.

        A stale day_start_equity in state JSON is exactly what caused the
        21-lot incident: risk sized off a number that no longer resembled the
        account. Any ref that is implausible relative to live equity is reset.
        """
        s = self.state
        ratio = self.cfg.equity_sanity_ratio

        if s.get("day_key") != self._today_key():
            log.info("New trading day -> resetting daily refs at equity %.2f", equity)
            s["day_key"] = self._today_key()
            s["day_start_equity"] = equity
            s["daily_halted"] = False

        for key in ("day_start_equity", "peak_equity"):
            ref = s.get(key)
            if not ref or ref <= 0 or ref > equity * ratio or ref < equity / ratio:
                log.warning("heal_ref: %s=%s implausible vs equity %.2f -> reset",
                            key, ref, equity)
                s[key] = equity

        s["peak_equity"] = max(s["peak_equity"], equity)

    def check(self, equity: float) -> Tuple[bool, str]:
        s = self.state
        self.heal_ref(equity)

        if s.get("kill_switch"):
            return False, "kill_switch engaged"

        dd = (s["peak_equity"] - equity) / s["peak_equity"] * 100.0
        if dd >= self.cfg.max_drawdown_pct:
            s["kill_switch"] = True
            return False, f"drawdown breaker {dd:.2f}% >= {self.cfg.max_drawdown_pct}%"

        day_pl = (equity - s["day_start_equity"]) / s["day_start_equity"] * 100.0
        if day_pl <= -self.cfg.daily_loss_halt_pct:
            s["daily_halted"] = True
            return False, f"daily loss halt {day_pl:.2f}%"
        if s.get("daily_halted"):
            return False, "daily loss halt (already tripped today)"

        return True, "ok"


# =============================================================================
# EXECUTION HELPERS
# =============================================================================

# Common ISO/market currency codes used as a fallback when a broker's symbol
# path does not contain the word "Forex". Metals and crypto are intentionally
# absent so XAUUSD, BTCUSD, etc. are not mistaken for currency pairs.
FIAT_CURRENCIES = frozenset({
    "AED", "ARS", "AUD", "BDT", "BGN", "BHD", "BRL", "BWP", "CAD",
    "CHF", "CLP", "CNH", "CNY", "COP", "CZK", "DKK", "DZD", "EGP",
    "EUR", "GBP", "GHS", "HKD", "HRK", "HUF", "IDR", "ILS", "INR",
    "ISK", "JOD", "JPY", "KES", "KRW", "KWD", "KZT", "MAD", "MXN",
    "MYR", "NGN", "NOK", "NZD", "OMR", "PEN", "PHP", "PKR", "PLN",
    "QAR", "RON", "RSD", "RUB", "SAR", "SEK", "SGD", "THB", "TND",
    "TRY", "TWD", "UAH", "UGX", "USD", "UYU", "VND", "ZAR", "ZMW",
})


def _symbol_is_tradable_forex(info) -> bool:
    """Identify a two-currency instrument that permits normal opening trades."""
    disabled = getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0)
    close_only = getattr(mt5, "SYMBOL_TRADE_MODE_CLOSEONLY", 3)
    trade_mode = getattr(info, "trade_mode", disabled)
    if trade_mode in (disabled, close_only):
        return False

    base = str(getattr(info, "currency_base", "") or "").upper()
    quote = str(getattr(info, "currency_profit", "") or "").upper()
    valid_codes = len(base) == 3 and len(quote) == 3 and base != quote
    fiat_pair = base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES
    return valid_codes and fiat_pair


def _excluded_forex_symbol(info, cfg: Config) -> bool:
    excludes = {x.strip().upper() for x in cfg.forex_exclude if x.strip()}
    if not excludes:
        return False
    name = str(getattr(info, "name", "") or "").upper()
    base = str(getattr(info, "currency_base", "") or "").upper()
    quote = str(getattr(info, "currency_profit", "") or "").upper()
    return name in excludes or f"{base}{quote}" in excludes


def discover_forex_symbols(cfg: Config) -> List[str]:
    """Discover one usable VT Markets symbol for every available FX pair.

    Symbols are grouped by their base/quote currencies so duplicate aliases
    such as EURUSD and EURUSD.a cannot both be traded by the same bot.
    """
    all_syms = mt5.symbols_get()
    if not all_syms:
        return []

    by_pair: Dict[Tuple[str, str], List] = {}
    for info in all_syms:
        if not _symbol_is_tradable_forex(info) or _excluded_forex_symbol(info, cfg):
            continue
        key = (str(info.currency_base).upper(), str(info.currency_profit).upper())
        by_pair.setdefault(key, []).append(info)

    selected: List[str] = []
    for pair in sorted(by_pair):
        # Prefer the account's already-selected/visible alias, then the shortest
        # broker name. This normally chooses the correct account-specific suffix.
        candidates = sorted(
            by_pair[pair],
            key=lambda s: (
                not bool(getattr(s, "select", False)),
                not bool(getattr(s, "visible", False)),
                len(s.name),
                s.name,
            ),
        )
        for info in candidates:
            if mt5.symbol_select(info.name, True):
                selected.append(info.name)
                break
        else:
            log.warning("Could not select any broker symbol for %s/%s", *pair)

    log.info("Discovered %d tradable Forex pairs", len(selected))
    if selected:
        log.info("Forex universe: %s", ", ".join(selected))
    return selected


def resolve_symbol(want: str) -> Optional[str]:
    """Resolve a manually requested symbol, including VT Markets suffixes."""
    if mt5.symbol_info(want) is not None and mt5.symbol_select(want, True):
        return want
    all_syms = mt5.symbols_get()
    if not all_syms:
        return None
    base = want.upper()
    matches = [s.name for s in all_syms if s.name.upper().startswith(base)]
    if not matches:
        return None
    for pick in sorted(matches, key=len):
        if mt5.symbol_select(pick, True):
            log.info("Symbol %s resolved to broker symbol %s", want, pick)
            return pick
    return None


def normalise_lot(symbol: str, lots: float, cfg: Config, *, round_down: bool = False) -> float:
    """Apply broker limits; risk sizing rounds down so it cannot exceed target risk."""
    info = mt5.symbol_info(symbol)
    if info is None or lots <= 0:
        return 0.0
    step = info.volume_step or 0.01
    upper = min(info.volume_max, cfg.max_lot)
    lots = min(lots, upper)
    units = lots / step
    units = math.floor(units + 1e-12) if round_down else round(units)
    lots = round(units * step, 8)
    if lots < info.volume_min:
        return 0.0 if round_down else round(info.volume_min, 8)
    return min(lots, upper)


def estimated_loss(symbol: str, direction: str, volume: float,
                   entry_price: float, stop_price: float) -> Optional[float]:
    """Estimate stop loss in account currency using the connected MT5 account."""
    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
    result = mt5.order_calc_profit(order_type, symbol, volume, entry_price, stop_price)
    if result is None:
        return None
    return abs(float(result))


def calc_lots(symbol: str, direction: str, entry_price: float, stop_price: float,
              equity: float, cfg: Config) -> float:
    info = mt5.symbol_info(symbol)
    if info is None or equity <= 0 or entry_price <= 0 or stop_price <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0

    if cfg.lot_mode == "fixed":
        lots = normalise_lot(symbol, cfg.fixed_lot, cfg)
        if lots <= 0:
            return 0.0
        loss = estimated_loss(symbol, direction, lots, entry_price, stop_price)
        if loss is not None:
            log.info("%s: fixed %.2f lots -> ~%.2f %s risk to broker SL",
                     symbol, lots, loss,
                     mt5.account_info().currency if mt5.account_info() else "")
        return lots

    risk_cash = equity * (cfg.risk_pct / 100.0)
    one_lot_loss = estimated_loss(symbol, direction, 1.0, entry_price, stop_price)

    # Fallback for a terminal/broker that cannot calculate this symbol yet.
    if one_lot_loss is None or one_lot_loss <= 0:
        tick_val = info.trade_tick_value or 0.0
        tick_sz = info.trade_tick_size or info.point
        if tick_val <= 0 or tick_sz <= 0:
            log.warning("%s: cannot estimate stop loss, trade skipped", symbol)
            return 0.0
        one_lot_loss = stop_distance * (tick_val / tick_sz)
        log.warning("%s: order_calc_profit unavailable; using tick-value fallback", symbol)

    raw_lots = risk_cash / one_lot_loss
    lots = normalise_lot(symbol, raw_lots, cfg, round_down=True)
    if lots <= 0:
        min_loss = estimated_loss(symbol, direction, info.volume_min,
                                  entry_price, stop_price)
        log.info("%s: broker minimum lot would risk %s; target is %.2f — skipped",
                 symbol, "unknown" if min_loss is None else f"{min_loss:.2f}", risk_cash)
        return 0.0

    actual_loss = estimated_loss(symbol, direction, lots, entry_price, stop_price)
    if actual_loss is not None:
        log.info("%s: %.2f%% target = %.2f; sized %.2f lots for ~%.2f %s risk",
                 symbol, cfg.risk_pct, risk_cash, lots, actual_loss,
                 mt5.account_info().currency if mt5.account_info() else "")
    return lots


def min_broker_distance(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0
    return max(info.trade_stops_level, info.spread) * info.point


def spread_points(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not info or not tick or info.point <= 0:
        return 10 ** 6
    return int(round((tick.ask - tick.bid) / info.point))


def filling_mode(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    mask = info.filling_mode
    if mask & 1:
        return mt5.ORDER_FILLING_FOK
    if mask & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def rnd(symbol: str, price: float) -> float:
    info = mt5.symbol_info(symbol)
    return round(price, info.digits if info else 5)


# =============================================================================
# SETUP  (armed BOS -> waiting for FVG -> pending order live)
# =============================================================================

@dataclass
class Setup:
    symbol: str
    direction: str
    bos_idx: int
    bos_time: int
    leg_start_idx: int
    created_epoch: int
    fvg: Optional[Dict] = None      # serialised FVG once found
    order_ticket: Optional[int] = None
    entry: Optional[float] = None
    struct_sl: Optional[float] = None
    hard_sl: Optional[float] = None
    tp: Optional[float] = None
    bars_waited: int = 0


# =============================================================================
# THE BOT
# =============================================================================

class BosFvgBot:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = self._load_state()
        self.guard = RiskGuard(cfg, self.state)
        self.trackers: Dict[str, StructureTracker] = {}
        self.setups: Dict[str, Setup] = {}
        self.symbols: List[str] = []
        self.last_bar_time: Dict[str, int] = {}
        self.last_htf_bar_time: Dict[str, int] = {}
        self.htf_trend_cache: Dict[str, Tuple[str, str]] = {}

    # -- state -------------------------------------------------------------
    def _load_state(self) -> Dict:
        default = {"day_key": None, "day_start_equity": 0.0, "peak_equity": 0.0,
                   "daily_halted": False, "kill_switch": False, "trades": 0}
        if os.path.exists(self.cfg.state_file):
            try:
                with open(self.cfg.state_file) as fh:
                    default.update(json.load(fh))
            except Exception as exc:
                log.warning("State file unreadable (%s) — starting clean", exc)
        return default

    def _save_state(self) -> None:
        try:
            with open(self.cfg.state_file, "w") as fh:
                json.dump(self.state, fh, indent=2)
        except Exception as exc:
            log.warning("Could not persist state: %s", exc)

    # -- connection --------------------------------------------------------
    def connect(self) -> bool:
        kwargs = {}
        if self.cfg.terminal_path:
            kwargs["path"] = self.cfg.terminal_path
        login = self.cfg.login or (int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None)
        password = self.cfg.password or os.getenv("MT5_PASSWORD")
        server = self.cfg.server or os.getenv("MT5_SERVER")
        if login and password and server:
            kwargs.update(login=login, password=password, server=server)

        if not mt5.initialize(**kwargs):
            log.error("mt5.initialize failed: %s", mt5.last_error())
            return False

        acct = mt5.account_info()
        if acct is None:
            log.error("No account info after initialize")
            return False
        log.info("Connected: #%s %s | %s | equity %.2f %s",
                 acct.login, acct.server, acct.company, acct.equity, acct.currency)
        if acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL and not self.cfg.dry_run:
            log.warning("*** LIVE ACCOUNT DETECTED — this strategy has not been "
                        "forward-tested by this process. Confirm you meant to do this. ***")

        if self.cfg.trade_all_forex:
            self.symbols = discover_forex_symbols(self.cfg)
        else:
            self.symbols = [s for s in (resolve_symbol(x) for x in self.cfg.symbols) if s]
        if not self.symbols:
            log.error("No tradable symbols resolved")
            return False
        CLOCK.resolve(self.symbols)
        for s in self.symbols:
            self.trackers[s] = StructureTracker(self.cfg)
        return True

    # -- position / order introspection ------------------------------------
    def _my_positions(self, symbol: Optional[str] = None) -> List:
        pos = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return [p for p in (pos or []) if p.magic == self.cfg.magic]

    def _my_orders(self, symbol: Optional[str] = None) -> List:
        ords = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        return [o for o in (ords or []) if o.magic == self.cfg.magic]

    # -- higher-timeframe trend filter -------------------------------------
    def htf_trend(self, symbol: str) -> Tuple[str, str]:
        """Return bullish/bearish/neutral from fully closed H1 candles only."""
        if not self.cfg.use_htf_trend_filter:
            return "disabled", "filter disabled"

        probe = mt5.copy_rates_from_pos(symbol, self.cfg.htf_timeframe, 1, 1)
        if probe is None or len(probe) != 1:
            return "neutral", "H1 data unavailable"
        newest = int(probe[0]["time"])
        cached = self.htf_trend_cache.get(symbol)
        if cached is not None and self.last_htf_bar_time.get(symbol) == newest:
            return cached

        needed = max(self.cfg.htf_bars_history, self.cfg.htf_slow_ema + 30)
        rates = mt5.copy_rates_from_pos(symbol, self.cfg.htf_timeframe, 0, needed)
        if rates is None or len(rates) < self.cfg.htf_slow_ema + 5:
            result = ("neutral", "insufficient H1 history")
            self.htf_trend_cache[symbol] = result
            self.last_htf_bar_time[symbol] = newest
            return result

        closed = rates[:-1]
        closes = [float(r["close"]) for r in closed]
        fast = ema_last(closes, self.cfg.htf_fast_ema)
        slow = ema_last(closes, self.cfg.htf_slow_ema)
        lookback = max(1, self.cfg.htf_slope_lookback)
        fast_prev = ema_last(closes[:-lookback], self.cfg.htf_fast_ema)
        if fast is None or slow is None or fast_prev is None:
            result = ("neutral", "EMA calculation unavailable")
        else:
            htf_bars = [
                Bar(i, int(r["time"]), float(r["open"]), float(r["high"]),
                    float(r["low"]), float(r["close"]))
                for i, r in enumerate(closed)
            ]
            htf_atr = atr(htf_bars)
            separation = abs(fast - slow)
            min_separation = htf_atr * self.cfg.htf_min_separation_atr_frac
            last_close = closes[-1]

            if separation < min_separation:
                result = ("neutral", "H1 EMAs too close/flat")
            elif last_close > fast > slow and fast > fast_prev:
                result = ("bullish",
                          f"H1 close {last_close:.5f} > EMA{self.cfg.htf_fast_ema} > "
                          f"EMA{self.cfg.htf_slow_ema}, fast EMA rising")
            elif last_close < fast < slow and fast < fast_prev:
                result = ("bearish",
                          f"H1 close {last_close:.5f} < EMA{self.cfg.htf_fast_ema} < "
                          f"EMA{self.cfg.htf_slow_ema}, fast EMA falling")
            else:
                result = ("neutral", "H1 close/EMA alignment or slope not confirmed")

        self.htf_trend_cache[symbol] = result
        self.last_htf_bar_time[symbol] = newest
        return result

    def htf_allows(self, symbol: str, direction: str) -> Tuple[bool, str]:
        trend, detail = self.htf_trend(symbol)
        if trend == "disabled":
            return True, detail
        wanted = "bullish" if direction == "buy" else "bearish"
        return trend == wanted, f"{trend}: {detail}"

    # -- order placement ---------------------------------------------------
    def place_pending(self, setup: Setup) -> bool:
        sym = setup.symbol
        info = mt5.symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        if not info or not tick:
            return False

        trade_mode = getattr(info, "trade_mode", getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0))
        if (setup.direction == "sell"
                and trade_mode == getattr(mt5, "SYMBOL_TRADE_MODE_LONGONLY", 1)):
            log.info("%s: broker currently allows long trades only — sell skipped", sym)
            return False
        if (setup.direction == "buy"
                and trade_mode == getattr(mt5, "SYMBOL_TRADE_MODE_SHORTONLY", 2)):
            log.info("%s: broker currently allows short trades only — buy skipped", sym)
            return False

        order_type = (mt5.ORDER_TYPE_SELL_LIMIT if setup.direction == "sell"
                      else mt5.ORDER_TYPE_BUY_LIMIT)

        # a limit must sit on the correct side of current price
        if setup.direction == "sell" and setup.entry <= tick.bid + min_broker_distance(sym):
            log.info("%s: sell-limit %.5f too close/below market %.5f — skipped",
                     sym, setup.entry, tick.bid)
            return False
        if setup.direction == "buy" and setup.entry >= tick.ask - min_broker_distance(sym):
            log.info("%s: buy-limit %.5f too close/above market %.5f — skipped",
                     sym, setup.entry, tick.ask)
            return False

        # Size to the actual broker SL, not the closer IFVG structural level.
        # This caps the worst-case hard-stop loss at the configured risk percent.
        lots = calc_lots(sym, setup.direction, setup.entry, setup.hard_sl,
                         mt5.account_info().equity, self.cfg)
        if lots <= 0:
            log.info("%s: lot size resolved to 0 — skipped", sym)
            return False

        expiry = CLOCK.server_now_epoch() + self.cfg.pending_expiry_bars * 5 * 60

        req = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": sym,
            "volume": lots,
            "type": order_type,
            "price": rnd(sym, setup.entry),
            "sl": rnd(sym, setup.hard_sl),
            "tp": rnd(sym, setup.tp),
            "deviation": 20,
            "magic": self.cfg.magic,
            "comment": f"BOSFVG {setup.direction}",
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiry,
            "type_filling": filling_mode(sym),
        }

        log.info("%s SETUP %s | entry %.5f | structSL %.5f | hardSL %.5f | TP %.5f | %.2f lots",
                 sym, setup.direction.upper(), setup.entry, setup.struct_sl,
                 setup.hard_sl, setup.tp, lots)

        if self.cfg.dry_run:
            log.info("[DRY RUN] pending order not sent")
            setup.order_ticket = -1
            return True

        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            # retry once with GTC in case the broker rejects timed expiry
            log.warning("%s: pending rejected (%s) — retrying GTC", sym,
                        None if res is None else res.retcode)
            req.pop("expiration", None)
            req["type_time"] = mt5.ORDER_TIME_GTC
            res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("%s: pending order failed: %s", sym,
                      None if res is None else (res.retcode, res.comment))
            return False

        setup.order_ticket = res.order
        log.info("%s: pending order #%s placed", sym, res.order)
        return True

    def cancel_order(self, ticket: int, why: str) -> None:
        if self.cfg.dry_run:
            log.info("[DRY RUN] would cancel #%s (%s)", ticket, why)
            return
        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        log.info("Cancel #%s (%s): %s", ticket, why, "ok" if ok else "failed")

    def close_position(self, pos, why: str) -> None:
        sym = pos.symbol
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            return
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": tick.bid if is_buy else tick.ask,
            "deviation": 30,
            "magic": self.cfg.magic,
            "comment": why[:31],
            "type_filling": filling_mode(sym),
        }
        log.info("%s: closing #%s — %s", sym, pos.ticket, why)
        if self.cfg.dry_run:
            return
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("%s: close failed %s", sym,
                      None if res is None else (res.retcode, res.comment))

    def modify_sl(self, pos, new_sl: float, why: str) -> None:
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket,
               "symbol": pos.symbol, "sl": rnd(pos.symbol, new_sl), "tp": pos.tp}
        log.info("%s: SL -> %.5f (%s)", pos.symbol, new_sl, why)
        if self.cfg.dry_run:
            return
        mt5.order_send(req)

    # -- IFVG exit ---------------------------------------------------------
    def check_ifvg_exit(self, symbol: str, bars: List[Bar]) -> None:
        """
        The FVG is inverted the moment a candle CLOSES through its far side.
        Close-based, so wicks are ignored by design.
        """
        positions = self._my_positions(symbol)
        if not positions or not bars:
            return
        rec = self.state.get("open_fvg", {}).get(symbol)
        if not rec:
            return

        last = bars[-1]
        if rec.get("last_checked_bar") == last.time:
            return
        rec["last_checked_bar"] = last.time

        for pos in positions:
            if pos.type == mt5.POSITION_TYPE_SELL and last.close > rec["top"]:
                self.close_position(pos, f"IFVG close {last.close:.5f}>{rec['top']:.5f}")
            elif pos.type == mt5.POSITION_TYPE_BUY and last.close < rec["bottom"]:
                self.close_position(pos, f"IFVG close {last.close:.5f}<{rec['bottom']:.5f}")
            elif self.cfg.breakeven_at_r > 0:
                self._maybe_breakeven(pos, rec)

    def _maybe_breakeven(self, pos, rec: Dict) -> None:
        r = abs(pos.price_open - rec["struct_sl"])
        if r <= 0:
            return
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return
        if pos.type == mt5.POSITION_TYPE_BUY:
            moved = (tick.bid - pos.price_open) / r
            if moved >= self.cfg.breakeven_at_r and pos.sl < pos.price_open:
                self.modify_sl(pos, pos.price_open, f"BE at {moved:.2f}R")
        else:
            moved = (pos.price_open - tick.ask) / r
            if moved >= self.cfg.breakeven_at_r and (pos.sl == 0 or pos.sl > pos.price_open):
                self.modify_sl(pos, pos.price_open, f"BE at {moved:.2f}R")

    # -- setup lifecycle ---------------------------------------------------
    def build_setup(self, symbol: str, setup: Setup, bars: List[Bar], a: float) -> bool:
        min_size = (self.cfg.min_fvg_points if self.cfg.min_fvg_points > 0
                    else max(a * self.cfg.fvg_auto_atr_frac,
                             self.cfg.min_stop_points * mt5.symbol_info(symbol).point * 0.25))

        gap = find_fvg(bars, setup.direction, setup.leg_start_idx, len(bars) - 1,
                       min_size, self.cfg.require_unfilled,
                       self.cfg.require_virgin, self.cfg.entry_edge)
        if gap is None:
            return False

        entry = gap.entry_price(self.cfg.entry_edge)
        buffer = max(a * self.cfg.sl_buffer_atr_frac,
                     mt5.symbol_info(symbol).point * 5)

        if setup.direction == "sell":
            struct_sl = gap.top + buffer
        else:
            struct_sl = gap.bottom - buffer

        r = abs(entry - struct_sl)
        point = mt5.symbol_info(symbol).point
        if r < self.cfg.min_stop_points * point:
            log.info("%s: stop %.1f pts below floor %d — setup rejected",
                     symbol, r / point, self.cfg.min_stop_points)
            return False
        if r < min_broker_distance(symbol):
            log.info("%s: stop inside broker minimum distance — setup rejected", symbol)
            return False

        tp = entry - r * self.cfg.rr_target if setup.direction == "sell" \
            else entry + r * self.cfg.rr_target

        if self.cfg.sl_mode == "ifvg":
            pad = r * self.cfg.disaster_pad_r
            hard_sl = struct_sl + pad if setup.direction == "sell" else struct_sl - pad
        else:
            hard_sl = struct_sl

        setup.fvg = {"direction": gap.direction, "bottom": gap.bottom, "top": gap.top,
                     "formed_time": gap.formed_time, "struct_sl": struct_sl}
        setup.entry, setup.struct_sl, setup.hard_sl, setup.tp = entry, struct_sl, hard_sl, tp
        return True

    def invalidate_checks(self, symbol: str, setup: Setup, bars: List[Bar]) -> Optional[str]:
        if not setup.fvg or not bars:
            return None
        last = bars[-1]
        top, bottom = setup.fvg["top"], setup.fvg["bottom"]

        if setup.direction == "sell":
            if last.close > top:
                return "FVG inverted before entry"
            if last.low <= setup.tp:
                return "TP level reached without entry"
        else:
            if last.close < bottom:
                return "FVG inverted before entry"
            if last.high >= setup.tp:
                return "TP level reached without entry"

        tracker = self.trackers[symbol]
        if tracker.bias != "none" and tracker.bias[:4] != ("bear" if setup.direction == "sell" else "bull"):
            return "structure flipped against setup"
        allowed, htf_reason = self.htf_allows(symbol, setup.direction)
        if not allowed:
            return f"H1 trend filter no longer aligned ({htf_reason})"
        return None

    # -- per-symbol cycle --------------------------------------------------
    def process_symbol(self, symbol: str) -> None:
        # With a full FX universe, avoid downloading 600 bars every five seconds.
        # Probe one closed candle first and only fetch history when a new bar exists.
        newest = latest_closed_bar_time(symbol, self.cfg)
        if newest is None or self.last_bar_time.get(symbol) == newest:
            return

        bars = fetch_bars(symbol, self.cfg)
        if not bars:
            return
        newest = bars[-1].time
        if self.last_bar_time.get(symbol) == newest:
            return
        self.last_bar_time[symbol] = newest

        a = atr(bars)
        self.check_ifvg_exit(symbol, bars)

        # sync: pending order gone?
        setup = self.setups.get(symbol)
        if setup and setup.order_ticket and setup.order_ticket > 0:
            still_open = any(o.ticket == setup.order_ticket for o in self._my_orders(symbol))
            if not still_open:
                filled = bool(self._my_positions(symbol))
                if filled and setup.fvg:
                    self.state.setdefault("open_fvg", {})[symbol] = {
                        "top": setup.fvg["top"], "bottom": setup.fvg["bottom"],
                        "struct_sl": setup.struct_sl, "last_checked_bar": 0}
                    self.state["trades"] = self.state.get("trades", 0) + 1
                    log.info("%s: pending filled -> IFVG monitor armed", symbol)
                else:
                    log.info("%s: pending order gone (expired/cancelled)", symbol)
                self.setups.pop(symbol, None)
                setup = None

        if not self._my_positions(symbol) and symbol in self.state.get("open_fvg", {}):
            self.state["open_fvg"].pop(symbol, None)

        # structure update / BOS detection
        event = self.trackers[symbol].update(bars)
        if event:
            log.info("%s: %s BOS on bar %s (broke %.5f)", symbol,
                     "BEARISH (new LL)" if event["dir"] == "sell" else "BULLISH (new HH)",
                     datetime.utcfromtimestamp(event["bos_time"]).strftime("%H:%M"),
                     event["broken_level"])
            if setup and setup.order_ticket and setup.order_ticket > 0 and setup.direction != event["dir"]:
                self.cancel_order(setup.order_ticket, "opposite BOS")
                self.setups.pop(symbol, None)
                setup = None
            if not setup:
                allowed, htf_reason = self.htf_allows(symbol, event["dir"])
                if not allowed:
                    log.info("%s: %s BOS rejected by H1 trend filter (%s)",
                             symbol, event["dir"].upper(), htf_reason)
                else:
                    log.info("%s: H1 trend filter confirmed %s (%s)",
                             symbol, event["dir"].upper(), htf_reason)
                    setup = Setup(symbol, event["dir"], event["bos_idx"], event["bos_time"],
                                  event["leg_start_idx"], int(time.time()))
                    self.setups[symbol] = setup

        if not setup:
            return

        # gate: don't arm new orders when blocked
        ok, reason = self.guard.check(mt5.account_info().equity)
        tradable = ok and in_session(self.cfg, CLOCK.utc_now())
        if not tradable and not (setup.order_ticket and setup.order_ticket > 0):
            if not ok:
                log.debug("%s: setup held — %s", symbol, reason)
            return

        if self._my_positions(symbol):
            return                       # one position per symbol
        # Count live pendings as risk slots too. Otherwise an all-pairs scan could
        # place many 1%-risk orders before any of them trigger simultaneously.
        has_live_pending = bool(setup.order_ticket and setup.order_ticket > 0)
        if not has_live_pending:
            active_risk_slots = len(self._my_positions()) + len(self._my_orders())
            if active_risk_slots >= self.cfg.max_concurrent_positions:
                log.debug("%s: active position/pending risk-slot cap reached", symbol)
                return
        if spread_points(symbol) > self.cfg.max_spread_points:
            log.info("%s: spread %d pts too wide", symbol, spread_points(symbol))
            return

        # already has a live pending: just police invalidation
        if setup.order_ticket and setup.order_ticket > 0:
            why = self.invalidate_checks(symbol, setup, bars)
            if why:
                self.cancel_order(setup.order_ticket, why)
                self.setups.pop(symbol, None)
            return

        # waiting for the FVG to print
        setup.bars_waited += 1
        if setup.bars_waited > self.cfg.max_bars_wait_for_fvg:
            log.info("%s: no qualifying FVG within %d bars — setup dropped",
                     symbol, self.cfg.max_bars_wait_for_fvg)
            self.setups.pop(symbol, None)
            return

        allowed, htf_reason = self.htf_allows(symbol, setup.direction)
        if not allowed:
            log.info("%s: setup dropped before entry; H1 filter not aligned (%s)",
                     symbol, htf_reason)
            self.setups.pop(symbol, None)
            return

        if self.build_setup(symbol, setup, bars, a):
            if not self.place_pending(setup):
                self.setups.pop(symbol, None)

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        if not self.connect():
            return
        sizing = (f"fixed {self.cfg.fixed_lot:.2f} lots" if self.cfg.lot_mode == "fixed"
                  else f"risk {self.cfg.risk_pct:.2f}%")
        log.info("Running | symbols=%s | sizing=%s | RR=%.2f | sl_mode=%s | "
                 "H1_filter=%s | dry_run=%s",
                 self.symbols, sizing, self.cfg.rr_target, self.cfg.sl_mode,
                 self.cfg.use_htf_trend_filter, self.cfg.dry_run)
        try:
            while True:
                CLOCK.refresh_if_stale(self.symbols)
                acct = mt5.account_info()
                if acct is None:
                    log.error("Lost account connection — retrying")
                    time.sleep(10)
                    continue

                ok, reason = self.guard.check(acct.equity)
                if not ok and self.state.get("kill_switch"):
                    log.error("KILL SWITCH: %s — flattening and exiting", reason)
                    for p in self._my_positions():
                        self.close_position(p, "kill switch")
                    for o in self._my_orders():
                        self.cancel_order(o.ticket, "kill switch")
                    self._save_state()
                    return

                for sym in self.symbols:
                    try:
                        self.process_symbol(sym)
                    except Exception as exc:
                        log.exception("%s: cycle error: %s", sym, exc)

                self._save_state()
                time.sleep(self.cfg.poll_seconds)

        except KeyboardInterrupt:
            log.info("Interrupted — pendings left in place, positions untouched.")
        finally:
            self._save_state()
            mt5.shutdown()


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="VT Markets 5M BOS->FVG bot")
    universe = ap.add_mutually_exclusive_group()
    universe.add_argument(
        "--all-forex", action="store_true",
        help="auto-discover every tradable Forex pair (default)",
    )
    universe.add_argument(
        "--symbols", help="manual comma-separated symbols, e.g. GBPUSD,XAUUSD",
    )
    ap.add_argument(
        "--exclude-symbols",
        help="comma-separated FX symbols/pair roots to omit, e.g. USDTRY,EURTRY.a",
    )
    ap.add_argument("--lots", type=float, help="fixed lot size (default 0.01)")
    ap.add_argument("--lot-mode", choices=["fixed", "risk"])
    ap.add_argument("--risk", type=float, help="risk %% per trade (risk mode only)")
    ap.add_argument("--rr", type=float, help="reward:risk target (default 1.5)")
    ap.add_argument("--sl-mode", choices=["ifvg", "structural"])
    ap.add_argument("--entry-edge", choices=["proximal", "mid", "distal"])
    ap.add_argument("--max-lot", type=float)
    ap.add_argument("--dry-run", action="store_true", help="log only, place nothing")
    ap.add_argument("--session-filter", action="store_true")
    ap.add_argument(
        "--no-htf-filter", action="store_true",
        help="disable the default H1 EMA50/EMA200 trend-quality filter",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--reset-kill-switch", action="store_true")
    args = ap.parse_args()

    cfg = CFG
    if args.symbols:
        cfg.trade_all_forex = False
        cfg.symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    elif args.all_forex:
        cfg.trade_all_forex = True
    if args.exclude_symbols:
        cfg.forex_exclude = tuple(
            s.strip().upper() for s in args.exclude_symbols.split(",") if s.strip()
        )
    if args.lots is not None:
        cfg.fixed_lot = args.lots
        cfg.lot_mode = "fixed"
    if args.lot_mode:
        cfg.lot_mode = args.lot_mode
    if args.risk is not None:
        if not 0 < args.risk <= 5.0:
            ap.error("--risk must be greater than 0 and no more than 5 percent")
        cfg.risk_pct = args.risk
        if not args.lot_mode:
            cfg.lot_mode = "risk"
    if args.rr is not None:
        cfg.rr_target = args.rr
    if args.sl_mode:
        cfg.sl_mode = args.sl_mode
    if args.entry_edge:
        cfg.entry_edge = args.entry_edge
    if args.max_lot is not None:
        cfg.max_lot = min(args.max_lot, 5.0)      # ceiling is not overridable upward
    cfg.dry_run = args.dry_run
    if args.session_filter:
        cfg.use_session_filter = True
    if args.no_htf_filter:
        cfg.use_htf_trend_filter = False

    setup_logging(cfg, args.verbose)

    bot = BosFvgBot(cfg)
    if args.reset_kill_switch:
        bot.state["kill_switch"] = False
        bot.state["daily_halted"] = False
        bot._save_state()
        log.info("Kill switch and daily halt cleared.")
    bot.run()


if __name__ == "__main__":
    main()