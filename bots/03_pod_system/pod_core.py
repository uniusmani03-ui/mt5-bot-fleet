"""
=============================================================================
 pod_core.py  —  shared plumbing for the POD strategy bots
=============================================================================
 Reusable MT5 helpers so each strategy file stays small: connection, symbol
 prep, risk-based lot sizing (with min-lot fallback), order send/close,
 PER-BOT profit/loss accounting (counts only THIS bot's own trades, so several
 bots can share one account without their loss limits colliding), CSV logging,
 a tiny status writer, and hand-coded indicators.

 Keep this file in the same folder as the bots. Windows + MT5 only.
 Install once:  pip install MetaTrader5 pandas numpy
=============================================================================
"""
import os
import csv
import json
import math
import time
from datetime import datetime, timezone

import pandas as pd
import MetaTrader5 as mt5


# ---------------- indicators (closed-candle, hand-coded) -------------------
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def donchian(df, n):
    """Highest high / lowest low of the n bars ENDING ONE BAR AGO (excludes the
    current bar, so a breakout of the prior range is detectable)."""
    return df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)


def _vol_decimals(step):
    return max(0, int(round(-math.log10(step)))) if step > 0 else 2


class Bot:
    """Holds one bot's identity (magic + files) and all broker interactions."""

    def __init__(self, name, magic, prefix, allow_min_lot=True):
        self.name = name
        self.magic = magic
        self.log_file = f"{prefix}_trades.csv"
        self.state_file = f"{prefix}_state.json"
        self.status_file = f"{prefix}_status.json"
        self.allow_min_lot = allow_min_lot
        self.max_lot = None   # optional hard cap on position size (set per bot)
        self.buf = []

    # ---- logging ----
    def log(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{self.name}] {msg}", flush=True)
        self.buf.append(line)
        self.buf[:] = self.buf[-80:]

    _CSV = ["timestamp_utc", "event", "symbol", "direction", "volume", "price",
            "sl", "tp", "retcode", "ticket", "profit", "comment"]

    def log_csv(self, event, symbol, direction="", volume=0.0, price=0.0, sl=0.0,
                tp=0.0, retcode="", ticket="", profit="", comment=""):
        row = {"timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
               "event": event, "symbol": symbol, "direction": direction, "volume": volume,
               "price": price, "sl": sl, "tp": tp, "retcode": retcode, "ticket": ticket,
               "profit": profit, "comment": comment}
        try:
            new = not os.path.exists(self.log_file)
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._CSV)
                if new:
                    w.writeheader()
                w.writerow(row)
        except Exception as exc:
            self.log(f"WARNING: CSV write failed -> {exc}")

    # ---- state ----
    def load_state(self):
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_state(self, d):
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(d, f)
        except Exception as exc:
            self.log(f"WARNING: state write failed -> {exc}")

    def heal_ref(self, state, balance):
        """Re-anchor a stale ref_balance (e.g. left over from a run on a different
        account). Without this, the loss limits can be sized to the wrong balance
        and fire instantly. Returns True if it re-anchored."""
        ref = state.get("ref_balance", 0)
        if ref <= 0 or balance > ref * 3 or ref > balance * 3:
            self.log(f"Saved ref balance {ref:.2f} doesn't match this account "
                     f"({balance:.2f}) — re-anchoring to {balance:.2f} and resetting limits.")
            state["ref_balance"] = balance
            state["start"] = datetime.now(timezone.utc).isoformat()
            state["day"] = ""
            state["day_start_realized"] = 0.0
            state["halted_today"] = False
            state["over"] = False
            self.save_state(state)
            return True
        return False

    # ---- connection / symbols ----
    def connect(self, login=None, password=None, server=None, path=None):
        kwargs = {}
        if path:
            kwargs["path"] = path
        if login and password and server:
            kwargs.update(login=int(login), password=password, server=server)
        if not mt5.initialize(**kwargs):
            self.log(f"ERROR: mt5.initialize() failed -> {mt5.last_error()}")
            return False
        term, acct = mt5.terminal_info(), mt5.account_info()
        if term is None or acct is None:
            self.log(f"ERROR: no terminal/account -> {mt5.last_error()}")
            return False
        if not term.trade_allowed:
            self.log("ERROR: Algo Trading disabled — click the green 'Algo Trading' button.")
            return False
        self.log(f"Connected. Account #{acct.login} | {acct.server} | "
                 f"Balance {acct.balance:.2f} {acct.currency} | Leverage 1:{acct.leverage}")
        return True

    def prepare(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            self.log(f"ERROR: symbol '{symbol}' not found -> {mt5.last_error()}")
            return None
        if not info.visible and not mt5.symbol_select(symbol, True):
            self.log(f"ERROR: could not select '{symbol}'")
            return None
        return mt5.symbol_info(symbol)

    def broker_utc_offset(self, symbol):
        for _ in range(20):
            t = mt5.symbol_info_tick(symbol)
            if t is not None and t.time != 0:
                return int(round((t.time - time.time()) / 3600.0))
            time.sleep(0.5)
        return 0

    # ---- sizing / execution ----
    def _filling(self, info):
        fm = info.filling_mode
        if fm & 2:
            return mt5.ORDER_FILLING_IOC
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def spread_ok(self, info, max_points):
        t = mt5.symbol_info_tick(info.name)
        if t is None:
            return False
        pts = (t.ask - t.bid) / info.point
        if pts > max_points:
            self.log(f"   {info.name}: spread {pts:.0f} > {max_points}; skip.")
            return False
        return True

    def calc_lot(self, info, sl_dist_price, risk_amount):
        tv, ts = info.trade_tick_value, info.trade_tick_size
        if ts <= 0 or tv <= 0 or sl_dist_price <= 0:
            return None
        loss_per_lot = (sl_dist_price / ts) * tv
        if loss_per_lot <= 0:
            return None
        step = info.volume_step
        vol = math.floor((risk_amount / loss_per_lot) / step) * step
        vol = round(vol, _vol_decimals(step))
        if vol < info.volume_min:
            if self.allow_min_lot:
                real = loss_per_lot * info.volume_min
                self.log(f"   {info.name}: risk {risk_amount:.2f} < min lot {info.volume_min}; "
                         f"using min = ~{real:.2f} risk ({real / max(risk_amount,1e-9):.1f}x target).")
                return info.volume_min
            self.log(f"   {info.name}: lot {vol} < min {info.volume_min}; skip.")
            return None
        vol = min(vol, info.volume_max)
        if self.max_lot is not None and vol > self.max_lot:
            self.log(f"   {info.name}: lot {vol} capped to MAX_LOT {self.max_lot} "
                     f"(tight stop on a large balance would size huge). Risk is reduced accordingly.")
            vol = round(min(vol, self.max_lot), _vol_decimals(info.volume_step))
        return vol

    def positions(self):
        out = []
        for p in (mt5.positions_get() or []):
            if p.magic == self.magic:
                out.append(p)
        return out

    def position(self, symbol):
        for p in (mt5.positions_get(symbol=symbol) or []):
            if p.magic == self.magic:
                return p
        return None

    def place(self, info, direction, sl_price, tp_price, risk_amount, slippage, comment,
              max_risk_amount=None):
        t = mt5.symbol_info_tick(info.name)
        if t is None:
            return False
        digits, point = info.digits, info.point
        min_dist = max(info.trade_stops_level, 1) * point
        if direction == "BUY":
            entry = t.ask
            if sl_price >= entry - min_dist:
                sl_price = entry - min_dist
            otype = mt5.ORDER_TYPE_BUY
            risk_dist = entry - sl_price
        else:
            entry = t.bid
            if sl_price <= entry + min_dist:
                sl_price = entry + min_dist
            otype = mt5.ORDER_TYPE_SELL
            risk_dist = sl_price - entry
        if risk_dist <= 0:
            self.log(f"   {info.name}: bad stop distance; abort.")
            return False
        vol = self.calc_lot(info, risk_dist, risk_amount)
        if vol is None:
            return False
        # Hard per-trade risk ceiling: on tiny accounts the min lot can risk far
        # more than intended. If so, SKIP rather than silently over-risk.
        if max_risk_amount is not None:
            loss_per_lot = (risk_dist / info.trade_tick_size) * info.trade_tick_value
            actual = loss_per_lot * vol
            if actual > max_risk_amount:
                self.log(f"   {info.name}: smallest lot risks ~{actual:.2f} > ceiling "
                         f"{max_risk_amount:.2f}; SKIP. (Account too small for this stop — "
                         f"a cent account fixes this.)")
                self.log_csv("skipped_risk", info.name, direction, vol,
                             comment=f"risk {actual:.2f}>{max_risk_amount:.2f}")
                return False
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": info.name, "volume": vol,
               "type": otype, "price": round(entry, digits), "sl": round(sl_price, digits),
               "tp": round(tp_price, digits) if tp_price else 0.0, "deviation": slippage,
               "magic": self.magic, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
               "type_filling": self._filling(info)}
        self.log(f">>> {direction} {info.name} vol={vol} @ {req['price']} "
                 f"SL={req['sl']} TP={req['tp']} (risk ~{risk_amount:.2f})")
        res = mt5.order_send(req)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            rc = res.retcode if res else mt5.last_error()
            self.log(f"   REJECTED ({rc}) {res.comment if res else ''}")
            self.log_csv("rejected", info.name, direction, vol, req["price"], req["sl"],
                         req["tp"], retcode=rc, comment=comment)
            return False
        self.log(f"   FILLED ticket={res.order} @ {res.price}")
        self.log_csv("filled", info.name, direction, res.volume, res.price, req["sl"],
                     req["tp"], retcode=res.retcode, ticket=res.order, comment=comment)
        return True

    def close(self, pos, slippage, comment):
        info = mt5.symbol_info(pos.symbol)
        t = mt5.symbol_info_tick(pos.symbol)
        if info is None or t is None:
            return False
        if pos.type == mt5.POSITION_TYPE_BUY:
            otype, price = mt5.ORDER_TYPE_SELL, t.bid
        else:
            otype, price = mt5.ORDER_TYPE_BUY, t.ask
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
               "type": otype, "position": pos.ticket, "price": round(price, info.digits),
               "deviation": slippage, "magic": self.magic, "comment": comment,
               "type_time": mt5.ORDER_TIME_GTC, "type_filling": self._filling(info)}
        res = mt5.order_send(req)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        self.log(f"   close {pos.symbol} #{pos.ticket}: {'OK' if ok else 'FAILED'} @ {price}")
        self.log_csv("closed" if ok else "close_failed", pos.symbol, volume=pos.volume,
                     price=price, ticket=pos.ticket, profit=round(pos.profit, 2), comment=comment)
        return ok

    def close_all(self, slippage, comment):
        for p in self.positions():
            self.close(p, slippage, comment)

    # ---- pending orders (limit / stop) ----
    _PEND = {"BUY_LIMIT": "ORDER_TYPE_BUY_LIMIT", "SELL_LIMIT": "ORDER_TYPE_SELL_LIMIT",
             "BUY_STOP": "ORDER_TYPE_BUY_STOP", "SELL_STOP": "ORDER_TYPE_SELL_STOP"}

    def place_pending(self, info, kind, price, sl_price, tp_price, risk_amount,
                      slippage, comment, max_risk_amount=None):
        if kind not in self._PEND:
            self.log(f"   bad pending kind {kind}"); return False
        risk_dist = abs(price - sl_price)
        if risk_dist <= 0:
            self.log(f"   {info.name}: bad pending stop distance; abort."); return False
        vol = self.calc_lot(info, risk_dist, risk_amount)
        if vol is None:
            return False
        if max_risk_amount is not None:
            lpl = (risk_dist / info.trade_tick_size) * info.trade_tick_value
            actual = lpl * vol
            if actual > max_risk_amount:
                self.log(f"   {info.name}: pending min-lot risk ~{actual:.2f} > ceiling "
                         f"{max_risk_amount:.2f}; SKIP. (cent account fixes this.)")
                return False
        req = {"action": mt5.TRADE_ACTION_PENDING, "symbol": info.name, "volume": vol,
               "type": getattr(mt5, self._PEND[kind]), "price": round(price, info.digits),
               "sl": round(sl_price, info.digits),
               "tp": round(tp_price, info.digits) if tp_price else 0.0,
               "magic": self.magic, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
               "type_filling": mt5.ORDER_FILLING_RETURN}
        res = mt5.order_send(req)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        rc = (res.retcode if res else mt5.last_error())
        self.log(f">>> {kind} {info.name} vol={vol} @ {req['price']} SL={req['sl']} "
                 f"TP={req['tp']} -> {'PLACED' if ok else 'REJECTED ' + str(rc)}")
        if ok:
            self.log_csv("pending", info.name, kind, vol, req["price"], req["sl"],
                         req["tp"], ticket=res.order, comment=comment)
        return ok

    def pending_orders(self):
        return [o for o in (mt5.orders_get() or []) if o.magic == self.magic]

    def cancel_all_pending(self):
        for o in self.pending_orders():
            res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
            self.log(f"   cancel pending #{o.ticket}: {'OK' if ok else 'FAIL'}")

    # ---- PER-BOT P&L (only this magic's trades) ----
    def own_pl(self, start_dt):
        realized = 0.0
        deals = mt5.history_deals_get(start_dt, datetime.now(timezone.utc))
        for d in (deals or []):
            if d.magic == self.magic:
                realized += d.profit + d.commission + d.swap
        floating = sum(p.profit for p in self.positions())
        return {"realized": round(realized, 2), "floating": round(floating, 2),
                "total": round(realized + floating, 2)}

    # ---- status snapshot for a dashboard ----
    def write_status(self, payload):
        try:
            tmp = self.status_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            for _ in range(6):
                try:
                    os.replace(tmp, self.status_file)
                    return
                except PermissionError:
                    time.sleep(0.05)
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as exc:
            self.log(f"WARNING: status write failed -> {exc}")
