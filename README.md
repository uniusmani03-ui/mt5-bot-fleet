# MT5 Algorithmic Trading Bot Fleet + Mission Control

> **Personal project**: built in my own time
> **Dates:** June 2026 to August 2026
> **Author:** Unais Usmani

This repository is a series of automated trading bots for MetaTrader 5 (Python), plus **Mission Control**, a FastAPI web dashboard that deploys, runs and monitors the whole fleet. The bots are in date order, so the repo also shows how my engineering approach changed after real failures.

## The fleet (in date order)

| # | Folder | Built | Strategy / purpose |
|---|---|---|---|
| 1 | `bots/01_scalper` | Jun 2026 | M1 EMA crossover + VWAP + RSI micro-scalper (first bot) |
| 2 | `bots/02_prop_breakout` | Jun 2026 | Liquidity-sweep / false-breakout fade with prop-firm style risk rules |
| 3 | `bots/03_pod_system` | Jun–Jul 2026 | Shared `pod_core` library + three strategy "pods" (low-risk Donchian trend, high-risk momentum, London session) + live dashboard |
| 4 | `bots/04_gold_session` | Jul 2026 | XAUUSD Asian-range session breakout + a backtester that replays the exact live logic |
| 5 | `bots/05_orb` | Jul 2026 | Multi-timeframe 30-minute Opening Range Breakout + backtester |
| 6 | `bots/06_bos_fvg` | Aug 2026 | 5-minute break-of-structure → fair-value-gap pullback with pending limit orders |
| 7 | `bots/07_adaptive_4h` | Aug 2026 | Adaptive multi-strategy bot (demo by default, live opt-in) |
| 8 | `bots/08_m1_scalper_ny` | Aug 2026 | M1 scalper and New York session bot |
| – | `mission_control/` | Jul 2026 | FastAPI + HTML dashboard: drag-and-drop bot deployment, per-bot processes, live consoles, stop-all kill switch |

## What I learned

The ORB bot went live before its backtest was finished and lost real money. The cause was a UTC+3 broker time-zone bug that made it read the wrong candle. A separate sizing incident led to an oversized position. Both shaped the safety design of the later bots and of my rebuilt research stack, [fxsys](../2026-08_fxsys-backtesting-framework):

- runtime time-zone detection
- hard lot ceilings
- daily-loss kill switches
- backtest-before-live, gated on statistical significance

## Stack

Python, MetaTrader5 API, pandas/NumPy, FastAPI, HTML/Tailwind, subprocess-based process management.

## Run

Each bot runs on Windows with MetaTrader 5 open:

```bash
pip install MetaTrader5 pandas numpy fastapi uvicorn python-multipart
python mission_control/mission_control.py     # http://localhost:8600
```

Account credentials come from the `MT5_LOGIN`, `MT5_PASSWORD` and `MT5_SERVER` environment variables. No credentials are stored in this repository. **Educational project: not financial advice.**
