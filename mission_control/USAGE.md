# Mission Control — MT5 Bot Fleet Manager

Drag & drop your bots, connect accounts, press play. No more VS Code copy-paste.

## Setup (once)

```
pip install fastapi uvicorn python-multipart
python mission_control.py
```

Open **http://localhost:8600** — must run on your Windows machine where MT5 is installed.

## How it works

- **Deploy a bot** — drag any `.py` file onto the fleet grid (or drop your whole fleet at once). Files land in the `bots/` folder next to this script, so you can also just copy them there.
- **Add an account** — right panel: label, login, password, server. Stored in `accounts.json`.
- **Play** — pick an account from the dropdown on a bot card (or "No account" to use the bot's hardcoded creds) and hit ▶. Each bot runs as its own process; click a card to watch its live console below.
- **Stop all bots** — the red button top-right kills the whole fleet.

## Connecting your existing bots to accounts

Mission Control passes the selected account via environment variables. Add these 4 lines to each bot (gold_session_pro, pod_london, etc.) where you call `mt5.initialize()` / `mt5.login()`:

```python
import os
LOGIN    = int(os.getenv("MT5_LOGIN", YOUR_CURRENT_LOGIN))
PASSWORD = os.getenv("MT5_PASSWORD", "your_current_password")
SERVER   = os.getenv("MT5_SERVER", "your_current_server")
```

Bots keep working standalone (falls back to your current values) but become account-switchable from the dashboard.

## Notes

- `accounts.json` stores passwords in plain text on your machine — fine for demo accounts, but don't commit it to git and think twice before putting live credentials in it.
- The dashboard is bound to `127.0.0.1` only, so nothing is exposed to your network.
- Crashed bots show a red rail + CRASHED status with the exit trace in the console.
