"""
MISSION CONTROL — MT5 bot fleet manager
Drag & drop bots, connect accounts, press play.

Run:  python mission_control.py
Then open http://localhost:8600
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).parent
BOTS_DIR = ROOT / "bots"
ACCOUNTS_FILE = ROOT / "accounts.json"
BOTS_DIR.mkdir(exist_ok=True)

IS_WINDOWS = os.name == "nt"
MAX_LOG_LINES = 500

app = FastAPI(title="Mission Control")

# ---------------------------------------------------------------- accounts

def load_accounts() -> list[dict]:
    if ACCOUNTS_FILE.exists():
        return json.loads(ACCOUNTS_FILE.read_text())
    return []

def save_accounts(accounts: list[dict]):
    ACCOUNTS_FILE.write_text(json.dumps(accounts, indent=2))

class Account(BaseModel):
    label: str          # e.g. "IC Markets Demo"
    login: int
    password: str
    server: str         # e.g. "ICMarketsSC-Demo"

# ---------------------------------------------------------------- bot runner

class BotProcess:
    """Wraps one running bot subprocess and captures its output."""

    def __init__(self, bot_file: Path, account: dict | None):
        self.bot_file = bot_file
        self.account = account
        self.logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.started_at = time.time()
        self.exit_code: int | None = None

        env = os.environ.copy()
        if account:
            env["MT5_LOGIN"] = str(account["login"])
            env["MT5_PASSWORD"] = account["password"]
            env["MT5_SERVER"] = account["server"]
        env["PYTHONUNBUFFERED"] = "1"

        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
        self.proc = subprocess.Popen(
            [sys.executable, str(bot_file)],
            cwd=str(bot_file.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=flags,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {line.rstrip()}")
        self.proc.wait()
        self.exit_code = self.proc.returncode
        self.logs.append(
            f"[{time.strftime('%H:%M:%S')}] ── process exited (code {self.exit_code}) ──"
        )

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def stop(self):
        if not self.running:
            return
        try:
            if IS_WINDOWS:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
            self.proc.wait(timeout=8)
        except Exception:
            self.proc.kill()

# bot filename -> BotProcess
RUNNING: dict[str, BotProcess] = {}

def bot_status(name: str) -> str:
    bp = RUNNING.get(name)
    if bp is None:
        return "stopped"
    if bp.running:
        return "running"
    return "crashed" if bp.exit_code not in (0, None) else "stopped"

# ---------------------------------------------------------------- API

@app.get("/")
def index():
    return FileResponse(ROOT / "dashboard.html")

@app.get("/api/state")
def get_state():
    bots = []
    for f in sorted(BOTS_DIR.glob("*.py")):
        bp = RUNNING.get(f.name)
        bots.append({
            "name": f.name,
            "status": bot_status(f.name),
            "account": bp.account["label"] if bp and bp.account else None,
            "uptime": int(time.time() - bp.started_at) if bp and bp.running else 0,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })
    accounts = [
        {"label": a["label"], "login": a["login"], "server": a["server"]}
        for a in load_accounts()
    ]
    return {"bots": bots, "accounts": accounts}

@app.post("/api/bots/upload")
async def upload_bot(file: UploadFile = File(...)):
    if not file.filename.endswith(".py"):
        raise HTTPException(400, "Only .py files can be deployed")
    dest = BOTS_DIR / Path(file.filename).name
    dest.write_bytes(await file.read())
    return {"ok": True, "name": dest.name}

@app.delete("/api/bots/{name}")
def delete_bot(name: str):
    if bot_status(name) == "running":
        raise HTTPException(400, "Stop the bot before removing it")
    target = BOTS_DIR / Path(name).name
    if not target.exists():
        raise HTTPException(404, "Bot not found")
    target.unlink()
    RUNNING.pop(name, None)
    return {"ok": True}

class StartRequest(BaseModel):
    account_label: str | None = None

@app.post("/api/bots/{name}/start")
def start_bot(name: str, req: StartRequest):
    if bot_status(name) == "running":
        raise HTTPException(400, "Already running")
    bot_file = BOTS_DIR / Path(name).name
    if not bot_file.exists():
        raise HTTPException(404, "Bot not found")
    account = None
    if req.account_label:
        account = next(
            (a for a in load_accounts() if a["label"] == req.account_label), None
        )
        if account is None:
            raise HTTPException(404, "Account not found")
    RUNNING[name] = BotProcess(bot_file, account)
    return {"ok": True}

@app.post("/api/bots/{name}/stop")
def stop_bot(name: str):
    bp = RUNNING.get(name)
    if bp is None or not bp.running:
        raise HTTPException(400, "Not running")
    bp.stop()
    return {"ok": True}

@app.get("/api/bots/{name}/logs")
def bot_logs(name: str):
    bp = RUNNING.get(name)
    return {"lines": list(bp.logs) if bp else []}

@app.post("/api/accounts")
def add_account(acc: Account):
    accounts = load_accounts()
    if any(a["label"] == acc.label for a in accounts):
        raise HTTPException(400, "An account with that label already exists")
    accounts.append(acc.model_dump())
    save_accounts(accounts)
    return {"ok": True}

@app.delete("/api/accounts/{label}")
def delete_account(label: str):
    accounts = [a for a in load_accounts() if a["label"] != label]
    save_accounts(accounts)
    return {"ok": True}

@app.post("/api/stop-all")
def stop_all():
    for bp in RUNNING.values():
        if bp.running:
            bp.stop()
    return {"ok": True}

# ---------------------------------------------------------------- main

if __name__ == "__main__":
    print("Mission Control → http://localhost:8600")
    uvicorn.run(app, host="127.0.0.1", port=8600, log_level="warning")
