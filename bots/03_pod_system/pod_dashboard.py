"""
=============================================================================
 POD DASHBOARD  -  live trading console for the MT5 bots  (Tailwind CSS)
=============================================================================
 Reads the status file the bot writes (pod2_status.json) plus the trade log
 (pod2_trade_log.csv) and serves a live dashboard in your browser. It ONLY
 reads files - it never touches MT5 - so it cannot interfere with trading.

 The page is styled with Tailwind via the Play CDN (loads in the browser, no
 build step). NO PYTHON INSTALL NEEDED (uses only the standard library).

 HOW TO RUN:
   1. Make sure the bot is running (so the status file exists/updates).
   2. In a SECOND terminal:  python pod_dashboard.py
   3. It opens  http://localhost:8000  automatically.
=============================================================================
"""
import os
import csv
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATUS_FILE = "pod2_status.json"
TRADE_LOG   = "pod2_trade_log.csv"
PORT        = 8000

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>POD // CONSOLE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    theme: { extend: {
      colors: {
        bg:'#070b0c', panel:'#0e1416', panel2:'#121a1c', line:'#1f2c2e',
        ink:'#cdd8d6', dim:'#6b7d7b', amber:'#ffb000', grn:'#34e29b',
        red:'#ff5d5d', blue:'#52b9ff'
      },
      fontFamily: { mono:['"IBM Plex Mono"','monospace'], disp:['"Chakra Petch"','monospace'] }
    }}
  }
</script>
<style>
  body{
    background:
      radial-gradient(1200px 600px at 70% -10%, #0c1718 0%, transparent 60%),
      repeating-linear-gradient(0deg, transparent 0 38px, rgba(255,255,255,.012) 38px 39px),
      #070b0c;
  }
  ::-webkit-scrollbar{width:8px}
  ::-webkit-scrollbar-thumb{background:#1f2c2e;border-radius:4px}
</style>
</head>
<body class="font-mono text-ink text-[13px] tracking-[.2px] p-[18px] min-h-screen">

  <!-- safelist: ensures JIT generates the dynamically-injected classes -->
  <div class="hidden text-grn text-red text-amber bg-grn bg-red bg-amber border-amber"></div>

  <header class="flex justify-between items-end border-b border-line pb-3 mb-4">
    <div>
      <h1 class="font-disp font-bold text-2xl tracking-[3px] text-[#eaf3f1]">POD<span class="text-amber">//</span>CONSOLE</h1>
      <div id="acct" class="text-dim text-[11px] tracking-[1.5px] uppercase mt-1">connecting…</div>
    </div>
    <div class="flex items-center gap-2 text-[11px] tracking-[1.5px] uppercase text-dim">
      <span id="updated">—</span>
      <span id="dot" class="w-2 h-2 rounded-full bg-grn animate-pulse" style="box-shadow:0 0 10px #34e29b"></span>
    </div>
  </header>

  <div id="metrics" class="grid grid-cols-2 md:grid-cols-5 gap-3.5"></div>

  <section class="mt-5">
    <h2 class="font-disp text-xs tracking-[2.5px] uppercase text-dim mb-2.5 flex items-center gap-2.5 after:content-[''] after:flex-1 after:h-px after:bg-line">Challenge</h2>
    <div id="bars" class="grid grid-cols-1 md:grid-cols-3 gap-3.5"></div>
  </section>

  <section class="mt-5">
    <h2 class="font-disp text-xs tracking-[2.5px] uppercase text-dim mb-2.5 flex items-center gap-2.5 after:content-[''] after:flex-1 after:h-px after:bg-line">Bot Brain</h2>
    <div id="cards" class="grid grid-cols-1 md:grid-cols-2 gap-3.5"></div>
  </section>

  <section class="mt-5">
    <h2 class="font-disp text-xs tracking-[2.5px] uppercase text-dim mb-2.5 flex items-center gap-2.5 after:content-[''] after:flex-1 after:h-px after:bg-line">Open Positions</h2>
    <div class="bg-panel border border-line rounded-md overflow-hidden">
      <table class="w-full text-[12px] border-collapse">
        <thead><tr class="text-dim text-[10px] tracking-[1.5px] uppercase">
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">Pair</th>
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">Side</th>
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">Vol</th>
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">Entry</th>
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">SL</th>
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">TP</th>
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">Now</th>
          <th class="text-left font-medium py-2 px-2.5 border-b border-line">P/L</th>
        </tr></thead>
        <tbody id="positions"></tbody>
      </table>
    </div>
  </section>

  <section class="mt-5">
    <h2 class="font-disp text-xs tracking-[2.5px] uppercase text-dim mb-2.5 flex items-center gap-2.5 after:content-[''] after:flex-1 after:h-px after:bg-line">Bot Log — live</h2>
    <div id="log" class="bg-[#060a0b] border border-line rounded-lg p-3 h-[300px] overflow-y-auto text-[12px]"></div>
  </section>

  <div id="foot" class="mt-[18px] text-line text-[10px] tracking-[1px] text-center uppercase">awaiting telemetry</div>

<script>
const $=id=>document.getElementById(id);
const money=(v,c)=>(v>0?'+':'')+v.toFixed(2)+(c?(' '+c):'');
const cls=v=>v>0?'text-grn':v<0?'text-red':'';

async function tick(){
  let s;
  try{ s = await (await fetch('/api/status?t='+Date.now())).json(); }
  catch(e){ $('dot').className='w-2 h-2 rounded-full bg-red'; $('foot').textContent='no status file — is the bot running?'; return; }
  if(!s || !s.account){ $('foot').textContent='status file empty — waiting for the bot…'; return; }

  const a=s.account, ch=s.challenge, cur=a.currency;
  $('acct').textContent = `ACCT #${a.login} · ${a.server} · ${cur}`;
  $('updated').textContent = 'UPDATED '+(s.updated||'');
  $('dot').className = 'w-2 h-2 rounded-full animate-pulse ' + (s._stale?'bg-red':'bg-grn');
  $('dot').style.boxShadow = s._stale ? '0 0 10px #ff5d5d' : '0 0 10px #34e29b';

  const cell=(l,v,c)=>`<div class="relative bg-panel border border-line rounded-md p-4 overflow-hidden
      before:content-[''] before:absolute before:left-0 before:top-0 before:bottom-0 before:w-0.5 before:bg-amber before:opacity-50">
      <div class="text-[10px] tracking-[1.5px] uppercase text-dim">${l}</div>
      <div class="font-disp text-[26px] font-bold mt-1.5 ${c||''}">${v}</div></div>`;
  $('metrics').innerHTML = [
    ['Balance', a.balance.toFixed(2), ''],
    ['Equity', a.equity.toFixed(2), ''],
    ['Floating', money(a.floating,''), cls(a.floating)],
    ['Day P/L', money(ch.day_pl,''), cls(ch.day_pl)],
    ['Total P/L', money(ch.total_pl,''), cls(ch.total_pl)],
  ].map(([l,v,c])=>cell(l,v,c)).join('');

  const tgtPct=Math.max(0,Math.min(100, ch.total_pl/ch.target*100));
  const dayPct=Math.max(0,Math.min(100, (-Math.min(0,ch.day_pl))/ch.daily_limit*100));
  const totPct=Math.max(0,Math.min(100, (-Math.min(0,ch.total_pl))/ch.total_limit*100));
  const bar=(lab,pct,color,left,right)=>`<div class="bg-panel border border-line rounded-md p-3 px-4">
     <div class="text-[10px] tracking-[1.5px] uppercase text-dim">${lab}</div>
     <div class="h-[9px] bg-[#0a0f10] border border-line rounded-[5px] mt-2.5 overflow-hidden">
        <div class="h-full rounded-[5px] transition-all duration-500" style="width:${pct}%;background:${color};box-shadow:0 0 10px ${color}"></div></div>
     <div class="flex justify-between text-[11px] text-dim mt-1.5"><span>${left}</span><span>${right}</span></div></div>`;
  $('bars').innerHTML =
     bar('Profit target', tgtPct, '#34e29b', money(ch.total_pl,''), '+'+ch.target+' '+cur)
   + bar('Daily loss used', dayPct, '#ffb000', money(ch.day_pl,''), '-'+ch.daily_limit+' '+cur)
   + bar('Total loss used', totPct, '#ff5d5d', money(ch.total_pl,''), '-'+ch.total_limit+' '+cur);

  $('cards').innerHTML = Object.entries(s.symbols||{}).map(([name,d])=>{
    const dig=name.includes('JPY')?3:5; const f=x=>(x==null?'—':Number(x).toFixed(dig));
    const active=d.in_trade;
    const chip=(on,t)=>`<span class="text-[10px] tracking-[1px] uppercase px-2.5 py-[3px] rounded-full border ${on?'bg-amber text-bg border-amber font-semibold':'border-line text-dim'}">${t}</span>`;
    return `<div class="rounded-lg p-4 border ${active?'border-amber':'border-line'}"
        style="background:linear-gradient(180deg,#121a1c,#0e1416);${active?'box-shadow:0 0 0 1px rgba(255,176,0,.25),0 0 24px rgba(255,176,0,.07)':''}">
      <div class="font-disp text-[17px] font-bold tracking-[2px] text-[#eaf3f1]">${name}</div>
      <div class="text-amber my-2.5 text-[12.5px] min-h-[34px]">${d.note||''}</div>
      <div class="grid grid-cols-2 gap-y-1.5 gap-x-[18px] text-[12px]">
        <span class="text-dim">Range high</span><b class="font-semibold">${f(d.range_high)}</b>
        <span class="text-dim">Range low</span><b class="font-semibold">${f(d.range_low)}</b>
        <span class="text-dim">Trades today</span><b class="font-semibold">${d.trades_today}</b>
        <span class="text-dim">Status</span><b class="font-semibold ${active?'text-amber':''}">${active?'IN TRADE':'flat'}</b>
      </div>
      <div class="flex gap-2 mt-3 flex-wrap">${chip(d.swept_high,'swept high')}${chip(d.swept_low,'swept low')}</div>
    </div>`;
  }).join('') || '<div class="text-dim italic p-3">no symbols</div>';

  const ps=s.positions||[];
  $('positions').innerHTML = ps.length ? ps.map(p=>{
    const dig=p.symbol.includes('JPY')?3:5; const f=x=>Number(x).toFixed(dig);
    const side=p.type==='BUY'?'text-grn':'text-red';
    return `<tr class="[&>td]:py-2.5 [&>td]:px-2.5 [&>td]:border-b [&>td]:border-[#131c1d]">
      <td>${p.symbol}</td><td class="${side}">${p.type}</td><td>${p.volume}</td>
      <td>${f(p.entry)}</td><td>${f(p.sl)}</td><td>${f(p.tp)}</td><td>${f(p.price)}</td>
      <td class="${cls(p.profit)}">${money(p.profit,cur)}</td></tr>`;
  }).join('') : '<tr><td colspan="8" class="text-dim italic py-3.5 px-2.5">no open positions</td></tr>';

  const log=$('log'); const atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<40;
  log.innerHTML=(s.log||[]).map(l=>{
    const k=/SHORT|LONG|FILLED|partial|exit|TARGET|LIMIT|Range set|setup/i.test(l);
    return `<div class="whitespace-pre-wrap py-px before:content-['▍_'] before:text-line ${k?'text-ink':'text-dim'}">${l.replace(/</g,'&lt;')}</div>`;
  }).join('');
  if(atBottom) log.scrollTop=log.scrollHeight;

  const set=s.settings||{};
  $('foot').textContent=`${(set.pairs||[]).join(' · ')}  |  risk ${set.risk_percent}%  |  confirm ${set.confirm_closes}  |  partial ${set.partial}  |  exit ${set.runner_exit}  |  window ${set.window}`;
}
tick(); setInterval(tick,2000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/status":
            data = {}
            try:
                import time as _t
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_stale"] = (_t.time() - os.path.getmtime(STATUS_FILE)) > 12
            except Exception:
                data = {}
            self._send(200, json.dumps(data), "application/json")
        elif path == "/api/trades":
            rows = []
            try:
                with open(TRADE_LOG, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))[-50:][::-1]
            except Exception:
                rows = []
            self._send(200, json.dumps(rows), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, *args):
        pass


def main():
    url = f"http://localhost:{PORT}"
    print(f"Pod dashboard running -> {url}")
    print("Leave this window open. Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
