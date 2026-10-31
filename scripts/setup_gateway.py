"""
TradeSense v7 — Setup GitHub Pages Gateway & Dedicated Tunnel Gist
==================================================================
Creates:
  1. A dedicated public GitHub Gist ('TradeSense Live Tunnel Endpoint')
     containing 'tradesense_tunnel.json'.
  2. A public repository 'uzair-aftab/trade-sense' on GitHub.
  3. Deploys the smart Slate-950 Gateway 'index.html' to GitHub Pages:
     https://uzair-aftab.github.io/trade-sense/
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = DATA_DIR / "friendly_urls.json"

GATEWAY_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TradeSense — Quantitative Terminal Gateway</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%23020617'/><path d='M6 22 L13 14 L18 18 L26 9' stroke='%2310b981' stroke-width='2.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/><circle cx='26' cy='9' r='2' fill='%2310b981'/></svg>">
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-200 min-h-screen flex items-center justify-center p-4 antialiased selection:bg-emerald-500/20">
  <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-7 shadow-2xl text-center space-y-4">
    <div class="mx-auto h-12 w-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 flex items-center justify-center shadow-md text-white font-black text-xl">
      TS
    </div>
    <h1 class="text-xl font-bold text-white tracking-tight">TradeSense v7 Terminal</h1>
    <p id="status-text" class="text-sm text-slate-400">Locating active TradeSense engine…</p>
    <div id="spinner" class="flex justify-center py-2">
      <div class="h-6 w-6 rounded-full border-2 border-emerald-500 border-t-transparent animate-spin"></div>
    </div>
    <div id="offline-box" class="hidden space-y-3 pt-2">
      <div class="p-3.5 rounded-lg bg-amber-950/40 border border-amber-800/60 text-xs text-amber-200 leading-relaxed text-left">
        The host machine is currently offline or sleeping. Ask Uzair to run <code class="font-mono text-amber-100 bg-amber-900/60 px-1 py-0.5 rounded">start_public_server.bat</code> on his laptop, then click <strong>Retry Connection</strong> below.
      </div>
      <div class="flex flex-col sm:flex-row gap-2 justify-center">
        <button onclick="connect()" class="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition cursor-pointer">Retry Connection</button>
        <a id="direct-link" href="#" class="px-4 py-2 rounded-lg border border-slate-700 hover:bg-slate-800 text-slate-300 text-xs font-medium transition">Open Tunnel Directly →</a>
      </div>
    </div>
  </div>
  <script>
    const GIST_ID = '__GIST_ID__';
    async function connect() {
      const statusEl = document.getElementById('status-text');
      const spinner = document.getElementById('spinner');
      const offlineBox = document.getElementById('offline-box');
      offlineBox.classList.add('hidden');
      spinner.classList.remove('hidden');
      statusEl.textContent = 'Locating active TradeSense engine…';
      try {
        const res = await fetch('https://api.github.com/gists/' + GIST_ID + '?t=' + Date.now(), { cache: 'no-store' });
        const gist = await res.json();
        const raw = gist.files['tradesense_tunnel.json'].content;
        const info = JSON.parse(raw);
        const targetUrl = (info.url || '').replace(/\\/$/, '');
        if (!targetUrl) throw new Error('No active tunnel URL in Gist');
        document.getElementById('direct-link').href = targetUrl;
        statusEl.textContent = 'Verifying server is online…';
        const img = new Image();
        let settled = false;
        const timer = setTimeout(() => {
          if (!settled) { settled = true; showOffline(); }
        }, 5000);
        img.onload = () => {
          if (!settled) {
            settled = true;
            clearTimeout(timer);
            statusEl.textContent = 'Connecting to TradeSense…';
            window.location.replace(targetUrl);
          }
        };
        img.onerror = () => {
          if (!settled) { settled = true; clearTimeout(timer); showOffline(); }
        };
        img.src = targetUrl + '/favicon.ico?t=' + Date.now();
      } catch (e) {
        showOffline();
      }
    }
    function showOffline() {
      document.getElementById('spinner').classList.add('hidden');
      document.getElementById('status-text').textContent = 'TradeSense server is currently offline';
      document.getElementById('offline-box').classList.remove('hidden');
    }
    connect();
  </script>
</body>
</html>
"""


def run_cmd(args, check=True, capture=True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=capture,
        text=True,
        check=check,
        **kwargs,
    )


def setup_gist() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if cfg.get("tradesense_gist_id"):
                print(f"[1/4] Found existing Gist ID: {cfg['tradesense_gist_id']}")
                return cfg["tradesense_gist_id"]
        except Exception:
            pass

    print("[1/4] Creating dedicated GitHub Gist for TradeSense tunnel...")
    temp_json = DATA_DIR / "tradesense_tunnel.json"
    temp_json.write_text(
        json.dumps({
            "url": "",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2),
        encoding="utf-8",
    )

    res = run_cmd([
        "gh", "gist", "create", str(temp_json),
        "--desc", "TradeSense Live Tunnel Endpoint",
        "--public",
    ])
    gist_url = res.stdout.strip()
    gist_id = gist_url.split("/")[-1]
    print(f"       Created Gist: {gist_url} (ID: {gist_id})")

    config_data = {
        "tradesense_gist_id": gist_id,
        "permanent_url": "https://uzair-aftab.github.io/trade-sense/",
    }
    CONFIG_FILE.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    return gist_id


def setup_github_pages(gist_id: str):
    print("[2/4] Ensuring GitHub Pages repository uzair-aftab/trade-sense exists...")
    check = run_cmd(["gh", "repo", "view", "uzair-aftab/trade-sense"], check=False)
    if check.returncode != 0:
        print("       Creating repository uzair-aftab/trade-sense...")
        run_cmd([
            "gh", "repo", "create", "uzair-aftab/trade-sense",
            "--public",
            "--description", "TradeSense Institutional Quantitative Terminal Gateway",
        ])

    print("[3/4] Generating and deploying smart gateway index.html...")
    html_content = GATEWAY_HTML_TEMPLATE.replace("__GIST_ID__", gist_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        run_cmd(["git", "clone", "https://github.com/uzair-aftab/trade-sense.git", str(tmppath)])
        (tmppath / "index.html").write_text(html_content, encoding="utf-8")

        run_cmd(["git", "config", "user.name", "Uzair Aftab"], cwd=str(tmppath))
        run_cmd(["git", "config", "user.email", "uzair.aftab.ca@gmail.com"], cwd=str(tmppath))
        run_cmd(["git", "add", "index.html"], cwd=str(tmppath))
        
        status_res = run_cmd(["git", "status", "--porcelain"], cwd=str(tmppath))
        if status_res.stdout.strip():
            run_cmd(["git", "commit", "-m", "Deploy TradeSense v7 Slate-950 Gateway"], cwd=str(tmppath))
            run_cmd(["git", "push", "origin", "main"], cwd=str(tmppath))
            print("       Pushed updated index.html to uzair-aftab/trade-sense (main)")
        else:
            print("       index.html already up to date")

    print("[4/4] Verifying GitHub Pages configuration...")
    pages_check = run_cmd(["gh", "api", "/repos/uzair-aftab/trade-sense/pages"], check=False)
    if pages_check.returncode != 0:
        print("       Enabling GitHub Pages on branch 'main'...")
        run_cmd([
            "gh", "api", "--method", "POST", "/repos/uzair-aftab/trade-sense/pages",
            "-f", "source[branch]=main",
            "-f", "source[path]=/",
        ], check=False)

    print("\n" + "=" * 76)
    print("  TRADESENSE GATEWAY READY!")
    print(f"  PERMANENT URL: https://uzair-aftab.github.io/trade-sense/")
    print(f"  BACKING GIST:  https://gist.github.com/uzair-aftab/{gist_id}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    gid = setup_gist()
    setup_github_pages(gid)
