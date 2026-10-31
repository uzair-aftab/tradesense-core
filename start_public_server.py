"""
TradeSense v7 — Public Server & Tunnel Sync Launcher
=====================================================
Starts TradeSense (FastAPI on port 8010) AND a free public HTTPS tunnel
(Cloudflare Quick Tunnel), and automatically updates the permanent friendly
gateway URL:
  https://uzair-aftab.github.io/trade-sense/
so you and your friends can access it with one permanent link anytime.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BIN_DIR = BASE_DIR / "bin"
CLOUDFLARED_EXE = BIN_DIR / "cloudflared.exe"
PUBLIC_URL_FILE = BASE_DIR / "data" / "public_url.txt"
GIST_PAYLOAD_FILE = BASE_DIR / "data" / "gist_payload.json"
CONFIG_FILE = BASE_DIR / "data" / "friendly_urls.json"

PERMANENT_FRIENDLY_URL = "https://uzair-aftab.github.io/tradesense-gateway/"
PORT = 8010

CLOUDFLARED_DOWNLOAD_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
)
TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")


def get_gist_id() -> str:
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if cfg.get("tradesense_gist_id"):
                return cfg["tradesense_gist_id"]
        except Exception:
            pass
    return "950ecdcb5db925cc7d5323908c45ec27"


def ensure_cloudflared() -> Path:
    if CLOUDFLARED_EXE.exists() and CLOUDFLARED_EXE.stat().st_size > 1_000_000:
        return CLOUDFLARED_EXE
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading official Cloudflare Quick Tunnel binary (cloudflared.exe)...", flush=True)
    urllib.request.urlretrieve(CLOUDFLARED_DOWNLOAD_URL, str(CLOUDFLARED_EXE))
    print(f"Downloaded to {CLOUDFLARED_EXE}", flush=True)
    return CLOUDFLARED_EXE


def update_permanent_gateway(gist_id: str, tunnel_url: str) -> bool:
    """Updates the GitHub Gist backing https://uzair-aftab.github.io/trade-sense/ via gh CLI."""
    try:
        inner = json.dumps({
            "url": tunnel_url,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        payload = {
            "description": "TradeSense Live Tunnel Endpoint",
            "files": {
                "tradesense_tunnel.json": {
                    "content": inner,
                }
            },
        }
        GIST_PAYLOAD_FILE.parent.mkdir(parents=True, exist_ok=True)
        GIST_PAYLOAD_FILE.write_text(json.dumps(payload), encoding="utf-8")
        res = subprocess.run(
            ["gh", "api", f"/gists/{gist_id}", "--method", "PATCH", "--input", str(GIST_PAYLOAD_FILE)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return res.returncode == 0
    except Exception as e:
        print(f"[Warning] Failed to update Gist: {e}", flush=True)
        return False


def free_port_8010() -> None:
    """Terminates any stale local process still listening on 127.0.0.1:8010."""
    try:
        res = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in res.stdout.splitlines():
            if f"127.0.0.1:{PORT}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0:
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        time.sleep(0.5)
    except Exception:
        pass


def stream_tunnel_output(proc: subprocess.Popen, gist_id: str) -> None:
    assert proc.stderr is not None
    for line in iter(proc.stderr.readline, ""):
        if not line:
            break
        match = TRYCLOUDFLARE_RE.search(line)
        if match:
            url = match.group(0)
            PUBLIC_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
            PUBLIC_URL_FILE.write_text(url, encoding="utf-8")
            synced = update_permanent_gateway(gist_id, url)
            sync_note = " (Synced to Gateway!)" if synced else " (Sync pending)"
            print("\n" + "=" * 78, flush=True)
            print(f"  PERMANENT GATEWAY URL (Share this!): {PERMANENT_FRIENDLY_URL}{sync_note}", flush=True)
            print(f"  DIRECT TUNNEL URL (This session):   {url}", flush=True)
            print(f"  LOCAL URL (This machine):           http://127.0.0.1:{PORT}", flush=True)
            print(f"  ONE-CLICK ACCOUNTS:                 Uzair, Waleed, or add new (+)", flush=True)
            print("=" * 78 + "\n", flush=True)


def main() -> None:
    gist_id = get_gist_id()
    cf_path = ensure_cloudflared()
    free_port_8010()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)

    print(f"Starting TradeSense FastAPI server on http://127.0.0.1:{PORT} ...", flush=True)
    uvicorn_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.web.server:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(BASE_DIR),
        env=env,
    )

    time.sleep(2.5)

    print("Starting Cloudflare HTTPS Quick Tunnel...", flush=True)
    tunnel_proc = subprocess.Popen(
        [str(cf_path), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    t = threading.Thread(target=stream_tunnel_output, args=(tunnel_proc, gist_id), daemon=True)
    t.start()

    try:
        while uvicorn_proc.poll() is None and tunnel_proc.poll() is None:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down TradeSense server and public tunnel...", flush=True)
    finally:
        for p in (tunnel_proc, uvicorn_proc):
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
