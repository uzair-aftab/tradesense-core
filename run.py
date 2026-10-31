"""
TradeSense v7 — Local Server Runner
===================================
Runs FastAPI on http://127.0.0.1:8010 with Uvicorn.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn


def main():
    port = 8010
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print(f"\n========================================================")
    print(f"  TRADESENSE v7 LOCAL TERMINAL")
    print(f"  URL: {url}")
    print(f"========================================================\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run("src.web.server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
