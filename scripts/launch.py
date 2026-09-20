"""Start the FastAPI backend and the Vite review UI together."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
API_PORT = "8000"
UI_PORT = "5173"


def _npm() -> str:
    found = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not found:
        raise SystemExit("npm is not on PATH. Install Node.js, then retry.")
    return found


def main() -> None:
    os.chdir(ROOT)
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", API_PORT],
        cwd=ROOT,
    )
    frontend_cmd = [_npm(), "run", "dev"]
    if not (FRONTEND / "node_modules").exists():
        print("Installing frontend packages (first run)...")
        subprocess.check_call([_npm(), "install"], cwd=FRONTEND)
    frontend = subprocess.Popen(frontend_cmd, cwd=FRONTEND)

    print()
    print(f"API  http://127.0.0.1:{API_PORT}")
    print(f"UI   http://127.0.0.1:{UI_PORT}")
    print("Ctrl+C stops both.")
    print()

    def _stop(_signum=None, _frame=None) -> None:
        for proc in (frontend, backend):
            if proc.poll() is None:
                proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    try:
        while True:
            if backend.poll() is not None:
                print("API exited; stopping UI.")
                _stop()
            if frontend.poll() is not None:
                print("UI exited; stopping API.")
                _stop()
            time.sleep(0.5)
    except KeyboardInterrupt:
        _stop()


if __name__ == "__main__":
    main()
