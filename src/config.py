"""Machine-local settings, read from a gitignored `.env` at the repository root.

Everyone on this project runs a differently named SQL Server. One machine has a
default instance reached as `XUSHOE`, another has a named instance reached as
`.\\SQLEXPRESS`, and the database itself has been called both `SafetyNet` and
`SafetyNetQA`. Hard-coding any of those means somebody edits a tracked file to
run the app and eventually commits their own machine's name.

So the values live in `.env`, which git ignores, and `.env.example` documents
them. Copy one to the other and fill it in; nothing tracked ever changes.

Real environment variables win over the file, so CI and one-off overrides like
`$env:SAFETYNET_SQL_SERVER="OTHER"` still work without editing anything.

The parser here is deliberately hand-rolled rather than pulling in
python-dotenv: it is a dozen lines, and a teammate who has not re-run
`pip install -r requirements.txt` should still get a working app.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Apply KEY=VALUE lines from `path`, without clobbering the real environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Unwrap in that order: outer padding, then quotes, then any padding the
        # quotes were protecting. `KEY="  x  "` is meant to yield `x`.
        value = value.strip().strip("\"'").strip()
        # An exported shell variable is the more specific instruction; the file
        # is the fallback, so it never overwrites one.
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

# Default instances answer to the bare machine name; named instances need
# .\NAME. Getting this wrong surfaces as ODBC error 08001.
SQL_SERVER = os.environ.get("SAFETYNET_SQL_SERVER", r".\SQLEXPRESS")

# The canonical `core` layer, and the Synthea-derived warehouse beside it.
SQL_DATABASE = os.environ.get("SAFETYNET_SQL_DATABASE", "SafetyNetQA")
HOPS_DATABASE = os.environ.get("SAFETYHOPS_SQL_DATABASE", "SafetyHops")

DRIVER = os.environ.get("SAFETYNET_SQL_DRIVER", "ODBC Driver 17 for SQL Server")
