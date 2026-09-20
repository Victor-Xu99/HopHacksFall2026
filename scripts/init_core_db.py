"""Create an empty core-only catalog. Does not touch an existing MIMIC SafetyNet database.

python -m scripts.init_core_db
python -m scripts.init_core_db --database SafetyNetQA
"""

from __future__ import annotations

import argparse

from src.data.sql_store import (
    DEFAULT_DATABASE,
    DEFAULT_SERVER,
    build_engine,
    ensure_core_schema,
    ensure_database,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a SQL database with only core.cases, events, scores, and reviews."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    args = parser.parse_args()

    created = ensure_database(server=args.server, database=args.database)
    engine = build_engine(server=args.server, database=args.database)
    try:
        ensure_core_schema(engine)
    finally:
        engine.dispose()

    action = "Created" if created else "Already existed"
    print(
        f"{action} [{args.database}] on {args.server}. "
        "Schema core has cases, events, scores, score_contributions, reviews."
    )


if __name__ == "__main__":
    main()
