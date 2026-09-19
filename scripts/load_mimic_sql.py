"""Load the MIMIC-IV demo into SQL Server as a raw layer.

Creates a database, one schema per MIMIC module (`hosp`, `icu`), and one table per
CSV, with column types inferred from the data rather than everything dumped into
NVARCHAR. Datetime columns are recognized by name and stored as DATETIME2 so that
date arithmetic works in T-SQL.

The module schema can be renamed so several sources can share one database without
colliding: `--schema-prefix mimic_` lands the tables in `mimic_hosp`/`mimic_icu`,
and `--schema-map icu=mimic_icu` renames individual modules outright. Nothing here
touches the canonical layer; raw schemas are never required to agree with it.

Usage:
    python -m scripts.load_mimic_sql
    python -m scripts.load_mimic_sql --zip C:\\path\\to\\mimic-iv-...-demo-2.2.zip
    python -m scripts.load_mimic_sql --database SafetyNet --schema-prefix mimic_

Windows authentication is used by default. Nothing here writes credentials to disk.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.dialects.mssql import BIGINT, BIT, DATETIME2, FLOAT, NVARCHAR
from sqlalchemy.engine import Engine
from sqlalchemy.types import UnicodeText

from src.data.sql_store import DEFAULT_SERVER, build_engine, ensure_database, ensure_schema

DEFAULT_ROOT = Path("data/mimic-iv-clinical-database-demo-2.2")
DEFAULT_DATABASE = "mimic_iv_demo"

# MIMIC names every timestamp column with one of these suffixes, plus `dod`.
DATETIME_SUFFIXES = ("time", "date")
DATETIME_EXACT = {"dod"}

# Columns worth indexing: every join in the adapter goes through these.
INDEX_COLUMNS = ("subject_id", "hadm_id", "stay_id", "itemid")

# NVARCHAR(4000) is the widest fixed-length unicode column SQL Server allows before
# it has to spill to MAX, which cannot participate in an index or fast_executemany.
NVARCHAR_LIMIT = 4000


def is_datetime_column(name: str) -> bool:
    lowered = name.lower()
    return lowered in DATETIME_EXACT or lowered.endswith(DATETIME_SUFFIXES)


def read_csv_typed(path: Path) -> pd.DataFrame:
    """Read a MIMIC CSV, parsing the timestamp columns."""
    header = pd.read_csv(path, nrows=0)
    datetime_columns = [c for c in header.columns if is_datetime_column(c)]
    frame = pd.read_csv(path, low_memory=False)
    for column in datetime_columns:
        # errors="coerce" keeps a single malformed value from aborting a whole table.
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def sql_type(series: pd.Series):
    """Map a pandas dtype to the narrowest reasonable SQL Server type."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME2
    if pd.api.types.is_bool_dtype(series):
        return BIT
    if pd.api.types.is_integer_dtype(series):
        return BIGINT
    if pd.api.types.is_float_dtype(series):
        return FLOAT
    # Everything else is text. Size it from the data so the column can be indexed.
    lengths = series.dropna().astype(str).str.len()
    widest = int(lengths.max()) if len(lengths) else 1
    if widest > NVARCHAR_LIMIT:
        return UnicodeText  # NVARCHAR(MAX)
    # Headroom, because the demo is a 100-patient sample of a much larger database.
    return NVARCHAR(max(16, min(NVARCHAR_LIMIT, int(widest * 2) + 16)))


def parse_schema_map(entries: Optional[List[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries or []:
        module, _, schema = entry.partition("=")
        if not module or not schema:
            raise ValueError(f"--schema-map entries look like module=schema, got {entry!r}")
        mapping[module] = schema
    return mapping


def target_schema(module: str, prefix: str = "", mapping: Optional[Dict[str, str]] = None) -> str:
    """Where a MIMIC module's tables land. An explicit mapping wins over the prefix."""
    if mapping and module in mapping:
        return mapping[module]
    return f"{prefix}{module}"


def load_table(
    engine: Engine, frame: pd.DataFrame, schema: str, table: str, chunksize: int
) -> None:
    dtypes = {name: sql_type(frame[name]) for name in frame.columns}
    # fast_executemany cannot bind NVARCHAR(MAX); those tables fall back to the
    # slower path rather than failing.
    has_max = any(t is UnicodeText for t in dtypes.values())
    frame.to_sql(
        table,
        engine,
        schema=schema,
        if_exists="replace",
        index=False,
        dtype=dtypes,
        chunksize=chunksize if not has_max else 200,
        method=None,
    )


def add_indexes(engine: Engine, schema: str, table: str, columns: List[str]) -> int:
    created = 0
    with engine.begin() as connection:
        for column in columns:
            name = f"ix_{table}_{column}"
            connection.execute(
                text(
                    f"IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{name}' "
                    f"AND object_id = OBJECT_ID('{schema}.{table}')) "
                    f"CREATE INDEX [{name}] ON [{schema}].[{table}] ([{column}])"
                )
            )
            created += 1
    return created


def discover_tables(root: Path) -> List[Path]:
    return sorted(root.glob("*/*.csv.gz"))


def extract_zip(zip_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
        top = {Path(n).parts[0] for n in archive.namelist()}
    if len(top) == 1:
        return destination / top.pop()
    return destination


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Extract this archive first.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument(
        "--no-fast", action="store_true", help="Disable pyodbc fast_executemany."
    )
    parser.add_argument("--only", nargs="*", help="Load only these tables, e.g. hosp/labevents.")
    parser.add_argument(
        "--schema-prefix",
        default="",
        help="Prefix every module schema, e.g. mimic_ gives mimic_hosp and mimic_icu.",
    )
    parser.add_argument(
        "--schema-map",
        nargs="*",
        metavar="MODULE=SCHEMA",
        help="Rename individual modules outright, e.g. icu=mimic_icu.",
    )
    args = parser.parse_args(argv)

    try:
        schema_map = parse_schema_map(args.schema_map)
    except ValueError as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    root = args.root
    if args.zip:
        print(f"extracting {args.zip}")
        root = extract_zip(args.zip, Path("data"))

    if not root.exists():
        print(f"error: {root} does not exist. Pass --zip or --root.", file=sys.stderr)
        return 1

    paths = discover_tables(root)
    if args.only:
        wanted = {w.replace("\\", "/") for w in args.only}
        paths = [p for p in paths if f"{p.parent.name}/{p.name[:-7]}" in wanted]
    if not paths:
        print(f"error: no .csv.gz tables found under {root}", file=sys.stderr)
        return 1

    print(f"{len(paths)} tables found under {root}")
    created = ensure_database(args.server, args.database)
    print(f"{'created' if created else 'found'} database [{args.database}]")
    engine = build_engine(args.server, args.database, fast=not args.no_fast)

    schemas = {p.parent.name: target_schema(p.parent.name, args.schema_prefix, schema_map) for p in paths}
    for schema in sorted(set(schemas.values())):
        ensure_schema(engine, schema)

    summary: List[Dict[str, object]] = []
    for path in paths:
        schema = schemas[path.parent.name]
        table = path.name[: -len(".csv.gz")]
        started = time.perf_counter()
        frame = read_csv_typed(path)
        load_table(engine, frame, schema, table, args.chunksize)
        indexable = [c for c in INDEX_COLUMNS if c in frame.columns]
        add_indexes(engine, schema, table, indexable)
        elapsed = time.perf_counter() - started
        summary.append(
            {
                "table": f"{schema}.{table}",
                "rows": len(frame),
                "columns": len(frame.columns),
                "indexes": len(indexable),
                "seconds": round(elapsed, 1),
            }
        )
        print(f"  {schema}.{table:<24} {len(frame):>8,} rows  {elapsed:5.1f}s")

    print("\nverifying row counts against the database")
    inspector = inspect(engine)
    mismatches = 0
    with engine.connect() as connection:
        for row in summary:
            schema, table = str(row["table"]).split(".")
            if not inspector.has_table(table, schema=schema):
                print(f"  MISSING {row['table']}")
                mismatches += 1
                continue
            actual = connection.execute(
                text(f"SELECT COUNT(*) FROM [{schema}].[{table}]")
            ).scalar_one()
            if actual != row["rows"]:
                print(f"  MISMATCH {row['table']}: csv {row['rows']} vs sql {actual}")
                mismatches += 1

    total = sum(int(r["rows"]) for r in summary)
    print(f"\n{len(summary)} tables, {total:,} rows loaded into [{args.database}]")
    print("row counts match the source CSVs" if not mismatches else f"{mismatches} problems")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
