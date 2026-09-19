"""The canonical layer: one shape every source maps into, so the engine stays source-agnostic.

The `SafetyNet` database holds two layers. Raw schemas (`mimic_hosp`, `mimic_icu`,
and whatever a future source brings) keep each source verbatim and are never
required to agree with one another. Schema `core` is the contract: an adapter's
only job is to produce `PatientCase` objects, and `write_cases` lands them in the
same four tables regardless of where they came from. `read_cases` hands back the
same objects, so `src/engine/*` never learns whether a cohort arrived from a CSV,
from MIMIC in SQL, or from a generator.

Four tables, because the loop this project describes needs all four:

  core.cases   one row per episode of care, keyed by (source, source_case_id)
  core.events  the timeline the watcher reads
  core.scores  what a model said, with per-feature contributions kept alongside
               so an explanation shown to a reviewer stays auditable afterwards
  core.reviews reviewer adjudications, which are the training labels next time

`label` is nullable on purpose. Harm is usually unknown rather than false, and a
schema that cannot say so would quietly turn "never reviewed" into "no harm".

Windows authentication only; nothing here reads or writes a credential.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyodbc as pyodbc
    from sqlalchemy import create_engine, event, text
    from sqlalchemy.engine import Engine
    _SQL_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    pyodbc = None  # type: ignore[assignment]
    create_engine = None  # type: ignore[assignment]
    event = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    Engine = None  # type: ignore[assignment,misc]
    _SQL_DEPS_AVAILABLE = False

from src.domain.models import PatientCase, PatientEvent

DEFAULT_SERVER = r".\SQLEXPRESS"
DEFAULT_DATABASE = "SafetyNet"
DRIVER = "ODBC Driver 17 for SQL Server"
CORE_SCHEMA = "core"

DECISIONS = ("harm", "no_harm", "unclear")

# Batch size for executemany. Large enough that round trips stop dominating,
# small enough that a batch's bound buffer stays reasonable.
BATCH_SIZE = 5000


@dataclass
class FeatureContribution:
    feature: str
    value: float
    weight: float
    contribution: float


@dataclass
class CaseScore:
    source_case_id: str
    score: float
    contributions: List[FeatureContribution] = field(default_factory=list)


# --------------------------------------------------------------------- plumbing


def connection_url(server: str = DEFAULT_SERVER, database: str = DEFAULT_DATABASE) -> str:
    return (
        f"mssql+pyodbc://@{server}/{database}"
        f"?driver={DRIVER.replace(' ', '+')}&trusted_connection=yes&TrustServerCertificate=yes"
    )


def odbc_connection_string(server: str = DEFAULT_SERVER, database: str = "master") -> str:
    return (
        f"DRIVER={{{DRIVER}}};SERVER={server};DATABASE={database};"
        "Trusted_Connection=yes;TrustServerCertificate=yes"
    )


def build_engine(
    server: str = DEFAULT_SERVER, database: str = DEFAULT_DATABASE, fast: bool = True
) -> Engine:
    engine = create_engine(connection_url(server, database), future=True)
    if fast:

        @event.listens_for(engine, "before_cursor_execute")
        def enable_fast_executemany(conn, cursor, statement, parameters, context, executemany):
            if executemany:
                cursor.fast_executemany = True

    return engine


def ensure_database(server: str = DEFAULT_SERVER, database: str = DEFAULT_DATABASE) -> bool:
    """CREATE DATABASE cannot run inside a transaction, so it goes through raw pyodbc."""
    with pyodbc.connect(odbc_connection_string(server), autocommit=True, timeout=10) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sys.databases WHERE name = ?", database
        ).fetchone()
        if exists:
            return False
        connection.execute(f"CREATE DATABASE [{database}]")
    return True


def ensure_schema(engine: Engine, schema: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = :s) "
                f"EXEC('CREATE SCHEMA [{schema}]')"
            ),
            {"s": schema},
        )


def available(server: str = DEFAULT_SERVER, database: str = DEFAULT_DATABASE) -> bool:
    """True when the canonical layer can be read right now, like mimic.mimic_available()."""
    if not _SQL_DEPS_AVAILABLE:
        return False
    try:
        with pyodbc.connect(
            odbc_connection_string(server, database), timeout=5
        ) as connection:
            return (
                connection.execute(
                    "SELECT OBJECT_ID('core.cases', 'U')"
                ).fetchone()[0]
                is not None
            )
    except pyodbc.Error:
        return False


# -------------------------------------------------------------------------- DDL

# Per-feature contributions live in a child table rather than a JSON column so a
# question like "which trigger drove last week's queue" stays a plain join.
CORE_DDL: Tuple[str, ...] = (
    """
    IF OBJECT_ID('core.cases', 'U') IS NULL
    CREATE TABLE core.cases (
        case_key        BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_cases PRIMARY KEY,
        source          NVARCHAR(64)  NOT NULL,
        source_case_id  NVARCHAR(128) NOT NULL,
        source_ordinal  INT           NOT NULL,
        age             INT           NULL,
        gender          NVARCHAR(16)  NULL,
        label           BIT           NULL,
        label_source    NVARCHAR(64)  NULL,
        scenario        NVARCHAR(128) NULL,
        loaded_at       DATETIME2(3)  NOT NULL CONSTRAINT df_cases_loaded_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT uq_cases_source_case UNIQUE (source, source_case_id)
    )
    """,
    """
    IF OBJECT_ID('core.events', 'U') IS NULL
    CREATE TABLE core.events (
        event_key       BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_events PRIMARY KEY,
        case_key        BIGINT        NOT NULL
            CONSTRAINT fk_events_case REFERENCES core.cases (case_key) ON DELETE CASCADE,
        case_ordinal    INT           NOT NULL,
        source_event_id NVARCHAR(128) NULL,
        event_type      NVARCHAR(32)  NOT NULL,
        event_time      DATETIME2(6)  NOT NULL,
        value           NVARCHAR(512) NULL,
        details         NVARCHAR(MAX) NULL,
        CONSTRAINT uq_events_case_ordinal UNIQUE (case_key, case_ordinal)
    )
    """,
    """
    IF OBJECT_ID('core.scores', 'U') IS NULL
    CREATE TABLE core.scores (
        score_key       BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_scores PRIMARY KEY,
        case_key        BIGINT        NOT NULL
            CONSTRAINT fk_scores_case REFERENCES core.cases (case_key) ON DELETE CASCADE,
        model_type      NVARCHAR(32)  NOT NULL,
        model_version   NVARCHAR(64)  NULL,
        run_id          NVARCHAR(64)  NOT NULL,
        score           FLOAT         NOT NULL,
        scored_at       DATETIME2(3)  NOT NULL CONSTRAINT df_scores_scored_at DEFAULT SYSUTCDATETIME()
    )
    """,
    """
    IF OBJECT_ID('core.score_contributions', 'U') IS NULL
    CREATE TABLE core.score_contributions (
        contribution_key BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_score_contributions PRIMARY KEY,
        score_key        BIGINT       NOT NULL
            CONSTRAINT fk_contributions_score REFERENCES core.scores (score_key) ON DELETE CASCADE,
        feature          NVARCHAR(128) NOT NULL,
        value            FLOAT         NOT NULL,
        weight           FLOAT         NOT NULL,
        contribution     FLOAT         NOT NULL
    )
    """,
    """
    IF OBJECT_ID('core.reviews', 'U') IS NULL
    CREATE TABLE core.reviews (
        review_key      BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT pk_reviews PRIMARY KEY,
        case_key        BIGINT        NOT NULL
            CONSTRAINT fk_reviews_case REFERENCES core.cases (case_key) ON DELETE CASCADE,
        reviewer        NVARCHAR(128) NOT NULL,
        decision        NVARCHAR(16)  NOT NULL
            CONSTRAINT ck_reviews_decision CHECK (decision IN ('harm', 'no_harm', 'unclear')),
        notes           NVARCHAR(MAX) NULL,
        reviewed_at     DATETIME2(3)  NOT NULL CONSTRAINT df_reviews_reviewed_at DEFAULT SYSUTCDATETIME()
    )
    """,
)

CORE_INDEXES: Tuple[Tuple[str, str, str], ...] = (
    ("ix_cases_source", "core.cases", "(source)"),
    ("ix_events_case_time", "core.events", "(case_key, event_time)"),
    ("ix_events_type", "core.events", "(event_type)"),
    ("ix_scores_run", "core.scores", "(run_id)"),
    ("ix_scores_case", "core.scores", "(case_key)"),
    ("ix_reviews_case", "core.reviews", "(case_key)"),
)


def ensure_core_schema(engine: Engine) -> None:
    """Idempotent: safe to re-run against a populated database."""
    ensure_schema(engine, CORE_SCHEMA)
    with engine.begin() as connection:
        for statement in CORE_DDL:
            connection.execute(text(statement))
        for name, table, columns in CORE_INDEXES:
            connection.execute(
                text(
                    f"IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{name}' "
                    f"AND object_id = OBJECT_ID('{table}')) "
                    f"CREATE INDEX [{name}] ON {table} {columns}"
                )
            )


# -------------------------------------------------------------------------- ETL

# A pooled connection can outlive the temp table it created, so the staging table
# is dropped defensively rather than assumed absent.
_STAGE_DDL = """
IF OBJECT_ID('tempdb..#case_stage') IS NOT NULL DROP TABLE #case_stage;
CREATE TABLE #case_stage (
    source_case_id NVARCHAR(128) NOT NULL PRIMARY KEY,
    source_ordinal INT           NOT NULL,
    age            INT           NULL,
    gender         NVARCHAR(16)  NULL,
    label          BIT           NULL,
    label_source   NVARCHAR(64)  NULL,
    scenario       NVARCHAR(128) NULL
)
"""

_MERGE_CASES = """
MERGE core.cases WITH (HOLDLOCK) AS target
USING (SELECT ? AS source, * FROM #case_stage) AS incoming
    ON target.source = incoming.source AND target.source_case_id = incoming.source_case_id
WHEN MATCHED THEN UPDATE SET
    source_ordinal = incoming.source_ordinal,
    age            = incoming.age,
    gender         = incoming.gender,
    label          = incoming.label,
    label_source   = incoming.label_source,
    scenario       = incoming.scenario,
    loaded_at      = SYSUTCDATETIME()
WHEN NOT MATCHED THEN INSERT
    (source, source_case_id, source_ordinal, age, gender, label, label_source, scenario)
    VALUES (incoming.source, incoming.source_case_id, incoming.source_ordinal,
            incoming.age, incoming.gender, incoming.label, incoming.label_source,
            incoming.scenario);
"""

_DELETE_STALE_EVENTS = """
DELETE e
FROM core.events AS e
JOIN core.cases AS c ON c.case_key = e.case_key
JOIN #case_stage AS s ON s.source_case_id = c.source_case_id
WHERE c.source = ?
"""

_SELECT_KEYS = """
SELECT c.source_case_id, c.case_key
FROM core.cases AS c
JOIN #case_stage AS s ON s.source_case_id = c.source_case_id
WHERE c.source = ?
"""

_INSERT_EVENTS = """
INSERT INTO core.events
    (case_key, case_ordinal, source_event_id, event_type, event_time, value, details)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

# DATETIME2(6) is bound explicitly because the default pyodbc datetime binding is
# DATETIME-shaped, which rounds sub-second values and would make the SQL timeline
# disagree with the CSV one. NVARCHAR(MAX) has to be declared as an unsized
# WVARCHAR or fast_executemany refuses to bind it.
_EVENT_INPUT_SIZES = (
    [
        None,
        None,
        None,
        None,
        (pyodbc.SQL_TYPE_TIMESTAMP, 26, 6),
        None,
        (pyodbc.SQL_WVARCHAR, 0, 0),
    ]
    if _SQL_DEPS_AVAILABLE
    else []
)


def _executemany(cursor, statement: str, rows: Sequence[tuple], batch_size: int) -> None:
    for start in range(0, len(rows), batch_size):
        cursor.executemany(statement, rows[start : start + batch_size])


def _case_rows(cases: Sequence[PatientCase]) -> List[tuple]:
    return [
        (
            case.patient_id,
            ordinal,
            int(case.age),
            case.gender,
            1 if case.is_harm_event else 0,
            case.label_source,
            case.scenario,
        )
        for ordinal, case in enumerate(cases)
    ]


def _event_rows(cases: Sequence[PatientCase], keys: Dict[str, int]) -> List[tuple]:
    rows: List[tuple] = []
    for case in cases:
        case_key = keys[case.patient_id]
        for ordinal, patient_event in enumerate(case.events):
            rows.append(
                (
                    case_key,
                    ordinal,
                    patient_event.event_id,
                    patient_event.event_type,
                    datetime.fromisoformat(patient_event.timestamp),
                    patient_event.value,
                    patient_event.details,
                )
            )
    return rows


def write_cases(
    cases: List[PatientCase],
    source: str,
    engine: Optional[Engine] = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Upsert a cohort into core.cases/core.events. Returns the number of cases written.

    A case's events are replaced wholesale rather than merged: a timeline is only
    meaningful as a whole, and a partial re-read of a source should not leave
    events behind from the previous one.
    """
    if not cases:
        return 0

    owns_engine = engine is None
    engine = engine or build_engine()
    ensure_core_schema(engine)

    connection = engine.raw_connection()
    try:
        cursor = connection.cursor()
        cursor.fast_executemany = True
        cursor.execute(_STAGE_DDL)
        _executemany(
            cursor,
            "INSERT INTO #case_stage VALUES (?, ?, ?, ?, ?, ?, ?)",
            _case_rows(cases),
            batch_size,
        )
        cursor.execute(_MERGE_CASES, source)
        cursor.execute(_DELETE_STALE_EVENTS, source)
        cursor.execute(_SELECT_KEYS, source)
        keys = {row[0]: int(row[1]) for row in cursor.fetchall()}

        rows = _event_rows(cases, keys)
        if rows:
            cursor.setinputsizes(_EVENT_INPUT_SIZES)
            _executemany(cursor, _INSERT_EVENTS, rows, batch_size)
        cursor.execute("DROP TABLE #case_stage")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        if owns_engine:
            engine.dispose()
    return len(cases)


def read_cases(
    source: Optional[str] = None, engine: Optional[Engine] = None
) -> List[PatientCase]:
    """Rebuild PatientCase objects from the canonical layer, in the order they were written."""
    owns_engine = engine is None
    engine = engine or build_engine()
    filter_clause = "WHERE c.source = :source" if source is not None else ""
    parameters = {"source": source} if source is not None else {}

    try:
        with engine.connect() as connection:
            case_rows = connection.execute(
                text(
                    "SELECT c.case_key, c.source_case_id, c.age, c.gender, c.label, "
                    f"c.label_source, c.scenario FROM core.cases AS c {filter_clause} "
                    "ORDER BY c.source, c.source_ordinal, c.case_key"
                ),
                parameters,
            ).fetchall()

            event_rows = connection.execute(
                text(
                    "SELECT e.case_key, e.source_event_id, e.event_type, e.event_time, "
                    "e.value, e.details FROM core.events AS e "
                    f"JOIN core.cases AS c ON c.case_key = e.case_key {filter_clause} "
                    "ORDER BY e.case_key, e.case_ordinal"
                ),
                parameters,
            ).fetchall()
    finally:
        if owns_engine:
            engine.dispose()

    events: Dict[int, List[PatientEvent]] = {}
    for case_key, event_id, event_type, event_time, value, details in event_rows:
        events.setdefault(int(case_key), []).append(
            PatientEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=event_time.isoformat(),
                value=value,
                details=details,
            )
        )

    cases: List[PatientCase] = []
    for case_key, source_case_id, age, gender, label, label_source, scenario in case_rows:
        cases.append(
            PatientCase(
                patient_id=source_case_id,
                age=int(age),
                gender=gender,
                events=events.get(int(case_key), []),
                # PatientCase cannot express "unknown", so an unadjudicated case
                # reads back as not-harm. The distinction survives in core.cases.
                is_harm_event=bool(label),
                scenario=scenario,
                label_source=label_source,
            )
        )
    return cases


# ------------------------------------------------------------------ scores, reviews


def write_scores(
    scores: Sequence[CaseScore],
    source: str,
    model_type: str,
    model_version: Optional[str] = None,
    run_id: Optional[str] = None,
    engine: Optional[Engine] = None,
    batch_size: int = BATCH_SIZE,
) -> str:
    """Record one scoring run with its per-feature explanation. Returns the run id."""
    run_id = run_id or uuid.uuid4().hex
    if not scores:
        return run_id

    owns_engine = engine is None
    engine = engine or build_engine()
    ensure_core_schema(engine)

    connection = engine.raw_connection()
    try:
        cursor = connection.cursor()
        cursor.fast_executemany = True
        cursor.execute(
            "SELECT source_case_id, case_key FROM core.cases WHERE source = ?", source
        )
        keys = {row[0]: int(row[1]) for row in cursor.fetchall()}

        missing = [s.source_case_id for s in scores if s.source_case_id not in keys]
        if missing:
            raise KeyError(f"{missing[0]!r} is not a case of source {source!r}")

        _executemany(
            cursor,
            "INSERT INTO core.scores (case_key, model_type, model_version, run_id, score) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (keys[s.source_case_id], model_type, model_version, run_id, float(s.score))
                for s in scores
            ],
            batch_size,
        )

        # One score row per case per run, so the run id is enough to map the
        # generated keys back without a round trip per row.
        cursor.execute("SELECT case_key, score_key FROM core.scores WHERE run_id = ?", run_id)
        score_keys = {int(row[0]): int(row[1]) for row in cursor.fetchall()}

        contribution_rows = [
            (
                score_keys[keys[s.source_case_id]],
                c.feature,
                float(c.value),
                float(c.weight),
                float(c.contribution),
            )
            for s in scores
            for c in s.contributions
        ]
        if contribution_rows:
            _executemany(
                cursor,
                "INSERT INTO core.score_contributions "
                "(score_key, feature, value, weight, contribution) VALUES (?, ?, ?, ?, ?)",
                contribution_rows,
                batch_size,
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        if owns_engine:
            engine.dispose()
    return run_id


def record_review(
    source: str,
    source_case_id: str,
    reviewer: str,
    decision: str,
    notes: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> int:
    """Log one adjudication. These decisions are the labels the next model trains on."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")

    owns_engine = engine is None
    engine = engine or build_engine()
    ensure_core_schema(engine)
    try:
        with engine.begin() as connection:
            case_key = connection.execute(
                text(
                    "SELECT case_key FROM core.cases WHERE source = :source "
                    "AND source_case_id = :case_id"
                ),
                {"source": source, "case_id": source_case_id},
            ).scalar_one_or_none()
            if case_key is None:
                raise KeyError(f"{source_case_id!r} is not a case of source {source!r}")
            return int(
                connection.execute(
                    text(
                        "INSERT INTO core.reviews (case_key, reviewer, decision, notes) "
                        "OUTPUT INSERTED.review_key VALUES (:k, :r, :d, :n)"
                    ),
                    {"k": case_key, "r": reviewer, "d": decision, "n": notes},
                ).scalar_one()
            )
    finally:
        if owns_engine:
            engine.dispose()


# ------------------------------------------------------------------------ counts


def source_counts(engine: Optional[Engine] = None) -> Dict[str, int]:
    """Case count per source, for callers deciding what they can offer."""
    owns_engine = engine is None
    engine = engine or build_engine()
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT source, COUNT(*) FROM core.cases GROUP BY source ORDER BY source")
            ).fetchall()
    except Exception:
        return {}
    finally:
        if owns_engine:
            engine.dispose()
    return {row[0]: int(row[1]) for row in rows}


def table_counts(tables: Iterable[str], engine: Optional[Engine] = None) -> Dict[str, int]:
    owns_engine = engine is None
    engine = engine or build_engine()
    counts: Dict[str, int] = {}
    try:
        with engine.connect() as connection:
            for table in tables:
                counts[table] = int(
                    connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
                )
    finally:
        if owns_engine:
            engine.dispose()
    return counts
