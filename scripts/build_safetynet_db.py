"""Build the whole `SafetyNet` database end to end and prove the SQL path is honest.

Three phases, any of which can be skipped on a re-run:

  raw        the 31 MIMIC tables verbatim into mimic_hosp / mimic_icu
  canonical  MIMIC and a synthetic cohort mapped into core.cases / core.events
  scores     one logistic scoring run per cohort, with its per-feature explanation

Then the verification that matters: the canonical MIMIC cohort read back out of SQL
has to produce the same cases, the same watcher features, and the same model scores
as reading the CSVs directly. A canonical layer that quietly rounds a timestamp or
turns a NULL into a zero would still look fine in a row count.

Usage:
    python -m scripts.build_safetynet_db
    python -m scripts.build_safetynet_db --skip-raw
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine

from scripts import load_mimic_sql
from src.data.generator import generate_dataset
from src.data.mimic import MIMIC_ROOT_DEFAULT, load_mimic_cases, mimic_available
from src.data.sql_store import (
    CaseScore,
    FeatureContribution,
    build_engine,
    ensure_core_schema,
    ensure_database,
    read_cases,
    source_counts,
    table_counts,
    write_cases,
    write_scores,
)
from src.domain.models import PatientCase
from src.engine.model import HarmScoringModel
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.watcher import StructuredDataWatcher

MIMIC_SOURCE = "mimic_iv_demo"
SYNTHETIC_SOURCE = "synthetic"
RAW_SCHEMA_PREFIX = "mimic_"

CORE_TABLES = (
    "core.cases",
    "core.events",
    "core.scores",
    "core.score_contributions",
    "core.reviews",
)


def mimic_extractors() -> List[StructuredDataWatcher]:
    """A whole admission is the unit and MIMIC carries no notes, so: watcher only."""
    return [StructuredDataWatcher(anchor="admission")]


def synthetic_extractors():
    return [StructuredDataWatcher(), ClinicalNoteReader()]


def _scored_cases(cases: Sequence[PatientCase], model: HarmScoringModel) -> List[CaseScore]:
    return [
        CaseScore(
            source_case_id=case.patient_id,
            score=model.predict_score(case),
            contributions=[
                FeatureContribution(c.feature, c.value, c.weight, c.contribution)
                for c in model.get_evidence(case)
            ],
        )
        for case in cases
    ]


def compare_to_csv(
    sql_cases: Sequence[PatientCase], csv_cases: Sequence[PatientCase]
) -> Dict[str, object]:
    """Equivalence report between a cohort read from SQL and the same cohort from CSV.

    Compares the objects, then the features the watcher derives from them, then the
    scores a model puts on them. Each layer can hide a difference the one above it
    would not show.
    """
    watcher = StructuredDataWatcher(anchor="admission")
    csv_by_id = {case.patient_id: case for case in csv_cases}
    sql_by_id = {case.patient_id: case for case in sql_cases}

    report: Dict[str, object] = {
        "csv_cases": len(csv_cases),
        "sql_cases": len(sql_cases),
        "same_ids": set(csv_by_id) == set(sql_by_id),
        "same_order": [c.patient_id for c in csv_cases] == [c.patient_id for c in sql_cases],
        "csv_events": sum(len(c.events) for c in csv_cases),
        "sql_events": sum(len(c.events) for c in sql_cases),
    }

    field_mismatches: List[str] = []
    event_mismatches: List[str] = []
    feature_mismatches: List[str] = []
    for case_id, csv_case in csv_by_id.items():
        sql_case = sql_by_id.get(case_id)
        if sql_case is None:
            field_mismatches.append(f"{case_id}: missing from SQL")
            continue

        for attribute in ("age", "gender", "is_harm_event", "scenario", "label_source"):
            expected, actual = getattr(csv_case, attribute), getattr(sql_case, attribute)
            if expected != actual:
                field_mismatches.append(f"{case_id}.{attribute}: csv {expected!r} vs sql {actual!r}")

        csv_events = [
            (e.event_id, e.event_type, e.timestamp, e.value, e.details) for e in csv_case.events
        ]
        sql_events = [
            (e.event_id, e.event_type, e.timestamp, e.value, e.details) for e in sql_case.events
        ]
        if csv_events != sql_events:
            differing = [
                f"{a} != {b}" for a, b in zip(csv_events, sql_events) if a != b
            ][:2]
            event_mismatches.append(
                f"{case_id}: {len(csv_events)} csv vs {len(sql_events)} sql events"
                + (f"; first differences {differing}" if differing else "")
            )

        csv_features = watcher.extract_features(csv_case)
        sql_features = watcher.extract_features(sql_case)
        if csv_features != sql_features:
            differing = {
                name: (value, sql_features.get(name))
                for name, value in csv_features.items()
                if value != sql_features.get(name)
            }
            feature_mismatches.append(f"{case_id}: {differing}")

    report["field_mismatches"] = field_mismatches
    report["event_mismatches"] = event_mismatches
    report["feature_mismatches"] = feature_mismatches

    # Both cohorts train their own model. Identical features in an identical order
    # make identical estimators, so a score difference here is the tell that
    # something upstream diverged in a way the per-case checks let through.
    csv_model = HarmScoringModel(mimic_extractors(), model_type="logistic")
    csv_model.train(list(csv_cases))
    sql_model = HarmScoringModel(mimic_extractors(), model_type="logistic")
    sql_model.train(list(sql_cases))

    deltas = [
        abs(csv_model.predict_score(csv_case) - sql_model.predict_score(sql_by_id[case_id]))
        for case_id, csv_case in csv_by_id.items()
        if case_id in sql_by_id
    ]
    report["max_score_delta"] = max(deltas) if deltas else 0.0
    report["same_weights"] = csv_model.get_feature_weights() == sql_model.get_feature_weights()
    report["roc_auc_csv"] = csv_model.report.roc_auc
    report["roc_auc_sql"] = sql_model.report.roc_auc
    report["equivalent"] = bool(
        report["same_ids"]
        and not field_mismatches
        and not event_mismatches
        and not feature_mismatches
        and report["same_weights"]
        and report["max_score_delta"] == 0.0
    )
    return report


def raw_schema_counts(engine: Engine) -> Dict[str, int]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT s.name, COUNT(DISTINCT t.object_id), SUM(p.rows) "
                "FROM sys.tables AS t "
                "JOIN sys.schemas AS s ON s.schema_id = t.schema_id "
                "JOIN sys.partitions AS p ON p.object_id = t.object_id AND p.index_id IN (0, 1) "
                "WHERE s.name LIKE :prefix + '%' "
                "GROUP BY s.name ORDER BY s.name"
            ),
            {"prefix": RAW_SCHEMA_PREFIX},
        ).fetchall()
    return {f"{row[0]} ({row[1]} tables)": int(row[2] or 0) for row in rows}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=load_mimic_sql.DEFAULT_SERVER)
    parser.add_argument("--database", default="SafetyNet")
    parser.add_argument("--root", type=Path, default=Path(MIMIC_ROOT_DEFAULT))
    parser.add_argument("--skip-raw", action="store_true", help="Leave the raw layer as it is.")
    parser.add_argument("--skip-scores", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--num-cases", type=int, default=600)
    parser.add_argument("--harm-ratio", type=float, default=0.15)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunksize", type=int, default=5000)
    args = parser.parse_args(argv)

    if not mimic_available(str(args.root)):
        print(f"error: MIMIC-IV demo not found under {args.root}")
        return 1

    timings: Dict[str, float] = {}
    created = ensure_database(args.server, args.database)
    print(f"{'created' if created else 'found'} database [{args.database}]")
    engine = build_engine(args.server, args.database)
    ensure_core_schema(engine)
    print("canonical layer present: " + ", ".join(CORE_TABLES))

    if not args.skip_raw:
        print(f"\n--- raw layer -> {RAW_SCHEMA_PREFIX}hosp / {RAW_SCHEMA_PREFIX}icu ---")
        started = time.perf_counter()
        status = load_mimic_sql.main(
            [
                "--server",
                args.server,
                "--database",
                args.database,
                "--root",
                str(args.root),
                "--chunksize",
                str(args.chunksize),
                "--schema-prefix",
                RAW_SCHEMA_PREFIX,
            ]
        )
        timings["raw layer"] = time.perf_counter() - started
        if status:
            print("error: the raw load reported problems; stopping before the canonical layer")
            return status

    print("\n--- canonical layer ---")
    started = time.perf_counter()
    mimic_cases = load_mimic_cases(str(args.root))
    timings["read MIMIC csv"] = time.perf_counter() - started

    started = time.perf_counter()
    written = write_cases(mimic_cases, MIMIC_SOURCE, engine=engine)
    timings["write mimic_iv_demo"] = time.perf_counter() - started
    print(
        f"  {MIMIC_SOURCE:<16} {written:>5} cases, "
        f"{sum(len(c.events) for c in mimic_cases):>7,} events  "
        f"{timings['write mimic_iv_demo']:5.1f}s"
    )

    started = time.perf_counter()
    synthetic_cases = generate_dataset(
        num_cases=args.num_cases,
        harm_ratio=args.harm_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        seed=args.seed,
    )
    written = write_cases(synthetic_cases, SYNTHETIC_SOURCE, engine=engine)
    timings["write synthetic"] = time.perf_counter() - started
    print(
        f"  {SYNTHETIC_SOURCE:<16} {written:>5} cases, "
        f"{sum(len(c.events) for c in synthetic_cases):>7,} events  "
        f"{timings['write synthetic']:5.1f}s"
    )

    if not args.skip_scores:
        print("\n--- scoring runs ---")
        started = time.perf_counter()
        for source, cases, extractors in (
            (MIMIC_SOURCE, mimic_cases, mimic_extractors()),
            (SYNTHETIC_SOURCE, synthetic_cases, synthetic_extractors()),
        ):
            model = HarmScoringModel(extractors, model_type="logistic")
            report = model.train(cases)
            run_id = write_scores(
                _scored_cases(cases, model),
                source=source,
                model_type="logistic",
                model_version="watcher+nlp v1" if len(extractors) > 1 else "watcher v1",
                engine=engine,
            )
            print(f"  {source:<16} run {run_id[:8]}  {report.summary()}")
        timings["scoring runs"] = time.perf_counter() - started

    if not args.skip_verify:
        print("\n--- CSV vs SQL equivalence for mimic_iv_demo ---")
        started = time.perf_counter()
        sql_cases = read_cases(MIMIC_SOURCE, engine=engine)
        equivalence = compare_to_csv(sql_cases, mimic_cases)
        timings["equivalence check"] = time.perf_counter() - started
        print(f"  cases      csv {equivalence['csv_cases']}  sql {equivalence['sql_cases']}")
        print(f"  events     csv {equivalence['csv_events']:,}  sql {equivalence['sql_events']:,}")
        print(f"  same ids   {equivalence['same_ids']}    same order {equivalence['same_order']}")
        for key in ("field_mismatches", "event_mismatches", "feature_mismatches"):
            problems = equivalence[key]
            print(f"  {key:<18} {len(problems)}")
            for problem in problems[:5]:
                print(f"     {problem}")
        print(
            f"  identical model weights {equivalence['same_weights']}, "
            f"max score delta {equivalence['max_score_delta']:.3e}, "
            f"ROC AUC csv {equivalence['roc_auc_csv']:.4f} sql {equivalence['roc_auc_sql']:.4f}"
        )
        print(f"  EQUIVALENT: {equivalence['equivalent']}")

    print("\n--- row counts ---")
    for name, rows in raw_schema_counts(engine).items():
        print(f"  {name:<28} {rows:>9,} rows")
    for name, rows in table_counts(CORE_TABLES, engine=engine).items():
        print(f"  {name:<28} {rows:>9,} rows")
    print("  cases by source: " + ", ".join(f"{k}={v}" for k, v in source_counts(engine).items()))

    print("\n--- timings ---")
    for name, seconds in timings.items():
        print(f"  {name:<22} {seconds:7.1f}s")
    print(f"  {'total':<22} {sum(timings.values()):7.1f}s")

    if not args.skip_verify and not equivalence["equivalent"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
