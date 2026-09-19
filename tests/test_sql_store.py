"""The canonical SQL layer has to be indistinguishable from the CSV path.

Skips cleanly when SQL Server or the MIMIC files are unavailable, the same way
test_app.py does, so the suite still runs on a machine with neither.
"""

from datetime import datetime, timedelta
from typing import List

import pytest
from sqlalchemy import text

from scripts.load_mimic_sql import parse_schema_map, target_schema
from src.data.mimic import load_mimic_cases, mimic_available
from src.data.sql_store import (
    CaseScore,
    FeatureContribution,
    build_engine,
    ensure_core_schema,
    read_cases,
    record_review,
    source_counts,
    write_cases,
    write_scores,
)
from src.data.sql_store import available as sql_available
from src.domain.models import PatientCase, PatientEvent
from scripts.build_safetynet_db import MIMIC_SOURCE, compare_to_csv

NEEDS_MIMIC = pytest.mark.skipif(not mimic_available(), reason="MIMIC-IV demo not extracted")
NEEDS_SQL = pytest.mark.skipif(
    not sql_available(), reason="SafetyNet canonical layer not reachable"
)

# Anything this module writes goes under its own source tag and is deleted again,
# so a run never touches the cohorts the app reads.
TEST_SOURCE = "pytest_roundtrip"


@pytest.fixture(scope="module")
def engine():
    engine = build_engine()
    ensure_core_schema(engine)
    yield engine
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM core.cases WHERE source = :s"), {"s": TEST_SOURCE}
        )
    engine.dispose()


@pytest.fixture(scope="module")
def csv_cases() -> List[PatientCase]:
    return load_mimic_cases()


@pytest.fixture(scope="module")
def equivalence(engine, csv_cases):
    return compare_to_csv(read_cases(MIMIC_SOURCE, engine=engine), csv_cases)


def sub_second_case(case_id: str = "roundtrip-1") -> PatientCase:
    """Microsecond timestamps, an empty string, and a long note in one case."""
    base = datetime(2026, 3, 4, 5, 6, 7, 123456)
    return PatientCase(
        patient_id=case_id,
        age=71,
        gender="F",
        events=[
            PatientEvent(
                event_id="e0",
                event_type="admission",
                timestamp=base.isoformat(),
                value="Inpatient admission",
                details="",
            ),
            PatientEvent(
                event_id="e1",
                event_type="lab",
                timestamp=(base + timedelta(hours=1.5, microseconds=999999)).isoformat(),
                value="Hemoglobin",
                details="6.4",
            ),
            PatientEvent(
                event_id="e2",
                event_type="note",
                timestamp=(base + timedelta(days=1)).isoformat(),
                value="Progress Note",
                details="Unanticipated deterioration overnight. " * 300,
            ),
        ],
        is_harm_event=True,
        scenario="roundtrip",
        label_source="hand_labeled",
    )


class TestSchemaOptions:
    def test_prefix_applies_to_every_module(self) -> None:
        assert target_schema("hosp", prefix="mimic_") == "mimic_hosp"
        assert target_schema("icu", prefix="mimic_") == "mimic_icu"

    def test_explicit_mapping_beats_the_prefix(self) -> None:
        mapping = parse_schema_map(["icu=raw_icu"])
        assert target_schema("icu", prefix="mimic_", mapping=mapping) == "raw_icu"
        assert target_schema("hosp", prefix="mimic_", mapping=mapping) == "mimic_hosp"

    def test_malformed_mapping_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_schema_map(["icu"])


class TestRoundTrip:
    @NEEDS_SQL
    def test_events_survive_verbatim(self, engine) -> None:
        original = sub_second_case()
        write_cases([original], TEST_SOURCE, engine=engine)
        restored = read_cases(TEST_SOURCE, engine=engine)

        assert len(restored) == 1
        assert restored[0] == original

    @NEEDS_SQL
    def test_sub_second_timestamps_are_not_rounded(self, engine) -> None:
        """DATETIME2 defaults would round these, and the watcher compares timestamps."""
        original = sub_second_case()
        write_cases([original], TEST_SOURCE, engine=engine)
        restored = read_cases(TEST_SOURCE, engine=engine)[0]
        assert [e.timestamp for e in restored.events] == [
            e.timestamp for e in original.events
        ]

    @NEEDS_SQL
    def test_writing_twice_does_not_duplicate_events(self, engine) -> None:
        case = sub_second_case()
        write_cases([case], TEST_SOURCE, engine=engine)
        write_cases([case], TEST_SOURCE, engine=engine)
        restored = read_cases(TEST_SOURCE, engine=engine)
        assert len(restored) == 1
        assert len(restored[0].events) == len(case.events)

    @NEEDS_SQL
    def test_scores_keep_their_contributions(self, engine) -> None:
        case = sub_second_case()
        write_cases([case], TEST_SOURCE, engine=engine)
        run_id = write_scores(
            [
                CaseScore(
                    source_case_id=case.patient_id,
                    score=0.73,
                    contributions=[FeatureContribution("hemoglobin_drop", 1.0, 0.8, 0.8)],
                )
            ],
            source=TEST_SOURCE,
            model_type="logistic",
            model_version="test",
            engine=engine,
        )
        with engine.connect() as connection:
            score, feature = connection.execute(
                text(
                    "SELECT s.score, c.feature FROM core.scores AS s "
                    "JOIN core.score_contributions AS c ON c.score_key = s.score_key "
                    "WHERE s.run_id = :run"
                ),
                {"run": run_id},
            ).one()
        assert score == pytest.approx(0.73)
        assert feature == "hemoglobin_drop"

    @NEEDS_SQL
    def test_reviews_record_an_adjudication(self, engine) -> None:
        case = sub_second_case()
        write_cases([case], TEST_SOURCE, engine=engine)
        review_key = record_review(
            TEST_SOURCE, case.patient_id, "reviewer_a", "unclear", "needs the chart", engine=engine
        )
        assert review_key > 0

    def test_unknown_decision_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="decision"):
            record_review(TEST_SOURCE, "roundtrip-1", "reviewer_a", "maybe")


@NEEDS_SQL
@NEEDS_MIMIC
class TestCsvEquivalence:
    """The main deliverable: SQL and CSV have to agree at every layer."""

    def test_same_case_count_and_ids(self, equivalence) -> None:
        assert equivalence["sql_cases"] == equivalence["csv_cases"]
        assert equivalence["same_ids"], "case ids differ between CSV and SQL"
        assert equivalence["same_order"], "cohort ordering differs between CSV and SQL"

    def test_same_events(self, equivalence) -> None:
        assert equivalence["sql_events"] == equivalence["csv_events"]
        assert equivalence["event_mismatches"] == []

    def test_same_demographics_and_labels(self, equivalence) -> None:
        assert equivalence["field_mismatches"] == []

    def test_same_watcher_features(self, equivalence) -> None:
        assert equivalence["feature_mismatches"] == []

    def test_same_model_scores(self, equivalence) -> None:
        assert equivalence["same_weights"], "models fit on the two cohorts differ"
        assert equivalence["max_score_delta"] == 0.0

    def test_verdict(self, equivalence) -> None:
        assert equivalence["equivalent"]


@NEEDS_SQL
def test_source_counts_sees_the_built_cohorts(engine) -> None:
    counts = source_counts(engine)
    assert counts.get(MIMIC_SOURCE, 0) > 0


@NEEDS_SQL
def test_reading_every_source_at_once(engine) -> None:
    everything = read_cases(engine=engine)
    assert len(everything) >= len(read_cases(MIMIC_SOURCE, engine=engine))
