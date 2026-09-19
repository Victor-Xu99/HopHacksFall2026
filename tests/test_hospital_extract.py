from pathlib import Path

import pytest
from sqlalchemy import text

from src.data.hospital_adapt import classify_folder, normalize_name, write_bundle
from src.data.hospital_extract import (
    HOSPITAL_SOURCE,
    ingest_extract,
    load_extract_dir,
    rank_discharged,
    write_extract_dir,
)
from src.data.sql_store import available as sql_available
from src.data.sql_store import (
    build_engine,
    discharged_unscored_ids,
    ensure_core_schema,
    read_cases,
    record_review,
    reviewed_case_ids,
)
from src.service import SOURCE_HOSPITAL, file_review, queue_sort_key, waiting_days

FIXTURES = Path(__file__).parent / "fixtures" / "hospital"
NEEDS_SQL = pytest.mark.skipif(not sql_available(), reason="SafetyNet SQL not reachable")


def test_write_extract_dir_requires_all_four_files(tmp_path):
    with pytest.raises(ValueError, match="stays.csv"):
        write_extract_dir(tmp_path, {"labs.csv": b"x"})


def test_classifies_our_sample_filenames():
    assigned, _report = classify_folder(FIXTURES / "in_house")
    assert assigned["stays"].name == "stays.csv"
    assert assigned["labs"].name == "labs.csv"


def test_narcan_becomes_naloxone():
    assert normalize_name("NARCAN") == "Naloxone"
    assert normalize_name("Creatinine") == "Creatinine"


def test_clarity_like_headers_and_filenames():
    cases, report = load_extract_dir(FIXTURES / "clarity_like")
    by_id = {case.patient_id: case for case in cases}
    assert report["files"]["stays"] == "encounters.csv"
    assert report["columns"]["stays"]["encounter_id"] == "PAT_ENC_CSN_ID"
    assert by_id["88421"].discharged_at.startswith("2026-09-07")
    assert any(event.value == "Naloxone" for event in by_id["88421"].events)
    assert any(event.value == "Creatinine" for event in by_id["88421"].events)
    assert any(event.value == "Lactate" for event in by_id["88421"].events)


def test_write_bundle_accepts_a_zip(tmp_path):
    import zipfile

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in (FIXTURES / "in_house").glob("*.csv"):
            zipped.write(path, arcname=path.name)
    folder = write_bundle(tmp_path / "out", [("bundle.zip", archive.read_bytes())])
    cases, _report = load_extract_dir(folder)
    assert {case.patient_id for case in cases} == {"88421", "88422", "88423"}


def test_mixed_dates_keeps_old_ids_and_adds_stale_and_fresh():
    cases, _report = load_extract_dir(FIXTURES / "mixed_dates")
    by_id = {case.patient_id: case for case in cases}
    assert "9001" in by_id and "9020" in by_id and "9023" in by_id
    assert by_id["9023"].discharged_at is None
    assert by_id["9020"].discharged_at.startswith("2026-08-20")
    assert by_id["9021"].discharged_at.startswith("2026-09-18")


def test_mixed_10_splits_discharged_and_in_house():
    cases, _report = load_extract_dir(FIXTURES / "mixed_10")
    discharged = {case.patient_id for case in cases if case.discharged_at}
    in_house = {case.patient_id for case in cases if not case.discharged_at}
    assert discharged == {"9001", "9002", "9003", "9004", "9005"}
    assert in_house == {"9011", "9012", "9013", "9014", "9015"}
    naloxone_in_house = next(case for case in cases if case.patient_id == "9014")
    assert any(event.value == "Naloxone" for event in naloxone_in_house.events)


@NEEDS_SQL
def test_mixed_10_ranks_only_the_five_discharges():
    engine = build_engine()
    ensure_core_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM core.cases WHERE source = :s"), {"s": HOSPITAL_SOURCE})
    result = ingest_extract(FIXTURES / "mixed_10", engine=engine)
    assert result["in_house"] == 5
    assert result["discharged"] == 5
    assert set(result["ready_to_rank"]) == {"9001", "9002", "9003", "9004", "9005"}
    ranked = rank_discharged(engine=engine)
    assert set(ranked["case_ids"]) == {"9001", "9002", "9003", "9004", "9005"}

    later = ingest_extract(FIXTURES / "mixed_10_later", engine=engine)
    assert later["in_house"] == 0
    assert later["discharged"] == 10
    assert set(later["ready_to_rank"]) == {"9011", "9012", "9013", "9014", "9015"}
    stored = {case.patient_id: case for case in read_cases(HOSPITAL_SOURCE, engine=engine)}
    assert stored["9014"].discharged_at is not None
    assert any(event.value == "Naloxone" for event in stored["9014"].events)
    ranked_later = rank_discharged(engine=engine)
    assert set(ranked_later["case_ids"]) == {"9011", "9012", "9013", "9014", "9015"}

    dates = ingest_extract(FIXTURES / "mixed_dates", engine=engine)
    stored = {case.patient_id: case for case in read_cases(HOSPITAL_SOURCE, engine=engine)}
    assert set(stored) >= {
        "9001",
        "9002",
        "9003",
        "9004",
        "9005",
        "9011",
        "9012",
        "9013",
        "9014",
        "9015",
        "9020",
        "9021",
        "9022",
        "9023",
        "9024",
    }
    assert any(event.value == "Naloxone" for event in stored["9001"].events)
    assert dates["in_house"] == 2
    assert dates["discharged"] == 13
    assert set(dates["ready_to_rank"]) == {"9020", "9021", "9022"}
    from src.service import waiting_days
    from datetime import datetime, timezone

    now = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    assert waiting_days(stored["9020"].discharged_at, now) >= 7
    assert waiting_days(stored["9021"].discharged_at, now) < 7
    engine.dispose()


@NEEDS_SQL
def test_done_hides_a_discharged_stay_from_the_queue():
    engine = build_engine()
    ensure_core_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM core.cases WHERE source = :s"), {"s": HOSPITAL_SOURCE})
    ingest_extract(FIXTURES / "mixed_10", engine=engine)
    rank_discharged(engine=engine)
    file_review(SOURCE_HOSPITAL, "9002", "unclear", "pytest", "cleared from queue")
    assert "9002" in reviewed_case_ids(HOSPITAL_SOURCE, engine=engine)
    still_open = set(discharged_unscored_ids(HOSPITAL_SOURCE, engine=engine))
    assert "9002" not in still_open
    from src.service import review_payload

    payload = review_payload(SOURCE_HOSPITAL, "logistic", 20)
    assert payload["can_review"] is True
    assert all(row["case_id"] != "9002" for row in payload["reviews"])
    engine.dispose()


def test_waiting_days_and_oldest_tie_break():
    from datetime import datetime, timezone

    now = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    assert waiting_days("2026-09-12T11:00:00", now) == 7
    assert waiting_days(None, now) is None
    first, second = sorted(
        [
            {"score": 0.9, "discharged_at": "2026-09-12T11:00:00"},
            {"score": 0.9, "discharged_at": "2026-09-05T16:00:00"},
        ],
        key=queue_sort_key,
    )
    assert first["discharged_at"].startswith("2026-09-05")
    assert second["discharged_at"].startswith("2026-09-12")


def test_file_review_rejects_in_memory_sources():
    with pytest.raises(ValueError, match="not stored"):
        file_review("synthetic", "anything", "unclear")


def test_in_house_extract_leaves_maria_undischarged():
    cases, _report = load_extract_dir(FIXTURES / "in_house")
    cases = {case.patient_id: case for case in cases}
    assert cases["88421"].discharged_at is None
    assert cases["88422"].discharged_at is not None
    assert any(event.value == "Creatinine" for event in cases["88421"].events)


def test_discharged_extract_sets_maria_discharge_time():
    cases, _report = load_extract_dir(FIXTURES / "discharged")
    cases = {case.patient_id: case for case in cases}
    assert cases["88421"].discharged_at.startswith("2026-09-07")
    assert cases["88423"].discharged_at is None
    assert any(event.value == "Naloxone" for event in cases["88421"].events)


@NEEDS_SQL
def test_pipeline_ranks_only_after_discharge():
    engine = build_engine()
    ensure_core_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM core.cases WHERE source = :s"), {"s": HOSPITAL_SOURCE})

    first = ingest_extract(FIXTURES / "in_house", engine=engine)
    assert first["in_house"] == 2
    assert first["discharged"] == 1
    assert first["ready_to_rank"] == ["88422"]

    ranked = rank_discharged(engine=engine)
    assert ranked["scored"] == 1
    assert ranked["case_ids"] == ["88422"]
    assert discharged_unscored_ids(HOSPITAL_SOURCE, engine=engine) == []

    second = ingest_extract(FIXTURES / "discharged", engine=engine)
    stored = {case.patient_id: case for case in read_cases(HOSPITAL_SOURCE, engine=engine)}
    assert stored["88421"].discharged_at is not None
    assert any(event.value == "Naloxone" for event in stored["88421"].events)
    assert any(event.value == "Creatinine" for event in stored["88421"].events)
    assert "88421" in second["ready_to_rank"]
    assert "88422" not in second["ready_to_rank"]

    ranked_again = rank_discharged(engine=engine)
    assert "88421" in ranked_again["case_ids"]
    assert "88422" not in ranked_again["case_ids"]

    engine.dispose()
