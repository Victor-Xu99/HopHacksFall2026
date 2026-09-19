"""Tests for the label-leakage audit."""

from typing import Dict, List

import pytest

from src.domain.models import PatientCase, PatientEvent
from src.engine.interfaces import FeatureExtractor
from src.engine.leakage import (
    GIVEAWAY_PURITY,
    MIN_SUPPORT,
    audit,
    audit_features,
    audit_values,
)


class FakeExtractor(FeatureExtractor):
    """Reports whichever features a case was constructed with."""

    def __init__(self, names: List[str]):
        self._names = names

    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        fired = {e.value for e in case.events if e.event_type == "flag"}
        return {name: (1.0 if name in fired else 0.0) for name in self._names}

    def feature_descriptions(self) -> Dict[str, str]:
        return {name: name for name in self._names}


def make_case(case_id: str, harm: bool, flags=(), meds=()) -> PatientCase:
    events = [
        PatientEvent(f"{case_id}-f{i}", "flag", "2024-01-01T00:00:00", flag, "")
        for i, flag in enumerate(flags)
    ]
    events += [
        PatientEvent(f"{case_id}-m{i}", "medication", "2024-01-01T01:00:00", med, "")
        for i, med in enumerate(meds)
    ]
    return PatientCase(
        patient_id=case_id, age=60, gender="F", events=events, is_harm_event=harm
    )


def test_perfectly_predictive_feature_is_flagged_as_leaking():
    # "tell" fires on labeled cases only, with support above the floor.
    cases = [make_case(f"h{i}", True, flags=["tell"]) for i in range(MIN_SUPPORT + 10)]
    cases += [make_case(f"n{i}", False) for i in range(50)]

    findings, _ = audit_features(cases, FakeExtractor(["tell"]))

    tell = next(f for f in findings if f.feature == "tell")
    assert tell.label_rate == pytest.approx(1.0)
    assert tell.separation == pytest.approx(1.0)
    assert tell.leaking


def test_partially_predictive_feature_is_not_flagged():
    # Fires on labeled and unlabeled cases alike, which is what real signals do.
    cases = [make_case(f"h{i}", True, flags=["hint"]) for i in range(40)]
    cases += [make_case(f"n{i}", False, flags=["hint"]) for i in range(40)]
    cases += [make_case(f"x{i}", False) for i in range(20)]

    findings, _ = audit_features(cases, FakeExtractor(["hint"]))

    hint = next(f for f in findings if f.feature == "hint")
    assert hint.separation < 1.0
    assert not hint.leaking


def test_perfect_but_rare_feature_is_not_flagged():
    """Below the support floor, perfect separation is chance rather than evidence."""
    rare = MIN_SUPPORT - 5
    cases = [make_case(f"h{i}", True, flags=["rare"]) for i in range(rare)]
    cases += [make_case(f"n{i}", False) for i in range(60)]

    findings, _ = audit_features(cases, FakeExtractor(["rare"]))

    found = next(f for f in findings if f.feature == "rare")
    assert found.separation == pytest.approx(1.0)
    assert not found.leaking


def test_feature_that_never_fires_is_reported_dead_not_leaking():
    cases = [make_case(f"h{i}", True) for i in range(20)]
    cases += [make_case(f"n{i}", False) for i in range(20)]

    findings, dead = audit_features(cases, FakeExtractor(["absent"]))

    assert dead == ["absent"]
    assert findings == []


def test_giveaway_medication_is_detected():
    cases = [
        make_case(f"h{i}", True, meds=["Protamine Sulfate"]) for i in range(MIN_SUPPORT + 5)
    ]
    cases += [make_case(f"n{i}", False, meds=["Cefazolin"]) for i in range(40)]

    giveaways = audit_values(cases)

    values = {g.value for g in giveaways}
    assert "Protamine Sulfate" in values
    assert "Cefazolin" not in values


def test_medication_given_to_both_classes_is_not_a_giveaway():
    """Naloxone on a case that turned out fine is exactly what breaks a tell."""
    cases = [make_case(f"h{i}", True, meds=["Naloxone"]) for i in range(30)]
    cases += [make_case(f"n{i}", False, meds=["Naloxone"]) for i in range(30)]

    assert audit_values(cases) == []


def test_repeated_event_counts_once_per_case():
    """A drug charted many times must not outvote one charted once."""
    cases = [make_case("h0", True, meds=["Heparin"] * 50)]
    cases += [make_case(f"n{i}", False, meds=["Heparin"]) for i in range(40)]

    giveaways = audit_values(cases, min_support=1)

    heparin = next(g for g in giveaways if g.value == "Heparin") if giveaways else None
    # 1 labeled of 41 cases carrying it; far below the purity bar.
    assert heparin is None


def test_report_is_clean_when_nothing_leaks():
    cases = [make_case(f"h{i}", True, flags=["hint"]) for i in range(40)]
    cases += [make_case(f"n{i}", False, flags=["hint"]) for i in range(40)]

    report = audit(cases, FakeExtractor(["hint"]))

    assert report.clean
    assert report.base_rate == pytest.approx(0.5)
    assert "No leakage detected" in report.summary()


def test_report_flags_cohort_when_a_feature_separates_perfectly():
    cases = [make_case(f"h{i}", True, flags=["tell"]) for i in range(MIN_SUPPORT + 5)]
    cases += [make_case(f"n{i}", False) for i in range(50)]

    report = audit(cases, FakeExtractor(["tell"]))

    assert not report.clean
    assert [f.feature for f in report.leaking_features] == ["tell"]
    assert "measure the generator" in report.summary()
