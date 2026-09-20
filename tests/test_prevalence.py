"""Resampling has to change the base rate without inventing or leaking cases."""

import pytest

from src.data.prevalence import (
    OIG_PREVENTABLE_HARM_RATE,
    downsample_to_prevalence,
    harm_rate,
    resample_with_summary,
)
from src.domain.models import PatientCase


def cohort(n_harm: int, n_clean: int):
    harm = [
        PatientCase(patient_id=f"h{i}", age=60, gender="F", is_harm_event=True)
        for i in range(n_harm)
    ]
    clean = [
        PatientCase(patient_id=f"c{i}", age=60, gender="M", is_harm_event=False)
        for i in range(n_clean)
    ]
    return harm + clean


class TestDownsampling:
    @pytest.mark.parametrize("target", (0.05, 0.10, 0.25))
    def test_hits_the_requested_rate(self, target):
        result = downsample_to_prevalence(cohort(1600, 1900), target)
        assert harm_rate(result) == pytest.approx(target, abs=0.01)

    def test_keeps_every_clean_case_when_thinning_harm(self):
        result = downsample_to_prevalence(cohort(1600, 1900), 0.10)
        assert sum(1 for c in result if not c.is_harm_event) == 1900

    def test_never_duplicates_a_case(self):
        result = downsample_to_prevalence(cohort(1600, 1900), 0.10)
        ids = [c.patient_id for c in result]
        assert len(ids) == len(set(ids))

    def test_raises_the_rate_by_dropping_clean_cases(self):
        result = downsample_to_prevalence(cohort(100, 1900), 0.25)
        assert harm_rate(result) == pytest.approx(0.25, abs=0.01)
        assert sum(1 for c in result if c.is_harm_event) == 100

    def test_is_deterministic_under_a_seed(self):
        source = cohort(1600, 1900)
        first = downsample_to_prevalence(source, 0.10, seed=3)
        second = downsample_to_prevalence(source, 0.10, seed=3)
        assert [c.patient_id for c in first] == [c.patient_id for c in second]

    def test_different_seeds_select_differently(self):
        source = cohort(1600, 1900)
        first = downsample_to_prevalence(source, 0.10, seed=1)
        second = downsample_to_prevalence(source, 0.10, seed=2)
        assert {c.patient_id for c in first} != {c.patient_id for c in second}

    def test_single_class_cohort_is_returned_untouched(self):
        only_clean = cohort(0, 50)
        assert len(downsample_to_prevalence(only_clean, 0.10)) == 50

    @pytest.mark.parametrize("bad", (0.0, 1.0, -0.2, 1.5))
    def test_rejects_impossible_targets(self, bad):
        with pytest.raises(ValueError, match="between 0 and 1"):
            downsample_to_prevalence(cohort(10, 10), bad)


class TestSummary:
    def test_reports_what_was_dropped(self):
        _, summary = resample_with_summary(cohort(1600, 1900), 0.10)
        assert summary.original_rate == pytest.approx(0.457, abs=0.01)
        assert summary.resulting_rate == pytest.approx(0.10, abs=0.01)
        assert summary.kept + summary.dropped == 3500
        assert "46" in summary.summary() or "45" in summary.summary()


def test_default_target_matches_the_oig_figure():
    assert OIG_PREVENTABLE_HARM_RATE == 0.10
