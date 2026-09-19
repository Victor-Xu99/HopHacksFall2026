import pytest

from src.data.generator import generate_case, generate_dataset
from src.domain.models import PatientCase, PatientEvent
from src.engine.model import HarmScoringModel
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.watcher import StructuredDataWatcher


def note_case(text: str) -> PatientCase:
    return PatientCase(
        patient_id="t",
        age=60,
        gender="F",
        events=[
            PatientEvent(
                event_id="n1",
                event_type="note",
                timestamp="2026-01-01T00:00:00",
                value="Progress Note",
                details=text,
            )
        ],
    )


@pytest.fixture(scope="module")
def reader() -> ClinicalNoteReader:
    return ClinicalNoteReader()


@pytest.fixture(scope="module")
def cohort():
    return generate_dataset(num_cases=400, harm_ratio=0.2, hard_negative_ratio=0.4, seed=11)


class TestStructuredWatcher:
    def test_antidote_before_treatment_is_not_flagged_as_ours(self):
        case = generate_case(is_harm=False, scenario="community_overdose_on_arrival")
        features = StructuredDataWatcher().extract_features(case)
        assert features["antidote_given"] == 1.0
        assert features["antidote_after_treatment_start"] == 0.0

    def test_antidote_after_treatment_is_flagged(self):
        case = generate_case(is_harm=True, scenario="opioid_oversedation")
        features = StructuredDataWatcher().extract_features(case)
        assert features["antidote_after_treatment_start"] == 1.0

    def test_planned_icu_transfer_is_not_an_unplanned_escalation(self):
        case = generate_case(is_harm=False, scenario="planned_icu_admission")
        assert StructuredDataWatcher().extract_features(case)["unplanned_icu_transfer"] == 0.0

    def test_unplanned_icu_transfer_fires(self):
        case = generate_case(is_harm=True, scenario="hospital_acquired_infection")
        assert StructuredDataWatcher().extract_features(case)["unplanned_icu_transfer"] == 1.0

    def test_unplanned_return_to_or(self):
        case = generate_case(is_harm=True, scenario="postop_hemorrhage")
        features = StructuredDataWatcher().extract_features(case)
        assert features["unplanned_return_to_or"] == 1.0
        assert features["hemoglobin_drop"] == 1.0

    def test_creatinine_doubling_detected(self):
        case = generate_case(is_harm=True, scenario="contrast_aki")
        assert StructuredDataWatcher().extract_features(case)["creatinine_doubled"] == 1.0

    def test_baseline_labs_are_not_post_treatment(self):
        case = generate_case(is_harm=False, scenario="routine")
        features = StructuredDataWatcher().extract_features(case)
        assert features["critical_post_treatment_lab"] == 0.0

    def test_readmission_detected(self):
        case = generate_case(is_harm=True, scenario="missed_diagnosis_readmission")
        assert StructuredDataWatcher().extract_features(case)["readmission_30d"] == 1.0


class TestNoteReader:
    def test_separates_framing_for_the_same_event(self, reader):
        unanticipated = note_case(
            "The bleeding after the operation was far more than we had any reason to expect "
            "and re-exploration should have happened sooner."
        )
        expected = note_case(
            "Bleeding after this operation is a known risk that was reviewed at consent, "
            "and the volume is within the anticipated range."
        )
        surprised = reader.extract_features(unanticipated)
        consented = reader.extract_features(expected)
        assert surprised["note_framing_margin"] > consented["note_framing_margin"]
        assert consented["note_expected_risk_language"] >= 1.0

    def test_negated_findings_do_not_count_as_harm(self, reader):
        features = reader.extract_features(
            note_case("No evidence of hemorrhage and sepsis was ruled out. No acute distress.")
        )
        assert features["note_harm_state_language"] == 0.0
        assert features["note_negated_harm_mentions"] > 0.0

    def test_unnegated_finding_counts(self, reader):
        features = reader.extract_features(note_case("Worsening hypoxia with ongoing hemorrhage."))
        assert features["note_harm_state_language"] > 0.0

    def test_negation_is_clause_scoped(self, reader):
        features = reader.extract_features(
            note_case("No acute distress, but there is worsening hypoxia this afternoon.")
        )
        assert features["note_harm_state_language"] > 0.0

    def test_explicit_error_language(self, reader):
        features = reader.extract_features(
            note_case("The wrong dose was given because of an order entry problem.")
        )
        assert features["note_explicit_error_language"] > 0.0

    def test_empty_chart_yields_zeros(self, reader):
        features = reader.extract_features(PatientCase(patient_id="x", age=1, gender="M"))
        assert set(features.values()) == {0.0}


class TestScoringModel:
    @pytest.mark.parametrize("model_type", ["logistic", "tree"])
    def test_learns_signal_and_reports_held_out_metrics(self, model_type, cohort, reader):
        model = HarmScoringModel(
            [StructuredDataWatcher(), reader], model_type=model_type, threshold=0.4
        )
        report = model.train(cohort)
        assert report.n_test > 0
        assert report.roc_auc > 0.85
        assert report.recall > 0.7

    def test_logistic_contributions_reconcile_with_the_score(self, cohort, reader):
        model = HarmScoringModel([StructuredDataWatcher(), reader], model_type="logistic")
        model.train(cohort)
        case = next(c for c in cohort if c.is_harm_event)
        breakdown = model.score_breakdown(case)
        assert breakdown["logit"] == pytest.approx(
            breakdown["baseline_log_odds"] + breakdown["sum_of_contributions"]
        )
        assert breakdown["probability"] == pytest.approx(model.predict_score(case))

    def test_expected_risk_language_is_learned_as_protective(self, cohort, reader):
        model = HarmScoringModel([StructuredDataWatcher(), reader], model_type="logistic")
        model.train(cohort)
        assert model.get_feature_weights()["note_expected_risk_language"] < 0

    def test_in_hospital_antidote_outweighs_bare_antidote(self, cohort, reader):
        model = HarmScoringModel([StructuredDataWatcher(), reader], model_type="logistic")
        model.train(cohort)
        weights = model.get_feature_weights()
        assert weights["antidote_after_treatment_start"] > weights["antidote_given"]

    def test_harm_outscores_its_matched_hard_negative(self, cohort, reader):
        model = HarmScoringModel([StructuredDataWatcher(), reader], model_type="logistic")
        model.train(cohort)
        harm = generate_case(is_harm=True, scenario="opioid_oversedation")
        lookalike = generate_case(is_harm=False, scenario="community_overdose_on_arrival")
        assert model.predict_score(harm) > model.predict_score(lookalike)

    def test_rejects_unknown_model_type(self):
        with pytest.raises(ValueError):
            HarmScoringModel([StructuredDataWatcher()], model_type="deep_net")

    def test_rejects_single_class_sample(self, reader):
        cases = [generate_case(is_harm=False) for _ in range(12)]
        model = HarmScoringModel([StructuredDataWatcher(), reader])
        with pytest.raises(ValueError, match="one class"):
            model.train(cases)

    def test_untrained_model_scores_zero(self, reader):
        model = HarmScoringModel([StructuredDataWatcher(), reader])
        assert model.predict_score(generate_case(is_harm=True)) == 0.0
        assert model.get_evidence(generate_case(is_harm=True)) == []

    def test_tree_exposes_readable_rules(self, cohort, reader):
        model = HarmScoringModel([StructuredDataWatcher(), reader], model_type="tree")
        model.train(cohort)
        assert "<=" in model.describe_rules()
