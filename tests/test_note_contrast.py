"""Guards on the note reader's measured behaviour.

The thresholds here sit deliberately below what the reader currently scores.
They are regression guards, not targets: they catch the layer going blind to
framing, which is the one failure that would make it pointless, without
breaking every time a corpus sentence is reworded.
"""

import pytest

from src.domain.models import PatientCase, PatientEvent
from src.engine.note_contrast import CONTRAST_PAIRS, contrast_sentences
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.note_eval import (
    FramingLexiconBaseline,
    LexiconOnlyNoteReader,
    evaluate_on_contrast,
    evaluate_sentence_classifier,
)


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
def trained_classifier():
    return ClinicalNoteReader().classifier


class TestContrastCorpus:
    def test_every_pair_changes_only_framing(self):
        for event, unanticipated, expected in CONTRAST_PAIRS:
            assert unanticipated != expected, f"{event} pair is identical"
            assert unanticipated.strip() and expected.strip()

    def test_no_duplicate_sentences_across_pairs(self):
        texts, _ = contrast_sentences()
        assert len(texts) == len(set(texts))

    def test_contrast_sentences_are_held_out_from_training(self):
        from src.engine.note_corpus import labeled_sentences

        corpus, _ = labeled_sentences()
        contrast, _ = contrast_sentences()
        assert not set(corpus) & set(contrast)

    def test_labels_alternate_unanticipated_then_expected(self):
        _, labels = contrast_sentences()
        assert labels[::2] == ["unanticipated"] * len(CONTRAST_PAIRS)
        assert labels[1::2] == ["expected_risk"] * len(CONTRAST_PAIRS)


class TestClassifierResponsToFraming:
    def test_classifier_beats_majority_baseline_in_cross_validation(self, trained_classifier):
        report = evaluate_sentence_classifier(trained_classifier, "tfidf")
        assert report.macro_f1_mean > report.majority_baseline

    def test_classifier_changes_its_answer_across_most_pairs(self, trained_classifier):
        report = evaluate_on_contrast(trained_classifier, "tfidf")
        assert report.flip_rate >= 0.55, (
            "reader has stopped distinguishing framing on held-out pairs; "
            f"only flipped {report.flipped}/{report.n_pairs}"
        )

    def test_classifier_generalizes_above_chance_on_unseen_prose(self, trained_classifier):
        report = evaluate_on_contrast(trained_classifier, "tfidf")
        assert report.accuracy >= 0.50

    def test_classifier_generalizes_better_than_the_keyword_lexicon(self, trained_classifier):
        """The lexicon was written against the seed corpus, so it wins in-domain
        and collapses outside it. Held-out prose is where the difference shows."""
        classifier = evaluate_on_contrast(trained_classifier, "tfidf")
        lexicon = evaluate_on_contrast(FramingLexiconBaseline().fit([]), "lexicon")
        assert classifier.accuracy > lexicon.accuracy


class TestLexiconOnlyReader:
    def test_emits_only_the_untrained_features(self):
        reader = LexiconOnlyNoteReader()
        assert set(reader.feature_descriptions()) == {
            "note_explicit_error_language",
            "note_harm_state_language",
            "note_negated_harm_mentions",
        }

    def test_counts_unnegated_harm_language(self):
        features = LexiconOnlyNoteReader().extract_features(
            note_case("Worsening hypoxia with ongoing hemorrhage.")
        )
        assert features["note_harm_state_language"] > 0
        assert features["note_negated_harm_mentions"] == 0

    def test_negated_harm_is_counted_separately(self):
        features = LexiconOnlyNoteReader().extract_features(
            note_case("No evidence of hemorrhage on repeat imaging.")
        )
        assert features["note_harm_state_language"] == 0
        assert features["note_negated_harm_mentions"] > 0

    def test_chart_without_notes_yields_zeros(self):
        case = PatientCase(patient_id="p3", age=44, gender="M")
        assert set(LexiconOnlyNoteReader().extract_features(case).values()) == {0.0}
