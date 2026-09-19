"""Reads free-text notes for harm-adjacent language.

Two passes over every sentence:

1. A trained classifier (TF-IDF n-grams + multinomial logistic regression, fit on
   the hand-labeled corpus in `note_corpus`) puts the sentence into one of
   unanticipated / expected_risk / neutral. This is what lets the reader tell
   "known risk, consented for" apart from "this wasn't supposed to happen" -- the
   distinction is carried by framing words, not by the clinical event named.
2. A lexicon pass catches explicit process-failure wording and handles negation,
   so "no evidence of bleeding" does not read as a bleed.

Both passes stay inspectable: `explain()` returns the sentences that fired and
which class they landed in, and `top_terms()` returns the n-grams the classifier
leans on.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.domain.models import PatientCase
from src.engine.interfaces import FeatureExtractor
from src.engine.note_corpus import labeled_sentences

# Explicit process failures. These are not about how bad the outcome was; they
# describe care that deviated from what was intended.
ERROR_PHRASES = (
    "wrong dose",
    "wrong site",
    "wrong patient",
    "wrong medication",
    "wrong concentration",
    "duplicate dose",
    "order entry",
    "omitted dose",
    "missed dose",
    "failure to",
    "delay in",
    "delayed recognition",
    "not recognized",
    "inadvertent",
    "inadvertently",
    "accidental",
    "unintended",
    "unintentional",
    "retained",
    "iatrogenic",
    "preventable",
    "avoidable",
    "should have been",
    "medication error",
    "documented but",
)

# Clinical deterioration wording. Raises severity but says nothing about whether
# the deterioration was expected, so it is kept as its own feature.
HARM_STATE_PHRASES = (
    "worsening",
    "deteriorat",
    "decompensat",
    "acute distress",
    "hemodynamically unstable",
    "required intubation",
    "reintubat",
    "respiratory arrest",
    "cardiac arrest",
    "code blue",
    "rapid response",
    "hemorrhag",
    "sepsis",
    "septic",
    "obtunded",
    "unresponsive",
    "hypoxic",
    "hypoxemia",
    "emergent",
    "took a turn for the worse",
    "abrupt turn",
)

NEGATION_CUES = (
    "no ",
    "not ",
    "non",
    "without",
    "denies",
    "denied",
    "negative for",
    "ruled out",
    "r/o",
    "free of",
    "absence of",
    "no evidence of",
    "resolved",
)

# Determiners, prepositions, and copulas only. Negation, modals, and hedges
# ("not", "should", "would", "never", "despite") are deliberately kept, because
# that is where framing lives. Without this list the classifier latches onto tense
# -- past for unanticipated, present for expected -- which is an artifact of how
# the corpus is written and would not survive contact with real charts.
CUSTOM_STOP_WORDS = frozenset(
    """
    a an the and or of to in on at for with by from as that this these those it its
    was were is are be been being am do does did has have had he she his her they
    them their we our you your i patient patients pt
    """.split()
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
CLAUSE_SPLIT = re.compile(r",|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b")

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "note_unanticipated_language": "Note language framed as unanticipated / not supposed to happen",
    "note_expected_risk_language": "Note language framing the event as a known, consented risk",
    "note_max_unanticipated_prob": "Strongest unanticipated-framing sentence in the chart",
    "note_framing_margin": "Unanticipated framing minus expected-risk framing",
    "note_explicit_error_language": "Explicit process failure wording (wrong dose, delay, failure to)",
    "note_harm_state_language": "Clinical deterioration wording, negation-aware",
    "note_negated_harm_mentions": "Harm words that were explicitly negated (ruled out, no evidence of)",
}


@dataclass
class SentenceFinding:
    text: str
    label: str
    probability: float
    error_terms: Tuple[str, ...] = ()
    harm_terms: Tuple[str, ...] = ()
    negated: bool = False


class ClinicalNoteReader(FeatureExtractor):
    """Scans free-text notes for harm-adjacent language."""

    def __init__(
        self,
        extra_training: Optional[Tuple[List[str], List[str]]] = None,
        min_confidence: float = 0.45,
    ):
        # Framing claims need to clear a bar. A sentence the classifier is merely
        # 40% sure about falls back to neutral instead of asserting intent.
        self.min_confidence = min_confidence
        texts, labels = labeled_sentences()
        if extra_training is not None:
            more_texts, more_labels = extra_training
            texts = texts + list(more_texts)
            labels = labels + list(more_labels)

        self.classifier: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 3),
                        sublinear_tf=True,
                        min_df=1,
                        stop_words=list(CUSTOM_STOP_WORDS),
                    ),
                ),
                ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
            ]
        )
        self.classifier.fit(texts, labels)
        self.classes_: List[str] = list(self.classifier.named_steps["lr"].classes_)

    def feature_descriptions(self) -> Dict[str, str]:
        return dict(FEATURE_DESCRIPTIONS)

    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        findings = self._analyze(case)
        features = {name: 0.0 for name in FEATURE_DESCRIPTIONS}
        if not findings:
            return features

        unanticipated = [f for f in findings if f.label == "unanticipated"]
        expected = [f for f in findings if f.label == "expected_risk"]

        features["note_unanticipated_language"] = float(len(unanticipated))
        features["note_expected_risk_language"] = float(len(expected))
        features["note_max_unanticipated_prob"] = max(
            (f.probability for f in unanticipated), default=0.0
        )
        features["note_framing_margin"] = features["note_max_unanticipated_prob"] - max(
            (f.probability for f in expected), default=0.0
        )
        features["note_explicit_error_language"] = float(
            sum(len(f.error_terms) for f in findings if not f.negated)
        )
        features["note_harm_state_language"] = float(
            sum(len(f.harm_terms) for f in findings if not f.negated)
        )
        features["note_negated_harm_mentions"] = float(
            sum(len(f.harm_terms) + len(f.error_terms) for f in findings if f.negated)
        )
        return features

    def explain(self, case: PatientCase) -> List[SentenceFinding]:
        """Only the sentences that carry signal, strongest framing first."""
        findings = [
            f
            for f in self._analyze(case)
            if f.label != "neutral" or f.error_terms or f.harm_terms
        ]
        findings.sort(key=lambda f: (f.label != "unanticipated", -f.probability))
        return findings

    def top_terms(self, label: str = "unanticipated", n: int = 12) -> List[Tuple[str, float]]:
        """The n-grams the trained classifier weights most for a class."""
        if label not in self.classes_:
            return []
        lr = self.classifier.named_steps["lr"]
        vocab = self.classifier.named_steps["tfidf"].get_feature_names_out()
        coefs = lr.coef_[self.classes_.index(label)]
        ranked = sorted(zip(vocab, coefs), key=lambda kv: kv[1], reverse=True)
        return [(term, float(weight)) for term, weight in ranked[:n]]

    def _analyze(self, case: PatientCase) -> List[SentenceFinding]:
        sentences: List[str] = []
        for event in case.events_of_type("note"):
            sentences.extend(self._split(event.details))
        if not sentences:
            return []

        probabilities = self.classifier.predict_proba(sentences)
        findings: List[SentenceFinding] = []
        for sentence, row in zip(sentences, probabilities):
            best_index = int(row.argmax())
            label = self.classes_[best_index]
            if row[best_index] < self.min_confidence:
                label = "neutral"
            error_terms, harm_terms, negated = self._lexicon_pass(sentence)
            findings.append(
                SentenceFinding(
                    text=sentence,
                    label=label,
                    probability=float(row[best_index]),
                    error_terms=error_terms,
                    harm_terms=harm_terms,
                    negated=negated,
                )
            )
        return findings

    @staticmethod
    def _split(text: str) -> List[str]:
        if not text:
            return []
        return [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 3]

    @staticmethod
    def _lexicon_pass(sentence: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], bool]:
        lowered = sentence.lower()
        error_terms = tuple(p for p in ERROR_PHRASES if p in lowered)
        harm_terms = tuple(p for p in HARM_STATE_PHRASES if p in lowered)
        if not error_terms and not harm_terms:
            return (), (), False

        # Negation is scoped to the clause holding the matched term, so
        # "no acute distress, worsening edema overnight" negates only the first
        # half. The sentence counts as negated only when every clause carrying a
        # matched term also carries a negation cue.
        matched = error_terms + harm_terms
        carrying_clauses = [
            clause
            for clause in CLAUSE_SPLIT.split(lowered)
            if clause.strip() and any(term in clause for term in matched)
        ]
        if not carrying_clauses:
            return error_terms, harm_terms, False

        negated = all(
            any(cue in clause for cue in NEGATION_CUES) for clause in carrying_clauses
        )
        return error_terms, harm_terms, negated
