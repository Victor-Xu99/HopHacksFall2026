"""Measures whether the note reader earns its place.

`ClinicalNoteReader` fits its sentence classifier on the whole seed corpus and
never scores itself, so until now the layer had no accuracy attached to it at
all. Three things are measured here:

  1. Cross-validated per-class performance of the trained classifier, read
     against the 1/3 a majority guess would get on a balanced three-class corpus.
  2. The same numbers for a rule-based framing lexicon that needs no training.
     If the classifier cannot beat a keyword list over 90 sentences, the honest
     move is to keep the keyword list.
  3. An end-to-end arm: PR AUC of the harm model with no note features, with
     lexicon-only note features, and with the full reader. This is the one that
     answers whether the layer is worth its complexity in the product.

The end-to-end arm runs on the synthetic cohort, where the same repository
writes both the notes and the labels. That circularity is survivable here only
because every arm is exposed to it equally -- the comparison between arms is
meaningful, the absolute numbers are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from src.domain.models import PatientCase
from src.engine.interfaces import FeatureExtractor
from src.engine.nlp_reader import ClinicalNoteReader
from src.engine.note_corpus import labeled_sentences

CLASS_ORDER: Tuple[str, ...] = ("expected_risk", "neutral", "unanticipated")

# Framing cues that mark an event as anticipated, consented, or routine. These
# are the obvious words a person would grep for, which is the point: the trained
# classifier has to beat the obvious approach to justify existing.
EXPECTED_CUES: Tuple[str, ...] = (
    "known risk",
    "known complication",
    "anticipated",
    "expected",
    "consent",
    "counseled",
    "typical",
    "common",
    "within normal limits",
    "per protocol",
    "planned",
    "scheduled",
    "natural history",
    "recognized",
    "complication rate",
    "reviewed with",
    "discussed",
)

# Cues that mark an event as unintended. Checked first, because the strongest
# signals of the class are negated forms of the expected cues ("not anticipated"
# contains "anticipated") and a naive count would score them for both sides.
UNANTICIPATED_CUES: Tuple[str, ...] = (
    "did not anticipate",
    "not anticipate",
    "not anticipated",
    "unanticipated",
    "unexpected",
    "not predicted",
    "not consistent with the expected",
    "not part of the expected",
    "out of proportion",
    "out of keeping",
    "should have been",
    "went wrong",
    "surprised",
    "abrupt turn",
    "turn for the worse",
    "inadvertent",
    "unintended",
    "avoidable",
    "preventable",
    "iatrogenic",
    "retained",
    "failure to",
    "delay in",
    "duplicate dose",
    "wrong ",
    "error",
)


class FramingLexiconBaseline(BaseEstimator, ClassifierMixin):
    """Keyword framing classifier with nothing learned from the corpus.

    Scores a sentence by counting cues for each side. Unanticipated cues are
    matched first and struck from the text before expected cues are counted, so
    "was not anticipated" does not also register as "anticipated". Ties, which
    includes the common case of no cues at all, fall to neutral.
    """

    def fit(self, X: Sequence[str], y: Optional[Sequence[str]] = None) -> "FramingLexiconBaseline":
        self.classes_ = np.array(CLASS_ORDER)
        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        return np.array([self._label(text) for text in X])

    @staticmethod
    def _label(text: str) -> str:
        lowered = text.lower()
        remainder = lowered
        unanticipated = 0
        for cue in UNANTICIPATED_CUES:
            if cue in remainder:
                unanticipated += 1
                remainder = remainder.replace(cue, " ")

        expected = sum(1 for cue in EXPECTED_CUES if cue in remainder)

        if unanticipated > expected:
            return "unanticipated"
        if expected > unanticipated:
            return "expected_risk"
        return "neutral"


class LexiconOnlyNoteReader(FeatureExtractor):
    """The note reader stripped to its untrained half.

    Emits only the features the lexicon pass produces, so the end-to-end
    ablation can price the TF-IDF classifier separately from the keyword and
    negation handling it sits on top of.
    """

    FEATURES = {
        "note_explicit_error_language": "Explicit process failure wording (wrong dose, delay, failure to)",
        "note_harm_state_language": "Clinical deterioration wording, negation-aware",
        "note_negated_harm_mentions": "Harm words that were explicitly negated (ruled out, no evidence of)",
    }

    def feature_descriptions(self) -> Dict[str, str]:
        return dict(self.FEATURES)

    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        features = {name: 0.0 for name in self.FEATURES}
        for event in case.events_of_type("note"):
            for sentence in ClinicalNoteReader._split(event.details):
                errors, harms, negated = ClinicalNoteReader._lexicon_pass(sentence)
                if negated:
                    features["note_negated_harm_mentions"] += len(errors) + len(harms)
                else:
                    features["note_explicit_error_language"] += len(errors)
                    features["note_harm_state_language"] += len(harms)
        return features


@dataclass
class NoteReaderReport:
    """Cross-validated performance for one sentence-level approach."""

    name: str
    n_samples: int
    n_folds: int
    accuracy: float
    macro_f1_mean: float
    macro_f1_std: float
    per_class: Dict[str, Dict[str, float]]
    confusion: List[List[int]]
    majority_baseline: float

    def summary(self) -> str:
        return (
            f"{self.name} | macro F1 {self.macro_f1_mean:.3f} "
            f"(+/- {self.macro_f1_std:.3f}) | accuracy {self.accuracy:.3f} "
            f"vs {self.majority_baseline:.3f} majority baseline"
        )

    def confusion_table(self) -> str:
        width = max(len(c) for c in CLASS_ORDER) + 2
        header = " " * width + "".join(f"{c[:12]:>14}" for c in CLASS_ORDER)
        rows = [header]
        for label, row in zip(CLASS_ORDER, self.confusion):
            rows.append(f"{label:<{width}}" + "".join(f"{v:>14}" for v in row))
        return "\n".join(rows)


def evaluate_sentence_classifier(
    estimator,
    name: str,
    n_folds: int = 5,
    random_state: int = 7,
) -> NoteReaderReport:
    """Stratified cross-validation over the seed corpus for any sklearn estimator."""
    texts, labels = labeled_sentences()
    X = np.array(texts, dtype=object)
    y = np.array(labels)

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_scores = cross_val_score(estimator, X, y, cv=splitter, scoring="f1_macro")
    predicted = cross_val_predict(estimator, X, y, cv=splitter)

    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=list(CLASS_ORDER), zero_division=0
    )
    per_class = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(fs),
            "support": int(s),
        }
        for label, p, r, fs, s in zip(CLASS_ORDER, precision, recall, f1, support)
    }

    counts = np.bincount([list(CLASS_ORDER).index(v) for v in y], minlength=len(CLASS_ORDER))

    return NoteReaderReport(
        name=name,
        n_samples=int(len(y)),
        n_folds=n_folds,
        accuracy=float((predicted == y).mean()),
        macro_f1_mean=float(fold_scores.mean()),
        macro_f1_std=float(fold_scores.std()),
        per_class=per_class,
        confusion=confusion_matrix(y, predicted, labels=list(CLASS_ORDER)).tolist(),
        majority_baseline=float(counts.max() / counts.sum()),
    )


def compare_sentence_level(n_folds: int = 5, random_state: int = 7) -> List[NoteReaderReport]:
    """The trained classifier against the untrained keyword list, same folds."""
    trained = ClinicalNoteReader().classifier
    return [
        evaluate_sentence_classifier(trained, "tfidf + logistic", n_folds, random_state),
        evaluate_sentence_classifier(
            FramingLexiconBaseline(), "framing keyword lexicon", n_folds, random_state
        ),
    ]


@dataclass
class ContrastReport:
    """Performance on held-out pairs that differ only in framing."""

    name: str
    n_pairs: int
    accuracy: float
    both_correct: int
    flipped: int
    misses: List[Tuple[str, str, str]]  # (sentence, truth, predicted)

    @property
    def both_correct_rate(self) -> float:
        return self.both_correct / self.n_pairs if self.n_pairs else float("nan")

    @property
    def flip_rate(self) -> float:
        """Pairs where the two sentences got different answers, right or wrong.

        Separated from accuracy on purpose: a reader can be sensitive to framing
        while still mislabelling which side is which, and that is a different
        failure from being blind to framing altogether.
        """
        return self.flipped / self.n_pairs if self.n_pairs else float("nan")

    def summary(self) -> str:
        return (
            f"{self.name:<26} both right {self.both_correct}/{self.n_pairs} "
            f"({self.both_correct_rate:.0%}) | responds to framing "
            f"{self.flipped}/{self.n_pairs} ({self.flip_rate:.0%}) | "
            f"sentence accuracy {self.accuracy:.0%}"
        )


def evaluate_on_contrast(predictor, name: str) -> ContrastReport:
    """Score any estimator exposing `predict` against the contrast pairs."""
    from src.engine.note_contrast import CONTRAST_PAIRS

    texts, truths = [], []
    for _, unanticipated, expected in CONTRAST_PAIRS:
        texts.extend((unanticipated, expected))
        truths.extend(("unanticipated", "expected_risk"))

    predictions = list(predictor.predict(np.array(texts, dtype=object)))

    both_correct = flipped = 0
    misses: List[Tuple[str, str, str]] = []
    for index in range(0, len(texts), 2):
        got = predictions[index : index + 2]
        want = truths[index : index + 2]
        if got == want:
            both_correct += 1
        if got[0] != got[1]:
            flipped += 1
        for text, truth, prediction in zip(texts[index : index + 2], want, got):
            if truth != prediction:
                misses.append((text, truth, prediction))

    correct = sum(1 for t, p in zip(truths, predictions) if t == p)
    return ContrastReport(
        name=name,
        n_pairs=len(CONTRAST_PAIRS),
        accuracy=correct / len(truths),
        both_correct=both_correct,
        flipped=flipped,
        misses=misses,
    )


def compare_on_contrast() -> List[ContrastReport]:
    """Both approaches against prose neither was built on."""
    return [
        evaluate_on_contrast(ClinicalNoteReader().classifier, "tfidf + logistic"),
        evaluate_on_contrast(FramingLexiconBaseline().fit([]), "framing keyword lexicon"),
    ]


@dataclass
class AblationArm:
    """One configuration of extractors and what the harm model scored with it."""

    name: str
    average_precision: float
    cv_average_precision_mean: float
    roc_auc: float
    pr_baseline: float

    @property
    def pr_lift(self) -> float:
        return self.average_precision / self.pr_baseline if self.pr_baseline else float("nan")

    def summary(self) -> str:
        return (
            f"{self.name:<34} PR AUC {self.average_precision:.3f} "
            f"(cv {self.cv_average_precision_mean:.3f}, {self.pr_lift:.2f}x base) "
            f"ROC {self.roc_auc:.3f}"
        )


def run_ablation(
    cases: Optional[Sequence[PatientCase]] = None,
    model_type: str = "logistic",
    seed: int = 7,
) -> List[AblationArm]:
    """Price each half of the note layer against the structured watcher alone."""
    from src.data.generator import generate_dataset
    from src.engine.model import HarmScoringModel
    from src.engine.watcher import StructuredDataWatcher

    if cases is None:
        cases = generate_dataset(num_cases=600, harm_ratio=0.15, hard_negative_ratio=0.35, seed=seed)
    cases = list(cases)

    configurations = (
        ("watcher only", []),
        ("watcher + lexicon only", [LexiconOnlyNoteReader()]),
        ("watcher + full note reader", [ClinicalNoteReader()]),
    )

    arms: List[AblationArm] = []
    for name, note_extractors in configurations:
        model = HarmScoringModel(
            [StructuredDataWatcher(), *note_extractors],
            model_type=model_type,
            random_state=seed,
        )
        report = model.train(cases)
        arms.append(
            AblationArm(
                name=name,
                average_precision=report.average_precision,
                cv_average_precision_mean=report.cv_average_precision_mean,
                roc_auc=report.roc_auc,
                pr_baseline=report.pr_baseline,
            )
        )
    return arms
