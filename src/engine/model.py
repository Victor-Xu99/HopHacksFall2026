"""The scoring layer: weights are learned, not hand-picked.

Signals from the watchers go in as a feature matrix and a labeled sample teaches
the model how much each one is worth. Two model families are supported, both
chosen because you can read their reasoning off the fitted object:

  logistic  - coefficients are log-odds per unit of feature, so a case's score
              decomposes exactly into per-feature contributions that sum to the
              logit. That decomposition is what the review queue displays.
  tree      - a shallow decision tree; the explanation is the path the case took,
              with each split's change in predicted risk attributed to it.

Performance numbers always come from a held-out split, plus stratified
cross-validation because a hand-labeled sample is usually small.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from src.domain.models import PatientCase
from src.engine.interfaces import Contribution, FeatureExtractor, ScoringModel

MODEL_TYPES = ("logistic", "tree")


@dataclass
class TrainingReport:
    """Held-out performance, so the confidence score is defensible."""

    model_type: str
    n_train: int
    n_test: int
    positive_rate: float
    roc_auc: float
    average_precision: float
    brier: float
    precision: float
    recall: float
    f1: float
    threshold: float
    confusion: Tuple[int, int, int, int]  # tn, fp, fn, tp
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_average_precision_mean: float = float("nan")
    cv_average_precision_std: float = float("nan")
    threshold_sweep: List[Dict[str, float]] = field(default_factory=list)

    @property
    def pr_baseline(self) -> float:
        """What PR AUC a coin weighted to the base rate would score.

        ROC AUC always nulls at 0.50; PR AUC nulls at the positive rate, so the
        same 0.74 is unremarkable on a balanced cohort and strong on a rare one.
        Reporting the number without this is reporting half of it.
        """
        return self.positive_rate

    @property
    def pr_lift(self) -> float:
        """PR AUC over its own baseline. 1.0 means no better than chance."""
        return self.average_precision / self.positive_rate if self.positive_rate else float("nan")

    def summary(self) -> str:
        return (
            f"{self.model_type} | held-out ROC AUC {self.roc_auc:.3f} | "
            f"PR AUC {self.average_precision:.3f} "
            f"({self.pr_lift:.2f}x over a {self.pr_baseline:.1%} base rate) | "
            f"recall {self.recall:.2f} at threshold {self.threshold:.2f}"
        )


class HarmScoringModel(ScoringModel):
    def __init__(
        self,
        extractors: List[FeatureExtractor],
        model_type: str = "logistic",
        threshold: float = 0.4,
        max_depth: int = 5,
        random_state: int = 7,
    ):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")

        self.extractors = extractors
        self.model_type = model_type
        self.threshold = threshold
        self.random_state = random_state
        self.feature_names: List[str] = []
        self.is_trained = False
        self.report: Optional[TrainingReport] = None

        if model_type == "logistic":
            self.model = LogisticRegression(
                class_weight="balanced", max_iter=2000, C=1.0, random_state=random_state
            )
        else:
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=random_state,
            )

        self._descriptions: Dict[str, str] = {}
        for extractor in extractors:
            self._descriptions.update(extractor.feature_descriptions())

    # ------------------------------------------------------------------ features

    def _extract_all(self, case: PatientCase) -> Dict[str, float]:
        combined: Dict[str, float] = {}
        for extractor in self.extractors:
            combined.update(extractor.extract_features(case))
        return combined

    def _row(self, features: Dict[str, float]) -> List[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def build_matrix(self, cases: Sequence[PatientCase]) -> Tuple[np.ndarray, np.ndarray]:
        dicts = [self._extract_all(case) for case in cases]
        if not self.feature_names:
            names: set = set()
            for d in dicts:
                names.update(d.keys())
            self.feature_names = sorted(names)
        X = np.array([self._row(d) for d in dicts], dtype=float)
        y = np.array([1 if case.is_harm_event else 0 for case in cases], dtype=int)
        return X, y

    # ------------------------------------------------------------------ training

    def train(self, cases: List[PatientCase], test_size: float = 0.3) -> Optional[TrainingReport]:
        if len(cases) < 10:
            raise ValueError("Need at least 10 labeled cases to fit and evaluate a model.")

        X, y = self.build_matrix(cases)
        if len(set(y)) < 2:
            raise ValueError("Labeled sample contains only one class; cannot learn weights.")

        stratify = y if min(np.bincount(y)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.report = self._evaluate(X, y, X_train, X_test, y_train, y_test)
        return self.report

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingReport:
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

        both_classes_present = len(set(y_test)) > 1
        roc = float(roc_auc_score(y_test, probabilities)) if both_classes_present else float("nan")
        ap = (
            float(average_precision_score(y_test, probabilities))
            if both_classes_present
            else float("nan")
        )

        folds = min(5, int(np.bincount(y).min()))
        if folds >= 2:
            # Both metrics from one pass over the folds. PR AUC needs the
            # cross-validated form more than ROC does: it is the noisier of the
            # two on a small positive class, which is exactly when it is read.
            cv = cross_validate(
                self._fresh_estimator(),
                X,
                y,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state),
                scoring=("roc_auc", "average_precision"),
            )
            cv_mean = float(cv["test_roc_auc"].mean())
            cv_std = float(cv["test_roc_auc"].std())
            cv_ap_mean = float(cv["test_average_precision"].mean())
            cv_ap_std = float(cv["test_average_precision"].std())
        else:
            cv_mean = cv_std = cv_ap_mean = cv_ap_std = float("nan")

        sweep = []
        for cut in np.arange(0.1, 0.95, 0.05):
            preds = (probabilities >= cut).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y_test, preds, average="binary", zero_division=0
            )
            sweep.append(
                {
                    "threshold": round(float(cut), 2),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "flagged": float(preds.sum()),
                }
            )

        return TrainingReport(
            model_type=self.model_type,
            n_train=int(len(y_train)),
            n_test=int(len(y_test)),
            positive_rate=float(y.mean()),
            roc_auc=roc,
            average_precision=ap,
            brier=float(brier_score_loss(y_test, probabilities)),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            threshold=self.threshold,
            confusion=(int(tn), int(fp), int(fn), int(tp)),
            cv_roc_auc_mean=cv_mean,
            cv_roc_auc_std=cv_std,
            cv_average_precision_mean=cv_ap_mean,
            cv_average_precision_std=cv_ap_std,
            threshold_sweep=sweep,
        )

    def _fresh_estimator(self):
        params = self.model.get_params()
        return type(self.model)(**params)

    # ------------------------------------------------------------------ scoring

    def predict_score(self, case: PatientCase) -> float:
        if not self.is_trained:
            return 0.0
        row = np.array([self._row(self._extract_all(case))], dtype=float)
        return float(self.model.predict_proba(row)[0][1])

    def get_feature_weights(self) -> Dict[str, float]:
        """Log-odds coefficients for logistic, split importances for the tree."""
        if not self.is_trained:
            return {}
        if self.model_type == "logistic":
            values = self.model.coef_[0]
        else:
            values = self.model.feature_importances_
        return {name: float(v) for name, v in zip(self.feature_names, values)}

    @property
    def intercept(self) -> float:
        if not self.is_trained or self.model_type != "logistic":
            return 0.0
        return float(self.model.intercept_[0])

    def get_evidence(self, case: PatientCase) -> List[Contribution]:
        """Per-feature push on this case's score, largest magnitude first."""
        if not self.is_trained:
            return []
        features = self._extract_all(case)
        if self.model_type == "logistic":
            contributions = self._logistic_contributions(features)
        else:
            contributions = self._tree_contributions(features)
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return contributions

    def score_breakdown(self, case: PatientCase) -> Dict[str, float]:
        """Baseline + contributions = logit, so the displayed reasoning is checkable."""
        contributions = self.get_evidence(case)
        total = sum(c.contribution for c in contributions)
        probability = self.predict_score(case)
        if self.model_type == "logistic":
            return {
                "baseline_log_odds": self.intercept,
                "sum_of_contributions": total,
                "logit": self.intercept + total,
                "probability": probability,
            }
        return {"sum_of_path_changes": total, "probability": probability}

    def _logistic_contributions(self, features: Dict[str, float]) -> List[Contribution]:
        weights = self.get_feature_weights()
        out: List[Contribution] = []
        for name in self.feature_names:
            value = float(features.get(name, 0.0))
            weight = weights.get(name, 0.0)
            product = value * weight
            if value == 0.0 or product == 0.0:
                continue  # a feature that did not fire says nothing about this case
            out.append(
                Contribution(
                    feature=name,
                    value=value,
                    weight=weight,
                    contribution=product,
                    description=self._descriptions.get(name, name),
                )
            )
        return out

    def _tree_contributions(self, features: Dict[str, float]) -> List[Contribution]:
        """Attribute each step of the decision path to the feature it split on."""
        row = np.array([self._row(features)], dtype=float)
        tree = self.model.tree_
        path = self.model.decision_path(row).indices

        def node_risk(node_id: int) -> float:
            counts = tree.value[node_id][0]
            total = counts.sum()
            return float(counts[1] / total) if total else 0.0

        aggregated: Dict[str, float] = {}
        for parent, child in zip(path[:-1], path[1:]):
            feature_index = tree.feature[parent]
            if feature_index < 0:
                continue
            name = self.feature_names[feature_index]
            aggregated[name] = aggregated.get(name, 0.0) + (node_risk(child) - node_risk(parent))

        out: List[Contribution] = []
        for name, delta in aggregated.items():
            if abs(delta) < 1e-9:
                continue
            value = float(features.get(name, 0.0))
            description = self._descriptions.get(name, name)
            if value == 0.0:
                # A tree branches on absence too, and that is part of the reasoning.
                description = f"{description} — absent"
            out.append(
                Contribution(
                    feature=name,
                    value=value,
                    weight=delta,
                    contribution=delta,
                    description=description,
                )
            )
        return out

    # ------------------------------------------------------------------ display

    def feature_table(self) -> List[Dict[str, object]]:
        weights = self.get_feature_weights()
        label = "log_odds_weight" if self.model_type == "logistic" else "split_importance"
        rows = [
            {
                "feature": name,
                "meaning": self._descriptions.get(name, name),
                label: round(weight, 4),
            }
            for name, weight in weights.items()
        ]
        rows.sort(key=lambda r: abs(float(r[label])), reverse=True)
        return rows

    def describe_rules(self) -> str:
        """Tree structure as text; the logistic model has no branching to show."""
        if not self.is_trained or self.model_type != "tree":
            return ""
        return export_text(self.model, feature_names=list(self.feature_names), decimals=2)


class LogisticScoringModel(HarmScoringModel):
    """Backwards-compatible alias for the logistic configuration."""

    def __init__(self, extractors: List[FeatureExtractor], **kwargs):
        kwargs.pop("model_type", None)
        super().__init__(extractors, model_type="logistic", **kwargs)
