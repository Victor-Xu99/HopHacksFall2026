"""Detects label leakage in a cohort before its scores are believed.

A trigger tool earns trust by being *imperfect* in a plausible way. Naloxone is
given to patients who turn out fine, and plenty of harm leaves no antidote
behind, so a signal that separates the label perfectly is evidence about the
dataset rather than about clinical reality.

Two checks, because leakage hides at two levels:

  perfect separation  a feature whose lift equals 1 / base_rate, meaning every
                      case it fires on carries the label
  giveaway values     a raw event string (a drug, a coded procedure) that maps
                      onto the label almost one-to-one, which is how a synthetic
                      generator's planting rule shows through the features

Neither check proves leakage on its own. A rare feature can separate perfectly
by chance, which is why `min_support` exists and why the report keeps counts
next to every rate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from src.domain.models import PatientCase
from src.engine.interfaces import FeatureExtractor

# A feature firing on fewer cases than this separates perfectly too easily for
# the result to mean anything.
MIN_SUPPORT = 30

# Lift is reported as a fraction of the ceiling (1 / base_rate). At 1.0 the
# feature is perfectly predictive; this is how close counts as suspicious.
SEPARATION_TOLERANCE = 0.995

# How pure a raw event string has to be before it reads as a planted tell.
GIVEAWAY_PURITY = 0.98


@dataclass
class FeatureFinding:
    feature: str
    support: int
    label_rate: float
    lift: float
    max_lift: float

    @property
    def separation(self) -> float:
        """1.0 when the feature is perfectly predictive of the label."""
        return self.lift / self.max_lift if self.max_lift else 0.0

    @property
    def leaking(self) -> bool:
        return self.support >= MIN_SUPPORT and self.separation >= SEPARATION_TOLERANCE


@dataclass
class ValueFinding:
    event_type: str
    value: str
    support: int
    label_rate: float


@dataclass
class LeakageReport:
    n_cases: int
    base_rate: float
    features: List[FeatureFinding] = field(default_factory=list)
    giveaways: List[ValueFinding] = field(default_factory=list)
    dead_features: List[str] = field(default_factory=list)

    @property
    def leaking_features(self) -> List[FeatureFinding]:
        return [f for f in self.features if f.leaking]

    @property
    def clean(self) -> bool:
        return not self.leaking_features and not self.giveaways

    def summary(self) -> str:
        if self.clean:
            return (
                f"No leakage detected across {len(self.features)} features "
                f"on {self.n_cases} cases."
            )
        return (
            f"{len(self.leaking_features)} of {len(self.features)} features separate the "
            f"label perfectly and {len(self.giveaways)} event values are near-pure. "
            f"Scores from this cohort measure the generator, not the clinical signal."
        )


def audit_features(
    cases: Sequence[PatientCase], extractor: FeatureExtractor
) -> Tuple[List[FeatureFinding], List[str]]:
    """Per-feature lift, plus the features this cohort cannot exercise at all."""
    total = len(cases)
    positives = sum(1 for c in cases if c.is_harm_event)
    base_rate = positives / total if total else 0.0
    # The most any feature can achieve: fire only on labeled cases.
    max_lift = 1 / base_rate if base_rate else 0.0

    fired: Counter = Counter()
    fired_labeled: Counter = Counter()
    for case in cases:
        for name, value in extractor.extract_features(case).items():
            if value > 0:
                fired[name] += 1
                if case.is_harm_event:
                    fired_labeled[name] += 1

    findings: List[FeatureFinding] = []
    dead: List[str] = []
    for name in extractor.feature_descriptions():
        support = fired.get(name, 0)
        if support == 0:
            dead.append(name)
            continue
        label_rate = fired_labeled[name] / support
        findings.append(
            FeatureFinding(
                feature=name,
                support=support,
                label_rate=label_rate,
                lift=label_rate / base_rate if base_rate else 0.0,
                max_lift=max_lift,
            )
        )
    findings.sort(key=lambda f: (-f.separation, -f.support))
    return findings, dead


def audit_values(
    cases: Sequence[PatientCase],
    event_types: Sequence[str] = ("medication", "procedure"),
    min_support: int = MIN_SUPPORT,
    purity: float = GIVEAWAY_PURITY,
) -> List[ValueFinding]:
    """Raw event strings that map onto the label almost one-to-one.

    Counted once per case, so a drug charted hourly does not outvote one charted
    once.
    """
    total: Counter = Counter()
    labeled: Counter = Counter()
    for case in cases:
        seen = {
            (e.event_type, e.value.strip())
            for e in case.events
            if e.event_type in event_types and e.value.strip()
        }
        for key in seen:
            total[key] += 1
            if case.is_harm_event:
                labeled[key] += 1

    findings = [
        ValueFinding(
            event_type=event_type,
            value=value,
            support=total[(event_type, value)],
            label_rate=labeled[(event_type, value)] / total[(event_type, value)],
        )
        for (event_type, value) in total
        if total[(event_type, value)] >= min_support
        and labeled[(event_type, value)] / total[(event_type, value)] >= purity
    ]
    findings.sort(key=lambda v: (-v.label_rate, -v.support))
    return findings


def audit(cases: Sequence[PatientCase], extractor: FeatureExtractor) -> LeakageReport:
    total = len(cases)
    features, dead = audit_features(cases, extractor)
    return LeakageReport(
        n_cases=total,
        base_rate=sum(1 for c in cases if c.is_harm_event) / total if total else 0.0,
        features=features,
        giveaways=audit_values(cases),
        dead_features=dead,
    )
