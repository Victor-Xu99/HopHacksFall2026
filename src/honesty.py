"""Judge-facing copy about how the numbers were made.

Presentation only. Scoring still trains on the selected source, except hospital
extracts, which have no labels and reuse the in-memory synthetic cohort.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.data.prevalence import OIG_PREVENTABLE_HARM_RATE

HOSPITAL_SCORE_SOURCE = "hospital"
HOSPITAL_TRAIN_SOURCE = "synthetic"

SOURCE_LABELS: Dict[str, str] = {
    "hospital": "Hospital extract",
    "synthetic": "Synthetic (in memory)",
    "safetyhops": "SafetyHops",
    "mimic": "MIMIC-IV demo (CSV)",
    "sql_mimic": "MIMIC-IV demo (SQL)",
    "sql_synthetic": "Synthetic (SQL)",
}

HEADLINE = (
    "We report numbers we can defend: leaked labels were broken, the harm rate "
    "was thinned to the OIG figure, and unlabeled hospital charts are ranked "
    "with a model trained on a labeled cohort."
)


@dataclass(frozen=True)
class HonestyPoint:
    title: str
    body: str


@dataclass(frozen=True)
class SourceChip:
    kind: str
    label: str
    value: str


POINTS = [
    HonestyPoint(
        title="We caught our own leakage",
        body=(
            "SafetyHops first scored ROC AUC 0.95 because every harm case carried "
            "its own evidence. We broke that tie (silent harm and benign triggers) "
            "and now report about 0.80."
        ),
    ),
    HonestyPoint(
        title="We fixed the base rate",
        body=(
            f"The warehouse plants harm at 46%. We thin SafetyHops to "
            f"{OIG_PREVENTABLE_HARM_RATE:.0%} — the OIG preventable-harm figure — "
            "by dropping cases, never duplicating them."
        ),
    ),
    HonestyPoint(
        title="Unlabeled charts still get a ranked list",
        body=(
            "Hospital extracts have no harm tags. Those stays are ranked with a "
            "model trained on the labeled synthetic cohort, then a person decides."
        ),
    ),
]


def train_source_id(score_source: str) -> str:
    if score_source == HOSPITAL_SCORE_SOURCE:
        return HOSPITAL_TRAIN_SOURCE
    return score_source


def source_label(source_id: str) -> str:
    return SOURCE_LABELS.get(source_id, source_id)


def chips_for(score_source: str) -> List[SourceChip]:
    train = train_source_id(score_source)
    items = [
        SourceChip("score", "Scoring", source_label(score_source)),
        SourceChip("train", "Trained on", source_label(train)),
    ]
    if score_source == "safetyhops":
        items.append(
            SourceChip(
                "rate",
                "Training harm rate",
                f"{OIG_PREVENTABLE_HARM_RATE:.0%} (OIG preventable)",
            )
        )
    return items


def as_json(score_source: str = "safetyhops") -> Dict[str, Any]:
    train = train_source_id(score_source)
    return {
        "headline": HEADLINE,
        "points": [asdict(point) for point in POINTS],
        "chips": [asdict(chip) for chip in chips_for(score_source)],
        "train_source": train,
        "score_source": score_source,
    }
