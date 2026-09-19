from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from src.domain.models import PatientCase


@dataclass
class Contribution:
    """One feature's push on a single case's score, in log-odds."""

    feature: str
    value: float
    weight: float
    contribution: float
    description: str = ""

    @property
    def direction(self) -> str:
        return "raises" if self.contribution > 0 else "lowers"


class FeatureExtractor(ABC):
    """Turns a PatientCase into named numeric signals."""

    @abstractmethod
    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        ...

    @abstractmethod
    def feature_descriptions(self) -> Dict[str, str]:
        """Plain-English label per feature, shown to the reviewer."""
        ...

    def feature_names(self) -> List[str]:
        return sorted(self.feature_descriptions().keys())


class ScoringModel(ABC):
    """Interface for the machine learning scoring model."""

    @abstractmethod
    def train(self, cases: List[PatientCase]) -> None:
        ...

    @abstractmethod
    def predict_score(self, case: PatientCase) -> float:
        ...

    @abstractmethod
    def get_feature_weights(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def get_evidence(self, case: PatientCase) -> List[Contribution]:
        ...
