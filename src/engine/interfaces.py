from abc import ABC, abstractmethod
from typing import Dict, List
from src.domain.models import PatientCase

class FeatureExtractor(ABC):
    """Interface for extracting features from a PatientCase."""
    
    @abstractmethod
    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        pass

class ScoringModel(ABC):
    """Interface for the machine learning scoring model."""
    
    @abstractmethod
    def train(self, features: List[Dict[str, float]], labels: List[int]) -> None:
        pass

    @abstractmethod
    def predict_score(self, features: Dict[str, float]) -> float:
        pass

    @abstractmethod
    def get_feature_weights(self) -> Dict[str, float]:
        pass
