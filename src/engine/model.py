from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from src.engine.interfaces import ScoringModel, FeatureExtractor
from src.domain.models import PatientCase

class LogisticScoringModel(ScoringModel):
    def __init__(self, extractors: List[FeatureExtractor]):
        self.model = LogisticRegression(class_weight='balanced')
        self.extractors = extractors
        self.feature_names = []
        self.is_trained = False

    def _extract_all(self, case: PatientCase) -> Dict[str, float]:
        combined_features = {}
        for ext in self.extractors:
            combined_features.update(ext.extract_features(case))
        return combined_features

    def train(self, cases: List[PatientCase]) -> None:
        if not cases:
            return
            
        all_features_dicts = [self._extract_all(case) for case in cases]
        self.feature_names = sorted(list(all_features_dicts[0].keys()))
        
        X = []
        y = []
        for d, case in zip(all_features_dicts, cases):
            row = [d.get(f, 0.0) for f in self.feature_names]
            X.append(row)
            y.append(1 if case.is_harm_event else 0)
            
        self.model.fit(X, y)
        self.is_trained = True

    def predict_score(self, case: PatientCase) -> float:
        if not self.is_trained:
            return 0.0
        d = self._extract_all(case)
        row = [d.get(f, 0.0) for f in self.feature_names]
        return self.model.predict_proba([row])[0][1]

    def get_feature_weights(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        weights = self.model.coef_[0]
        return {name: float(weight) for name, weight in zip(self.feature_names, weights)}

    # Extra method to expose evidence for a specific case
    def get_evidence(self, case: PatientCase) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        features = self._extract_all(case)
        weights = self.get_feature_weights()
        evidence = {}
        for feature_name, val in features.items():
            if val > 0:
                evidence[feature_name] = val * weights.get(feature_name, 0.0)
        return evidence
