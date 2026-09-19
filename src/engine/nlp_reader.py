from typing import Dict
from src.domain.models import PatientCase
from src.engine.interfaces import FeatureExtractor

class ClinicalNoteReader(FeatureExtractor):
    """Scans free-text notes for harm-adjacent language."""
    
    def __init__(self):
        # In a real app, this would be an LLM or complex NLP model
        self.harm_keywords = [
            "unexpected complication",
            "worsening symptoms",
            "unplanned return",
            "acute distress"
        ]

    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        features = {
            "harm_language_score": 0.0
        }
        
        score = 0.0
        for event in case.events:
            if event.event_type == "note":
                text = event.details.lower()
                for kw in self.harm_keywords:
                    if kw in text:
                        score += 1.0
        
        features["harm_language_score"] = score
        return features
