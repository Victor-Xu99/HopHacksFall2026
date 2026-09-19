from typing import Dict
from src.domain.models import PatientCase
from src.engine.interfaces import FeatureExtractor

class StructuredDataWatcher(FeatureExtractor):
    """Monitors structured data (labs, meds, transfers) for red flags."""
    
    def extract_features(self, case: PatientCase) -> Dict[str, float]:
        features = {
            "has_naloxone": 0.0,
            "has_icu_transfer": 0.0
        }
        
        for event in case.events:
            if event.event_type == "medication" and "naloxone" in event.value.lower():
                features["has_naloxone"] = 1.0
            elif event.event_type == "transfer" and "icu" in event.value.lower():
                features["has_icu_transfer"] = 1.0
                
        return features
