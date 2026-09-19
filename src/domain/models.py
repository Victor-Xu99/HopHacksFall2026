from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

EVENT_TYPES = ("lab", "medication", "transfer", "procedure", "admission", "note")


@dataclass
class PatientEvent:
    event_id: str
    event_type: str  # one of EVENT_TYPES
    timestamp: str  # ISO 8601
    value: str  # lab name, drug name, unit, procedure name, note title
    details: str  # numeric result for labs, free text otherwise

    @property
    def when(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)

    @property
    def numeric_details(self) -> Optional[float]:
        """Lab results are stored as text; return the value when it parses."""
        try:
            return float(self.details.strip().split()[0])
        except (ValueError, IndexError, AttributeError):
            return None


@dataclass
class PatientCase:
    patient_id: str
    age: int
    gender: str
    events: List[PatientEvent] = field(default_factory=list)
    is_harm_event: bool = False  # Ground truth for training
    # Which scenario produced this case. Useful for error analysis in the demo,
    # never fed to the model.
    scenario: str = "routine"
    # "synthetic" for generated cases, "hand_labeled" for reviewer-adjudicated ones.
    label_source: str = "synthetic"

    def events_of_type(self, *event_types: str) -> List[PatientEvent]:
        return [e for e in self.events if e.event_type in event_types]

    def sorted_events(self) -> List[PatientEvent]:
        return sorted(self.events, key=lambda e: e.timestamp)
