from dataclasses import dataclass, field
from typing import List

@dataclass
class PatientEvent:
    event_id: str
    event_type: str  # 'lab', 'medication', 'transfer', 'note'
    timestamp: str
    value: str
    details: str

@dataclass
class PatientCase:
    patient_id: str
    age: int
    gender: str
    events: List[PatientEvent] = field(default_factory=list)
    is_harm_event: bool = False  # Ground truth for training
