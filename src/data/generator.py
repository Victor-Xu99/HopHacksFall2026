import random
from faker import Faker
from src.domain.models import PatientCase, PatientEvent
from datetime import datetime, timedelta

fake = Faker()

def generate_case(is_harm: bool) -> PatientCase:
    patient_id = fake.uuid4()
    age = random.randint(18, 90)
    gender = random.choice(["M", "F"])
    
    events = []
    base_time = datetime.now() - timedelta(days=random.randint(1, 30))
    
    # Normal events
    events.append(PatientEvent(
        event_id=fake.uuid4(),
        event_type="medication",
        timestamp=(base_time + timedelta(hours=1)).isoformat(),
        value="Acetaminophen",
        details="Standard pain management"
    ))
    
    events.append(PatientEvent(
        event_id=fake.uuid4(),
        event_type="lab",
        timestamp=(base_time + timedelta(hours=2)).isoformat(),
        value="WBC",
        details=str(round(random.uniform(4.5, 10.0), 1))
    ))
    
    events.append(PatientEvent(
        event_id=fake.uuid4(),
        event_type="note",
        timestamp=(base_time + timedelta(hours=24)).isoformat(),
        value="Discharge Note",
        details="Patient recovering well, no acute distress."
    ))

    if is_harm:
        # Inject harm signals
        harm_type = random.choice(["naloxone", "icu_transfer", "note_flag"])
        
        if harm_type == "naloxone":
            events.append(PatientEvent(
                event_id=fake.uuid4(),
                event_type="medication",
                timestamp=(base_time + timedelta(hours=12)).isoformat(),
                value="Naloxone",
                details="Given for suspected opioid overdose."
            ))
        elif harm_type == "icu_transfer":
            events.append(PatientEvent(
                event_id=fake.uuid4(),
                event_type="transfer",
                timestamp=(base_time + timedelta(hours=14)).isoformat(),
                value="ICU",
                details="Unplanned transfer due to decompensation."
            ))
        elif harm_type == "note_flag":
            events.append(PatientEvent(
                event_id=fake.uuid4(),
                event_type="note",
                timestamp=(base_time + timedelta(hours=18)).isoformat(),
                value="Progress Note",
                details="Unexpected complication. Patient experiencing worsening symptoms not consistent with expected recovery."
            ))

    return PatientCase(
        patient_id=patient_id,
        age=age,
        gender=gender,
        events=events,
        is_harm_event=is_harm
    )

def generate_dataset(num_cases: int, harm_ratio: float = 0.2) -> list[PatientCase]:
    cases = []
    for _ in range(num_cases):
        is_harm = random.random() < harm_ratio
        cases.append(generate_case(is_harm))
    return cases
