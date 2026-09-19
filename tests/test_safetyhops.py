import pytest

from src.data.safetyhops import HARM_CONDITIONS, hops_available, load_safetyhops_cases
from src.engine.watcher import StructuredDataWatcher

NEEDS_HOPS = pytest.mark.skipif(not hops_available(), reason="SafetyHops database not reachable")


@pytest.fixture(scope="module")
def hops_cases():
    return load_safetyhops_cases()


@NEEDS_HOPS
def test_loads_one_case_per_encounter(hops_cases):
    assert len(hops_cases) == 3500
    assert sum(c.is_harm_event for c in hops_cases) > 1000
    assert any(not c.is_harm_event for c in hops_cases)


@NEEDS_HOPS
def test_labels_come_from_harm_conditions_not_chronic_disease(hops_cases):
    harm = [c for c in hops_cases if c.is_harm_event]
    assert all(c.scenario in HARM_CONDITIONS for c in harm[:50])
    assert all(c.label_source == "safetyhops_condition_proxy" for c in hops_cases[:5])


@NEEDS_HOPS
def test_watcher_sees_labs_and_antidotes(hops_cases):
    watcher = StructuredDataWatcher(anchor="admission")
    fired = {name: 0 for name in watcher.feature_descriptions()}
    for case in hops_cases:
        for name, value in watcher.extract_features(case).items():
            if value > 0:
                fired[name] += 1
    assert fired["antidote_given"] > 0
    assert fired["abnormal_post_treatment_lab"] > 0
    assert fired["transfusion_given"] > 0
    assert fired["rapid_response_called"] == 0
