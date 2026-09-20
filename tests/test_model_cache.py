from pathlib import Path

from typing import Tuple

from src.data.generator import generate_dataset
from src.engine.model import HarmScoringModel
from src.engine.model_cache import invalidate, load_cached, save_cached
from src.engine.watcher import StructuredDataWatcher
from src.domain.models import PatientCase


def _tiny_model() -> Tuple[HarmScoringModel, PatientCase]:
    cases = generate_dataset(num_cases=40, harm_ratio=0.3, hard_negative_ratio=0.2, seed=3)
    model = HarmScoringModel([StructuredDataWatcher()], model_type="logistic", threshold=0.4)
    model.train(cases)
    return model, cases[0]


def test_round_trip_preserves_score(tmp_path: Path) -> None:
    model, case = _tiny_model()
    before = model.predict_score(case)
    saved = save_cached("safetyhops", "logistic", model, directory=tmp_path)
    assert saved is not None
    loaded = load_cached("safetyhops", "logistic", directory=tmp_path)
    assert loaded is not None
    assert loaded.predict_score(case) == before


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_cached("safetyhops", "tree", directory=tmp_path) is None


def test_invalidate_removes_one_file(tmp_path: Path) -> None:
    model, _case = _tiny_model()
    save_cached("safetyhops", "logistic", model, directory=tmp_path)
    save_cached("safetyhops", "tree", model, directory=tmp_path)
    removed = invalidate("safetyhops", "logistic", directory=tmp_path)
    assert removed == 1
    assert load_cached("safetyhops", "logistic", directory=tmp_path) is None
    assert load_cached("safetyhops", "tree", directory=tmp_path) is not None
