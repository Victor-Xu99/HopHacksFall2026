from src.data.prevalence import OIG_PREVENTABLE_HARM_RATE
from src.honesty import POINTS, as_json, train_source_id


def test_hospital_trains_on_synthetic_not_itself() -> None:
    assert train_source_id("hospital") == "synthetic"
    assert train_source_id("safetyhops") == "safetyhops"


def test_story_covers_leakage_and_oig_rate() -> None:
    payload = as_json("safetyhops")
    text = payload["headline"] + " ".join(point["body"] for point in payload["points"])
    assert "0.95" in text or "0.80" in text
    assert f"{OIG_PREVENTABLE_HARM_RATE:.0%}" in text
    assert payload["train_source"] == "safetyhops"
    assert payload["score_source"] == "safetyhops"
    kinds = [chip["kind"] for chip in payload["chips"]]
    assert kinds == ["score", "train", "rate"]
    assert len(POINTS) == 3


def test_hospital_chips_name_synthetic_training() -> None:
    payload = as_json("hospital")
    assert payload["train_source"] == "synthetic"
    values = {chip["kind"]: chip["value"] for chip in payload["chips"]}
    assert "Hospital" in values["score"]
    assert "Synthetic" in values["train"]
