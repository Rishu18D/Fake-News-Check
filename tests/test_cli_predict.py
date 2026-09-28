import joblib
import pytest

from detect_fake_news import classify_probability, main, predict_one
from train_model import build_pipeline


def test_classify_probability_uses_uncertain_band():
    assert classify_probability(0.30, threshold=0.5, uncertainty_margin=0.10) == "REAL"
    assert classify_probability(0.50, threshold=0.5, uncertainty_margin=0.10) == "UNCERTAIN"
    assert classify_probability(0.55, threshold=0.5, uncertainty_margin=0.10) == "UNCERTAIN"
    assert classify_probability(0.70, threshold=0.5, uncertainty_margin=0.10) == "FAKE"


def test_classify_probability_rejects_non_finite_settings():
    with pytest.raises(ValueError):
        classify_probability(0.5, threshold=float("nan"))
    with pytest.raises(ValueError):
        classify_probability(0.5, uncertainty_margin=float("inf"))


def test_predict_one_returns_expected_keys(tmp_path):
    pipeline = build_pipeline(max_features=100, min_df=1, max_df=1.0, ngram_max=1, C=1.0)
    pipeline.fit(
        [
            "reuters central bank announces policy",
            "official reuters government vote",
            "fake conspiracy celebrity rumor",
            "viral hoax miracle cure claim",
        ],
        [0, 0, 1, 1],
    )
    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    result = predict_one(model_path, "reuters policy announcement", 0.5)
    assert set(result) == {"label", "prob_fake", "threshold", "uncertainty_margin", "model_path"}
    assert result["label"] in {"REAL", "FAKE", "UNCERTAIN", "Insufficient Confidence"}
    assert 0.0 <= result["prob_fake"] <= 1.0

    low_confidence = predict_one(
        model_path,
        "reuters policy announcement",
        0.5,
        confidence_threshold=1.0,
    )
    assert low_confidence["label"] == "Insufficient Confidence"


def test_component_requires_advanced_model(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["detect_fake_news.py", "--component", "transformer", "--text", "A news story"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
