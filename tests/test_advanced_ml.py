from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import advanced_models
import train_advanced


class FakeClassifier:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, texts):
        return np.asarray([[1 - self.probability, self.probability] for _ in texts])


def test_advanced_models_import_without_optional_neural_libraries():
    assert callable(advanced_models.predict_advanced)
    assert callable(advanced_models.explain_transformer)
    assert "torch" not in sys.modules or not hasattr(advanced_models, "torch")


def test_probability_validation_and_prediction_thresholds(tmp_path):
    bundle = {
        "components": {"baseline": {"available": True}},
        "baseline": FakeClassifier(0.8),
    }
    advanced_models.save_bundle(bundle, tmp_path)

    result = advanced_models.predict_advanced(tmp_path, "Some sample article.", threshold=0.5)
    assert result["label"] == "FAKE"
    assert result["prob_fake"] == pytest.approx(0.8)
    assert result["decision_component"] == "baseline"
    assert result["provenance"]["inference_network_access"] is False

    uncertain = advanced_models.predict_advanced(
        tmp_path, "Some sample article.", threshold=0.76, uncertainty_margin=0.1
    )
    assert uncertain["label"] == "UNCERTAIN"

    insufficient = advanced_models.predict_advanced(
        tmp_path,
        "Some sample article.",
        threshold=0.5,
        uncertainty_margin=0,
        confidence_threshold=0.85,
    )
    assert insufficient["label"] == "Insufficient Confidence"

    low_confidence = advanced_models.predict_advanced(
        tmp_path,
        "Some sample article.",
        threshold=0.5,
        uncertainty_margin=0,
        confidence_threshold=0.9,
    )
    assert low_confidence["label"] == "Insufficient Confidence"

    with pytest.raises(ValueError, match="threshold"):
        advanced_models.predict_advanced(tmp_path, "article", threshold=1.1)
    with pytest.raises(ValueError, match="uncertainty_margin"):
        advanced_models.predict_advanced(tmp_path, "article", uncertainty_margin=0.6)
    with pytest.raises(ValueError, match="empty"):
        advanced_models.predict_advanced(tmp_path, "  ")
    with pytest.raises(TypeError, match="string"):
        advanced_models.predict_advanced(tmp_path, 10)


def test_checksum_is_verified_and_artifact_availability(tmp_path):
    assert not advanced_models.advanced_artifacts_available(tmp_path)
    advanced_models.save_bundle(
        {"components": {"baseline": {"available": True}}, "baseline": FakeClassifier(0.4)},
        tmp_path,
    )
    assert advanced_models.advanced_artifacts_available(tmp_path)
    bundle_path = tmp_path / advanced_models.BUNDLE_FILENAME
    bundle_path.write_bytes(bundle_path.read_bytes() + b"corruption")
    assert not advanced_models.advanced_artifacts_available(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        advanced_models.predict_advanced(tmp_path, "article")


def test_component_availability_checks_required_files_and_prediction_is_selective(
    tmp_path, monkeypatch
):
    transformer_dir = tmp_path / "transformer"
    (transformer_dir / "tokenizer").mkdir(parents=True)
    (transformer_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    encoder_dir = transformer_dir / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text("{}", encoding="utf-8")
    (encoder_dir / "model.safetensors").write_bytes(b"test weights")
    (transformer_dir / "head.pt").write_bytes(b"test weights")
    sentence_dir = tmp_path / "sentence_transformer"
    sentence_dir.mkdir()
    (sentence_dir / "modules.json").write_text(
        '[{"path":"","type":"sentence_transformers.models.Transformer"}]',
        encoding="utf-8",
    )
    (sentence_dir / "config.json").write_text("{}", encoding="utf-8")
    (sentence_dir / "model.safetensors").write_bytes(b"test weights")
    (sentence_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    bundle = {
        "components": {
            "baseline": {"available": True},
            "embedding": {"available": True},
            "transformer": {"available": True},
        },
        "baseline": FakeClassifier(0.8),
        "embedding_classifier": FakeClassifier(0.6),
        "neural_config": {"hidden_size": 4},
    }
    advanced_models.save_bundle(bundle, tmp_path)

    availability = advanced_models.advanced_component_availability(tmp_path)
    assert availability["baseline"] is True
    assert availability["embedding"] is True
    assert availability["transformer"] is True
    assert availability["fusion"] is False
    assert availability["ensemble"] is False

    def unexpected_component_call(*args, **kwargs):
        raise AssertionError("unselected component was evaluated")

    monkeypatch.setattr(advanced_models, "_predict_embedding", unexpected_component_call)
    monkeypatch.setattr(advanced_models, "_predict_neural", unexpected_component_call)
    result = advanced_models.predict_advanced(
        tmp_path, "A sample article.", component="baseline"
    )
    assert result["decision_component"] == "baseline"
    assert result["raw_components"] == {"baseline": pytest.approx(0.8)}


def test_incomplete_transformer_is_not_available(tmp_path):
    advanced_models.save_bundle(
        {
            "components": {
                "baseline": {"available": True},
                "transformer": {"available": True},
            },
            "baseline": FakeClassifier(0.4),
            "neural_config": {"hidden_size": 4},
        },
        tmp_path,
    )

    availability = advanced_models.advanced_component_availability(tmp_path)
    assert availability["baseline"] is True
    assert availability["transformer"] is False


def test_sentence_transformer_availability_rejects_empty_module_manifest(tmp_path):
    sentence_dir = tmp_path / "sentence_transformer"
    sentence_dir.mkdir()
    (sentence_dir / "modules.json").write_text("[]", encoding="utf-8")
    advanced_models.save_bundle(
        {
            "components": {"embedding": {"available": True}},
            "embedding_classifier": FakeClassifier(0.4),
        },
        tmp_path,
    )

    assert advanced_models.advanced_component_availability(tmp_path)["embedding"] is False


def test_metrics_and_calibration_have_expected_shapes():
    truth = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    calibration = advanced_models.calibration_bins(truth, probabilities, n_bins=4)
    assert calibration["n_bins"] == 4
    assert len(calibration["bins"]) == 4
    assert 0 <= calibration["ece"] <= 1
    assert advanced_models.expected_calibration_error(truth, probabilities, 4) == pytest.approx(
        calibration["ece"]
    )

    metrics = advanced_models.evaluate_probabilities(truth, probabilities, threshold=0.5, n_bins=4)
    assert metrics["accuracy"] == 1
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["f1"] == 1
    assert metrics["roc_auc"] == 1
    assert metrics["pr_auc"] == 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert 0 <= metrics["brier_score"] <= 1
    assert len(metrics["calibration_bins"]) == 4

    with pytest.raises(ValueError, match="equal length"):
        advanced_models.calibration_bins([0], [0.2, 0.8])


def test_platt_calibration_returns_finite_probability_scores():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    raw_probabilities = np.asarray([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    calibrator = advanced_models.fit_platt_calibrator(labels, raw_probabilities)
    calibrated = advanced_models.apply_platt_calibrator(calibrator, raw_probabilities)
    assert calibrated.shape == raw_probabilities.shape
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    assert np.all(np.diff(calibrated) > 0)
    assert np.array_equal(
        advanced_models.apply_platt_calibrator(None, raw_probabilities),
        raw_probabilities,
    )


def test_platt_calibration_rejects_single_class_validation_data():
    with pytest.raises(ValueError, match="both classes"):
        advanced_models.fit_platt_calibrator([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])


def test_stratified_split_removes_cross_label_and_same_label_duplicates():
    # Build enough unique examples for all three stratified partitions.
    texts = ["ambiguous story"]
    labels = [0]
    for index in range(20):
        texts.extend([f"real report {index}", f"fake report {index}"])
        labels.extend([0, 1])
    texts.extend(["AMBIGUOUS   STORY", "real report 0"])
    labels.extend([1, 0])
    split = advanced_models.split_labeled_texts(
        texts, labels, test_size=0.2, validation_size=0.2, random_state=7
    )
    assert split["removed_cross_label_duplicates"] == 2
    assert split["removed_duplicate_or_empty_rows"] == 1
    assert len(set(split["train_indices"]) & set(split["test_indices"])) == 0
    assert len(set(split["train_indices"]) & set(split["validation_indices"])) == 0
    assert len(set(split["test_indices"]) & set(split["validation_indices"])) == 0
    for indices in (
        split["train_indices"],
        split["validation_indices"],
        split["test_indices"],
    ):
        assert set(split["labels"][indices]) == {0, 1}


def test_augmentation_is_deterministic_and_retains_labels():
    original_texts = ["This is a remarkably useful article.", "Another detailed statement."]
    labels = np.asarray([0, 1])
    first = advanced_models.augment_training_texts(original_texts, labels, seed=3)
    second = advanced_models.augment_training_texts(original_texts, labels, seed=3)
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert first[0][:2] == original_texts
    assert set(first[1]) == {0, 1}


def test_cpu_training_defaults_and_stratified_neural_caps():
    args = train_advanced.parse_args([])
    assert args.epochs == 1
    assert args.batch_size == 4
    assert args.max_length == 128
    assert args.neural_train_limit == 1000
    assert args.neural_eval_limit == 300

    labels = np.asarray([0] * 100 + [1] * 100)
    selected = train_advanced._bounded_stratified_indices(labels, 40, random_state=9)
    assert len(selected) == 40
    assert np.bincount(labels[selected]).tolist() == [20, 20]


def test_training_cli_writes_baseline_artifacts_without_optional_dependencies(
    tmp_path, monkeypatch
):
    real_path = tmp_path / "real.csv"
    fake_path = tmp_path / "fake.csv"
    output_dir = tmp_path / "advanced"
    pd.DataFrame(
        {
            "title": [f"Real headline {index}" for index in range(40)],
            "text": [
                f"Real reporting describes public policy and community services detail {index}."
                for index in range(40)
            ],
        }
    ).to_csv(real_path, index=False)
    pd.DataFrame(
        {
            "title": [f"Fake headline {index}" for index in range(40)],
            "text": [
                f"Fake reporting repeats fabricated claims about secret events detail {index}."
                for index in range(40)
            ],
        }
    ).to_csv(fake_path, index=False)

    original_import = builtins.__import__

    def no_optional_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "sentence_transformers"}:
            raise ImportError("blocked by offline test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optional_import)
    train_advanced.main(
        [
            "--real",
            str(real_path),
            "--fake",
            str(fake_path),
            "--outdir",
            str(output_dir),
            "--min-df",
            "1",
            "--max-df",
            "1.0",
            "--max-features",
            "200",
            "--cv-folds",
            "2",
            "--neural-eval-limit",
            "8",
        ]
    )

    assert (output_dir / "advanced_metrics.json").is_file()
    assert (output_dir / "model_bundle.joblib").is_file()
    assert (output_dir / "model_bundle.joblib.sha256").is_file()
    metrics = json.loads(
        (output_dir / "advanced_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["components"]["baseline"]["available"] is True
    assert metrics["components"]["transformer"]["available"] is False
    assert metrics["holdout_test"]["baseline"]["status"] == "available"
    baseline_report = metrics["holdout_test"]["baseline"]
    assert metrics["protocol"]["neural_sample_counts"]["test"] == 8
    assert "raw_holdout_test" in baseline_report
    assert "calibrated_holdout_test" in baseline_report
    for score_report in (
        baseline_report["raw_holdout_test"],
        baseline_report["calibrated_holdout_test"],
    ):
        assert {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "confusion_matrix",
            "calibration_bins",
        }.issubset(score_report)
    assert (output_dir / "calibration_curve.png").is_file()
    assert advanced_models.advanced_artifacts_available(output_dir)
    prediction = advanced_models.predict_advanced(output_dir, "A sample article.")
    assert prediction["prob_fake"] >= 0
    assert prediction["raw_prob_fake"] >= 0
    assert prediction["calibration_method"] == "Platt scaling"
