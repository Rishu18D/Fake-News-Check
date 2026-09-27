#!/usr/bin/env python3
"""Train optional neural and embedding models alongside the existing baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from advanced_models import (
    _build_fusion_network,
    _pool_hidden,
    augment_training_texts,
    evaluate_probabilities,
    save_bundle,
    split_labeled_texts,
)
from text_clean import clean_text
from train_model import build_pipeline, pick_text_column, read_csv_any


def _read_dataset(real_path: Path, fake_path: Path, text_column: str, include_title: bool):
    real = read_csv_any(real_path)
    fake = read_csv_any(fake_path)
    real_column = pick_text_column(real, text_column)
    fake_column = pick_text_column(fake, text_column)
    if real_column != fake_column:
        raise ValueError(
            f"Real and fake files resolved to different text columns: "
            f"{real_column!r} vs {fake_column!r}"
        )
    frames = []
    for frame, label in ((real, 0), (fake, 1)):
        article = frame[real_column].fillna("").astype(str)
        if include_title and "title" in frame.columns:
            article = (frame["title"].fillna("").astype(str) + " " + article).str.strip()
        frames.append((article.tolist(), [label] * len(article)))
    combined_texts = frames[0][0] + frames[1][0]
    combined_labels = np.asarray(frames[0][1] + frames[1][1], dtype=np.int64)
    usable = [(text, int(label)) for text, label in zip(combined_texts, combined_labels, strict=True) if clean_text(text)]
    return [item[0] for item in usable], np.asarray([item[1] for item in usable], dtype=np.int64)


def _write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _component_report(probabilities: np.ndarray | None, y_test: np.ndarray, threshold: float, reason: str | None = None):
    if probabilities is None:
        return {"status": "skipped", "reason": reason or "Component was not trained"}
    return {
        "status": "available",
        "holdout_test": evaluate_probabilities(y_test, probabilities, threshold),
    }


def _bounded_stratified_indices(
    labels: np.ndarray, limit: int, random_state: int
) -> np.ndarray:
    """Choose a reproducible, class-stratified subset for CPU-bound neural runs."""
    indices = np.arange(len(labels))
    if len(indices) <= limit:
        return indices
    _, selected = train_test_split(
        indices,
        test_size=limit,
        stratify=labels,
        random_state=random_state,
    )
    return np.sort(selected)


def _fit_baseline(
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
    test_texts: list[str],
    args: argparse.Namespace,
):
    estimator = build_pipeline(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        C=args.C,
    )
    minority_count = int(np.bincount(train_labels, minlength=2).min())
    folds = min(args.cv_folds, minority_count)
    if folds < 2:
        raise ValueError("At least two examples per class are required in the training split for CV")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.random_state)
    cv_payload: dict[str, Any] = {
        "protocol": "StratifiedKFold over training split only",
        "folds": folds,
        "scoring": ["accuracy", "f1_macro", "roc_auc"],
    }
    if args.search:
        search = GridSearchCV(
            estimator,
            {
                "tfidf__ngram_range": [(1, 1), (1, min(args.ngram_max, 2))],
                "classifier__C": [0.5, args.C, max(args.C * 2, args.C + 0.1)],
            },
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(train_texts, train_labels)
        estimator = search.best_estimator_
        cv_payload.update(
            {
                "search": "bounded GridSearchCV on training split only",
                "best_parameters": search.best_params_,
                "best_cv_macro_f1": float(search.best_score_),
            }
        )
    else:
        scores = cross_validate(
            estimator,
            train_texts,
            train_labels,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        cv_payload["mean_scores"] = {
            key.removeprefix("test_"): float(np.mean(value))
            for key, value in scores.items()
            if key.startswith("test_")
        }

    augmented_texts, augmented_labels = augment_training_texts(
        train_texts, train_labels, seed=args.random_state
    )
    estimator.fit(augmented_texts, augmented_labels)
    test_probabilities = estimator.predict_proba(test_texts)[:, 1]

    no_augmentation = build_pipeline(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        C=args.C,
    )
    no_augmentation.fit(train_texts, train_labels)
    no_aug_probabilities = no_augmentation.predict_proba(test_texts)[:, 1]
    validation_probabilities = estimator.predict_proba(validation_texts)[:, 1]
    return (
        estimator,
        test_probabilities,
        validation_probabilities,
        no_aug_probabilities,
        cv_payload,
        len(augmented_texts) - len(train_texts),
    )


def _require_neural_training():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Transformer training requires the optional packages in requirements-advanced.txt "
            "(torch and transformers)."
        ) from error
    return torch, AutoModel, AutoTokenizer


def _tokenize_many(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, Any]:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


def _batches(size: int, batch_size: int, shuffle: bool, torch: Any, generator: Any):
    indices = torch.randperm(size, generator=generator).tolist() if shuffle else list(range(size))
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def _fit_neural_model(
    model: Any,
    train_tokens: dict[str, Any],
    train_labels: np.ndarray,
    validation_tokens: dict[str, Any],
    validation_labels: np.ndarray,
    device: str,
    args: argparse.Namespace,
    train_sentences: Any = None,
    validation_sentences: Any = None,
):
    torch, _, _ = _require_neural_training()
    model.to(device)
    class_counts = np.bincount(train_labels, minlength=2)
    weights = torch.tensor(
        [len(train_labels) / (2 * max(int(count), 1)) for count in class_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_labels_tensor = torch.as_tensor(train_labels, dtype=torch.long)
    validation_labels_tensor = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(args.random_state)
    best_loss = float("inf")
    best_state = None
    for _epoch in range(args.epochs):
        model.train()
        for indices in _batches(
            len(train_labels), args.batch_size, True, torch, generator
        ):
            batch = {
                key: value[indices].to(device)
                for key, value in train_tokens.items()
                if key in ("input_ids", "attention_mask")
            }
            optimizer.zero_grad(set_to_none=True)
            if train_sentences is None:
                encoded = model.encoder(**batch)
                logits = model.classifier(_pool_hidden(encoded, batch["attention_mask"]))
            else:
                sentence_batch = train_sentences[indices].to(device)
                _, logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    sentence_embeddings=sentence_batch,
                )
            loss = criterion(logits, train_labels_tensor[indices].to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for indices in _batches(
                len(validation_labels), args.batch_size, False, torch, generator
            ):
                batch = {
                    key: value[indices].to(device)
                    for key, value in validation_tokens.items()
                    if key in ("input_ids", "attention_mask")
                }
                if validation_sentences is None:
                    encoded = model.encoder(**batch)
                    logits = model.classifier(_pool_hidden(encoded, batch["attention_mask"]))
                else:
                    _, logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        sentence_embeddings=validation_sentences[indices].to(device),
                    )
                validation_losses.append(
                    float(criterion(logits, validation_labels_tensor[indices]).item())
                )
        mean_validation_loss = float(np.mean(validation_losses))
        if mean_validation_loss < best_loss:
            best_loss = mean_validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def _neural_probabilities(
    model: Any,
    tokens: dict[str, Any],
    device: str,
    batch_size: int,
    torch: Any,
    sentence_vectors: Any = None,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for indices in _batches(len(tokens["input_ids"]), batch_size, False, torch, torch.Generator()):
            batch = {
                key: value[indices].to(device)
                for key, value in tokens.items()
                if key in ("input_ids", "attention_mask")
            }
            if sentence_vectors is None:
                encoded = model.encoder(**batch)
                logits = model.classifier(_pool_hidden(encoded, batch["attention_mask"]))
            else:
                _, logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    sentence_embeddings=sentence_vectors[indices].to(device),
                )
            predictions.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
    return np.asarray(predictions, dtype=float)


def _train_transformer(
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
    validation_labels: np.ndarray,
    test_texts: list[str],
    artifact_dir: Path,
    args: argparse.Namespace,
):
    torch, AutoModel, AutoTokenizer = _require_neural_training()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_checkpoint)
    train_tokens = _tokenize_many(tokenizer, train_texts, args.max_length)
    validation_tokens = _tokenize_many(tokenizer, validation_texts, args.max_length)
    test_tokens = _tokenize_many(tokenizer, test_texts, args.max_length)
    encoder = AutoModel.from_pretrained(args.transformer_checkpoint)

    class TransformerClassifier(torch.nn.Module):
        def __init__(self, base_encoder: Any):
            super().__init__()
            self.encoder = base_encoder
            self.classifier = torch.nn.Linear(int(base_encoder.config.hidden_size), 2)

    model = TransformerClassifier(encoder)
    model, validation_loss = _fit_neural_model(
        model,
        train_tokens,
        train_labels,
        validation_tokens,
        validation_labels,
        device,
        args,
    )
    probabilities = _neural_probabilities(
        model, test_tokens, device, args.batch_size, torch
    )
    validation_probabilities = _neural_probabilities(
        model, validation_tokens, device, args.batch_size, torch
    )
    model_directory = artifact_dir / "transformer"
    model_directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(model_directory / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(model_directory / "tokenizer")
    torch.save(model.classifier.state_dict(), model_directory / "head.pt")
    return {
        "test_probabilities": probabilities,
        "validation_probabilities": validation_probabilities,
        "validation_loss": validation_loss,
        "model_directory": "transformer",
        "device": device,
        "hidden_size": int(model.encoder.config.hidden_size),
    }


def _train_sentence_embedding(
    sentence_model: Any,
    texts_by_split: dict[str, list[str]],
    labels: dict[str, np.ndarray],
    test_texts: list[str],
    device: str,
    batch_size: int,
    artifact_dir: Path,
):
    encoded = {
        split: np.asarray(
            sentence_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        for split, texts in texts_by_split.items()
    }
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42, C=1.0
    )
    classifier.fit(encoded["train"], labels["train"])
    test_probabilities = classifier.predict_proba(encoded["test"])[:, 1]
    validation_probabilities = classifier.predict_proba(encoded["validation"])[:, 1]
    model_directory = artifact_dir / "sentence_transformer"
    sentence_model.save(str(model_directory))
    return classifier, test_probabilities, validation_probabilities, encoded, model_directory


def _train_fusion(
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
    validation_labels: np.ndarray,
    test_texts: list[str],
    sentence_vectors: dict[str, np.ndarray],
    artifact_dir: Path,
    args: argparse.Namespace,
):
    torch, AutoModel, AutoTokenizer = _require_neural_training()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_checkpoint)
    train_tokens = _tokenize_many(tokenizer, train_texts, args.max_length)
    validation_tokens = _tokenize_many(tokenizer, validation_texts, args.max_length)
    test_tokens = _tokenize_many(tokenizer, test_texts, args.max_length)
    encoder = AutoModel.from_pretrained(args.transformer_checkpoint)
    hidden_size = int(encoder.config.hidden_size)
    sentence_size = int(sentence_vectors["train"].shape[1])
    fusion_size = min(256, hidden_size)
    fusion_class = _build_fusion_network(
        torch, torch.nn, encoder, hidden_size, sentence_size, fusion_size
    )
    model = fusion_class()
    model, validation_loss = _fit_neural_model(
        model,
        train_tokens,
        train_labels,
        validation_tokens,
        validation_labels,
        device,
        args,
        train_sentences=torch.as_tensor(sentence_vectors["train"], dtype=torch.float32),
        validation_sentences=torch.as_tensor(
            sentence_vectors["validation"], dtype=torch.float32
        ),
    )
    test_probability = _neural_probabilities(
        model,
        test_tokens,
        device,
        args.batch_size,
        torch,
        torch.as_tensor(sentence_vectors["test"], dtype=torch.float32),
    )
    validation_probability = _neural_probabilities(
        model,
        validation_tokens,
        device,
        args.batch_size,
        torch,
        torch.as_tensor(sentence_vectors["validation"], dtype=torch.float32),
    )
    model_directory = artifact_dir / "fusion_model"
    model_directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(model_directory / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(model_directory / "tokenizer")
    torch.save(model.state_dict(), model_directory / "fusion.pt")
    return {
        "test_probabilities": test_probability,
        "validation_probabilities": validation_probability,
        "validation_loss": validation_loss,
        "model_directory": "fusion_model",
        "device": device,
        "hidden_size": hidden_size,
        "sentence_size": sentence_size,
        "fusion_size": fusion_size,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the optional advanced local fake-news classifiers."
    )
    parser.add_argument("--real", default="data/True.csv", help="CSV containing REAL news.")
    parser.add_argument("--fake", default="data/Fake.csv", help="CSV containing FAKE news.")
    parser.add_argument("--text-col", default="text", help="Preferred article text column.")
    parser.add_argument("--outdir", default="outputs/advanced", help="Advanced artifacts directory.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--ngram-max", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--C", type=float, default=2.0)
    parser.add_argument("--search", action="store_true", help="Run bounded baseline GridSearchCV on training data.")
    parser.add_argument(
        "--transformer-checkpoint",
        default="distilroberta-base",
        help="Hugging Face sequence encoder checkpoint; downloaded at training time if absent.",
    )
    parser.add_argument(
        "--sentence-checkpoint",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint; downloaded at training time if absent.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--neural-train-limit",
        type=int,
        default=1000,
        help="Maximum stratified training samples for each CPU-expensive neural component.",
    )
    parser.add_argument(
        "--neural-eval-limit",
        type=int,
        default=300,
        help="Maximum stratified validation and holdout samples used for neural comparison.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-title", action="store_true", help="Ignore title columns.")
    args = parser.parse_args(argv)
    if (
        args.cv_folds < 2
        or args.epochs < 1
        or args.batch_size < 1
        or args.max_length < 8
        or args.neural_train_limit < 2
        or args.neural_eval_limit < 2
    ):
        parser.error(
            "cv-folds >= 2, epochs >= 1, batch-size >= 1, max-length >= 8, "
            "and neural sample limits >= 2 are required"
        )
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifact_dir = Path(args.outdir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_texts, source_labels = _read_dataset(
        Path(args.real), Path(args.fake), args.text_col, not args.no_title
    )
    split = split_labeled_texts(
        source_texts,
        source_labels,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    texts = split["texts"]
    labels = split["labels"]
    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]
    test_indices = split["test_indices"]
    train_texts = [texts[index] for index in train_indices]
    validation_texts = [texts[index] for index in validation_indices]
    test_texts = [texts[index] for index in test_indices]
    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]
    test_labels = labels[test_indices]
    neural_train_indices = _bounded_stratified_indices(
        train_labels, args.neural_train_limit, args.random_state
    )
    neural_validation_indices = _bounded_stratified_indices(
        validation_labels, args.neural_eval_limit, args.random_state + 1
    )
    neural_test_indices = _bounded_stratified_indices(
        test_labels, args.neural_eval_limit, args.random_state + 2
    )
    neural_train_texts = [train_texts[index] for index in neural_train_indices]
    neural_train_labels = train_labels[neural_train_indices]
    neural_validation_texts = [
        validation_texts[index] for index in neural_validation_indices
    ]
    neural_validation_labels = validation_labels[neural_validation_indices]
    neural_test_texts = [test_texts[index] for index in neural_test_indices]
    neural_test_labels = test_labels[neural_test_indices]
    component_state: dict[str, Any] = {
        "baseline": {"available": True},
        "embedding": {"available": False, "reason": "Optional component not trained"},
        "transformer": {"available": False, "reason": "Optional component not trained"},
        "fusion": {"available": False, "reason": "Optional component not trained"},
        "ensemble": {"available": False, "reason": "Requires baseline and transformer validation scores"},
    }

    baseline, baseline_test, baseline_validation, no_aug_test, cv, n_augmented = _fit_baseline(
        train_texts, train_labels, validation_texts, test_texts, args
    )
    components: dict[str, Any] = {"baseline": baseline}
    validation_probabilities: dict[str, np.ndarray] = {"baseline": baseline_validation}
    test_probabilities: dict[str, np.ndarray] = {"baseline": baseline_test}
    reports: dict[str, Any] = {
        "baseline": _component_report(
            baseline_test[neural_test_indices], neural_test_labels, args.threshold
        ),
        "transformer": _component_report(None, neural_test_labels, args.threshold, "Optional model was not trained"),
        "embedding": _component_report(None, neural_test_labels, args.threshold, "Optional model was not trained"),
        "fused": _component_report(None, neural_test_labels, args.threshold, "Optional model was not trained"),
        "ensemble": _component_report(None, neural_test_labels, args.threshold, "Transformer unavailable"),
        "adversarial_ablation": {
            "augmentation_off_baseline": evaluate_probabilities(
                neural_test_labels, no_aug_test[neural_test_indices], args.threshold
            ),
            "augmentation_on_baseline": evaluate_probabilities(
                neural_test_labels, baseline_test[neural_test_indices], args.threshold
            ),
            "training_examples_added": int(n_augmented),
            "validation_and_test_unchanged": True,
        },
    }
    sentence_model = None
    sentence_vectors = None
    sentence_failure = None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sentence_failure = (
            "sentence-transformers is not installed; install requirements-advanced.txt "
            "to train embedding and fusion components"
        )
        component_state["embedding"]["reason"] = sentence_failure
        component_state["fusion"]["reason"] = sentence_failure
        print(f"Skipping sentence embedding/fusion models: {sentence_failure}", file=sys.stderr)
    else:
        neural_device = "cuda" if _require_neural_training()[0].cuda.is_available() else "cpu"
        sentence_model = SentenceTransformer(args.sentence_checkpoint, device=neural_device)
        sentence_component, embedding_test, embedding_validation, sentence_vectors, _ = _train_sentence_embedding(
            sentence_model,
            {
                "train": neural_train_texts,
                "validation": neural_validation_texts,
                "test": neural_test_texts,
            },
            {
                "train": neural_train_labels,
                "validation": neural_validation_labels,
                "test": neural_test_labels,
            },
            neural_test_texts,
            neural_device,
            args.batch_size,
            artifact_dir,
        )
        components["embedding_classifier"] = sentence_component
        validation_probabilities["embedding"] = embedding_validation
        test_probabilities["embedding"] = embedding_test
        component_state["embedding"] = {"available": True}
        reports["embedding"] = _component_report(
            embedding_test, neural_test_labels, args.threshold
        )

    transformer_result = None
    try:
        transformer_result = _train_transformer(
            neural_train_texts,
            neural_train_labels,
            neural_validation_texts,
            neural_validation_labels,
            neural_test_texts,
            artifact_dir,
            args,
        )
    except ImportError as error:
        component_state["transformer"]["reason"] = str(error)
        component_state["fusion"]["reason"] = str(error)
        reports["transformer"] = _component_report(None, neural_test_labels, args.threshold, str(error))
        print(f"Skipping transformer/fusion models: {error}", file=sys.stderr)
    if transformer_result is not None:
        validation_probabilities["transformer"] = transformer_result["validation_probabilities"]
        test_probabilities["transformer"] = transformer_result["test_probabilities"]
        component_state["transformer"] = {"available": True}
        reports["transformer"] = _component_report(
            transformer_result["test_probabilities"], neural_test_labels, args.threshold
        )
        if sentence_model is not None and sentence_vectors is not None:
            try:
                fusion_result = _train_fusion(
                    neural_train_texts,
                    neural_train_labels,
                    neural_validation_texts,
                    neural_validation_labels,
                    neural_test_texts,
                    sentence_vectors,
                    artifact_dir,
                    args,
                )
            except ImportError as error:
                component_state["fusion"]["reason"] = str(error)
                reports["fused"] = _component_report(None, neural_test_labels, args.threshold, str(error))
            else:
                validation_probabilities["fusion"] = fusion_result["validation_probabilities"]
                test_probabilities["fusion"] = fusion_result["test_probabilities"]
                component_state["fusion"] = {"available": True}
                reports["fused"] = _component_report(
                    fusion_result["test_probabilities"], neural_test_labels, args.threshold
                )
                transformer_result["neural_config"] = {
                    "hidden_size": fusion_result["hidden_size"],
                    "sentence_size": fusion_result["sentence_size"],
                    "fusion_size": fusion_result["fusion_size"],
                    "max_length": args.max_length,
                }
        else:
            reports["fused"] = _component_report(
                None,
                neural_test_labels,
                args.threshold,
                sentence_failure or "Sentence-transformer component was unavailable",
            )

    if {"baseline", "transformer"}.issubset(validation_probabilities):
        ensemble = LogisticRegression(max_iter=1000, random_state=args.random_state)
        ensemble_validation = np.column_stack(
            (
                validation_probabilities["baseline"][neural_validation_indices],
                validation_probabilities["transformer"],
            )
        )
        ensemble.fit(ensemble_validation, neural_validation_labels)
        ensemble_test = ensemble.predict_proba(
            np.column_stack(
                (
                    test_probabilities["baseline"][neural_test_indices],
                    test_probabilities["transformer"],
                )
            )
        )[:, 1]
        components["ensemble"] = ensemble
        component_state["ensemble"] = {
            "available": True,
            "fit_on": "validation predictions from training-only component models",
        }
        reports["ensemble"] = _component_report(ensemble_test, neural_test_labels, args.threshold)

    bundle: dict[str, Any] = {
        "format_version": 1,
        "components": component_state,
        "component_paths": {
            "transformer": "transformer",
            "fusion_model": "fusion_model",
            "sentence_transformer": "sentence_transformer",
        },
        "baseline": baseline,
        "embedding_classifier": components.get("embedding_classifier"),
        "ensemble": components.get("ensemble"),
        "neural_config": {
            "hidden_size": transformer_result["hidden_size"] if transformer_result else None,
            "sentence_size": (
                transformer_result.get("neural_config", {}).get("sentence_size")
                if transformer_result
                else None
            ),
            "fusion_size": (
                transformer_result.get("neural_config", {}).get("fusion_size")
                if transformer_result
                else None
            ),
            "max_length": args.max_length,
        },
    }
    save_bundle(bundle, artifact_dir)
    metrics = {
        "protocol": {
            "split": "single deterministic stratified train/validation/test split",
            "random_state": args.random_state,
            "test_fraction": args.test_size,
            "validation_fraction": args.validation_size,
            "split_counts": {
                "train": len(train_indices),
                "validation": len(validation_indices),
                "test": len(test_indices),
            },
            "neural_sample_counts": {
                "train": len(neural_train_indices),
                "validation": len(neural_validation_indices),
                "test": len(neural_test_indices),
                "sampling": (
                    "Deterministic stratified caps for CPU-bound neural components; "
                    "baseline fits the complete training split and all comparison metrics "
                    "use the same sampled holdout observations."
                ),
            },
            "cross_label_duplicates_removed_before_split": split[
                "removed_cross_label_duplicates"
            ],
            "duplicate_or_empty_rows_removed_before_split": split[
                "removed_duplicate_or_empty_rows"
            ],
            "cross_validation": cv,
            "test_used_only_for_final_reporting": True,
            "validation_used_for": [
                "neural early stopping",
                "validation-only baseline/transformer ensemble fitting",
            ],
            "threshold_tuned_on_test": False,
            "class_imbalance": (
                "Baseline uses class_weight=balanced; neural losses use training-split "
                "class weights; embedding classifier uses class_weight=balanced. "
                "No oversampling crosses split boundaries."
            ),
            "compute_defaults": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_token_length": args.max_length,
                "max_neural_training_samples": args.neural_train_limit,
                "max_neural_validation_or_test_samples": args.neural_eval_limit,
            },
            "augmentation": (
                "Deterministic adjacent-character typos are appended to training examples "
                "only; validation and test texts are not modified."
            ),
            "model_downloads": (
                "Training may download open-source Hugging Face checkpoint weights on first "
                "run. Saved model, tokenizer, and sentence-embedding files are local; "
                "inference uses local_files_only."
            ),
            "device": (
                transformer_result["device"]
                if transformer_result
                else "cpu (neural components skipped)"
            ),
        },
        "holdout_test": reports,
        "components": component_state,
        "bundle_sha256": (artifact_dir / "model_bundle.joblib.sha256").read_text(
            encoding="ascii"
        ).split()[0],
        "versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    _write_json(metrics, artifact_dir / "advanced_metrics.json")
    print(f"Saved advanced bundle and metrics to {artifact_dir}")


if __name__ == "__main__":
    main()
