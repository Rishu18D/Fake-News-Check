"""Optional local-only advanced models and shared training utilities.

Heavy neural dependencies are deliberately imported only inside the functions that
need them so the baseline application remains usable without the optional stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

LABELS = ("REAL", "FAKE")
BUNDLE_FILENAME = "model_bundle.joblib"
SIDECAR_FILENAME = "model_bundle.joblib.sha256"


def _validate_probability_options(threshold: float, uncertainty_margin: float) -> None:
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1")
    if (
        not isinstance(uncertainty_margin, (int, float))
        or not math.isfinite(uncertainty_margin)
        or not 0 <= uncertainty_margin <= 0.5
    ):
        raise ValueError("uncertainty_margin must be a finite number between 0 and 0.5")


def _validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    return text.strip()


def calibration_bins(
    y_true: list[int] | np.ndarray,
    y_prob_fake: list[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return equal-width expected calibration error and per-bin diagnostics."""
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_prob_fake, dtype=float)
    if truth.ndim != 1 or probability.ndim != 1 or len(truth) != len(probability):
        raise ValueError("y_true and y_prob_fake must be one-dimensional arrays of equal length")
    if len(truth) == 0:
        raise ValueError("calibration metrics require at least one observation")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true values must be 0 (REAL) or 1 (FAKE)")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.digitize(probability, edges[1:-1], right=False), n_bins - 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        if count:
            confidence = float(probability[mask].mean())
            accuracy = float(truth[mask].mean())
            ece += count / len(truth) * abs(accuracy - confidence)
        else:
            confidence = 0.0
            accuracy = 0.0
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability_fake": confidence,
                "observed_fake_rate": accuracy,
            }
        )
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins}


def expected_calibration_error(
    y_true: list[int] | np.ndarray,
    y_prob_fake: list[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    return float(calibration_bins(y_true, y_prob_fake, n_bins)["ece"])


def evaluate_probabilities(
    y_true: list[int] | np.ndarray,
    y_prob_fake: list[float] | np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return binary holdout metrics for the FAKE-positive class."""
    _validate_probability_options(threshold, 0.1)
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_prob_fake, dtype=float)
    bins = calibration_bins(truth, probability, n_bins)
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None,
        "pr_auc": float(np.clip(average_precision_score(truth, probability), 0.0, 1.0))
        if len(np.unique(truth)) == 2
        else None,
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).astype(int).tolist(),
        "brier_score": float(np.mean((probability - truth) ** 2)),
        "calibration_error": bins["ece"],
        "calibration_bins": bins["bins"],
    }


def _probability_logits(probabilities: list[float] | np.ndarray) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt_calibrator(
    y_true: list[int] | np.ndarray,
    y_prob_fake: list[float] | np.ndarray,
):
    """Fit a Platt sigmoid to validation-only probabilities."""
    from sklearn.linear_model import LogisticRegression

    truth = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob_fake, dtype=float)
    if truth.ndim != 1 or probabilities.ndim != 1 or len(truth) != len(probabilities):
        raise ValueError("calibration labels and probabilities must have equal one-dimensional shapes")
    if len(truth) < 4 or len(np.unique(truth)) != 2:
        raise ValueError(
            "Platt calibration requires at least four validation examples and both classes"
        )
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("calibration probabilities must be finite values between 0 and 1")
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(_probability_logits(probabilities), truth)
    return calibrator


def apply_platt_calibrator(
    calibrator: Any | None,
    y_prob_fake: list[float] | np.ndarray,
) -> np.ndarray:
    """Apply a saved Platt model, using unchanged raw scores when unavailable."""
    probabilities = np.asarray(y_prob_fake, dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    if calibrator is None:
        return probabilities.copy()
    return np.asarray(calibrator.predict_proba(_probability_logits(probabilities))[:, 1], dtype=float)


def baseline_feature_indicators(classifier: Any, text: str, top_n: int = 4) -> list[dict[str, Any]]:
    """Return strongest present TF-IDF LR indicators; these are not evidence."""
    if not hasattr(classifier, "named_steps"):
        return []
    steps = classifier.named_steps
    if not {"cleaner", "tfidf", "classifier"}.issubset(steps):
        return []
    features = steps["tfidf"].get_feature_names_out()
    transformed = steps["tfidf"].transform(steps["cleaner"].transform([text]))
    coefficients = np.asarray(steps["classifier"].coef_).reshape(-1)
    contributions = transformed.multiply(coefficients).tocoo()
    ranked = sorted(
        (
            (str(features[index]), float(value))
            for index, value in zip(contributions.col, contributions.data, strict=True)
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_n]
    return [
        {
            "feature": feature,
            "direction": "FAKE-associated" if contribution > 0 else "REAL-associated",
            "contribution": contribution,
        }
        for feature, contribution in ranked
    ]


def split_labeled_texts(
    texts: list[str],
    labels: list[int] | np.ndarray,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """De-duplicate first, remove cross-label duplicates, then make a stratified split."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("at least one text sample is required")
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")
    clean = [re.sub(r"\s+", " ", str(value)).strip().casefold() for value in texts]
    y = np.asarray(labels, dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be 0 (REAL) or 1 (FAKE)")

    label_sets: dict[str, set[int]] = {}
    for text, label in zip(clean, y, strict=True):
        if text:
            label_sets.setdefault(text, set()).add(int(label))
    conflicts = {text for text, found_labels in label_sets.items() if len(found_labels) > 1}
    conflicting_rows = sum(text in conflicts for text in clean)
    kept_texts: list[str] = []
    kept_labels: list[int] = []
    seen: set[str] = set()
    for text, label in zip(clean, y, strict=True):
        if text and text not in conflicts and text not in seen:
            kept_texts.append(text)
            kept_labels.append(int(label))
            seen.add(text)
    if len(set(kept_labels)) != 2:
        raise ValueError("both REAL and FAKE classes must remain after de-duplication")
    all_indices = np.arange(len(kept_texts))
    train_validation, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        stratify=kept_labels,
        random_state=random_state,
    )
    relative_validation_size = validation_size / (1 - test_size)
    train_indices, validation_indices = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        stratify=np.asarray(kept_labels)[train_validation],
        random_state=random_state,
    )
    return {
        "texts": kept_texts,
        "labels": np.asarray(kept_labels, dtype=np.int64),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "validation_indices": np.asarray(validation_indices, dtype=np.int64),
        "test_indices": np.asarray(test_indices, dtype=np.int64),
        "removed_cross_label_duplicates": conflicting_rows,
        "removed_duplicate_or_empty_rows": len(texts) - len(kept_texts) - conflicting_rows,
    }


def adversarial_perturb_text(text: str, seed: int = 42) -> str:
    """Apply one deterministic, label-preserving adjacent-character typo."""
    words = text.split()
    candidates = [i for i, word in enumerate(words) if len(word) >= 5 and word.isalpha()]
    if not candidates:
        return text
    rng = random.Random(seed)
    token_index = rng.choice(candidates)
    word = words[token_index]
    char_index = rng.randrange(1, len(word) - 1)
    chars = list(word)
    chars[char_index], chars[char_index + 1] = chars[char_index + 1], chars[char_index]
    words[token_index] = "".join(chars)
    return " ".join(words)


def augment_training_texts(
    texts: list[str], labels: list[int] | np.ndarray, seed: int = 42
) -> tuple[list[str], np.ndarray]:
    """Append one deterministic typo-perturbed copy per training example."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    augmented = list(texts)
    augmented_labels = [int(value) for value in labels]
    for index, (text, label) in enumerate(zip(texts, labels, strict=True)):
        changed = adversarial_perturb_text(text, seed=seed + index)
        if changed != text:
            augmented.append(changed)
            augmented_labels.append(int(label))
    return augmented, np.asarray(augmented_labels, dtype=np.int64)


def auto_device() -> str:
    """Return CUDA when a usable PyTorch CUDA runtime exists, otherwise CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_bundle(bundle: dict[str, Any], artifact_dir: str | Path) -> Path:
    """Persist a joblib bundle and SHA256 sidecar."""
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / BUNDLE_FILENAME
    joblib.dump(bundle, bundle_path)
    (directory / SIDECAR_FILENAME).write_text(
        f"{sha256_file(bundle_path)}  {BUNDLE_FILENAME}\n", encoding="ascii"
    )
    return bundle_path


def _load_bundle(artifact_dir: str | Path) -> tuple[dict[str, Any], Path]:
    directory = Path(artifact_dir)
    bundle_path = directory if directory.is_file() else directory / BUNDLE_FILENAME
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Advanced model bundle not found: {bundle_path}")
    sidecar = bundle_path.with_name(SIDECAR_FILENAME)
    if sidecar.exists():
        expected = sidecar.read_text(encoding="ascii").strip().split()[0]
        actual = sha256_file(bundle_path)
        if expected.lower() != actual:
            raise ValueError(f"Advanced model bundle checksum mismatch: {bundle_path}")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Advanced model bundle has an unexpected format: {bundle_path}")
    return bundle, bundle_path.parent


def _component_artifacts_available(
    bundle: dict[str, Any], root: Path, component: str
) -> bool:
    try:
        if not _component_enabled(bundle, component):
            return False

        if component == "baseline":
            return callable(getattr(bundle.get("baseline"), "predict_proba", None))
        if component == "embedding":
            model_dir = _resolve_component_path(root, "sentence_transformer", bundle)
            return (
                callable(getattr(bundle.get("embedding_classifier"), "predict_proba", None))
                and _has_sentence_transformer_artifacts(model_dir)
            )
        if component == "transformer":
            model_dir = _resolve_component_path(root, component, bundle)
            neural_config = bundle.get("neural_config")
            return (
                isinstance(neural_config, dict)
                and _has_tokenizer_artifacts(model_dir / "tokenizer")
                and _has_pretrained_model_artifacts(model_dir / "encoder")
                and _is_nonempty_file(model_dir / "head.pt")
                and isinstance(neural_config.get("hidden_size"), int)
            )
        if component == "fusion":
            model_dir = _resolve_component_path(root, "fusion_model", bundle)
            sentence_dir = _resolve_component_path(root, "sentence_transformer", bundle)
            neural_config = bundle.get("neural_config")
            return (
                isinstance(neural_config, dict)
                and _has_tokenizer_artifacts(model_dir / "tokenizer")
                and _has_pretrained_model_artifacts(model_dir / "encoder")
                and _is_nonempty_file(model_dir / "fusion.pt")
                and _has_sentence_transformer_artifacts(sentence_dir)
                and isinstance(neural_config.get("hidden_size"), int)
                and isinstance(neural_config.get("sentence_size"), int)
                and isinstance(neural_config.get("fusion_size"), int)
            )
        if component == "ensemble":
            return (
                callable(getattr(bundle.get("ensemble"), "predict_proba", None))
                and _component_artifacts_available(bundle, root, "baseline")
                and _component_artifacts_available(bundle, root, "transformer")
            )
    except (OSError, TypeError, ValueError):
        return False
    return False


def _has_tokenizer_artifacts(directory: Path) -> bool:
    return directory.is_dir() and any(
        _is_nonempty_file(directory / name)
        for name in (
            "tokenizer.json",
            "vocab.txt",
            "vocab.json",
            "spiece.model",
            "sentencepiece.bpe.model",
            "tokenizer.model",
        )
    )


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _has_pretrained_model_artifacts(directory: Path) -> bool:
    config_path = directory / "config.json"
    if not _is_nonempty_file(config_path):
        return False
    try:
        if not isinstance(json.loads(config_path.read_text(encoding="utf-8")), dict):
            return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if any(
        _is_nonempty_file(directory / name)
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = directory / index_name
        if not _is_nonempty_file(index_path):
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        shards = set(weight_map.values())
        if all(isinstance(shard, str) for shard in shards) and all(
            _is_nonempty_file(directory / shard) for shard in shards
        ):
            return True
    return False


def _has_sentence_transformer_artifacts(directory: Path) -> bool:
    modules_path = directory / "modules.json"
    if not _is_nonempty_file(modules_path):
        return False
    try:
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(modules, list) or not modules:
        return False
    transformer_dirs = []
    for module in modules:
        if not isinstance(module, dict) or not isinstance(module.get("path"), str):
            return False
        module_dir = (directory / module["path"]).resolve()
        if module_dir != directory.resolve() and directory.resolve() not in module_dir.parents:
            return False
        if not module_dir.is_dir():
            return False
        if "transformer" in str(module.get("type", "")).casefold():
            transformer_dirs.append(module_dir)
    return any(
        _has_pretrained_model_artifacts(module_dir)
        and (
            _has_tokenizer_artifacts(module_dir)
            or _has_tokenizer_artifacts(directory)
        )
        for module_dir in transformer_dirs
    )


def advanced_component_availability(artifact_dir: str | Path) -> dict[str, bool]:
    """Return components whose bundle entries and required local files are present."""
    try:
        bundle, root = _load_bundle(artifact_dir)
    except (FileNotFoundError, OSError, ValueError, IndexError, EOFError):
        return dict.fromkeys(
            ("baseline", "embedding", "transformer", "fusion", "ensemble"), False
        )
    return {
        component: _component_artifacts_available(bundle, root, component)
        for component in ("baseline", "embedding", "transformer", "fusion", "ensemble")
    }


def advanced_artifacts_available(artifact_dir: str | Path) -> bool:
    """Return whether any advanced bundle component has usable local artifacts."""
    return any(advanced_component_availability(artifact_dir).values())


def _resolve_component_path(root: Path, component: str, bundle: dict[str, Any]) -> Path:
    paths = bundle.get("component_paths", {})
    if not isinstance(paths, dict):
        raise ValueError("component_paths must be a mapping")
    configured = paths.get(component)
    if configured is not None and not isinstance(configured, (str, Path)):
        raise ValueError(f"Invalid local path configured for {component}")
    path = Path(configured) if configured else Path(component)
    return path if path.is_absolute() else (root / path).resolve()


def _require_neural_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Advanced transformer inference requires the optional packages in "
            "requirements-advanced.txt (torch and transformers)."
        ) from error
    return torch, AutoModel, AutoTokenizer


@lru_cache(maxsize=4)
def _cached_sentence_model(model_dir: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "Sentence embedding inference requires sentence-transformers; "
            "install requirements-advanced.txt."
        ) from error
    return SentenceTransformer(model_dir, device=device, local_files_only=True)


@lru_cache(maxsize=4)
def _cached_transformer_parts(model_dir: str, device: str, hidden_size: int):
    torch, AutoModel, AutoTokenizer = _require_neural_dependencies()
    directory = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(directory / "encoder", local_files_only=True).to(device)
    encoder.eval()
    head = torch.nn.Linear(hidden_size, 2).to(device)
    head.load_state_dict(
        torch.load(directory / "head.pt", map_location=device, weights_only=True)
    )
    head.eval()
    return torch, tokenizer, encoder, head


@lru_cache(maxsize=4)
def _cached_fusion_parts(
    model_dir: str,
    sentence_dir: str,
    device: str,
    hidden_size: int,
    sentence_size: int,
    fusion_size: int,
):
    torch, AutoModel, AutoTokenizer = _require_neural_dependencies()
    directory = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(directory / "encoder", local_files_only=True).to(device)
    fusion_class = _build_fusion_network(
        torch, torch.nn, encoder, hidden_size, sentence_size, fusion_size
    )
    model = fusion_class().to(device)
    model.load_state_dict(
        torch.load(directory / "fusion.pt", map_location=device, weights_only=True)
    )
    model.eval()
    sentence_model = _cached_sentence_model(sentence_dir, device)
    return torch, tokenizer, model, sentence_model


def _pool_hidden(output: Any, attention_mask: Any) -> Any:
    token_vectors = output.last_hidden_state
    weights = attention_mask.unsqueeze(-1).to(token_vectors.dtype)
    return (token_vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def _build_fusion_network(torch: Any, nn: Any, encoder: Any, hidden_size: int, sentence_size: int, fusion_size: int):
    class FusionNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.transformer_head = nn.Linear(hidden_size, 2)
            self.transformer_projection = nn.Linear(hidden_size, fusion_size)
            self.sentence_projection = nn.Linear(sentence_size, fusion_size)
            heads = 4 if fusion_size % 4 == 0 else 2 if fusion_size % 2 == 0 else 1
            self.attention = nn.MultiheadAttention(fusion_size, num_heads=heads, batch_first=True)
            self.gate = nn.Linear(fusion_size * 2, fusion_size)
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(fusion_size),
                nn.Dropout(0.1),
                nn.Linear(fusion_size, 2),
            )

        def forward(self, input_ids: Any, attention_mask: Any, sentence_embeddings: Any):
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = _pool_hidden(encoded, attention_mask)
            transformer_features = torch.tanh(self.transformer_projection(pooled))
            sentence_features = torch.tanh(self.sentence_projection(sentence_embeddings))
            modalities = torch.stack((transformer_features, sentence_features), dim=1)
            attended, _ = self.attention(modalities, modalities, modalities, need_weights=False)
            attended_features = attended.mean(dim=1)
            gate = torch.sigmoid(self.gate(torch.cat((transformer_features, sentence_features), dim=1)))
            fused = gate * attended_features + (1 - gate) * transformer_features
            return self.transformer_head(pooled), self.fusion_head(fused)

    return FusionNetwork


def _encode_transformer(text: str, tokenizer: Any, model: Any, torch: Any, device: str, max_length: int):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _predict_neural(
    text: str,
    component: str,
    root: Path,
    bundle: dict[str, Any],
    device: str,
) -> float:
    torch, _, _ = _require_neural_dependencies()
    config = bundle.get("neural_config", {})
    model_dir = _resolve_component_path(root, component, bundle)
    tokenizer_dir = model_dir / "tokenizer"
    encoder_dir = model_dir / "encoder"
    if not tokenizer_dir.is_dir() or not encoder_dir.is_dir():
        raise FileNotFoundError(
            f"Local {component} tokenizer/model files are missing under {model_dir}. "
            "Training downloads model files once; inference requires them locally and never downloads."
        )
    hidden_size = int(config["hidden_size"])
    if component == "transformer":
        weights_path = model_dir / "head.pt"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Local transformer classifier weights are missing: {weights_path}")
        torch, tokenizer, encoder, head = _cached_transformer_parts(
            str(model_dir), device, hidden_size
        )
        encoded = _encode_transformer(
            text, tokenizer, encoder, torch, device, int(config.get("max_length", 256))
        )
        with torch.inference_mode():
            output = encoder(**encoded)
            logits = head(_pool_hidden(output, encoded["attention_mask"]))
        return float(torch.softmax(logits, dim=1)[0, 1].item())

    sentence_dir = _resolve_component_path(root, "sentence_transformer", bundle)
    if not sentence_dir.is_dir():
        raise FileNotFoundError(
            f"Local sentence-transformer files are missing: {sentence_dir}. "
            "Inference is offline; train once with network access to save these files."
        )
    weights_path = model_dir / "fusion.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Local fusion classifier weights are missing: {weights_path}")
    torch, tokenizer, fusion_model, sentence_model = _cached_fusion_parts(
        str(model_dir),
        str(sentence_dir),
        device,
        hidden_size,
        int(config["sentence_size"]),
        int(config["fusion_size"]),
    )
    sentence_vector = sentence_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    encoded = _encode_transformer(
        text,
        tokenizer,
        fusion_model.encoder,
        torch,
        device,
        int(config.get("max_length", 256)),
    )
    sentence_tensor = torch.as_tensor(sentence_vector, dtype=torch.float32, device=device)
    with torch.inference_mode():
        _, logits = fusion_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            sentence_embeddings=sentence_tensor,
        )
    return float(torch.softmax(logits, dim=1)[0, 1].item())


def _predict_embedding(
    text: str, root: Path, bundle: dict[str, Any], device: str
) -> float:
    sentence_dir = _resolve_component_path(root, "sentence_transformer", bundle)
    if not sentence_dir.is_dir():
        raise FileNotFoundError(
            f"Local sentence-transformer files are missing: {sentence_dir}. "
            "Inference is offline; train once with network access to save these files."
        )
    sentence_model = _cached_sentence_model(str(sentence_dir), device)
    vector = sentence_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return _probability(bundle["embedding_classifier"], vector)


def _probability(classifier: Any, text: Any) -> float:
    samples = [text] if isinstance(text, str) else text
    probabilities = classifier.predict_proba(samples)
    classes = list(getattr(classifier, "classes_", [0, 1]))
    fake_index = classes.index(1) if 1 in classes else 1
    return float(np.asarray(probabilities)[0, fake_index])


def _component_enabled(bundle: dict[str, Any], name: str) -> bool:
    components = bundle.get("components", {})
    if not isinstance(components, dict):
        return False
    value = components.get(name, False)
    return bool(value.get("available", False)) if isinstance(value, dict) else bool(value)


def predict_advanced(
    artifact_dir: str | Path,
    text: str,
    threshold: float = 0.5,
    uncertainty_margin: float = 0.10,
    confidence_threshold: float = 0.60,
    component: str | None = None,
) -> dict[str, Any]:
    """Run the selected saved component and return an uncertainty-aware label."""
    clean_input = _validate_text(text)
    _validate_probability_options(threshold, uncertainty_margin)
    if (
        not isinstance(confidence_threshold, (int, float))
        or not math.isfinite(confidence_threshold)
        or not 0.5 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must be a finite number between 0.5 and 1")
    bundle, root = _load_bundle(artifact_dir)
    components = ("baseline", "embedding", "transformer", "fusion", "ensemble")
    if component is not None and component not in components:
        raise ValueError(f"Unsupported model component: {component}")
    available = {
        name: _component_artifacts_available(bundle, root, name) for name in components
    }
    if component is None:
        source = next(
            (
                name
                for name in ("ensemble", "fusion", "transformer", "baseline", "embedding")
                if available[name]
            ),
            None,
        )
        if source is None:
            raise ValueError("Advanced bundle contains no usable inference components")
    else:
        source = component
        if not available[source]:
            raise ValueError(f"Requested model component is unavailable or incomplete: {source}")

    dependencies = ("baseline", "transformer") if source == "ensemble" else (source,)
    raw_scores: dict[str, float] = {}
    device = auto_device()
    if "baseline" in dependencies:
        raw_scores["baseline"] = _probability(bundle["baseline"], clean_input)
    if "embedding" in dependencies:
        raw_scores["embedding"] = _predict_embedding(
            clean_input, root, bundle, device
        )
    if "transformer" in dependencies:
        raw_scores["transformer"] = _predict_neural(
            clean_input, "transformer", root, bundle, device
        )
    if "fusion" in dependencies:
        raw_scores["fusion"] = _predict_neural(
            clean_input, "fusion_model", root, bundle, device
        )

    calibrators = bundle.get("calibrators", {})
    calibrated_scores = {
        name: float(apply_platt_calibrator(calibrators.get(name), [probability])[0])
        for name, probability in raw_scores.items()
    }
    if source == "ensemble":
        ensemble_input = np.asarray(
            [[calibrated_scores["baseline"], calibrated_scores["transformer"]]], dtype=float
        )
        raw_final_probability = float(bundle["ensemble"].predict_proba(ensemble_input)[0, 1])
        final_calibrator = calibrators.get("ensemble")
        fake_probability = float(apply_platt_calibrator(final_calibrator, [raw_final_probability])[0])
        source = "ensemble"
        calibration_method = "Platt scaling" if final_calibrator is not None else "uncalibrated"
    else:
        raw_final_probability = raw_scores[source]
        fake_probability = calibrated_scores[source]
        calibration_method = "Platt scaling" if calibrators.get(source) is not None else "uncalibrated"

    confidence = max(fake_probability, 1 - fake_probability)
    if confidence < confidence_threshold:
        label = "Insufficient Confidence"
    elif abs(fake_probability - threshold) <= uncertainty_margin / 2:
        label = "UNCERTAIN"
    else:
        label = "FAKE" if fake_probability >= threshold else "REAL"
    indicators = (
        baseline_feature_indicators(bundle["baseline"], clean_input)
        if "baseline" in raw_scores
        else []
    )
    return {
        "label": label,
        "prob_fake": fake_probability,
        "raw_prob_fake": raw_final_probability,
        "calibration_method": calibration_method,
        "threshold": float(threshold),
        "uncertainty_margin": float(uncertainty_margin),
        "confidence_threshold": float(confidence_threshold),
        "confidence": confidence,
        "decision_component": source,
        "components": calibrated_scores,
        "raw_components": raw_scores,
        "indicators": indicators,
        "provenance": {
            "artifact_dir": str(root),
            "device": device,
            "inference_network_access": False,
        },
    }


def explain_transformer(
    artifact_dir: str | Path,
    text: str,
    target: str = "FAKE",
    steps: int = 32,
) -> dict[str, Any]:
    """Integrated-gradients token attributions using the saved transformer classifier."""
    clean_input = _validate_text(text)
    if target not in LABELS:
        raise ValueError("target must be REAL or FAKE")
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer of at least 2")
    bundle, root = _load_bundle(artifact_dir)
    if not _component_enabled(bundle, "transformer"):
        raise ValueError("The advanced bundle has no saved transformer classifier")
    torch, AutoModel, AutoTokenizer = _require_neural_dependencies()
    model_dir = _resolve_component_path(root, "transformer", bundle)
    tokenizer_dir, encoder_dir = model_dir / "tokenizer", model_dir / "encoder"
    if not tokenizer_dir.is_dir() or not encoder_dir.is_dir():
        raise FileNotFoundError(f"Local transformer files are missing under {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(encoder_dir, local_files_only=True)
    device = auto_device()
    encoder.to(device).eval()
    head = torch.nn.Linear(int(bundle["neural_config"]["hidden_size"]), 2).to(device)
    head.load_state_dict(
        torch.load(model_dir / "head.pt", map_location=device, weights_only=True)
    )
    head.eval()
    tokens = tokenizer(
        clean_input,
        truncation=True,
        max_length=int(bundle["neural_config"].get("max_length", 256)),
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    embeddings = encoder.get_input_embeddings()(input_ids).detach()
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    baseline = encoder.get_input_embeddings()(baseline_ids).detach()
    target_id = LABELS.index(target)
    total_gradient = torch.zeros_like(embeddings)
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        interpolated = (baseline + alpha * (embeddings - baseline)).detach().requires_grad_(True)
        output = encoder(inputs_embeds=interpolated, attention_mask=attention_mask)
        score = head(_pool_hidden(output, attention_mask))[0, target_id]
        gradient = torch.autograd.grad(score, interpolated)[0]
        total_gradient += gradient
    attribution = ((embeddings - baseline) * total_gradient / steps).sum(dim=-1)[0]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    token_attributions = [
        {"token": token, "attribution": float(value)}
        for token, value in zip(token_strings, attribution.tolist(), strict=True)
    ]
    return {"target": target, "tokens": token_attributions, "method": "integrated_gradients"}
