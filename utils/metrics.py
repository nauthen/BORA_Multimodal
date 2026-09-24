from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from utils.ordinal import BOUNDARY_NAMES, DATASET_LABEL_TO_RANK, RANK_NAMES


CLASS_NAMES = ["none", "strong", "medium", "weak"]


def compute_ordinal_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    true_labels = np.asarray(list(y_true), dtype=int)
    pred_labels = np.asarray(list(y_pred), dtype=int)
    label_to_rank = np.asarray(DATASET_LABEL_TO_RANK, dtype=int)
    true_ranks = label_to_rank[true_labels]
    pred_ranks = label_to_rank[pred_labels]
    errors = np.abs(true_ranks - pred_ranks)
    qwk = cohen_kappa_score(true_ranks, pred_ranks, labels=[0, 1, 2, 3], weights="quadratic")
    return {
        "rank_mae": float(errors.mean()) if errors.size else 0.0,
        "qwk": float(np.nan_to_num(qwk, nan=0.0)),
        "within_one_accuracy": float((errors <= 1).mean()) if errors.size else 0.0,
        "severe_error_rate": float((errors >= 2).mean()) if errors.size else 0.0,
    }


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
    }


def save_metrics_csv(metrics: Dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_history_csv(history: List[Dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_history_plot(history: List[Dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
        if key in history[0]:
            plt.plot(epochs, [row[key] for row in history], label=key)
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_confusion_outputs(y_true: Iterable[int], y_pred: Iterable[int], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(list(y_true), list(y_pred), labels=list(range(len(CLASS_NAMES))))
    np.savetxt(output_path / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=200)
    plt.close()


def save_bora_predictions(
    sample_keys: List[str],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    rank_probabilities: np.ndarray,
    audio_reliability: np.ndarray,
    video_reliability: np.ndarray,
    audio_gate_weights: np.ndarray,
    video_gate_weights: np.ndarray,
    path: str | Path,
) -> None:
    true_labels = np.asarray(list(y_true), dtype=int)
    pred_labels = np.asarray(list(y_pred), dtype=int)
    label_to_rank = np.asarray(DATASET_LABEL_TO_RANK, dtype=int)
    arrays = [rank_probabilities, audio_reliability, video_reliability, audio_gate_weights, video_gate_weights]
    if any(array.shape[0] != len(sample_keys) for array in arrays):
        raise ValueError("BORA prediction arrays must have one row per sample key.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_key", "true_label", "predicted_label", "true_rank", "predicted_rank"]
    fieldnames.extend(f"p_rank_{name}" for name in RANK_NAMES)
    for prefix in ("audio_reliability", "video_reliability", "audio_gate", "video_gate"):
        fieldnames.extend(f"{prefix}_{boundary}" for boundary in BOUNDARY_NAMES)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_key in enumerate(sample_keys):
            row = {
                "sample_key": sample_key,
                "true_label": int(true_labels[index]),
                "predicted_label": int(pred_labels[index]),
                "true_rank": int(label_to_rank[true_labels[index]]),
                "predicted_rank": int(label_to_rank[pred_labels[index]]),
            }
            for rank_index, name in enumerate(RANK_NAMES):
                row[f"p_rank_{name}"] = float(rank_probabilities[index, rank_index])
            for boundary_index, boundary in enumerate(BOUNDARY_NAMES):
                row[f"audio_reliability_{boundary}"] = float(audio_reliability[index, boundary_index])
                row[f"video_reliability_{boundary}"] = float(video_reliability[index, boundary_index])
                row[f"audio_gate_{boundary}"] = float(audio_gate_weights[index, boundary_index])
                row[f"video_gate_{boundary}"] = float(video_gate_weights[index, boundary_index])
            writer.writerow(row)


def save_bora_gate_summary(
    y_true: Iterable[int],
    audio_reliability: np.ndarray,
    video_reliability: np.ndarray,
    audio_gate_weights: np.ndarray,
    video_gate_weights: np.ndarray,
    path: str | Path,
) -> None:
    true_labels = np.asarray(list(y_true), dtype=int)
    true_ranks = np.asarray(DATASET_LABEL_TO_RANK, dtype=int)[true_labels]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "true_rank",
        "true_rank_name",
        "boundary",
        "sample_count",
        "audio_gate_mean",
        "audio_gate_std",
        "video_gate_mean",
        "video_gate_std",
        "audio_reliability_mean",
        "audio_reliability_std",
        "video_reliability_mean",
        "video_reliability_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, rank_name in enumerate(RANK_NAMES):
            rank_mask = true_ranks == rank
            for boundary_index, boundary in enumerate(BOUNDARY_NAMES):
                row = {
                    "true_rank": rank,
                    "true_rank_name": rank_name,
                    "boundary": boundary,
                    "sample_count": int(rank_mask.sum()),
                }
                for name, values in (
                    ("audio_gate", audio_gate_weights),
                    ("video_gate", video_gate_weights),
                    ("audio_reliability", audio_reliability),
                    ("video_reliability", video_reliability),
                ):
                    selected = values[rank_mask, boundary_index]
                    row[f"{name}_mean"] = float(selected.mean()) if selected.size else float("nan")
                    row[f"{name}_std"] = float(selected.std()) if selected.size else float("nan")
                writer.writerow(row)
