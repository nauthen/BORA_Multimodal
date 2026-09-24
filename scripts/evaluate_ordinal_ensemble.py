from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_train_config
from dataset.multimodal_dataset import (
    MultimodalFishDataset,
    multimodal_collate_fn,
    resolve_num_workers,
)
from models.multimodal_model import MultimodalDeepFusionModel
from utils.ordinal import (
    DATASET_LABEL_TO_RANK,
    RANK_TO_DATASET_LABEL,
    fit_ordinal_thresholds,
    labels_to_ranks,
    predict_ranks_with_thresholds,
)


def _read_split(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "video_path": row["video_path"],
            "audio_path": row["audio_path"],
            "label": int(row["label"]),
            "sample_key": row["sample_key"],
        }
        for row in rows
    ]


def _loader(
    entries: list[dict[str, object]],
    split: str,
    cfg,
    temporal_offset: float | None = None,
) -> DataLoader:
    # Decord creates native decoder state per preload thread. Capping this
    # avoids nondeterministic native crashes on high-core notebook hosts while
    # retaining most of the parallel preload throughput.
    workers = min(32, resolve_num_workers(cfg.dataset.num_workers))
    dataset = MultimodalFishDataset(
        entries=entries,
        split=split,
        sample_rate=cfg.audio_features.sample_rate,
        image_size=cfg.video_features.image_size,
        cache_audio=cfg.dataset.cache_audio,
        cache_video=cfg.dataset.cache_video and cfg.dataset.video_cache_mode == "ram",
        num_workers=workers,
        num_frames=cfg.video_features.num_frames,
        temporal_offset=temporal_offset,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )


@torch.no_grad()
def _predict(
    model: MultimodalDeepFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_keys: list[str] = []
    model.eval()
    for batch in tqdm(loader, desc="Ordinal ensemble inference", unit="batch"):
        output = model(
            waveform=batch["waveform"].to(device, non_blocking=True),
            video_form=batch["video_form"].to(device, non_blocking=True),
        )
        probabilities.append(output["rank_probabilities"].cpu().numpy())
        labels.append(batch["target"].numpy())
        sample_keys.extend(str(key) for key in batch["sample_key"])
    return np.concatenate(probabilities), np.concatenate(labels), sample_keys


def _ensemble_predictions(
    checkpoints: Iterable[Path],
    cfg,
    loader: DataLoader,
    device: torch.device,
    snapshot_output_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model = MultimodalDeepFusionModel(cfg).to(device)
    ensemble: list[np.ndarray] = []
    labels = None
    sample_keys = None
    for path in checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.set_epoch(int(checkpoint["epoch"]))
        current_probabilities, current_labels, current_keys = _predict(model, loader, device)
        if labels is not None and not np.array_equal(labels, current_labels):
            raise RuntimeError("Prediction order changed between checkpoints.")
        if sample_keys is not None and sample_keys != current_keys:
            raise RuntimeError("Sample-key order changed between checkpoints.")
        ensemble.append(current_probabilities)
        labels = current_labels
        sample_keys = current_keys
    if not ensemble or labels is None or sample_keys is None:
        raise ValueError("At least one checkpoint is required.")
    stacked = np.stack(ensemble)
    if snapshot_output_path is not None:
        snapshot_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snapshot_output_path,
            probabilities=stacked,
            labels=labels,
            sample_keys=np.asarray(sample_keys),
        )
    return np.mean(stacked, axis=0), labels, sample_keys


def _decode(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: torch.Tensor,
) -> dict[str, object]:
    probability_tensor = torch.from_numpy(probabilities)
    calibrated_ranks = predict_ranks_with_thresholds(probability_tensor, thresholds).numpy()
    raw_ranks = probabilities.argmax(axis=1)
    true_ranks = np.asarray(DATASET_LABEL_TO_RANK, dtype=np.int64)[labels]
    label_table = np.asarray(RANK_TO_DATASET_LABEL, dtype=np.int64)
    return {
        "raw_accuracy": float(np.mean(raw_ranks == true_ranks)),
        "calibrated_accuracy": float(np.mean(calibrated_ranks == true_ranks)),
        "raw_ranks": raw_ranks,
        "calibrated_ranks": calibrated_ranks,
        "raw_labels": label_table[raw_ranks],
        "calibrated_labels": label_table[calibrated_ranks],
        "true_ranks": true_ranks,
    }


def _save_predictions(
    path: Path,
    sample_keys: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    decoded: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_key", "true_label", "true_rank", "raw_label", "raw_rank",
                "calibrated_label", "calibrated_rank", "p_none", "p_weak", "p_medium", "p_strong",
            ]
        )
        for index, key in enumerate(sample_keys):
            writer.writerow(
                [
                    key,
                    int(labels[index]),
                    int(decoded["true_ranks"][index]),
                    int(decoded["raw_labels"][index]),
                    int(decoded["raw_ranks"][index]),
                    int(decoded["calibrated_labels"][index]),
                    int(decoded["calibrated_ranks"][index]),
                    *[float(value) for value in probabilities[index]],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-calibrated ordinal snapshot ensemble.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--fixed-thresholds",
        type=float,
        nargs=3,
        metavar=("NONE_WEAK", "WEAK_MEDIUM", "MEDIUM_STRONG"),
        help="Skip validation fitting and apply three previously locked validation thresholds.",
    )
    parser.add_argument(
        "--temporal-offset",
        type=float,
        choices=(0.25, 0.5, 0.75),
        help="Use a deterministic stratified temporal TTA view instead of the training grid.",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoints = [path.resolve() for path in args.checkpoint]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "checkpoints": [str(path) for path in checkpoints],
    }
    if args.fixed_thresholds is None:
        val_entries = _read_split(args.split_dir / "val.csv")
        val_probabilities, val_labels, val_keys = _ensemble_predictions(
            checkpoints,
            cfg,
            _loader(val_entries, "val", cfg, temporal_offset=args.temporal_offset),
            device,
            snapshot_output_path=args.output_dir / "val_snapshot_probabilities.npz",
        )
        val_ranks = labels_to_ranks(torch.from_numpy(val_labels))
        thresholds = fit_ordinal_thresholds(torch.from_numpy(val_probabilities), val_ranks)
        val_decoded = _decode(val_probabilities, val_labels, thresholds)
        _save_predictions(
            args.output_dir / "val_predictions.csv", val_keys, val_labels, val_probabilities, val_decoded
        )
        summary.update(
            {
                "thresholds": thresholds.tolist(),
                "val_raw_accuracy": val_decoded["raw_accuracy"],
                "val_calibrated_accuracy": val_decoded["calibrated_accuracy"],
            }
        )
    else:
        if not args.evaluate_test:
            parser.error("--fixed-thresholds requires --evaluate-test.")
        thresholds = torch.tensor(args.fixed_thresholds, dtype=torch.float32)
        summary.update({"thresholds": thresholds.tolist(), "threshold_source": "locked_validation"})

    if args.evaluate_test:
        test_entries = _read_split(args.split_dir / "test.csv")
        test_probabilities, test_labels, test_keys = _ensemble_predictions(
            checkpoints,
            cfg,
            _loader(test_entries, "test", cfg, temporal_offset=args.temporal_offset),
            device,
            snapshot_output_path=args.output_dir / "test_snapshot_probabilities.npz",
        )
        test_decoded = _decode(test_probabilities, test_labels, thresholds)
        _save_predictions(
            args.output_dir / "test_predictions.csv",
            test_keys,
            test_labels,
            test_probabilities,
            test_decoded,
        )
        summary.update(
            {
                "test_raw_accuracy": test_decoded["raw_accuracy"],
                "test_calibrated_accuracy": test_decoded["calibrated_accuracy"],
            }
        )
    with (args.output_dir / "ordinal_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
