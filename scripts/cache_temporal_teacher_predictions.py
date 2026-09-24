from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import load_train_config
from models.multimodal_model import MultimodalDeepFusionModel
from scripts.evaluate_ordinal_ensemble import _loader, _read_split


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Cache BORA and direct teacher probabilities.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    entries = _read_split(args.split_dir / f"{args.split}.csv")
    loader = _loader(entries, args.split, cfg)
    model = MultimodalDeepFusionModel(cfg).to(device)
    all_bora: list[np.ndarray] = []
    all_audio: list[np.ndarray] = []
    all_video: list[np.ndarray] = []
    labels = None
    sample_keys = None
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.set_epoch(int(checkpoint["epoch"]))
        model.eval()
        bora_batches: list[np.ndarray] = []
        audio_batches: list[np.ndarray] = []
        video_batches: list[np.ndarray] = []
        current_labels: list[np.ndarray] = []
        current_keys: list[str] = []
        for batch in tqdm(loader, desc=f"Teacher cache {checkpoint_path.name}", unit="batch"):
            output = model(
                waveform=batch["waveform"].to(device, non_blocking=True),
                video_form=batch["video_form"].to(device, non_blocking=True),
            )
            bora_batches.append(output["rank_probabilities"].cpu().numpy())
            audio_batches.append(output["audio_teacher_probabilities"].cpu().numpy()[..., [0, 3, 2, 1]])
            video_batches.append(output["video_teacher_probabilities"].cpu().numpy()[..., [0, 3, 2, 1]])
            current_labels.append(batch["target"].numpy())
            current_keys.extend(str(key) for key in batch["sample_key"])
        current_label_array = np.concatenate(current_labels)
        if labels is not None:
            if not np.array_equal(labels, current_label_array) or sample_keys != current_keys:
                raise RuntimeError("Sample order changed between checkpoints.")
        labels = current_label_array
        sample_keys = current_keys
        all_bora.append(np.concatenate(bora_batches))
        all_audio.append(np.concatenate(audio_batches))
        all_video.append(np.concatenate(video_batches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        bora_probabilities=np.stack(all_bora),
        audio_teacher_probabilities=np.stack(all_audio),
        video_teacher_probabilities=np.stack(all_video),
        labels=labels,
        sample_keys=np.asarray(sample_keys),
    )
    print(args.output)


if __name__ == "__main__":
    main()
