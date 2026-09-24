import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils.checkpoint_integrity import (
    load_wrapped_single_modal_checkpoint,
    validate_checkpoint_split_integrity,
)


def _records() -> dict[str, list[dict]]:
    return {
        "train": [
            {
                "audio_path": "/data/audio/2024-01-01/AM1/none/fish_audio_001.wav",
                "video_path": "/data/video/2024-01-01/AM1/none/fish_video_001.mp4",
                "sample_key": "2024-01-01/AM1/none/fish_sample_001",
                "label": 0,
            }
        ],
        "val": [
            {
                "audio_path": "/data/audio/2024-01-01/AM1/weak/fish_audio_002.wav",
                "video_path": "/data/video/2024-01-01/AM1/weak/fish_video_002.mp4",
                "sample_key": "2024-01-01/AM1/weak/fish_sample_002",
                "label": 3,
            }
        ],
        "test": [
            {
                "audio_path": "/data/audio/2024-01-02/PM1/strong/fish_audio_003.wav",
                "video_path": "/data/video/2024-01-02/PM1/strong/fish_video_003.mp4",
                "sample_key": "2024-01-02/PM1/strong/fish_sample_003",
                "label": 1,
            }
        ],
    }


def _write_sidecars(checkpoint: Path, records: dict[str, list[dict]]) -> None:
    split_dir = checkpoint.parent / "splits"
    split_dir.mkdir()
    for split_name, split_records in records.items():
        with (split_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["audio_path", "video_path", "label", "sample_key"])
            writer.writeheader()
            writer.writerows(split_records)


def test_wrapper_checkpoint_loads_prefix_modules_strictly(tmp_path: Path) -> None:
    source_frontend = nn.Linear(3, 2)
    source_backbone = nn.Linear(2, 4)
    state = {}
    state.update({f"module.frontend.{key}": value for key, value in source_frontend.state_dict().items()})
    state.update({f"module.backbone.{key}": value for key, value in source_backbone.state_dict().items()})
    checkpoint = tmp_path / "audio_best.pt"
    torch.save({"model_state_dict": state}, checkpoint)

    frontend = nn.Linear(3, 2)
    backbone = nn.Linear(2, 4)
    load_wrapped_single_modal_checkpoint(
        checkpoint, {"frontend": frontend, "backbone": backbone}, "Audio"
    )
    for expected, actual in zip(source_frontend.parameters(), frontend.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(source_backbone.parameters(), backbone.parameters()):
        assert torch.equal(expected, actual)


def test_wrapper_checkpoint_rejects_wrong_architecture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "video_best.pt"
    torch.save({"model_state_dict": {"backbone.weight": torch.randn(2, 2)}}, checkpoint)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_wrapped_single_modal_checkpoint(checkpoint, {"backbone": nn.Linear(3, 4)}, "Video")


def test_split_integrity_accepts_path_independent_exact_match(tmp_path: Path) -> None:
    checkpoint = tmp_path / "audio_best.pt"
    checkpoint.touch()
    records = _records()
    _write_sidecars(checkpoint, records)
    validate_checkpoint_split_integrity(checkpoint, records, "Audio")


def test_split_integrity_accepts_legacy_single_modal_sidecars(tmp_path: Path) -> None:
    checkpoint = tmp_path / "video_best.pt"
    checkpoint.touch()
    records = _records()
    split_dir = checkpoint.parent / "splits"
    split_dir.mkdir()
    for split_name, split_records in records.items():
        with (split_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as file:
            fields = ["video_path", "audio_path", "label", "class_name", "date", "session", "sample_id"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for record in split_records:
                key_parts = record["sample_key"].split("/")
                writer.writerow(
                    {
                        "video_path": f"/different/root/{key_parts[-1].replace('_sample_', '_video_')}.mp4",
                        "audio_path": f"/different/root/{key_parts[-1].replace('_sample_', '_audio_')}.wav",
                        "label": record["label"],
                        "class_name": key_parts[-2],
                        "date": key_parts[-4],
                        "session": key_parts[-3],
                        "sample_id": key_parts[-1],
                    }
                )
    validate_checkpoint_split_integrity(checkpoint, records, "Video")


def test_split_integrity_rejects_mismatch_and_missing_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "audio_best.pt"
    checkpoint.touch()
    records = _records()
    _write_sidecars(checkpoint, records)
    mismatched = _records()
    mismatched["test"][0]["label"] = 2
    with pytest.raises(ValueError):
        validate_checkpoint_split_integrity(checkpoint, mismatched, "Audio")
    (checkpoint.parent / "splits" / "val.csv").unlink()
    with pytest.raises(FileNotFoundError):
        validate_checkpoint_split_integrity(checkpoint, records, "Audio")
