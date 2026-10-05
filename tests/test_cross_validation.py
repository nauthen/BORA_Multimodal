import csv
from pathlib import Path

import pytest

import main as entrypoint
from config import TrainConfig


def _cross_validation_config() -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "optimizer": "adam",
            "monitor": "accuracy",
            "evaluation_mode": "cross_validation",
            "audio": {
                "backbone": "TinyPANNS_ECA",
                "checkpoint_path": "/holdout/audio_best.pt",
                "cv_checkpoint_path": "/cv/audio/fold_{fold:02d}/audio_best.pt",
            },
            "video": {
                "backbone": "MobileViTXXS",
                "checkpoint_path": "/holdout/video_best.pt",
                "cv_checkpoint_path": "/cv/video/fold_{fold}/video_best.pt",
            },
            "fusion": {"type": "temporal_bora_fusion", "proj_dim": 16, "bora": {"temporal_num_heads": 4}},
            "video_features": {"num_frames": 2},
            "dataset": {"split_strategy": "random_sample", "num_folds": 3},
        }
    )


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, checked: list, trained: list) -> None:
    def fake_load_splits(dataset_cfg, mode, fold_index=None):
        assert mode == "cross_validation"
        return {"fold": fold_index}

    def fake_validate(path, splits, description):
        checked.append((path, splits["fold"], description))

    class FakeTrainer:
        def __init__(self, cfg, loaders, output_dir, splits, run_name):
            trained.append(
                {
                    "audio": cfg.audio.checkpoint_path,
                    "video": cfg.video.checkpoint_path,
                    "fold": splits["fold"],
                    "output": Path(output_dir).name,
                    "run_name": run_name,
                }
            )

        def fit(self):
            return {"test_accuracy": 0.9 + 0.01 * len(trained)}

    monkeypatch.setattr(entrypoint, "load_splits", fake_load_splits)
    monkeypatch.setattr(entrypoint, "validate_checkpoint_split_integrity", fake_validate)
    monkeypatch.setattr(entrypoint, "create_dataloaders", lambda **kwargs: {"splits": kwargs["splits"]})
    monkeypatch.setattr(entrypoint, "MultimodalTrainer", FakeTrainer)


def test_cross_validation_uses_each_folds_own_teachers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cross_validation_config()
    checked: list = []
    trained: list = []
    _patch_pipeline(monkeypatch, checked, trained)

    results = entrypoint._run_cross_validation(cfg, tmp_path)

    # Every fold's teachers are verified before the first fold trains.
    assert checked == [
        ("/cv/audio/fold_00/audio_best.pt", 0, "Audio fold_00"),
        ("/cv/video/fold_0/video_best.pt", 0, "Video fold_00"),
        ("/cv/audio/fold_01/audio_best.pt", 1, "Audio fold_01"),
        ("/cv/video/fold_1/video_best.pt", 1, "Video fold_01"),
        ("/cv/audio/fold_02/audio_best.pt", 2, "Audio fold_02"),
        ("/cv/video/fold_2/video_best.pt", 2, "Video fold_02"),
    ]
    assert [run["fold"] for run in trained] == [0, 1, 2]
    assert trained[1]["audio"] == "/cv/audio/fold_01/audio_best.pt"
    assert trained[1]["video"] == "/cv/video/fold_1/video_best.pt"
    assert [run["output"] for run in trained] == ["fold_00", "fold_01", "fold_02"]
    assert cfg.audio.checkpoint_path == "/holdout/audio_best.pt"
    assert len(results) == 3

    with (tmp_path / "cross_validation" / "summary_mean_std.csv").open(newline="", encoding="utf-8") as file:
        summary = {row["metric"]: float(row["mean"]) for row in csv.DictReader(file)}
    assert summary["test_accuracy"] == pytest.approx(0.92)


def test_cross_validation_stops_before_training_when_a_fold_teacher_mismatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checked: list = []
    trained: list = []
    _patch_pipeline(monkeypatch, checked, trained)

    def failing_validate(path, splits, description):
        if splits["fold"] == 2:
            raise ValueError(f"{description} checkpoint split mismatch")

    monkeypatch.setattr(entrypoint, "validate_checkpoint_split_integrity", failing_validate)
    with pytest.raises(ValueError, match="Audio fold_02"):
        entrypoint._run_cross_validation(_cross_validation_config(), tmp_path)
    assert trained == []
