import csv
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

import main as entrypoint
import utils.artifact_upload as artifact_upload
from config import ArtifactUploadConfig, TrainConfig
from utils.metrics import save_metrics_csv


def _cross_validation_config(**dataset) -> TrainConfig:
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
            "dataset": {"split_strategy": "random_sample", "num_folds": 3, **dataset},
        }
    )


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, events: list) -> None:
    def fake_load_splits(dataset_cfg, mode, fold_index=None):
        assert mode == "cross_validation"
        return {"fold": fold_index}

    def fake_validate(path, splits, description):
        events.append(("check", path, splits["fold"], description))

    class FakeTrainer:
        def __init__(self, cfg, loaders, output_dir, splits, run_name):
            self.output_dir = Path(output_dir)
            self.fold = splits["fold"]
            events.append(("train", cfg.audio.checkpoint_path, cfg.video.checkpoint_path, self.fold, self.output_dir.name))

        def fit(self):
            result = {"test_accuracy": 0.9 + 0.01 * (self.fold + 1)}
            save_metrics_csv(result, self.output_dir / "result.csv")
            return result

    def fake_upload(upload_cfg, train_cfg, fold_dir, archive_root, fold_index):
        events.append(("upload", fold_index, Path(fold_dir).name))

    monkeypatch.setattr(entrypoint, "load_splits", fake_load_splits)
    monkeypatch.setattr(entrypoint, "validate_checkpoint_split_integrity", fake_validate)
    monkeypatch.setattr(entrypoint, "create_dataloaders", lambda **kwargs: {"splits": kwargs["splits"]})
    monkeypatch.setattr(entrypoint, "MultimodalTrainer", FakeTrainer)
    monkeypatch.setattr(entrypoint, "upload_cv_fold_if_enabled", fake_upload)


def _summary(cv_dir: Path) -> dict:
    with (cv_dir / "summary_mean_std.csv").open(newline="", encoding="utf-8") as file:
        return {row["metric"]: row for row in csv.DictReader(file)}


def test_cross_validation_trains_and_uploads_each_fold_with_its_own_teachers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _cross_validation_config()
    events: list = []
    _patch_pipeline(monkeypatch, events)

    results = entrypoint._run_cross_validation(cfg, tmp_path)

    checks = [event for event in events if event[0] == "check"]
    assert checks[:2] == [
        ("check", "/cv/audio/fold_00/audio_best.pt", 0, "Audio fold_00"),
        ("check", "/cv/video/fold_0/video_best.pt", 0, "Video fold_00"),
    ]
    # All teachers are verified before the first fold trains.
    assert events.index(("check", "/cv/video/fold_2/video_best.pt", 2, "Video fold_02")) < events.index(
        ("train", "/cv/audio/fold_00/audio_best.pt", "/cv/video/fold_0/video_best.pt", 0, "fold_00")
    )
    # Each fold is uploaded as soon as it finishes, before the next fold trains.
    order = [(event[0], event[3] if event[0] == "train" else event[1]) for event in events if event[0] != "check"]
    assert order == [("train", 0), ("upload", 0), ("train", 1), ("upload", 1), ("train", 2), ("upload", 2)]
    assert cfg.audio.checkpoint_path == "/holdout/audio_best.pt"
    assert sorted(results) == [0, 1, 2]

    summary = _summary(tmp_path / "cross_validation")
    assert float(summary["test_accuracy"]["mean"]) == pytest.approx(0.92)
    assert summary["test_accuracy"]["folds"] == "0 1 2"


def test_cross_validation_resumes_selected_folds_and_summarises_all_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # fold_00 survived from an earlier run (or was unzipped from its uploaded archive).
    save_metrics_csv({"test_accuracy": 0.5}, tmp_path / "cross_validation" / "fold_00" / "result.csv")
    events: list = []
    _patch_pipeline(monkeypatch, events)

    results = entrypoint._run_cross_validation(_cross_validation_config(cv_folds=[2]), tmp_path)

    assert sorted(results) == [2]
    assert {event[2] for event in events if event[0] == "check"} == {2}
    assert [event[3] for event in events if event[0] == "train"] == [2]
    summary = _summary(tmp_path / "cross_validation")
    assert summary["test_accuracy"]["folds"] == "0 2"
    assert float(summary["test_accuracy"]["mean"]) == pytest.approx((0.5 + 0.93) / 2)
    with (tmp_path / "cross_validation" / "fold_results.csv").open(newline="", encoding="utf-8") as file:
        assert [row["fold"] for row in csv.DictReader(file)] == ["0", "2"]


def test_cross_validation_stops_before_training_when_a_fold_teacher_mismatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list = []
    _patch_pipeline(monkeypatch, events)

    def failing_validate(path, splits, description):
        if splits["fold"] == 2:
            raise ValueError(f"{description} checkpoint split mismatch")

    monkeypatch.setattr(entrypoint, "validate_checkpoint_split_integrity", failing_validate)
    with pytest.raises(ValueError, match="Audio fold_02"):
        entrypoint._run_cross_validation(_cross_validation_config(), tmp_path)
    assert not [event for event in events if event[0] == "train"]


@pytest.mark.parametrize("cv_folds", [[], [1, 1], [3], [-1]])
def test_cv_folds_must_be_unique_folds_in_range(cv_folds: list) -> None:
    with pytest.raises(ValidationError, match="cv_folds"):
        _cross_validation_config(cv_folds=cv_folds)


def test_cv_folds_default_to_every_fold() -> None:
    assert _cross_validation_config().dataset.selected_folds() == [0, 1, 2]
    assert _cross_validation_config(cv_folds=[2, 0]).dataset.selected_folds() == [2, 0]


def test_fold_archive_restores_at_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fold_dir = tmp_path / "outputs" / "exp" / "cross_validation" / "fold_01"
    save_metrics_csv({"test_accuracy": 0.9}, fold_dir / "result.csv")
    uploaded: list = []

    def fake_upload_zip(upload_cfg, token, zip_path, artifact_name):
        uploaded.append((artifact_name, zipfile.ZipFile(zip_path).namelist()))

    monkeypatch.setattr(artifact_upload, "get_token", lambda: "token")
    monkeypatch.setattr(artifact_upload, "_upload_zip", fake_upload_zip)
    name = artifact_upload.upload_cv_fold_if_enabled(
        ArtifactUploadConfig(repo_id="user/repo"), _cross_validation_config(), fold_dir, tmp_path, 1
    )

    assert name.startswith("MultimodalDL_TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion_cross_validation_fold_01_")
    assert "outputs/exp/cross_validation/fold_01/result.csv" in uploaded[0][1]


def test_failed_fold_upload_is_retried_then_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fold_dir = tmp_path / "outputs" / "exp" / "cross_validation" / "fold_00"
    save_metrics_csv({"test_accuracy": 0.9}, fold_dir / "result.csv")
    attempts: list = []

    def failing_upload_zip(upload_cfg, token, zip_path, artifact_name):
        attempts.append(artifact_name)
        raise ConnectionError("network down")

    monkeypatch.setattr(artifact_upload, "get_token", lambda: "token")
    monkeypatch.setattr(artifact_upload, "_upload_zip", failing_upload_zip)
    name = artifact_upload.upload_cv_fold_if_enabled(
        ArtifactUploadConfig(repo_id="user/repo"), _cross_validation_config(), fold_dir, tmp_path, 0,
        attempts=3, retry_seconds=0.0,
    )

    assert name is None
    assert len(attempts) == 3
    assert (fold_dir / "result.csv").is_file()
