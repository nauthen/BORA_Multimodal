import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import TrainConfig, fold_config


def _valid_bora_config() -> dict:
    return {
        "optimizer": "adam",
        "monitor": "accuracy",
        "evaluation_mode": "holdout",
        "audio": {
            "backbone": "PANNS_Cnn6",
            "checkpoint_path": "audio_best.pt",
            "freeze": False,
        },
        "video": {
            "backbone": "SwinTiny",
            "checkpoint_path": "video_best.pt",
            "freeze": False,
        },
        "fusion": {"type": "bora_fusion"},
        "dataset": {"split_strategy": "random_sample"},
    }


def test_bora_config_accepts_locked_global_protocol() -> None:
    config = TrainConfig.model_validate(_valid_bora_config())
    assert config.fusion.type == "bora_fusion"
    assert config.audio.backbone == "PANNS_Cnn6"
    assert config.video.backbone == "SwinTiny"


def test_bora_config_accepts_efficientnet_video() -> None:
    raw = _valid_bora_config()
    raw["video"]["backbone"] = "EfficientNetB0"
    config = TrainConfig.model_validate(raw)
    assert config.video.backbone == "EfficientNetB0"


@pytest.mark.parametrize("audio", ["PANNS_Cnn6", "PANNS_Cnn6_DW_ECA", "TinyPANNS_ECA"])
@pytest.mark.parametrize("video", ["EfficientNetB0", "MobileViTXXS", "MobileNetV2", "SwinTiny"])
def test_bora_config_accepts_every_teacher_pair(audio: str, video: str) -> None:
    raw = _valid_bora_config()
    raw["audio"]["backbone"] = audio
    raw["video"]["backbone"] = video
    config = TrainConfig.model_validate(raw)
    assert (config.audio.backbone, config.video.backbone) == (audio, video)


def test_bora_config_accepts_mobilevit_xxs_video() -> None:
    raw = _valid_bora_config()
    raw["video"]["backbone"] = "MobileViTXXS"
    config = TrainConfig.model_validate(raw)
    assert config.video.backbone == "MobileViTXXS"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("audio", "checkpoint_path"), ""),
        (("audio", "backbone"), "PANNS_Cnn10"),
        (("audio", "freeze"), True),
        (("video", "backbone"), "ResNet18"),
        (("dataset", "split_strategy"), "group_random"),
        (("optimizer",), "adamw"),
        (("monitor",), "f1_macro"),
    ],
)
def test_bora_config_rejects_protocol_drift(path: tuple[str, ...], value: object) -> None:
    raw = _valid_bora_config()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)


def test_cross_validation_requires_one_teacher_per_fold() -> None:
    raw = _valid_bora_config()
    raw["evaluation_mode"] = "cross_validation"
    with pytest.raises(ValidationError, match="one audio teacher per fold"):
        TrainConfig.model_validate(raw)

    raw["audio"]["cv_checkpoint_path"] = "cv/audio/fold_{fold:02d}/audio_best.pt"
    raw["video"]["cv_checkpoint_path"] = "cv/video/video_best.pt"
    with pytest.raises(ValidationError, match="one video teacher per fold"):
        TrainConfig.model_validate(raw)

    raw["video"]["cv_checkpoint_path"] = "cv/video/fold_{fold}/{run}/video_best.pt"
    with pytest.raises(ValidationError, match="not a valid fold template"):
        TrainConfig.model_validate(raw)

    raw["video"]["cv_checkpoint_path"] = "cv/video/fold_{fold}/video_best.pt"
    config = TrainConfig.model_validate(raw)
    fold = fold_config(config, 3)
    assert fold.audio.checkpoint_path == "cv/audio/fold_03/audio_best.pt"
    assert fold.video.checkpoint_path == "cv/video/fold_3/video_best.pt"
    assert config.audio.checkpoint_path == "audio_best.pt"


def test_holdout_rejects_fold_template_as_teacher() -> None:
    raw = _valid_bora_config()
    raw["audio"]["checkpoint_path"] = "cv/audio/fold_{fold}/audio_best.pt"
    with pytest.raises(ValidationError, match="holdout needs one fixed teacher"):
        TrainConfig.model_validate(raw)


@pytest.mark.parametrize("evaluation_mode", ["holdout", "cross_validation"])
def test_repository_config_loads_in_both_modes(evaluation_mode: str) -> None:
    root = Path(__file__).resolve().parent.parent / "config"
    raw = json.loads((root / "train_config.json").read_text(encoding="utf-8"))
    raw["evaluation_mode"] = evaluation_mode
    raw["audio"]["cv_checkpoint_path"] = "/cv/audio/fold_{fold:02d}/audio_best.pt"
    raw["video"]["cv_checkpoint_path"] = "/cv/video/fold_{fold:02d}/video_best.pt"
    assert TrainConfig.model_validate(raw).evaluation_mode == evaluation_mode
