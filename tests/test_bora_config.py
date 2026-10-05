import json
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from config import ABLATIONS, TrainConfig, experiment_name, fold_config, load_train_config


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


def _valid_temporal_config() -> dict:
    raw = _valid_bora_config()
    raw["fusion"] = {
        "type": "temporal_bora_fusion",
        "proj_dim": 16,
        "bora": {"temporal_num_heads": 4, "categorical_loss_weight": 1.0, "motion_loss_weight": 0.1, "nominal_loss_weight": 0.5},
    }
    raw["video_features"] = {"num_frames": 2}
    return raw


def test_temporal_bora_defaults_to_full_model() -> None:
    config = TrainConfig.model_validate(_valid_temporal_config())
    bora = config.fusion.bora
    assert config.ablation == "none"
    assert (bora.gate_confidence, bora.temporal_motion, bora.decoders) == ("margin", "explicit", "dual")
    assert experiment_name(config) == "PANNS_Cnn6_SwinTiny_temporal_bora_fusion"


@pytest.mark.parametrize(
    ("ablation", "suffix"),
    [
        ("no_motion", "_nomotion"),
        ("no_confidence", "_noconf"),
        ("ordinal_only", "_ordinalonly"),
        ("nominal_only", "_nominalonly"),
    ],
)
def test_ablation_field_changes_only_its_component(ablation: str, suffix: str) -> None:
    raw = _valid_temporal_config()
    baseline = TrainConfig.model_validate(raw)
    variant = TrainConfig.model_validate({**raw, "ablation": ablation})
    assert variant.ablation == ablation
    assert experiment_name(variant) == experiment_name(baseline) + suffix

    baseline_bora = baseline.fusion.bora.model_dump()
    variant_bora = variant.fusion.bora.model_dump()
    changed = {key for key in baseline_bora if baseline_bora[key] != variant_bora[key]}
    assert changed == set(ABLATIONS[ablation])
    assert {key: variant_bora[key] for key in changed} == ABLATIONS[ablation]
    assert variant.model_dump(exclude={"fusion", "ablation"}) == baseline.model_dump(exclude={"fusion", "ablation"})
    # A saved config_snapshot.json reloads to the same run.
    assert TrainConfig.model_validate(variant.model_dump()) == variant
    assert raw["fusion"]["bora"] == _valid_temporal_config()["fusion"]["bora"]


def test_ablation_field_lists_every_preset() -> None:
    assert get_args(TrainConfig.model_fields["ablation"].annotation) == ("none", *ABLATIONS)


def test_unknown_ablation_is_rejected() -> None:
    raw = _valid_temporal_config()
    raw["ablation"] = "no_transformer"
    with pytest.raises(ValidationError, match="no_motion"):
        TrainConfig.model_validate(raw)


@pytest.mark.parametrize(
    "bora",
    [
        {"temporal_motion": "none", "motion_loss_weight": 0.1},
        {"decoders": "ordinal", "nominal_loss_weight": 0.5},
        {"decoders": "nominal", "nominal_loss_weight": 0.5},
        {"decoders": "nominal", "nominal_loss_weight": 0.0, "categorical_loss_weight": 0.0},
        {"gate_confidence": "entropy"},
    ],
)
def test_ablation_rejects_inconsistent_settings(bora: dict) -> None:
    raw = _valid_temporal_config()
    raw["fusion"]["bora"].update(bora)
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"ablation": "no_confidence"},
        {"ablation": "ordinal_only"},
        {"fusion": {"type": "bora_fusion", "bora": {"temporal_motion": "none"}}},
        {"fusion": {"type": "gated_fusion", "bora": {"decoders": "ordinal"}}},
    ],
)
def test_ablations_require_temporal_fusion(change: dict) -> None:
    raw = {**_valid_bora_config(), **change}
    with pytest.raises(ValidationError, match="temporal_bora_fusion only"):
        TrainConfig.model_validate(raw)


def test_cross_validation_requires_one_teacher_per_fold() -> None:
    raw = _valid_temporal_config()
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
    raw = _valid_temporal_config()
    raw["audio"]["checkpoint_path"] = "cv/audio/fold_{fold}/audio_best.pt"
    with pytest.raises(ValidationError, match="holdout needs one fixed teacher"):
        TrainConfig.model_validate(raw)


def test_repository_config_is_the_full_holdout_baseline() -> None:
    config = load_train_config(Path(__file__).resolve().parent.parent / "config" / "train_config.json")
    assert (config.evaluation_mode, config.ablation) == ("holdout", "none")
    assert experiment_name(config) == "TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion"


@pytest.mark.parametrize("ablation", ["none", *ABLATIONS])
def test_repository_config_supports_every_ablation_in_both_modes(ablation: str) -> None:
    root = Path(__file__).resolve().parent.parent / "config"
    raw = json.loads((root / "train_config.json").read_text(encoding="utf-8"))
    holdout = TrainConfig.model_validate({**raw, "ablation": ablation})
    assert holdout.evaluation_mode == "holdout"

    raw["audio"]["cv_checkpoint_path"] = "/cv/audio/fold_{fold:02d}/audio_best.pt"
    raw["video"]["cv_checkpoint_path"] = "/cv/video/fold_{fold:02d}/video_best.pt"
    cross_validation = TrainConfig.model_validate({**raw, "ablation": ablation, "evaluation_mode": "cross_validation"})
    assert cross_validation.evaluation_mode == "cross_validation"
    assert experiment_name(cross_validation) == experiment_name(holdout)
