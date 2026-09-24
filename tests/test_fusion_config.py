import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import TrainConfig, experiment_name


def _valid_fusion_config() -> dict:
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
        "fusion": {"type": "temporal_reliability_fusion"},
        "video_features": {"num_frames": 8},
        "dataset": {"split_strategy": "random_sample"},
    }


def test_fusion_config_accepts_locked_protocol() -> None:
    config = TrainConfig.model_validate(_valid_fusion_config())
    assert config.fusion.type == "temporal_reliability_fusion"
    assert config.audio.backbone == "PANNS_Cnn6"
    assert config.video.backbone == "SwinTiny"


def test_fusion_config_accepts_efficientnet_video() -> None:
    raw = _valid_fusion_config()
    raw["video"]["backbone"] = "EfficientNetB0"
    config = TrainConfig.model_validate(raw)
    assert config.video.backbone == "EfficientNetB0"


def test_fusion_config_accepts_mobilevit_xxs_video() -> None:
    raw = _valid_fusion_config()
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
        (("evaluation_mode",), "cross_validation"),
        (("optimizer",), "adamw"),
        (("monitor",), "f1_macro"),
    ],
)
def test_fusion_config_rejects_protocol_drift(path: tuple[str, ...], value: object) -> None:
    raw = _valid_fusion_config()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)


def test_fusion_gate_confidence_defaults_to_margin_and_names_ablation_run() -> None:
    raw = _valid_fusion_config()
    raw["fusion"] = {"type": "temporal_reliability_fusion", "proj_dim": 16, "bora": {"temporal_num_heads": 4}}
    config = TrainConfig.model_validate(raw)
    assert config.fusion.bora.gate_confidence == "margin"
    assert experiment_name(config) == "PANNS_Cnn6_SwinTiny_temporal_reliability_fusion"

    raw["fusion"]["bora"]["gate_confidence"] = "none"
    ablation = TrainConfig.model_validate(raw)
    assert experiment_name(ablation) == "PANNS_Cnn6_SwinTiny_temporal_reliability_fusion_noconf"

    raw["fusion"]["bora"]["gate_confidence"] = "entropy"
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)


def test_ablation_config_file_differs_from_main_only_in_gate_confidence() -> None:
    root = Path(__file__).resolve().parent.parent / "config"
    main = json.loads((root / "train_config.json").read_text(encoding="utf-8"))
    ablation = json.loads((root / "train_config.ablation_no_confidence.json").read_text(encoding="utf-8"))
    assert ablation["fusion"]["bora"].pop("gate_confidence") == "none"
    main["fusion"]["bora"].pop("gate_confidence", None)
    assert ablation == main
