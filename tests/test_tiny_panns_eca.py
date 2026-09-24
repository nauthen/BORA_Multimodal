import torch

from config import TrainConfig
from models.audio import TinyPANNS_ECA, build_audio_backbone


def test_tiny_panns_eca_is_registered_and_exposes_fusion_features() -> None:
    model = build_audio_backbone("TinyPANNS_ECA", classes_num=4)

    assert isinstance(model, TinyPANNS_ECA)
    assert model.get_name() == "tiny_panns_eca"
    assert model.fc_audioset.in_features == 256
    assert model(torch.randn(2, 1, 64, 64)).shape == (2, 4)


def test_tiny_panns_eca_state_keys_match_trained_eca_dependency() -> None:
    keys = set(TinyPANNS_ECA().state_dict())

    assert "block1.eca.channel_conv.weight" in keys
    assert "block5.eca.channel_conv.weight" in keys
    assert not any(".eca.conv.weight" in key for key in keys)


def test_bora_config_accepts_tiny_panns_eca() -> None:
    config = TrainConfig.model_validate(
        {
            "optimizer": "adam",
            "monitor": "accuracy",
            "evaluation_mode": "holdout",
            "audio": {
                "backbone": "TinyPANNS_ECA",
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
    )

    assert config.audio.backbone == "TinyPANNS_ECA"
