from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


DEFAULT_IMAGE_CACHE_ROOT = "video_image_cache"
VALID_CACHE_MODES = {"none", "ram", "disk"}
BORA_AUDIO_BACKBONES = {"PANNS_Cnn6", "PANNS_Cnn6_DW_ECA", "TinyPANNS_ECA"}


class AudioFeaturesConfig(BaseModel):
    sample_rate: int = 64000
    window_size: int = 2048
    hop_size: int = 1024
    mel_bins: int = 128
    fmin: int = 1
    fmax: int = 32000
    time_drop_width: int = 64
    time_stripes_num: int = 2
    freq_drop_width: int = 8
    freq_stripes_num: int = 2
    freeze_parameters: bool = True


class VideoFeaturesConfig(BaseModel):
    image_size: int = 224
    num_frames: int = Field(default=1, ge=1, le=32)


class SplitterConfig(BaseModel):
    dataset_path: str = "/marimo/Fish_Feeding_Intensity_Dataset"
    seed: int = 42
    test_sample_per_class: int = 700
    save_results: bool = False
    include_video: bool = True
    split_strategy: Literal["random_sample", "time_series", "group_random"] = "random_sample"
    evaluation_mode: Literal["holdout", "cross_validation"] = "holdout"
    num_folds: int = 5
    fold_index: Optional[int] = None
    cv_val_ratio: float = 0.2
    output_dir: str = "outputs"


class ModalityConfig(BaseModel):
    backbone: str
    pretrained: bool = False
    freeze: bool = False
    checkpoint_path: str = ""


class BoraConfig(BaseModel):
    gate_temperature: float = Field(default=0.7, gt=0.0)
    aux_loss_weight: float = Field(default=0.3, ge=0.0)
    reliability_loss_weight: float = Field(default=0.1, ge=0.0)
    warmup_epochs: int = Field(default=5, ge=0)
    encoder_lr_scale: float = Field(default=0.1, gt=0.0)
    corruption_probability: float = Field(default=0.30, ge=0.0, le=1.0)
    modality_dropout_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    audio_snr_db: tuple[float, float] = (0.0, 20.0)
    audio_gain: tuple[float, float] = (0.3, 1.0)
    temporal_mask_ratio: tuple[float, float] = (0.1, 0.3)
    video_brightness: tuple[float, float] = (0.5, 1.5)
    video_blur_sigma: tuple[float, float] = (0.1, 2.0)
    video_occlusion_ratio: tuple[float, float] = (0.1, 0.3)
    categorical_loss_weight: float = Field(default=0.0, ge=0.0)
    motion_loss_weight: float = Field(default=0.0, ge=0.0)
    nominal_loss_weight: float = Field(default=0.0, ge=0.0)
    teacher_preservation_weight: float = Field(default=0.0, ge=0.0)
    temporal_num_layers: int = Field(default=2, ge=1, le=6)
    temporal_num_heads: int = Field(default=4, ge=1)
    temporal_max_frames: int = Field(default=16, ge=2, le=64)

    @model_validator(mode="after")
    def validate_ranges(self) -> "BoraConfig":
        if self.corruption_probability + self.modality_dropout_probability > 1.0:
            raise ValueError("BORA corruption_probability + modality_dropout_probability must be <= 1.0.")
        ranges = {
            "audio_snr_db": self.audio_snr_db,
            "audio_gain": self.audio_gain,
            "temporal_mask_ratio": self.temporal_mask_ratio,
            "video_brightness": self.video_brightness,
            "video_blur_sigma": self.video_blur_sigma,
            "video_occlusion_ratio": self.video_occlusion_ratio,
        }
        for name, (low, high) in ranges.items():
            if low > high:
                raise ValueError(f"BORA {name} lower bound must be <= upper bound.")
        for name in ("audio_gain", "video_brightness", "video_blur_sigma"):
            if ranges[name][0] < 0.0:
                raise ValueError(f"BORA {name} values must be non-negative.")
        for name in ("temporal_mask_ratio", "video_occlusion_ratio"):
            low, high = ranges[name]
            if low < 0.0 or high > 1.0:
                raise ValueError(f"BORA {name} values must lie in [0, 1].")
        return self


class FusionConfig(BaseModel):
    type: Literal[
        "raw_concat",
        "linear_concat",
        "linear_mean",
        "gated_fusion",
        "self_attention",
        "bora_fusion",
        "temporal_bora_fusion",
    ] = "raw_concat"
    proj_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3
    num_heads: int = 4
    activation: Literal["relu", "gelu"] = "relu"
    use_batchnorm: bool = True
    bora: BoraConfig = Field(default_factory=BoraConfig)


class DatasetConfig(BaseModel):
    dataset_path: str = "/marimo/Fish_Feeding_Intensity_Dataset"
    cache_audio: bool = True
    cache_video: bool = True
    video_cache_mode: Literal["none", "ram", "disk"] = "ram"
    num_workers: int = -1
    prefetch_factor: Optional[int] = None
    seed: int = 42
    split_strategy: Literal["random_sample", "time_series", "group_random"] = "random_sample"
    test_sample_per_class: int = 700
    num_folds: int = 5
    cv_val_ratio: float = 0.2


class TrainConfig(BaseModel):
    seed: int = 42
    device: str = "cuda"
    num_classes: int = 4
    evaluation_mode: Literal["holdout", "cross_validation"] = "holdout"
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    monitor: Literal["accuracy", "f1_macro", "loss"] = "accuracy"
    early_stopping: bool = True
    patience: int = 30
    delta: float = 0.0
    output_dir: str = "outputs"
    audio: ModalityConfig = Field(default_factory=lambda: ModalityConfig(backbone="PANNS_Cnn6"))
    video: ModalityConfig = Field(default_factory=lambda: ModalityConfig(backbone="EfficientNetB0", pretrained=True))
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)
    video_features: VideoFeaturesConfig = Field(default_factory=VideoFeaturesConfig)

    @model_validator(mode="after")
    def validate_bora_run(self) -> "TrainConfig":
        if self.fusion.type not in {"bora_fusion", "temporal_bora_fusion"}:
            return self
        if self.num_classes != 4:
            raise ValueError("BORA-Fuse requires num_classes=4.")
        if self.video.backbone not in {"SwinTiny", "EfficientNetB0"}:
            raise ValueError("BORA-Fuse requires video backbone SwinTiny or EfficientNetB0.")
        if self.audio.backbone not in BORA_AUDIO_BACKBONES:
            raise ValueError(
                "BORA-Fuse requires audio backbone PANNS_Cnn6, "
                "PANNS_Cnn6_DW_ECA, or TinyPANNS_ECA."
            )
        if self.evaluation_mode != "holdout" or self.dataset.split_strategy != "random_sample":
            raise ValueError("BORA-Fuse supports random_sample holdout evaluation only.")
        if self.optimizer != "adam":
            raise ValueError("BORA-Fuse uses Adam with encoder/head parameter groups; optimizer must be 'adam'.")
        if self.monitor != "accuracy":
            raise ValueError("BORA-Fuse selects its best checkpoint by accuracy; monitor must be 'accuracy'.")
        if not self.audio.checkpoint_path.strip() or not self.video.checkpoint_path.strip():
            raise ValueError("BORA-Fuse requires both audio.checkpoint_path and video.checkpoint_path.")
        if self.audio.freeze or self.video.freeze:
            raise ValueError("BORA-Fuse fine-tunes both encoders; audio.freeze and video.freeze must be false.")
        if self.fusion.type == "bora_fusion" and self.video_features.num_frames != 1:
            raise ValueError("Global BORA-Fuse requires video_features.num_frames=1.")
        if self.fusion.type == "temporal_bora_fusion":
            if self.video_features.num_frames < 2:
                raise ValueError("Temporal BORA-Fuse requires video_features.num_frames>=2.")
            if self.video_features.num_frames > self.fusion.bora.temporal_max_frames:
                raise ValueError(
                    "video_features.num_frames cannot exceed fusion.bora.temporal_max_frames."
                )
            if self.fusion.proj_dim % self.fusion.bora.temporal_num_heads != 0:
                raise ValueError(
                    "fusion.proj_dim must be divisible by fusion.bora.temporal_num_heads."
                )
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


def load_train_config(path: str | Path = "config/train_config.json") -> TrainConfig:
    return TrainConfig.from_json(path)
