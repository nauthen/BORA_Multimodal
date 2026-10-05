from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


DEFAULT_IMAGE_CACHE_ROOT = "video_image_cache"
VALID_CACHE_MODES = {"none", "ram", "disk"}
BORA_AUDIO_BACKBONES = {"PANNS_Cnn6", "PANNS_Cnn6_DW_ECA", "TinyPANNS_ECA"}
BORA_VIDEO_BACKBONES = {"SwinTiny", "EfficientNetB0", "MobileViTXXS", "MobileNetV2"}


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
    # Cross-validation teacher template with a {fold} placeholder, e.g.
    # ".../fold_{fold:02d}/checkpoint/audio_best.pt" (splits/ sidecar beside it).
    # BORA cross-validation needs one teacher per fold: a holdout teacher was
    # trained on most of every CV test fold.
    cv_checkpoint_path: str = ""


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
    # Ablation switch for the temporal reliability gate: "margin" modulates the
    # learned reliability with the auxiliary boundary margin 2|sigmoid(o)-0.5|;
    # "none" feeds the raw learned reliability to the gate.
    gate_confidence: Literal["margin", "none"] = "margin"
    # Ablation switch for the explicit frame-difference cue: "explicit" adds the
    # projected |v_t - v_(t-1)| to the temporal tokens and the event gate and
    # trains the motion regressor; "none" removes all three.
    temporal_motion: Literal["explicit", "none"] = "explicit"
    # Ablation switch for the decoders: "dual" couples the ordinal (CORN) and
    # nominal decoders; "ordinal" or "nominal" keeps only that decoder.
    decoders: Literal["dual", "ordinal", "nominal"] = "dual"

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


FOLD_PLACEHOLDER = "{fold"


def _validate_teacher_paths(name: str, modality: ModalityConfig, evaluation_mode: str) -> None:
    if evaluation_mode == "holdout":
        if not modality.checkpoint_path.strip():
            raise ValueError(f"BORA-Fuse requires {name}.checkpoint_path.")
        if FOLD_PLACEHOLDER in modality.checkpoint_path:
            raise ValueError(
                f"{name}.checkpoint_path contains a fold placeholder; holdout needs one fixed teacher "
                f"checkpoint (per-fold templates belong in {name}.cv_checkpoint_path)."
            )
        return
    template = modality.cv_checkpoint_path.strip()
    if FOLD_PLACEHOLDER not in template:
        raise ValueError(
            f"BORA-Fuse cross-validation needs one {name} teacher per fold: set {name}.cv_checkpoint_path "
            "to a template containing {fold}, e.g. '.../fold_{fold:02d}/checkpoint/" + name + "_best.pt'. "
            "A holdout teacher was trained on most of every cross-validation test fold."
        )
    try:
        template.format(fold=0)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"{name}.cv_checkpoint_path is not a valid fold template: {template!r} ({exc}).") from exc


def _validate_ablation(fusion: FusionConfig) -> None:
    bora = fusion.bora
    is_ablation = (
        bora.gate_confidence != "margin" or bora.temporal_motion != "explicit" or bora.decoders != "dual"
    )
    if is_ablation and fusion.type != "temporal_bora_fusion":
        raise ValueError(
            "Ablations (the ablation preset or gate_confidence/temporal_motion/decoders) apply to "
            "temporal_bora_fusion only."
        )
    if bora.temporal_motion == "none" and bora.motion_loss_weight != 0.0:
        raise ValueError("temporal_motion='none' removes the motion regressor; motion_loss_weight must be 0.")
    if bora.decoders != "dual" and bora.nominal_loss_weight != 0.0:
        raise ValueError(
            f"decoders='{bora.decoders}' requires nominal_loss_weight=0: an ordinal-only model has no "
            "nominal decoder, and in a nominal-only model categorical_loss already supervises the "
            "nominal logits."
        )
    if bora.decoders == "nominal" and bora.categorical_loss_weight <= 0.0:
        raise ValueError("decoders='nominal' is trained through categorical_loss; categorical_loss_weight must be > 0.")


# Leave-one-component-out ablations of Temporal BORA-Fuse, selected with the
# "ablation" field of train_config.json. Each preset is written into fusion.bora
# on load, so a variant differs from the baseline only by the removed component
# and the loss attached to it (docs/ablation_design.md).
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_motion": {"temporal_motion": "none", "motion_loss_weight": 0.0},
    "no_confidence": {"gate_confidence": "none"},
    "ordinal_only": {"decoders": "ordinal", "nominal_loss_weight": 0.0},
    "nominal_only": {"decoders": "nominal", "nominal_loss_weight": 0.0},
}


class TrainConfig(BaseModel):
    seed: int = 42
    device: str = "cuda"
    num_classes: int = 4
    evaluation_mode: Literal["holdout", "cross_validation"] = "holdout"
    # "none" = full model; otherwise one key of ABLATIONS.
    ablation: Literal["none", "no_motion", "no_confidence", "ordinal_only", "nominal_only"] = "none"
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

    @model_validator(mode="before")
    @classmethod
    def apply_ablation_preset(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("ablation", "none") not in ABLATIONS:
            # "none", or an unknown name that field validation reports.
            return data
        fusion = data.get("fusion", {})
        fusion = fusion.model_dump() if isinstance(fusion, BaseModel) else dict(fusion)
        bora = fusion.get("bora", {})
        bora = bora.model_dump() if isinstance(bora, BaseModel) else dict(bora)
        fusion["bora"] = {**bora, **ABLATIONS[data["ablation"]]}
        return {**data, "fusion": fusion}

    @model_validator(mode="after")
    def validate_bora_run(self) -> "TrainConfig":
        _validate_ablation(self.fusion)
        if self.fusion.type not in {"bora_fusion", "temporal_bora_fusion"}:
            return self
        if self.num_classes != 4:
            raise ValueError("BORA-Fuse requires num_classes=4.")
        if self.video.backbone not in BORA_VIDEO_BACKBONES:
            raise ValueError(
                "BORA-Fuse requires video backbone "
                f"{', '.join(sorted(BORA_VIDEO_BACKBONES))}."
            )
        if self.audio.backbone not in BORA_AUDIO_BACKBONES:
            raise ValueError(
                "BORA-Fuse requires audio backbone PANNS_Cnn6, "
                "PANNS_Cnn6_DW_ECA, or TinyPANNS_ECA."
            )
        if self.dataset.split_strategy != "random_sample":
            raise ValueError("BORA-Fuse supports the random_sample split strategy only.")
        if self.optimizer != "adam":
            raise ValueError("BORA-Fuse uses Adam with encoder/head parameter groups; optimizer must be 'adam'.")
        if self.monitor != "accuracy":
            raise ValueError("BORA-Fuse selects its best checkpoint by accuracy; monitor must be 'accuracy'.")
        for name, modality in (("audio", self.audio), ("video", self.video)):
            _validate_teacher_paths(name, modality, self.evaluation_mode)
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


def experiment_name(cfg: TrainConfig) -> str:
    """Run name used for output directories and uploaded artifacts; ablations get a suffix."""
    name = f"{cfg.audio.backbone}_{cfg.video.backbone}_{cfg.fusion.type}"
    if cfg.fusion.type != "temporal_bora_fusion":
        return name
    bora = cfg.fusion.bora
    if bora.temporal_motion == "none":
        name += "_nomotion"
    if bora.gate_confidence == "none":
        name += "_noconf"
    if bora.decoders != "dual":
        name += f"_{bora.decoders}only"
    return name


def fold_config(cfg: TrainConfig, fold_index: int) -> TrainConfig:
    """Copy of cfg whose teacher checkpoints point at one cross-validation fold."""
    resolved = cfg.model_copy(deep=True)
    for modality in (resolved.audio, resolved.video):
        if modality.cv_checkpoint_path.strip():
            modality.checkpoint_path = modality.cv_checkpoint_path.strip().format(fold=fold_index)
    return resolved
