from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from config import RELIABILITY_FUSION_TYPES, TrainConfig
from features.audio_frontend import AudioFrontend
from models.audio import build_audio_backbone
from models.fusion import build_fusion_head
from models.video import build_video_backbone
from utils.checkpoint_integrity import load_wrapped_single_modal_checkpoint

logger = logging.getLogger(__name__)


def _strict_load_checkpoint(module: nn.Module, checkpoint_path: str, prefix: str) -> None:
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"{prefix} checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    missing, unexpected = module.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"{prefix} checkpoint strict load failed. Missing keys={missing}, unexpected keys={unexpected}"
        )
    logger.info("%s checkpoint strict load completed: %s", prefix, path)


class FeatureHook(nn.Module):
    def __init__(self, module: nn.Module, classifier: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.module = module
        self.feature_dim = int(feature_dim)
        self._features: Optional[torch.Tensor] = None
        self._logits: Optional[torch.Tensor] = None
        self._hook = classifier.register_forward_hook(self._capture_classifier_input)

    def _capture_classifier_input(self, _module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        feat = inputs[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, start_dim=1)
        self._features = feat
        self._logits = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features = None
        self._logits = None
        _ = self.module(x)
        if self._features is None:
            raise RuntimeError("Feature hook did not capture classifier input.")
        return self._features

    @property
    def last_logits(self) -> torch.Tensor:
        if self._logits is None:
            raise RuntimeError("Feature hook has no classifier logits; run forward first.")
        return self._logits


def _video_classifier_and_dim(video_backbone: nn.Module) -> tuple[nn.Module, int]:
    model = video_backbone.model
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            if len(classifier) >= 3 and isinstance(classifier[2], nn.Linear):
                return classifier[2], int(classifier[2].in_features)
            if len(classifier) >= 2 and isinstance(classifier[1], nn.Linear):
                return classifier[1], int(classifier[1].in_features)
        if isinstance(classifier, nn.Linear):
            return classifier, int(classifier.in_features)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc, int(model.fc.in_features)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return model.head, int(model.head.in_features)
    # timm ClassifierHead (e.g. MobileViT): pool -> drop -> fc; hook fc to get pooled features.
    if hasattr(model, "head") and isinstance(getattr(model.head, "fc", None), nn.Linear):
        return model.head.fc, int(model.head.fc.in_features)
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        return model.heads.head, int(model.heads.head.in_features)
    raise ValueError(f"Cannot locate video classifier module for {video_backbone.__class__.__name__}.")


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


class MultimodalDeepFusionModel(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.audio_frontend = AudioFrontend(cfg.audio_features)
        audio_backbone = build_audio_backbone(
            name=cfg.audio.backbone,
            classes_num=cfg.num_classes,
            pretrained=cfg.audio.pretrained,
        )
        if not hasattr(audio_backbone, "fc_audioset") or not isinstance(audio_backbone.fc_audioset, nn.Linear):
            raise ValueError(f"Audio backbone '{cfg.audio.backbone}' must expose fc_audioset for feature extraction.")
        if cfg.fusion.type in RELIABILITY_FUSION_TYPES:
            load_wrapped_single_modal_checkpoint(
                cfg.audio.checkpoint_path,
                {"frontend": self.audio_frontend, "backbone": audio_backbone},
                "Audio",
            )
        else:
            _strict_load_checkpoint(audio_backbone, cfg.audio.checkpoint_path, "Audio backbone")
        self.audio_branch = FeatureHook(
            module=nn.Sequential(self.audio_frontend, audio_backbone),
            classifier=audio_backbone.fc_audioset,
            feature_dim=audio_backbone.fc_audioset.in_features,
        )

        video_backbone = build_video_backbone(
            name=cfg.video.backbone,
            classes_num=cfg.num_classes,
            # The teacher checkpoint is a complete single-modal wrapper state. Avoid an
            # unnecessary ImageNet download before strict-loading those weights.
            pretrained=False if cfg.fusion.type in RELIABILITY_FUSION_TYPES else cfg.video.pretrained,
        )
        classifier, video_dim = _video_classifier_and_dim(video_backbone)
        if cfg.fusion.type in RELIABILITY_FUSION_TYPES:
            load_wrapped_single_modal_checkpoint(
                cfg.video.checkpoint_path,
                {"backbone": video_backbone},
                "Video",
            )
        else:
            _strict_load_checkpoint(video_backbone, cfg.video.checkpoint_path, "Video backbone")
        self.video_branch = FeatureHook(module=video_backbone, classifier=classifier, feature_dim=video_dim)

        if cfg.audio.freeze:
            _set_requires_grad(self.audio_branch, False)
        if cfg.video.freeze:
            _set_requires_grad(self.video_branch, False)

        self.fusion = build_fusion_head(
            dim_audio=self.audio_branch.feature_dim,
            dim_video=self.video_branch.feature_dim,
            num_classes=cfg.num_classes,
            cfg=cfg.fusion,
        )
        logger.info(
            "Initialized multimodal model: audio=%s (%dD), video=%s (%dD), fusion=%s",
            cfg.audio.backbone,
            self.audio_branch.feature_dim,
            cfg.video.backbone,
            self.video_branch.feature_dim,
            cfg.fusion.type,
        )

    def set_epoch(self, epoch: int | None) -> None:
        if hasattr(self.fusion, "set_epoch"):
            self.fusion.set_epoch(epoch)

    def forward(self, waveform: torch.Tensor, video_form: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_feat = self.audio_branch(waveform)
        audio_teacher_logits = self.audio_branch.last_logits
        if video_form.ndim == 5:
            batch_size, num_frames = video_form.shape[:2]
            flat_video = video_form.reshape(batch_size * num_frames, *video_form.shape[2:])
            video_feat = self.video_branch(flat_video).reshape(batch_size, num_frames, -1)
            video_teacher_logits = self.video_branch.last_logits.reshape(batch_size, num_frames, -1)
        elif video_form.ndim == 4:
            video_feat = self.video_branch(video_form)
            video_teacher_logits = self.video_branch.last_logits
            if self.cfg.fusion.type in RELIABILITY_FUSION_TYPES:
                video_feat = video_feat.unsqueeze(1)
                video_teacher_logits = video_teacher_logits.unsqueeze(1)
        else:
            raise ValueError(
                f"Expected video_form [B,C,H,W] or [B,T,C,H,W], got {video_form.shape}."
            )
        if self.cfg.fusion.type in RELIABILITY_FUSION_TYPES:
            fusion_output = self.fusion(
                audio_feat,
                video_feat,
                audio_teacher_logits=audio_teacher_logits,
                video_teacher_logits=video_teacher_logits,
            )
        else:
            fusion_output = self.fusion(audio_feat, video_feat)
        if isinstance(fusion_output, dict):
            output = dict(fusion_output)
        else:
            output = {"clipwise_output": fusion_output}
        output.update({
            "audio_features": audio_feat,
            "video_features": video_feat,
        })
        return output

    def get_name(self) -> str:
        return f"{self.cfg.audio.backbone}_{self.cfg.video.backbone}_{self.cfg.fusion.type}"
