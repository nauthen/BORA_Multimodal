from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FusionConfig
from utils.ordinal import conditional_logits_to_rank_probabilities, rank_probabilities_to_dataset_order


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _classifier_block(input_dim: int, hidden_dim: int, num_classes: int, cfg: FusionConfig) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
    if cfg.use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.extend([_activation(cfg.activation), nn.Dropout(cfg.dropout), nn.Linear(hidden_dim, num_classes)])
    return nn.Sequential(*layers)


def _direct_classifier(input_dim: int, num_classes: int, cfg: FusionConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    if cfg.use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    layers.extend([_activation(cfg.activation), nn.Dropout(cfg.dropout), nn.Linear(input_dim, num_classes)])
    return nn.Sequential(*layers)


class RawConcatFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.classifier = _classifier_block(dim_audio + dim_video, cfg.hidden_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([audio_feat, video_feat], dim=-1))


class LinearConcatFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.classifier = _direct_classifier(cfg.proj_dim * 2, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        return self.classifier(torch.cat([z_a, z_v], dim=-1))


class LinearMeanFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.classifier = _direct_classifier(cfg.proj_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        return self.classifier((z_a + z_v) / 2.0)


class GatedFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.gate_layer = nn.Sequential(nn.Linear(cfg.proj_dim * 2, cfg.proj_dim), nn.Sigmoid())
        self.classifier = _direct_classifier(cfg.proj_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        gate = self.gate_layer(torch.cat([z_a, z_v], dim=-1))
        return self.classifier(gate * z_a + (1.0 - gate) * z_v)


class SelfAttentionFusion(nn.Module):
    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        if cfg.proj_dim % cfg.num_heads != 0:
            raise ValueError(f"fusion.proj_dim={cfg.proj_dim} must be divisible by num_heads={cfg.num_heads}.")
        self.proj_audio = nn.Linear(dim_audio, cfg.proj_dim)
        self.proj_video = nn.Linear(dim_video, cfg.proj_dim)
        self.attention = nn.MultiheadAttention(embed_dim=cfg.proj_dim, num_heads=cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.proj_dim)
        self.classifier = _classifier_block(cfg.proj_dim * 2, cfg.hidden_dim, num_classes, cfg)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_audio(audio_feat)
        z_v = self.proj_video(video_feat)
        tokens = torch.stack([z_a, z_v], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens)
        fused_tokens = self.norm(tokens + attended)
        return self.classifier(fused_tokens.flatten(start_dim=1))


class BORAFusion(nn.Module):
    """Boundary-specific Ordinal Reliability-Aware fusion over global audio/video embeddings."""

    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        if num_classes != 4:
            raise ValueError("BORA-Fuse requires exactly four feeding-intensity classes.")
        self.cfg = cfg
        self.warmup_epochs = cfg.bora.warmup_epochs
        self._warmup_active = False
        self.proj_audio = nn.Sequential(
            nn.Linear(dim_audio, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            _activation(cfg.activation),
            nn.Dropout(cfg.dropout),
        )
        self.proj_video = nn.Sequential(
            nn.Linear(dim_video, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            _activation(cfg.activation),
            nn.Dropout(cfg.dropout),
        )
        reliability_hidden = max(16, min(cfg.hidden_dim, cfg.proj_dim))
        self.audio_ordinal_head = nn.Linear(cfg.proj_dim, 3)
        self.video_ordinal_head = nn.Linear(cfg.proj_dim, 3)
        self.audio_reliability_head = nn.Sequential(
            nn.Linear(cfg.proj_dim, reliability_hidden),
            _activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(reliability_hidden, 3),
            nn.Sigmoid(),
        )
        self.video_reliability_head = nn.Sequential(
            nn.Linear(cfg.proj_dim, reliability_hidden),
            _activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(reliability_hidden, 3),
            nn.Sigmoid(),
        )
        self.audio_boundary_projections = nn.ModuleList(
            nn.Linear(cfg.proj_dim, cfg.proj_dim) for _ in range(3)
        )
        self.video_boundary_projections = nn.ModuleList(
            nn.Linear(cfg.proj_dim, cfg.proj_dim) for _ in range(3)
        )
        self.boundary_norms = nn.ModuleList(nn.LayerNorm(cfg.proj_dim) for _ in range(3))
        self.boundary_classifiers = nn.ModuleList(nn.Linear(cfg.proj_dim, 1) for _ in range(3))

    def set_epoch(self, epoch: int | None) -> None:
        self._warmup_active = epoch is not None and epoch < self.warmup_epochs

    def _gate_weights(
        self, audio_reliability: torch.Tensor, video_reliability: torch.Tensor
    ) -> torch.Tensor:
        if self._warmup_active:
            return torch.full(
                (*audio_reliability.shape, 2),
                0.5,
                dtype=audio_reliability.dtype,
                device=audio_reliability.device,
            )
        reliability = torch.stack([audio_reliability, video_reliability], dim=-1).clamp_min(1e-6)
        return F.softmax(torch.log(reliability) / self.cfg.bora.gate_temperature, dim=-1)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        z_audio = self.proj_audio(audio_feat)
        z_video = self.proj_video(video_feat)
        audio_ordinal = self.audio_ordinal_head(z_audio)
        video_ordinal = self.video_ordinal_head(z_video)
        audio_reliability = self.audio_reliability_head(z_audio)
        video_reliability = self.video_reliability_head(z_video)
        weights = self._gate_weights(audio_reliability, video_reliability)

        boundary_logits = []
        for boundary in range(3):
            audio_boundary = self.audio_boundary_projections[boundary](z_audio)
            video_boundary = self.video_boundary_projections[boundary](z_video)
            fused = (
                weights[:, boundary, 0:1] * audio_boundary
                + weights[:, boundary, 1:2] * video_boundary
            )
            fused = self.boundary_norms[boundary](fused)
            boundary_logits.append(self.boundary_classifiers[boundary](fused))
        ordinal_logits = torch.cat(boundary_logits, dim=1)
        rank_probabilities = conditional_logits_to_rank_probabilities(ordinal_logits)
        dataset_probabilities = rank_probabilities_to_dataset_order(rank_probabilities)

        return {
            "clipwise_output": torch.log(dataset_probabilities.clamp_min(1e-8)),
            "rank_probabilities": rank_probabilities,
            "ordinal_logits": ordinal_logits,
            "audio_ordinal_logits": audio_ordinal,
            "video_ordinal_logits": video_ordinal,
            "audio_reliability": audio_reliability,
            "video_reliability": video_reliability,
            "gate_weights": weights[:, :, 0],
            "audio_gate_weights": weights[:, :, 0],
            "video_gate_weights": weights[:, :, 1],
        }


class TemporalBORAFusion(nn.Module):
    """Motion-aware, boundary-conditioned temporal extension of BORA-Fuse."""

    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        if num_classes != 4:
            raise ValueError("Temporal BORA-Fuse requires exactly four classes.")
        if cfg.proj_dim % cfg.bora.temporal_num_heads != 0:
            raise ValueError("proj_dim must be divisible by temporal_num_heads.")
        self.cfg = cfg
        self.warmup_epochs = cfg.bora.warmup_epochs
        self._warmup_active = False
        dim = cfg.proj_dim
        hidden = max(16, min(cfg.hidden_dim, dim))
        self.proj_audio = nn.Sequential(
            nn.Linear(dim_audio, dim), nn.LayerNorm(dim), _activation(cfg.activation), nn.Dropout(cfg.dropout)
        )
        self.proj_video = nn.Sequential(
            nn.Linear(dim_video, dim), nn.LayerNorm(dim), _activation(cfg.activation), nn.Dropout(cfg.dropout)
        )
        self.position = nn.Parameter(torch.zeros(1, cfg.bora.temporal_max_frames, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=cfg.bora.temporal_num_heads,
            dim_feedforward=max(dim * 2, cfg.hidden_dim * 2),
            dropout=min(cfg.dropout, 0.2),
            activation="gelu" if cfg.activation == "gelu" else "relu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=cfg.bora.temporal_num_layers, norm=nn.LayerNorm(dim)
        )
        self.motion_projection = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.event_gate = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(hidden, 1)
        )
        self.boundary_queries = nn.Parameter(torch.empty(3, dim))
        nn.init.trunc_normal_(self.boundary_queries, std=0.02)
        self.attention_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.audio_query = nn.Linear(dim, dim * 3)
        self.audio_ordinal_head = nn.Linear(dim, 3)
        self.video_ordinal_head = nn.Linear(dim, 3)
        self.audio_reliability_head = nn.Sequential(
            nn.Linear(dim, hidden), _activation(cfg.activation), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 3), nn.Sigmoid()
        )
        self.video_reliability_head = nn.Sequential(
            nn.Linear(dim, hidden), _activation(cfg.activation), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.audio_boundary_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
        self.video_boundary_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
        self.boundary_interactions = nn.ModuleList(
            nn.Sequential(nn.Linear(dim * 3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(cfg.dropout))
            for _ in range(3)
        )
        self.boundary_classifiers = nn.ModuleList(nn.Linear(dim, 1) for _ in range(3))
        self.motion_regressor = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.teacher_residual_scale = nn.Parameter(torch.full((3,), -1.1))
        # Dual decoder: an exact-class head over pooled boundary evidence. CORN
        # factorizes the class decision into sequential conditionals and can
        # lose exact-class signal even at near-perfect ranking quality; this
        # decoder observes all three boundary-conditioned temporal evidences
        # simultaneously and is coupled back through a learned logit residual.
        self.nominal_head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.LayerNorm(hidden),
            _activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, num_classes),
        )
        self.nominal_residual_scale = nn.Parameter(torch.tensor(-1.1))

    def set_epoch(self, epoch: int | None) -> None:
        self._warmup_active = epoch is not None and epoch < self.warmup_epochs

    def _gate_weights(
        self, audio_reliability: torch.Tensor, video_reliability: torch.Tensor
    ) -> torch.Tensor:
        if self._warmup_active:
            return torch.full(
                (*audio_reliability.shape, 2), 0.5,
                dtype=audio_reliability.dtype, device=audio_reliability.device,
            )
        reliability = torch.stack([audio_reliability, video_reliability], dim=-1).clamp_min(1e-6)
        return F.softmax(torch.log(reliability) / self.cfg.bora.gate_temperature, dim=-1)

    @staticmethod
    def _teacher_probabilities_to_conditional_logits(
        dataset_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # Dataset order [none, strong, medium, weak] -> rank order
        # [none, weak, medium, strong], followed by CORN conditionals.
        rank = dataset_probabilities[..., [0, 3, 2, 1]]
        survival_0 = rank[..., 1:].sum(dim=-1)
        survival_1 = rank[..., 2:].sum(dim=-1)
        survival_2 = rank[..., 3]
        conditionals = torch.stack(
            [
                survival_0,
                survival_1 / survival_0.clamp_min(1e-6),
                survival_2 / survival_1.clamp_min(1e-6),
            ],
            dim=-1,
        ).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(conditionals)

    def forward(
        self,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_teacher_logits: torch.Tensor | None = None,
        video_teacher_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if video_feat.ndim != 3:
            raise ValueError(f"Temporal BORA expects video features [B, T, D], got {video_feat.shape}.")
        if video_feat.size(1) > self.position.size(1):
            raise ValueError(
                f"Received {video_feat.size(1)} frames, maximum is {self.position.size(1)}."
            )
        audio = self.proj_audio(audio_feat)
        frames = self.proj_video(video_feat)
        motion = torch.cat(
            [torch.zeros_like(frames[:, :1]), torch.abs(frames[:, 1:] - frames[:, :-1])], dim=1
        )
        motion_context = self.motion_projection(motion)
        tokens = self.temporal_encoder(
            frames + motion_context + self.position[:, : frames.size(1)]
        )
        audio_expanded = audio.unsqueeze(1).expand(-1, tokens.size(1), -1)
        event_logits = self.event_gate(torch.cat([tokens, audio_expanded, motion_context], dim=-1)).squeeze(-1)
        event_probabilities = torch.sigmoid(event_logits)

        queries = self.audio_query(audio).view(audio.size(0), 3, -1) + self.boundary_queries.unsqueeze(0)
        attention_logits = torch.einsum(
            "btd,bkd->bkt", F.normalize(tokens, dim=-1), F.normalize(queries, dim=-1)
        ) * self.attention_logit_scale.exp().clamp(max=100.0)
        attention_logits = attention_logits + torch.log(event_probabilities.clamp_min(1e-6)).unsqueeze(1)
        temporal_attention = F.softmax(attention_logits, dim=-1)
        video_boundaries = torch.einsum("bkt,btd->bkd", temporal_attention, tokens)
        video_global = video_boundaries.mean(dim=1)

        audio_ordinal = self.audio_ordinal_head(audio)
        video_ordinal = self.video_ordinal_head(video_global)
        teacher_audio_ordinal = None
        teacher_video_ordinal = None
        teacher_audio_probabilities = None
        teacher_video_probabilities = None
        if audio_teacher_logits is not None and video_teacher_logits is not None:
            teacher_audio_probabilities = F.softmax(audio_teacher_logits, dim=-1)
            teacher_video_probabilities = F.softmax(video_teacher_logits, dim=-1).mean(dim=1)
            teacher_audio_ordinal = self._teacher_probabilities_to_conditional_logits(
                teacher_audio_probabilities
            )
            teacher_video_ordinal = self._teacher_probabilities_to_conditional_logits(
                teacher_video_probabilities
            )
            residual_scale = torch.sigmoid(self.teacher_residual_scale).unsqueeze(0)
            audio_ordinal = audio_ordinal + residual_scale * teacher_audio_ordinal
            video_ordinal = video_ordinal + residual_scale * teacher_video_ordinal
        audio_reliability = self.audio_reliability_head(audio)
        video_reliability = self.video_reliability_head(video_boundaries).squeeze(-1)
        # Reliability alone can saturate close to one. Modulate it with each
        # auxiliary boundary margin, an entropy-like uncertainty signal, so an
        # uncertain modality cannot dominate merely because its MLP is biased
        # high. The floor retains gradients for genuinely complementary cues.
        audio_confidence = 2.0 * torch.abs(torch.sigmoid(audio_ordinal) - 0.5)
        video_confidence = 2.0 * torch.abs(torch.sigmoid(video_ordinal) - 0.5)
        audio_gate_reliability = audio_reliability * (0.25 + 0.75 * audio_confidence)
        video_gate_reliability = video_reliability * (0.25 + 0.75 * video_confidence)
        weights = self._gate_weights(audio_gate_reliability, video_gate_reliability)

        boundary_logits = []
        boundary_features = []
        for boundary in range(3):
            audio_boundary = self.audio_boundary_projections[boundary](audio)
            video_boundary = self.video_boundary_projections[boundary](video_boundaries[:, boundary])
            weighted = (
                weights[:, boundary, 0:1] * audio_boundary
                + weights[:, boundary, 1:2] * video_boundary
            )
            interaction = self.boundary_interactions[boundary](
                torch.cat(
                    [weighted, torch.abs(audio_boundary - video_boundary), audio_boundary * video_boundary],
                    dim=-1,
                )
            )
            learned_logit = self.boundary_classifiers[boundary](interaction)
            boundary_features.append(interaction)
            if teacher_audio_ordinal is not None and teacher_video_ordinal is not None:
                teacher_logit = (
                    weights[:, boundary, 0] * teacher_audio_ordinal[:, boundary]
                    + weights[:, boundary, 1] * teacher_video_ordinal[:, boundary]
                ).unsqueeze(1)
                learned_logit = learned_logit + torch.sigmoid(
                    self.teacher_residual_scale[boundary]
                ) * teacher_logit
            boundary_logits.append(learned_logit)
        ordinal_logits = torch.cat(boundary_logits, dim=1)
        rank_probabilities = conditional_logits_to_rank_probabilities(ordinal_logits)
        dataset_probabilities = rank_probabilities_to_dataset_order(rank_probabilities)
        pooled_boundary_evidence = torch.stack(boundary_features, dim=1).mean(dim=1)
        nominal_logits = self.nominal_head(
            torch.cat([pooled_boundary_evidence, audio, video_global], dim=-1)
        )
        ordinal_log_probabilities = torch.log(dataset_probabilities.clamp_min(1e-8))
        combined_logits = ordinal_log_probabilities + torch.sigmoid(
            self.nominal_residual_scale
        ) * nominal_logits
        clipwise_output = F.log_softmax(combined_logits, dim=-1)
        motion_score = self.motion_regressor(motion_context.mean(dim=1)).squeeze(-1)
        result = {
            "clipwise_output": clipwise_output,
            "rank_probabilities": rank_probabilities,
            "ordinal_logits": ordinal_logits,
            "audio_ordinal_logits": audio_ordinal,
            "video_ordinal_logits": video_ordinal,
            "audio_reliability": audio_reliability,
            "video_reliability": video_reliability,
            "audio_gate_reliability": audio_gate_reliability,
            "video_gate_reliability": video_gate_reliability,
            "gate_weights": weights[:, :, 0],
            "audio_gate_weights": weights[:, :, 0],
            "video_gate_weights": weights[:, :, 1],
            "temporal_attention": temporal_attention,
            "event_probabilities": event_probabilities,
            "motion_score": motion_score,
            "teacher_residual_scales": torch.sigmoid(self.teacher_residual_scale),
            "nominal_logits": nominal_logits,
            "nominal_residual_gate": torch.sigmoid(self.nominal_residual_scale),
        }
        if teacher_audio_probabilities is not None and teacher_video_probabilities is not None:
            result["audio_teacher_probabilities"] = teacher_audio_probabilities
            result["video_teacher_probabilities"] = teacher_video_probabilities
            # Raw logits for the label-anchored preservation loss.
            if audio_teacher_logits is not None:
                result["audio_teacher_logits"] = audio_teacher_logits
            if video_teacher_logits is not None:
                result["video_teacher_logits_mean"] = video_teacher_logits.mean(dim=1)
        return result


FUSION_HEADS = {
    "raw_concat": RawConcatFusion,
    "linear_concat": LinearConcatFusion,
    "linear_mean": LinearMeanFusion,
    "gated_fusion": GatedFusion,
    "self_attention": SelfAttentionFusion,
    "bora_fusion": BORAFusion,
    "temporal_bora_fusion": TemporalBORAFusion,
}


def build_fusion_head(dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> nn.Module:
    if cfg.type not in FUSION_HEADS:
        raise ValueError(f"Unsupported fusion type '{cfg.type}'. Expected one of {sorted(FUSION_HEADS)}.")
    return FUSION_HEADS[cfg.type](dim_audio=dim_audio, dim_video=dim_video, num_classes=num_classes, cfg=cfg)
