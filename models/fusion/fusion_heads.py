from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FusionConfig


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


class TemporalReliabilityFusion(nn.Module):
    """Motion-aware temporal fusion with reliability-gated audio/video evidence and one nominal decoder."""

    def __init__(self, dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> None:
        super().__init__()
        if cfg.proj_dim % cfg.bora.temporal_num_heads != 0:
            raise ValueError("proj_dim must be divisible by temporal_num_heads.")
        self.cfg = cfg
        self.num_queries = 3
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
        self.pooling_queries = nn.Parameter(torch.empty(self.num_queries, dim))
        nn.init.trunc_normal_(self.pooling_queries, std=0.02)
        self.attention_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.audio_query = nn.Linear(dim, dim * self.num_queries)
        # Auxiliary per-modality classifiers: deep supervision for each branch and
        # the source of the label-free margin confidence used by the gate.
        self.audio_aux_head = nn.Linear(dim, num_classes)
        self.video_aux_head = nn.Linear(dim, num_classes)
        self.audio_reliability_head = nn.Sequential(
            nn.Linear(dim, hidden), _activation(cfg.activation), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.video_reliability_head = nn.Sequential(
            nn.Linear(dim, hidden), _activation(cfg.activation), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.audio_query_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(self.num_queries))
        self.video_query_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(self.num_queries))
        self.query_interactions = nn.ModuleList(
            nn.Sequential(nn.Linear(dim * 3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(cfg.dropout))
            for _ in range(self.num_queries)
        )
        self.motion_regressor = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        # Single decoder over every query-specific evidence (concatenated so the
        # queries keep distinct gradients) plus global audio and pooled video.
        self.nominal_head = nn.Sequential(
            nn.Linear(dim * (self.num_queries + 2), hidden),
            nn.LayerNorm(hidden),
            _activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, num_classes),
        )
        # Teacher log-probabilities act as a prior in logit space; sigmoid(0) = 0.5.
        self.teacher_prior_scale = nn.Parameter(torch.tensor(0.0))

    def set_epoch(self, epoch: int | None) -> None:
        self._warmup_active = epoch is not None and epoch < self.warmup_epochs

    def _gate_weights(
        self, audio_reliability: torch.Tensor, video_reliability: torch.Tensor
    ) -> torch.Tensor:
        if self._warmup_active:
            return torch.full(
                (*video_reliability.shape, 2), 0.5,
                dtype=video_reliability.dtype, device=video_reliability.device,
            )
        reliability = torch.stack(
            [audio_reliability.expand_as(video_reliability), video_reliability], dim=-1
        ).clamp_min(1e-6)
        return F.softmax(torch.log(reliability) / self.cfg.bora.gate_temperature, dim=-1)

    @staticmethod
    def _margin_confidence(logits: torch.Tensor) -> torch.Tensor:
        # Top-1 minus top-2 probability; for two classes this equals 2|p - 0.5|.
        top2 = F.softmax(logits, dim=-1).topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]

    def forward(
        self,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_teacher_logits: torch.Tensor | None = None,
        video_teacher_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if video_feat.ndim != 3:
            raise ValueError(f"Temporal fusion expects video features [B, T, D], got {video_feat.shape}.")
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

        queries = (
            self.audio_query(audio).view(audio.size(0), self.num_queries, -1)
            + self.pooling_queries.unsqueeze(0)
        )
        attention_logits = torch.einsum(
            "btd,bkd->bkt", F.normalize(tokens, dim=-1), F.normalize(queries, dim=-1)
        ) * self.attention_logit_scale.exp().clamp(max=100.0)
        attention_logits = attention_logits + torch.log(event_probabilities.clamp_min(1e-6)).unsqueeze(1)
        temporal_attention = F.softmax(attention_logits, dim=-1)
        video_queries = torch.einsum("bkt,btd->bkd", temporal_attention, tokens)
        video_global = video_queries.mean(dim=1)

        audio_aux_logits = self.audio_aux_head(audio)
        video_aux_logits = self.video_aux_head(video_queries)
        audio_reliability = self.audio_reliability_head(audio)
        video_reliability = self.video_reliability_head(video_queries).squeeze(-1)
        if self.cfg.bora.gate_confidence == "margin":
            # Learned reliability tends to saturate near one; the label-free margin
            # of each auxiliary decision keeps an uncertain modality from
            # dominating. The floor retains gradients for complementary cues.
            audio_confidence = self._margin_confidence(audio_aux_logits).unsqueeze(-1)
            video_confidence = self._margin_confidence(video_aux_logits)
            audio_gate_reliability = audio_reliability * (0.25 + 0.75 * audio_confidence)
            video_gate_reliability = video_reliability * (0.25 + 0.75 * video_confidence)
        else:
            # Ablation: the gate sees only the raw learned reliability.
            audio_gate_reliability = audio_reliability
            video_gate_reliability = video_reliability
        weights = self._gate_weights(audio_gate_reliability, video_gate_reliability)

        query_features = []
        for query in range(self.num_queries):
            audio_query = self.audio_query_projections[query](audio)
            video_query = self.video_query_projections[query](video_queries[:, query])
            weighted = (
                weights[:, query, 0:1] * audio_query
                + weights[:, query, 1:2] * video_query
            )
            query_features.append(
                self.query_interactions[query](
                    torch.cat([weighted, torch.abs(audio_query - video_query), audio_query * video_query], dim=-1)
                )
            )
        logits = self.nominal_head(torch.cat([*query_features, audio, video_global], dim=-1))
        result: dict[str, torch.Tensor] = {}
        if audio_teacher_logits is not None and video_teacher_logits is not None:
            mean_weights = weights.mean(dim=1)
            teacher_audio_log_probabilities = F.log_softmax(audio_teacher_logits, dim=-1)
            teacher_video_log_probabilities = torch.log(
                F.softmax(video_teacher_logits, dim=-1).mean(dim=1).clamp_min(1e-8)
            )
            teacher_prior = (
                mean_weights[:, 0:1] * teacher_audio_log_probabilities
                + mean_weights[:, 1:2] * teacher_video_log_probabilities
            )
            logits = logits + torch.sigmoid(self.teacher_prior_scale) * teacher_prior
            # Raw logits for the label-anchored preservation loss.
            result["audio_teacher_logits"] = audio_teacher_logits
            result["video_teacher_logits_mean"] = video_teacher_logits.mean(dim=1)
        clipwise_output = F.log_softmax(logits, dim=-1)
        motion_score = self.motion_regressor(motion_context.mean(dim=1)).squeeze(-1)
        result.update({
            "clipwise_output": clipwise_output,
            "class_probabilities": clipwise_output.exp(),
            "audio_aux_logits": audio_aux_logits,
            "video_aux_logits": video_aux_logits,
            "audio_reliability": audio_reliability,
            "video_reliability": video_reliability,
            "audio_gate_reliability": audio_gate_reliability.expand_as(video_gate_reliability),
            "video_gate_reliability": video_gate_reliability,
            "audio_gate_weights": weights[:, :, 0],
            "video_gate_weights": weights[:, :, 1],
            "temporal_attention": temporal_attention,
            "event_probabilities": event_probabilities,
            "motion_score": motion_score,
            "teacher_prior_gate": torch.sigmoid(self.teacher_prior_scale),
        })
        return result


FUSION_HEADS = {
    "raw_concat": RawConcatFusion,
    "linear_concat": LinearConcatFusion,
    "linear_mean": LinearMeanFusion,
    "gated_fusion": GatedFusion,
    "self_attention": SelfAttentionFusion,
    "temporal_reliability_fusion": TemporalReliabilityFusion,
}


def build_fusion_head(dim_audio: int, dim_video: int, num_classes: int, cfg: FusionConfig) -> nn.Module:
    if cfg.type not in FUSION_HEADS:
        raise ValueError(f"Unsupported fusion type '{cfg.type}'. Expected one of {sorted(FUSION_HEADS)}.")
    return FUSION_HEADS[cfg.type](dim_audio=dim_audio, dim_video=dim_video, num_classes=num_classes, cfg=cfg)
