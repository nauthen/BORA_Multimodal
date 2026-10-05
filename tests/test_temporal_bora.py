import numpy as np
import pytest
import torch

from config import FusionConfig, TrainConfig
from dataset.video_loader import _stratified_frame_indices, _uniform_frame_indices
from models.fusion import TemporalBORAFusion
from transforms import VideoClipTransform
from utils.corruption import corrupt_bora_batch
from utils.ordinal import bora_loss, rank_probabilities_to_dataset_order


def _fusion_config() -> FusionConfig:
    return FusionConfig(
        type="temporal_bora_fusion",
        proj_dim=16,
        hidden_dim=16,
        dropout=0.0,
        activation="gelu",
        bora={
            "warmup_epochs": 2,
            "temporal_num_layers": 1,
            "temporal_num_heads": 4,
            "temporal_max_frames": 8,
            "categorical_loss_weight": 0.5,
            "motion_loss_weight": 0.1,
        },
    )


def test_uniform_frame_indices_cover_clip_and_repeat_short_clips() -> None:
    assert _uniform_frame_indices(50, 4).tolist() == [0, 16, 33, 49]
    assert _uniform_frame_indices(2, 4).tolist() == [0, 0, 1, 1]
    assert _uniform_frame_indices(0, 3).tolist() == [0, 0, 0]


def test_stratified_temporal_views_are_valid_and_distinct() -> None:
    original = _uniform_frame_indices(frame_count=50, num_frames=8)
    quarter = _stratified_frame_indices(frame_count=50, num_frames=8, temporal_offset=0.25)
    center = _stratified_frame_indices(frame_count=50, num_frames=8, temporal_offset=0.5)
    late = _stratified_frame_indices(frame_count=50, num_frames=8, temporal_offset=0.75)
    for indexes in (original, quarter, center, late):
        assert indexes.shape == (8,)
        assert (indexes >= 0).all() and (indexes < 50).all()
        assert (indexes[:-1] <= indexes[1:]).all()
    assert len({tuple(view) for view in (original, quarter, center, late)}) == 4


def test_clip_transform_preserves_shape_and_frame_consistency() -> None:
    frame = np.full((3, 20, 24), 127, dtype=np.uint8)
    clip = np.stack([frame, frame, frame, frame])
    torch.manual_seed(3)
    output = VideoClipTransform(image_size=16, train=True)(clip)
    assert output.shape == (4, 3, 16, 16)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.equal(output[0], output[1])


def test_temporal_bora_forward_warmup_adaptive_and_gradients() -> None:
    head = TemporalBORAFusion(10, 12, 4, _fusion_config())
    audio = torch.randn(6, 10, requires_grad=True)
    video = torch.randn(6, 5, 12, requires_grad=True)

    head.set_epoch(0)
    warmup = head(audio, video)
    assert warmup["clipwise_output"].shape == (6, 4)
    assert warmup["ordinal_logits"].shape == (6, 3)
    assert warmup["temporal_attention"].shape == (6, 3, 5)
    assert warmup["event_probabilities"].shape == (6, 5)
    assert torch.allclose(warmup["audio_gate_weights"], torch.full((6, 3), 0.5))
    assert torch.allclose(warmup["video_gate_weights"], torch.full((6, 3), 0.5))
    assert torch.allclose(warmup["temporal_attention"].sum(-1), torch.ones(6, 3))

    head.set_epoch(3)
    audio_teacher = torch.randn(6, 4, requires_grad=True)
    video_teacher = torch.randn(6, 5, 4, requires_grad=True)
    output = head(
        audio,
        video,
        audio_teacher_logits=audio_teacher,
        video_teacher_logits=video_teacher,
    )
    assert output["audio_teacher_probabilities"].shape == (6, 4)
    assert output["video_teacher_probabilities"].shape == (6, 4)
    assert output["nominal_logits"].shape == (6, 4)
    assert 0.0 < float(output["nominal_residual_gate"]) < 1.0
    assert torch.allclose(output["clipwise_output"].exp().sum(1), torch.ones(6), atol=1e-5)
    assert torch.allclose(output["audio_teacher_probabilities"].sum(1), torch.ones(6), atol=1e-6)
    assert torch.allclose(output["video_teacher_probabilities"].sum(1), torch.ones(6), atol=1e-6)
    assert torch.allclose(
        output["audio_gate_weights"] + output["video_gate_weights"], torch.ones(6, 3), atol=1e-6
    )
    loss, parts = bora_loss(
        output,
        torch.tensor([0, 1, 2, 3, 0, 1]),
        aux_loss_weight=0.3,
        reliability_loss_weight=0.1,
        categorical_loss_weight=0.5,
        motion_loss_weight=0.1,
        nominal_loss_weight=0.5,
        teacher_preservation_weight=0.3,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) >= {
        "fused",
        "categorical",
        "motion",
        "reliability",
        "nominal",
        "teacher_preservation",
    }
    assert audio.grad is not None and torch.isfinite(audio.grad).all()
    assert video.grad is not None and torch.isfinite(video.grad).all()
    assert audio_teacher.grad is not None and torch.isfinite(audio_teacher.grad).all()
    assert video_teacher.grad is not None and torch.isfinite(video_teacher.grad).all()
    assert head.event_gate[0].weight.grad is not None
    assert head.motion_regressor[0].weight.grad is not None
    assert head.audio_reliability_head[0].weight.grad is not None
    assert head.teacher_residual_scale.grad is not None
    assert head.nominal_head[0].weight.grad is not None
    assert head.nominal_residual_scale.grad is not None


def test_temporal_corruption_accepts_clips() -> None:
    cfg = _fusion_config().bora
    waveform = torch.randn(5, 128)
    video = torch.randn(5, 4, 3, 16, 16)
    output_audio, output_video = corrupt_bora_batch(
        waveform, video, cfg, generator=torch.Generator().manual_seed(11)
    )
    assert output_audio.shape == waveform.shape
    assert output_video.shape == video.shape
    assert torch.isfinite(output_audio).all()
    assert torch.isfinite(output_video).all()


def test_temporal_bora_config_requires_multiple_frames() -> None:
    raw = {
        "optimizer": "adam",
        "monitor": "accuracy",
        "evaluation_mode": "holdout",
        "audio": {"backbone": "PANNS_Cnn6", "checkpoint_path": "audio.pt"},
        "video": {"backbone": "SwinTiny", "checkpoint_path": "video.pt"},
        "fusion": {"type": "temporal_bora_fusion", "proj_dim": 16, "bora": {"temporal_num_heads": 4}},
        "video_features": {"num_frames": 1},
        "dataset": {"split_strategy": "random_sample"},
    }
    with pytest.raises(ValueError, match="num_frames>=2"):
        TrainConfig.model_validate(raw)
    raw["video_features"]["num_frames"] = 8
    assert TrainConfig.model_validate(raw).video_features.num_frames == 8


@pytest.mark.parametrize("gate_confidence", ["margin", "none"])
def test_temporal_bora_gate_confidence_ablation(gate_confidence: str) -> None:
    cfg = _fusion_config()
    cfg.bora.gate_confidence = gate_confidence
    torch.manual_seed(5)
    head = TemporalBORAFusion(10, 12, 4, cfg)
    head.set_epoch(3)
    output = head(torch.randn(6, 10), torch.randn(6, 5, 12))

    if gate_confidence == "none":
        assert torch.equal(output["audio_gate_reliability"], output["audio_reliability"])
        assert torch.equal(output["video_gate_reliability"], output["video_reliability"])
    else:
        assert (output["audio_gate_reliability"] <= output["audio_reliability"] + 1e-7).all()
        assert (output["video_gate_reliability"] <= output["video_reliability"] + 1e-7).all()
        assert not torch.equal(output["audio_gate_reliability"], output["audio_reliability"])

    loss, _ = bora_loss(
        output,
        torch.tensor([0, 1, 2, 3, 0, 1]),
        aux_loss_weight=0.3,
        reliability_loss_weight=0.1,
        nominal_loss_weight=0.5,
    )
    loss.backward()
    assert head.audio_reliability_head[0].weight.grad is not None
    assert head.video_reliability_head[0].weight.grad is not None
    assert torch.isfinite(head.audio_reliability_head[0].weight.grad).all()


def test_full_model_keeps_construction_order() -> None:
    # Construction order fixes the seeded initialization; ablation switches must
    # not reorder the full model's modules or parameters.
    head = TemporalBORAFusion(10, 12, 4, _fusion_config())
    assert [name for name, _ in head.named_children()] == [
        "proj_audio",
        "proj_video",
        "temporal_encoder",
        "motion_projection",
        "event_gate",
        "audio_query",
        "audio_ordinal_head",
        "video_ordinal_head",
        "audio_reliability_head",
        "video_reliability_head",
        "audio_boundary_projections",
        "video_boundary_projections",
        "boundary_interactions",
        "boundary_classifiers",
        "motion_regressor",
        "nominal_head",
    ]
    assert list(head._parameters) == [
        "position",
        "boundary_queries",
        "attention_logit_scale",
        "teacher_residual_scale",
        "nominal_residual_scale",
    ]


@pytest.mark.parametrize(
    ("temporal_motion", "decoders"),
    [("explicit", "dual"), ("none", "dual"), ("explicit", "ordinal"), ("explicit", "nominal")],
)
def test_temporal_bora_ablation_variants(temporal_motion: str, decoders: str) -> None:
    cfg = _fusion_config()
    cfg.bora.temporal_motion = temporal_motion
    cfg.bora.decoders = decoders
    torch.manual_seed(7)
    head = TemporalBORAFusion(10, 12, 4, cfg)
    head.set_epoch(3)
    use_motion = temporal_motion == "explicit"
    use_ordinal = decoders != "nominal"
    use_nominal = decoders != "ordinal"

    assert hasattr(head, "motion_projection") == use_motion
    assert hasattr(head, "motion_regressor") == use_motion
    assert head.event_gate[0].in_features == 16 * (3 if use_motion else 2)
    assert hasattr(head, "boundary_classifiers") == use_ordinal
    assert hasattr(head, "nominal_head") == use_nominal
    assert hasattr(head, "nominal_residual_scale") == (use_ordinal and use_nominal)

    output = head(
        torch.randn(6, 10),
        torch.randn(6, 2, 12),
        audio_teacher_logits=torch.randn(6, 4),
        video_teacher_logits=torch.randn(6, 2, 4),
    )
    assert ("motion_score" in output) == use_motion
    assert ("ordinal_logits" in output) == use_ordinal
    assert ("nominal_logits" in output) == use_nominal
    assert ("nominal_residual_gate" in output) == (use_ordinal and use_nominal)
    assert output["clipwise_output"].shape == (6, 4)
    assert torch.allclose(output["clipwise_output"].exp().sum(1), torch.ones(6), atol=1e-5)
    assert output["rank_probabilities"].shape == (6, 4)
    assert torch.allclose(output["rank_probabilities"].sum(1), torch.ones(6), atol=1e-5)
    # The prediction comes from the kept decoder alone.
    dataset_probabilities = rank_probabilities_to_dataset_order(output["rank_probabilities"])
    if decoders == "ordinal":
        assert torch.allclose(output["clipwise_output"].exp(), dataset_probabilities, atol=1e-5)
    if decoders == "nominal":
        assert torch.allclose(output["clipwise_output"], torch.log_softmax(output["nominal_logits"], dim=-1))
        assert torch.allclose(output["clipwise_output"].exp(), dataset_probabilities)

    loss, parts = bora_loss(
        output,
        torch.tensor([0, 1, 2, 3, 0, 1]),
        aux_loss_weight=0.3,
        reliability_loss_weight=0.1,
        categorical_loss_weight=0.5,
        motion_loss_weight=0.1 if use_motion else 0.0,
        nominal_loss_weight=0.5 if decoders == "dual" else 0.0,
        teacher_preservation_weight=0.3,
        ordinal_loss_weight=1.0 if use_ordinal else 0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert (float(parts["fused"]) > 0.0) == use_ordinal
    assert head.event_gate[0].weight.grad is not None
    assert head.audio_reliability_head[0].weight.grad is not None
    if use_ordinal:
        assert head.boundary_classifiers[0].weight.grad is not None
    if use_nominal:
        assert head.nominal_head[0].weight.grad is not None
    if use_motion:
        assert head.motion_regressor[0].weight.grad is not None


def test_bora_loss_requires_ordinal_logits_unless_disabled() -> None:
    cfg = _fusion_config()
    cfg.bora.decoders = "nominal"
    output = TemporalBORAFusion(10, 12, 4, cfg)(torch.randn(4, 10), torch.randn(4, 2, 12))
    labels = torch.tensor([0, 1, 2, 3])
    with pytest.raises(KeyError, match="ordinal_logits"):
        bora_loss(output, labels, aux_loss_weight=0.3, reliability_loss_weight=0.1, motion_loss_weight=0.1)
    loss, parts = bora_loss(
        output,
        labels,
        aux_loss_weight=0.3,
        reliability_loss_weight=0.1,
        categorical_loss_weight=1.0,
        motion_loss_weight=0.1,
        ordinal_loss_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert float(parts["fused"]) == 0.0
