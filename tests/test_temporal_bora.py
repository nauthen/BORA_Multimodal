import numpy as np
import pytest
import torch

from config import FusionConfig, TrainConfig
from dataset.video_loader import _stratified_frame_indices, _uniform_frame_indices
from models.fusion import TemporalBORAFusion
from transforms import VideoClipTransform
from utils.corruption import corrupt_bora_batch
from utils.ordinal import bora_loss


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
