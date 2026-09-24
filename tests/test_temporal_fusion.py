import numpy as np
import pytest
import torch

from config import FusionConfig, TrainConfig
from dataset.video_loader import _stratified_frame_indices, _uniform_frame_indices
from models.fusion import TemporalReliabilityFusion
from transforms import VideoClipTransform
from utils.corruption import corrupt_bora_batch
from utils.losses import fusion_loss


LABELS = torch.tensor([0, 1, 2, 3, 0, 1])


def _fusion_config() -> FusionConfig:
    return FusionConfig(
        type="temporal_reliability_fusion",
        proj_dim=16,
        hidden_dim=16,
        dropout=0.0,
        activation="gelu",
        bora={
            "warmup_epochs": 2,
            "temporal_num_layers": 1,
            "temporal_num_heads": 4,
            "temporal_max_frames": 8,
            "motion_loss_weight": 0.1,
        },
    )


def _loss(output: dict) -> tuple[torch.Tensor, dict]:
    return fusion_loss(
        output,
        LABELS,
        aux_loss_weight=0.3,
        reliability_loss_weight=0.1,
        motion_loss_weight=0.1,
        teacher_preservation_weight=0.3,
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


def test_temporal_fusion_forward_warmup_adaptive_and_gradients() -> None:
    head = TemporalReliabilityFusion(10, 12, 4, _fusion_config())
    audio = torch.randn(6, 10, requires_grad=True)
    video = torch.randn(6, 5, 12, requires_grad=True)

    head.set_epoch(0)
    warmup = head(audio, video)
    assert warmup["clipwise_output"].shape == (6, 4)
    assert warmup["audio_aux_logits"].shape == (6, 4)
    assert warmup["video_aux_logits"].shape == (6, 3, 4)
    assert warmup["audio_reliability"].shape == (6, 1)
    assert warmup["video_reliability"].shape == (6, 3)
    assert warmup["temporal_attention"].shape == (6, 3, 5)
    assert warmup["event_probabilities"].shape == (6, 5)
    assert torch.allclose(warmup["audio_gate_weights"], torch.full((6, 3), 0.5))
    assert torch.allclose(warmup["video_gate_weights"], torch.full((6, 3), 0.5))
    assert torch.allclose(warmup["temporal_attention"].sum(-1), torch.ones(6, 3))
    assert "audio_teacher_logits" not in warmup

    head.set_epoch(3)
    audio_teacher = torch.randn(6, 4, requires_grad=True)
    video_teacher = torch.randn(6, 5, 4, requires_grad=True)
    output = head(audio, video, audio_teacher_logits=audio_teacher, video_teacher_logits=video_teacher)
    assert torch.allclose(output["clipwise_output"].exp().sum(1), torch.ones(6), atol=1e-5)
    assert torch.allclose(output["class_probabilities"].sum(1), torch.ones(6), atol=1e-5)
    assert torch.allclose(
        output["audio_gate_weights"] + output["video_gate_weights"], torch.ones(6, 3), atol=1e-6
    )
    assert 0.0 < float(output["teacher_prior_gate"]) < 1.0

    loss, parts = _loss(output)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"decision", "auxiliary", "reliability", "motion", "teacher_preservation"}
    assert audio.grad is not None and torch.isfinite(audio.grad).all()
    assert video.grad is not None and torch.isfinite(video.grad).all()
    assert audio_teacher.grad is not None and torch.isfinite(audio_teacher.grad).all()
    assert video_teacher.grad is not None and torch.isfinite(video_teacher.grad).all()
    for module in (
        head.event_gate[0],
        head.motion_regressor[0],
        head.audio_aux_head,
        head.video_aux_head,
        head.audio_reliability_head[0],
        head.video_reliability_head[0],
        head.nominal_head[0],
    ):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
    assert head.teacher_prior_scale.grad is not None
    assert head.pooling_queries.grad is not None


def test_teacher_prior_changes_decision_only_when_teachers_are_given() -> None:
    torch.manual_seed(1)
    head = TemporalReliabilityFusion(10, 12, 4, _fusion_config()).eval()
    head.set_epoch(3)
    audio, video = torch.randn(6, 10), torch.randn(6, 5, 12)
    plain = head(audio, video)
    confident_teacher = torch.full((6, 4), -10.0)
    confident_teacher[:, 2] = 10.0
    guided = head(
        audio,
        video,
        audio_teacher_logits=confident_teacher,
        video_teacher_logits=confident_teacher.unsqueeze(1).expand(-1, 5, -1),
    )
    assert (guided["class_probabilities"][:, 2] > plain["class_probabilities"][:, 2]).all()
    assert torch.equal(guided["audio_gate_weights"], plain["audio_gate_weights"])


def test_margin_confidence_is_top1_minus_top2_and_matches_binary_margin() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0, -3.0], [0.0, 0.0, 0.0, 0.0]])
    probabilities = torch.softmax(logits, dim=-1)
    top = probabilities.sort(dim=-1, descending=True).values
    margin = TemporalReliabilityFusion._margin_confidence(logits)
    assert torch.allclose(margin, top[:, 0] - top[:, 1])
    assert margin[1] == 0.0
    binary = torch.tensor([[1.3, -0.4]])
    p = torch.softmax(binary, dim=-1)[0, 0]
    assert torch.isclose(TemporalReliabilityFusion._margin_confidence(binary)[0], 2 * torch.abs(p - 0.5))


@pytest.mark.parametrize("gate_confidence", ["margin", "none"])
def test_temporal_fusion_gate_confidence_ablation(gate_confidence: str) -> None:
    cfg = _fusion_config()
    cfg.bora.gate_confidence = gate_confidence
    torch.manual_seed(5)
    head = TemporalReliabilityFusion(10, 12, 4, cfg)
    head.set_epoch(3)
    output = head(torch.randn(6, 10), torch.randn(6, 5, 12))
    audio_reliability = output["audio_reliability"].expand(-1, 3)

    if gate_confidence == "none":
        assert torch.equal(output["audio_gate_reliability"], audio_reliability)
        assert torch.equal(output["video_gate_reliability"], output["video_reliability"])
    else:
        assert (output["audio_gate_reliability"] <= audio_reliability + 1e-7).all()
        assert (output["video_gate_reliability"] <= output["video_reliability"] + 1e-7).all()
        assert not torch.equal(output["video_gate_reliability"], output["video_reliability"])

    loss, _ = fusion_loss(output, LABELS, aux_loss_weight=0.3, reliability_loss_weight=0.1)
    loss.backward()
    assert head.audio_reliability_head[0].weight.grad is not None
    assert head.video_reliability_head[0].weight.grad is not None
    assert torch.isfinite(head.audio_reliability_head[0].weight.grad).all()


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


def test_temporal_fusion_config_requires_multiple_frames() -> None:
    raw = {
        "optimizer": "adam",
        "monitor": "accuracy",
        "evaluation_mode": "holdout",
        "audio": {"backbone": "PANNS_Cnn6", "checkpoint_path": "audio.pt"},
        "video": {"backbone": "SwinTiny", "checkpoint_path": "video.pt"},
        "fusion": {"type": "temporal_reliability_fusion", "proj_dim": 16, "bora": {"temporal_num_heads": 4}},
        "video_features": {"num_frames": 1},
        "dataset": {"split_strategy": "random_sample"},
    }
    with pytest.raises(ValueError, match="num_frames>=2"):
        TrainConfig.model_validate(raw)
    raw["video_features"]["num_frames"] = 8
    assert TrainConfig.model_validate(raw).video_features.num_frames == 8
