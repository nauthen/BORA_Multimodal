import pytest
import torch

from config import FusionConfig
from models.fusion import BORAFusion, build_fusion_head
from utils.ordinal import bora_loss


def _config(fusion_type: str = "bora_fusion") -> FusionConfig:
    return FusionConfig(
        type=fusion_type,
        proj_dim=8,
        hidden_dim=6,
        dropout=0.0,
        use_batchnorm=False,
        bora={"warmup_epochs": 5, "gate_temperature": 0.7},
    )


def test_bora_forward_shapes_and_warmup_gate() -> None:
    head = BORAFusion(5, 7, 4, _config()).eval()
    head.set_epoch(0)
    output = head(torch.randn(4, 5), torch.randn(4, 7))
    assert output["clipwise_output"].shape == (4, 4)
    for key in (
        "ordinal_logits",
        "audio_ordinal_logits",
        "video_ordinal_logits",
        "audio_reliability",
        "video_reliability",
        "gate_weights",
    ):
        assert output[key].shape == (4, 3)
    assert torch.allclose(output["rank_probabilities"].sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.allclose(output["audio_gate_weights"], torch.full((4, 3), 0.5))
    assert torch.allclose(output["video_gate_weights"], torch.full((4, 3), 0.5))


def test_adaptive_gate_uses_predicted_reliability() -> None:
    head = BORAFusion(5, 7, 4, _config()).eval()
    with torch.no_grad():
        head.audio_reliability_head[-2].weight.zero_()
        head.audio_reliability_head[-2].bias.fill_(2.0)
        head.video_reliability_head[-2].weight.zero_()
        head.video_reliability_head[-2].bias.fill_(-2.0)
    head.set_epoch(5)
    output = head(torch.randn(2, 5), torch.randn(2, 7))
    assert torch.all(output["audio_gate_weights"] > output["video_gate_weights"])
    assert torch.allclose(
        output["audio_gate_weights"] + output["video_gate_weights"], torch.ones(2, 3)
    )


def test_bora_loss_is_finite_and_backpropagates_to_features_and_heads() -> None:
    head = BORAFusion(5, 7, 4, _config())
    audio_encoder = torch.nn.Linear(3, 5)
    video_encoder = torch.nn.Linear(4, 7)
    audio = audio_encoder(torch.randn(6, 3))
    video = video_encoder(torch.randn(6, 4))
    output = head(audio, video)
    loss, parts = bora_loss(output, torch.tensor([0, 3, 2, 1, 0, 1]), 0.3, 0.1)
    optimizer = torch.optim.Adam(
        [*audio_encoder.parameters(), *video_encoder.parameters(), *head.parameters()], lr=1e-3
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert audio_encoder.weight.grad is not None and torch.count_nonzero(audio_encoder.weight.grad) > 0
    assert video_encoder.weight.grad is not None and torch.count_nonzero(video_encoder.weight.grad) > 0
    assert head.proj_audio[0].weight.grad is not None
    assert head.proj_video[0].weight.grad is not None
    assert head.audio_ordinal_head.weight.grad is not None
    assert head.audio_reliability_head[-2].weight.grad is not None
    assert head.boundary_classifiers[0].weight.grad is not None


@pytest.mark.parametrize(
    "fusion_type", ["raw_concat", "linear_concat", "linear_mean", "gated_fusion", "self_attention"]
)
def test_existing_fusion_heads_keep_tensor_contract(fusion_type: str) -> None:
    config = _config(fusion_type)
    head = build_fusion_head(5, 7, 4, config)
    output = head(torch.randn(4, 5), torch.randn(4, 7))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 4)
