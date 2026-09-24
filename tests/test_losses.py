import pytest
import torch

from utils.losses import fusion_loss
from utils.ordinal import labels_to_ranks, ranks_to_labels


def _output(batch: int = 4, queries: int = 3, classes: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "clipwise_output": torch.log_softmax(torch.randn(batch, classes, requires_grad=True), dim=-1),
        "audio_aux_logits": torch.randn(batch, classes, requires_grad=True),
        "video_aux_logits": torch.randn(batch, queries, classes, requires_grad=True),
        "audio_reliability": torch.rand(batch, 1, requires_grad=True),
        "video_reliability": torch.rand(batch, queries, requires_grad=True),
        "motion_score": torch.rand(batch, requires_grad=True),
        "audio_teacher_logits": torch.randn(batch, classes, requires_grad=True),
        "video_teacher_logits_mean": torch.randn(batch, classes, requires_grad=True),
    }


def test_fusion_loss_is_finite_and_weights_switch_terms_off() -> None:
    output = _output()
    labels = torch.tensor([0, 1, 2, 3])
    loss, parts = fusion_loss(output, labels, 0.3, 0.1, 0.1, 0.3)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())

    decision_only, parts_off = fusion_loss(output, labels, 0.0, 0.0, 0.0, 0.0)
    assert torch.isclose(decision_only, parts_off["decision"])
    assert float(parts_off["motion"]) == 0.0
    assert float(parts_off["teacher_preservation"]) == 0.0


def test_reliability_target_is_detached_from_aux_heads() -> None:
    output = _output()
    labels = torch.tensor([0, 1, 2, 3])
    loss, parts = fusion_loss(output, labels, aux_loss_weight=0.0, reliability_loss_weight=1.0)
    loss.backward()
    # With the auxiliary CE switched off, aux heads could only receive gradient via the
    # reliability target, which must be detached.
    for key in ("audio_aux_logits", "video_aux_logits"):
        grad = output[key].grad
        assert grad is None or torch.count_nonzero(grad) == 0
    assert output["audio_reliability"].grad is not None
    assert torch.count_nonzero(output["audio_reliability"].grad) > 0
    assert float(parts["reliability"]) >= 0.0


def test_teacher_preservation_requires_teacher_logits() -> None:
    output = _output()
    del output["audio_teacher_logits"]
    with pytest.raises(KeyError, match="audio_teacher_logits"):
        fusion_loss(output, torch.tensor([0, 1, 2, 3]), 0.3, 0.1, teacher_preservation_weight=0.3)


def test_dataset_label_rank_mapping_round_trips() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    ranks = labels_to_ranks(labels)
    assert ranks.tolist() == [0, 3, 2, 1]
    assert torch.equal(ranks_to_labels(ranks), labels)
    with pytest.raises(ValueError):
        labels_to_ranks(torch.tensor([4]))
