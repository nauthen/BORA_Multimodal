from types import SimpleNamespace

import torch.nn as nn

from tasks.trainer import _optimizer


class _TinyMultimodalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_branch = nn.Linear(2, 3)
        self.video_branch = nn.Linear(4, 3)
        self.fusion = nn.Linear(6, 4)


def test_fusion_optimizer_uses_scaled_encoder_and_full_head_learning_rates() -> None:
    model = _TinyMultimodalModel()
    config = SimpleNamespace(
        learning_rate=1e-3,
        fusion=SimpleNamespace(
            type="temporal_reliability_fusion",
            bora=SimpleNamespace(encoder_lr_scale=0.1),
        ),
    )
    optimizer = _optimizer(model, config)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-3]

    encoder_ids = {
        id(parameter)
        for branch in (model.audio_branch, model.video_branch)
        for parameter in branch.parameters()
    }
    head_ids = {id(parameter) for parameter in model.fusion.parameters()}
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == encoder_ids
    assert {id(parameter) for parameter in optimizer.param_groups[1]["params"]} == head_ids
    assert encoder_ids.isdisjoint(head_ids)
