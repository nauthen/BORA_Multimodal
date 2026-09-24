import torch
import torch.nn.functional as F

from utils.ordinal import (
    conditional_logits_to_rank_probabilities,
    corn_loss,
    fit_ordinal_thresholds,
    labels_to_ranks,
    ordinal_targets_and_mask,
    predict_ranks_with_thresholds,
    rank_probabilities_to_survival,
    ranks_to_labels,
)


def test_label_rank_mapping_round_trip() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    ranks = labels_to_ranks(labels)
    assert ranks.tolist() == [0, 3, 2, 1]
    assert torch.equal(ranks_to_labels(ranks), labels)


def test_ordinal_targets_and_active_mask() -> None:
    ranks = torch.tensor([0, 1, 2, 3])
    targets, active = ordinal_targets_and_mask(ranks)
    assert targets.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    assert active.tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
        [True, True, True],
    ]


def test_corn_loss_uses_only_active_boundaries() -> None:
    logits = torch.tensor([[0.0, 100.0, -100.0], [0.0, 0.0, 100.0]])
    ranks = torch.tensor([0, 1])
    loss = corn_loss(logits, ranks)
    expected = torch.stack(
        [
            F.binary_cross_entropy_with_logits(logits[0, 0], torch.tensor(0.0)),
            F.binary_cross_entropy_with_logits(logits[1, 0], torch.tensor(1.0)),
            F.binary_cross_entropy_with_logits(logits[1, 1], torch.tensor(0.0)),
        ]
    ).mean()
    assert torch.allclose(loss, expected)


def test_rank_probabilities_are_valid_and_monotonic() -> None:
    logits = torch.randn(32, 3)
    probabilities = conditional_logits_to_rank_probabilities(logits)
    cumulative = torch.cumprod(torch.sigmoid(logits), dim=1)
    assert probabilities.shape == (32, 4)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(32), atol=1e-6)
    assert torch.all(cumulative[:, :-1] >= cumulative[:, 1:])


def test_validation_fitted_thresholds_improve_shifted_boundaries() -> None:
    probabilities = torch.tensor(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.15, 0.45, 0.25, 0.15],
            [0.10, 0.20, 0.50, 0.20],
            [0.05, 0.10, 0.25, 0.60],
        ]
    ).repeat_interleave(4, dim=0)
    ranks = torch.arange(4).repeat_interleave(4)
    survival = rank_probabilities_to_survival(probabilities)
    assert torch.all(survival[:, :-1] >= survival[:, 1:])
    default = predict_ranks_with_thresholds(probabilities, torch.full((3,), 0.5))
    thresholds = fit_ordinal_thresholds(probabilities, ranks)
    calibrated = predict_ranks_with_thresholds(probabilities, thresholds)
    assert (calibrated == ranks).float().mean() > (default == ranks).float().mean()
    assert torch.equal(calibrated, ranks)
