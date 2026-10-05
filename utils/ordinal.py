from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# Dataset order: none=0, strong=1, medium=2, weak=3.
# Ordinal order: none=0, weak=1, medium=2, strong=3.
DATASET_LABEL_TO_RANK = (0, 3, 2, 1)
RANK_TO_DATASET_LABEL = (0, 3, 2, 1)
RANK_NAMES = ("none", "weak", "medium", "strong")
BOUNDARY_NAMES = ("none_to_weak", "weak_to_medium", "medium_to_strong")


def _lookup(values: torch.Tensor, mapping: tuple[int, ...]) -> torch.Tensor:
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"Expected integer tensor, got dtype={values.dtype}.")
    table = torch.tensor(mapping, dtype=torch.long, device=values.device)
    if values.numel() and (int(values.min()) < 0 or int(values.max()) >= len(mapping)):
        raise ValueError(f"Values must lie in [0, {len(mapping) - 1}].")
    return table[values.long()]


def labels_to_ranks(labels: torch.Tensor) -> torch.Tensor:
    return _lookup(labels, DATASET_LABEL_TO_RANK)


def ranks_to_labels(ranks: torch.Tensor) -> torch.Tensor:
    return _lookup(ranks, RANK_TO_DATASET_LABEL)


def ordinal_targets_and_mask(ranks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CORN conditional targets and active-task mask, both shaped [B, 3]."""
    if ranks.ndim != 1:
        raise ValueError(f"Expected ranks with shape [B], got {tuple(ranks.shape)}.")
    boundaries = torch.arange(3, device=ranks.device).unsqueeze(0)
    rank_column = ranks.long().unsqueeze(1)
    targets = (rank_column > boundaries).to(torch.float32)
    active = rank_column >= boundaries
    return targets, active


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have equal shapes, got {values.shape} and {mask.shape}.")
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def corn_loss(conditional_logits: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    if conditional_logits.ndim != 2 or conditional_logits.size(1) != 3:
        raise ValueError(f"Expected conditional logits [B, 3], got {tuple(conditional_logits.shape)}.")
    targets, active = ordinal_targets_and_mask(ranks)
    element_loss = F.binary_cross_entropy_with_logits(conditional_logits, targets, reduction="none")
    return masked_mean(element_loss, active)


def conditional_logits_to_rank_probabilities(conditional_logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN conditional logits to probabilities in ordinal rank order."""
    if conditional_logits.ndim != 2 or conditional_logits.size(1) != 3:
        raise ValueError(f"Expected conditional logits [B, 3], got {tuple(conditional_logits.shape)}.")
    cumulative = torch.cumprod(torch.sigmoid(conditional_logits), dim=1)
    probabilities = torch.stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2],
        ],
        dim=1,
    )
    return probabilities.clamp_min(0.0)


def rank_probabilities_to_dataset_order(rank_probabilities: torch.Tensor) -> torch.Tensor:
    if rank_probabilities.ndim != 2 or rank_probabilities.size(1) != 4:
        raise ValueError(f"Expected rank probabilities [B, 4], got {tuple(rank_probabilities.shape)}.")
    indexes = torch.tensor(DATASET_LABEL_TO_RANK, dtype=torch.long, device=rank_probabilities.device)
    return rank_probabilities.index_select(1, indexes)


def dataset_probabilities_to_rank_order(dataset_probabilities: torch.Tensor) -> torch.Tensor:
    if dataset_probabilities.ndim != 2 or dataset_probabilities.size(1) != 4:
        raise ValueError(f"Expected dataset probabilities [B, 4], got {tuple(dataset_probabilities.shape)}.")
    indexes = torch.tensor(RANK_TO_DATASET_LABEL, dtype=torch.long, device=dataset_probabilities.device)
    return dataset_probabilities.index_select(1, indexes)


def rank_probabilities_to_survival(rank_probabilities: torch.Tensor) -> torch.Tensor:
    """Return P(rank > k) for the three ordinal boundaries."""
    if rank_probabilities.ndim != 2 or rank_probabilities.size(1) != 4:
        raise ValueError(f"Expected rank probabilities [B, 4], got {tuple(rank_probabilities.shape)}.")
    return torch.stack(
        [
            rank_probabilities[:, 1:].sum(dim=1),
            rank_probabilities[:, 2:].sum(dim=1),
            rank_probabilities[:, 3],
        ],
        dim=1,
    )


def predict_ranks_with_thresholds(
    rank_probabilities: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Decode ranks using nested, validation-calibrated boundary decisions."""
    if thresholds.shape != (3,):
        raise ValueError(f"Expected thresholds [3], got {tuple(thresholds.shape)}.")
    survival = rank_probabilities_to_survival(rank_probabilities)
    thresholds = thresholds.to(device=survival.device, dtype=survival.dtype)
    above_0 = survival[:, 0] > thresholds[0]
    above_1 = above_0 & (survival[:, 1] > thresholds[1])
    above_2 = above_1 & (survival[:, 2] > thresholds[2])
    return above_0.long() + above_1.long() + above_2.long()


def fit_ordinal_thresholds(
    rank_probabilities: torch.Tensor,
    ranks: torch.Tensor,
    grid_size: int = 181,
    max_iterations: int = 8,
) -> torch.Tensor:
    """Fit three nested decision thresholds using validation accuracy only.

    Coordinate search keeps the decoder rank-consistent and uses 0.5 as a
    deterministic tie-breaker, limiting unnecessary calibration drift.
    """
    if ranks.ndim != 1 or ranks.size(0) != rank_probabilities.size(0):
        raise ValueError("ranks must have shape [B] and align with rank_probabilities.")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    survival = rank_probabilities_to_survival(rank_probabilities.detach())
    target = ranks.to(device=survival.device, dtype=torch.long)
    grid = torch.linspace(0.05, 0.95, grid_size, device=survival.device, dtype=survival.dtype)
    thresholds = torch.full((3,), 0.5, device=survival.device, dtype=survival.dtype)
    best_accuracy = (predict_ranks_with_thresholds(rank_probabilities, thresholds) == target).float().mean()

    for _ in range(max_iterations):
        changed = False
        for boundary in range(3):
            candidate_scores = []
            for candidate in grid:
                proposal = thresholds.clone()
                proposal[boundary] = candidate
                prediction = predict_ranks_with_thresholds(rank_probabilities, proposal)
                candidate_scores.append((prediction == target).float().mean())
            scores = torch.stack(candidate_scores)
            maximum = scores.max()
            tied = torch.nonzero(scores == maximum, as_tuple=False).flatten()
            tie_distances = torch.abs(grid.index_select(0, tied) - 0.5)
            selected = tied[tie_distances.argmin()]
            candidate_threshold = grid[selected]
            if maximum > best_accuracy or (
                torch.isclose(maximum, best_accuracy) and not torch.isclose(candidate_threshold, thresholds[boundary])
            ):
                changed = changed or not torch.isclose(candidate_threshold, thresholds[boundary])
                thresholds[boundary] = candidate_threshold
                best_accuracy = maximum
        if not changed:
            break
    return thresholds.detach()


def bora_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    aux_loss_weight: float,
    reliability_loss_weight: float,
    categorical_loss_weight: float = 0.0,
    motion_loss_weight: float = 0.0,
    nominal_loss_weight: float = 0.0,
    teacher_preservation_weight: float = 0.0,
    ordinal_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ranks = labels_to_ranks(labels)
    targets, active = ordinal_targets_and_mask(ranks)
    # CORN loss of the fused ordinal decoder; a nominal-only ablation has none.
    if ordinal_loss_weight > 0.0:
        if "ordinal_logits" not in output:
            raise KeyError("ordinal_logits is required when ordinal_loss_weight > 0.")
        fused = corn_loss(output["ordinal_logits"], ranks)
    else:
        fused = output["audio_ordinal_logits"].new_zeros(())
    audio = corn_loss(output["audio_ordinal_logits"], ranks)
    video = corn_loss(output["video_ordinal_logits"], ranks)

    audio_element_bce = F.binary_cross_entropy_with_logits(
        output["audio_ordinal_logits"], targets, reduction="none"
    )
    video_element_bce = F.binary_cross_entropy_with_logits(
        output["video_ordinal_logits"], targets, reduction="none"
    )
    audio_target = torch.exp(-audio_element_bce).detach()
    video_target = torch.exp(-video_element_bce).detach()
    audio_reliability = F.smooth_l1_loss(
        output["audio_reliability"], audio_target, reduction="none"
    )
    video_reliability = F.smooth_l1_loss(
        output["video_reliability"], video_target, reduction="none"
    )
    reliability = 0.5 * (
        masked_mean(audio_reliability, active) + masked_mean(video_reliability, active)
    )
    categorical = F.nll_loss(output["clipwise_output"], labels.long())
    if motion_loss_weight > 0.0:
        if "motion_score" not in output:
            raise KeyError("motion_score is required when motion_loss_weight > 0.")
        motion = F.smooth_l1_loss(output["motion_score"], ranks.to(torch.float32) / 3.0)
    else:
        motion = fused.new_zeros(())
    # Dual-decoder: direct nominal supervision so the exact-class decision is
    # not forced exclusively through the CORN conditional factorization.
    if nominal_loss_weight > 0.0:
        if "nominal_logits" not in output:
            raise KeyError("nominal_logits is required when nominal_loss_weight > 0.")
        nominal = F.cross_entropy(output["nominal_logits"], labels.long())
    else:
        nominal = fused.new_zeros(())
    # Teacher preservation: label-anchored cross-entropy keeps the reused
    # single-modal classifiers discriminative while the encoders fine-tune,
    # preventing the silent drift observed without this supervision.
    preservation_terms: list[torch.Tensor] = []
    for key in ("audio_teacher_logits", "video_teacher_logits_mean"):
        if teacher_preservation_weight > 0.0:
            if key not in output:
                raise KeyError(f"{key} is required when teacher_preservation_weight > 0.")
            preservation_terms.append(F.cross_entropy(output[key], labels.long()))
    if preservation_terms:
        preservation = torch.stack(preservation_terms).mean()
    else:
        preservation = fused.new_zeros(())
    total = (
        ordinal_loss_weight * fused
        + aux_loss_weight * (audio + video)
        + reliability_loss_weight * reliability
        + categorical_loss_weight * categorical
        + motion_loss_weight * motion
        + nominal_loss_weight * nominal
        + teacher_preservation_weight * preservation
    )
    return total, {
        "fused": fused.detach(),
        "audio_aux": audio.detach(),
        "video_aux": video.detach(),
        "reliability": reliability.detach(),
        "categorical": categorical.detach(),
        "motion": motion.detach(),
        "nominal": nominal.detach(),
        "teacher_preservation": preservation.detach(),
    }
