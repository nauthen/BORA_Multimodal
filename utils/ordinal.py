from __future__ import annotations

import torch


# Dataset order: none=0, strong=1, medium=2, weak=3.
# Intensity rank order: none=0, weak=1, medium=2, strong=3.
# Used for ordinal evaluation metrics and the motion-intensity target only.
DATASET_LABEL_TO_RANK = (0, 3, 2, 1)
RANK_TO_DATASET_LABEL = (0, 3, 2, 1)
RANK_NAMES = ("none", "weak", "medium", "strong")


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
