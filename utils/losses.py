from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from utils.ordinal import labels_to_ranks


def fusion_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    aux_loss_weight: float,
    reliability_loss_weight: float,
    motion_loss_weight: float = 0.0,
    teacher_preservation_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Nominal fusion objective: decision CE plus auxiliary, reliability, motion and teacher terms."""
    labels = labels.long()
    decision = F.nll_loss(output["clipwise_output"], labels)

    # Deep supervision of each modality branch; the video head sees every query.
    video_aux_logits = output["video_aux_logits"]
    num_queries = video_aux_logits.size(1)
    audio_aux_ce = F.cross_entropy(output["audio_aux_logits"], labels, reduction="none")
    video_aux_ce = F.cross_entropy(
        video_aux_logits.reshape(-1, video_aux_logits.size(-1)),
        labels.repeat_interleave(num_queries),
        reduction="none",
    ).view(-1, num_queries)
    auxiliary = audio_aux_ce.mean() + video_aux_ce.mean()

    # Reliability regresses the probability each auxiliary head assigns to the
    # true class, exp(-CE); the target is detached so only the reliability heads learn from it.
    audio_target = torch.exp(-audio_aux_ce).detach().unsqueeze(-1)
    video_target = torch.exp(-video_aux_ce).detach()
    reliability = 0.5 * (
        F.smooth_l1_loss(output["audio_reliability"], audio_target)
        + F.smooth_l1_loss(output["video_reliability"], video_target)
    )

    if motion_loss_weight > 0.0:
        ranks = labels_to_ranks(labels)
        motion = F.smooth_l1_loss(output["motion_score"], ranks.to(torch.float32) / 3.0)
    else:
        motion = decision.new_zeros(())

    # Teacher preservation: label-anchored cross-entropy keeps the reused
    # single-modal classifiers discriminative while the encoders fine-tune.
    if teacher_preservation_weight > 0.0:
        for key in ("audio_teacher_logits", "video_teacher_logits_mean"):
            if key not in output:
                raise KeyError(f"{key} is required when teacher_preservation_weight > 0.")
        preservation = 0.5 * (
            F.cross_entropy(output["audio_teacher_logits"], labels)
            + F.cross_entropy(output["video_teacher_logits_mean"], labels)
        )
    else:
        preservation = decision.new_zeros(())

    total = (
        decision
        + aux_loss_weight * auxiliary
        + reliability_loss_weight * reliability
        + motion_loss_weight * motion
        + teacher_preservation_weight * preservation
    )
    return total, {
        "decision": decision.detach(),
        "auxiliary": auxiliary.detach(),
        "reliability": reliability.detach(),
        "motion": motion.detach(),
        "teacher_preservation": preservation.detach(),
    }
