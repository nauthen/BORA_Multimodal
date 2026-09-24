from __future__ import annotations

import math
from typing import Optional

import torch
import torchvision.transforms.functional as TF

from config import BoraConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _uniform(
    bounds: tuple[float, float], device: torch.device, generator: Optional[torch.Generator]
) -> float:
    low, high = bounds
    value = torch.rand((), device=device, generator=generator).item()
    return float(low + (high - low) * value)


def _corrupt_audio_sample(
    waveform: torch.Tensor, cfg: BoraConfig, generator: Optional[torch.Generator]
) -> torch.Tensor:
    choice = int(torch.randint(0, 3, (), device=waveform.device, generator=generator).item())
    if choice == 0:
        snr_db = _uniform(cfg.audio_snr_db, waveform.device, generator)
        signal_rms = waveform.square().mean().sqrt().clamp_min(1e-6)
        noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise = torch.randn(
            waveform.shape,
            dtype=waveform.dtype,
            device=waveform.device,
            generator=generator,
        )
        return waveform + noise * noise_rms
    if choice == 1:
        return waveform * _uniform(cfg.audio_gain, waveform.device, generator)

    ratio = _uniform(cfg.temporal_mask_ratio, waveform.device, generator)
    width = min(waveform.numel(), max(1, int(round(waveform.numel() * ratio))))
    max_start = waveform.numel() - width
    start = int(torch.randint(0, max_start + 1, (), device=waveform.device, generator=generator).item())
    result = waveform.clone()
    result[start : start + width] = 0
    return result


def _denormalize_video(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return image * std + mean


def _normalize_video(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std


def _corrupt_video_sample(
    normalized_image: torch.Tensor, cfg: BoraConfig, generator: Optional[torch.Generator]
) -> torch.Tensor:
    image = _denormalize_video(normalized_image).clamp(0.0, 1.0)
    choice = int(torch.randint(0, 3, (), device=image.device, generator=generator).item())
    if choice == 0:
        image = image * _uniform(cfg.video_brightness, image.device, generator)
    elif choice == 1:
        sigma = max(1e-3, _uniform(cfg.video_blur_sigma, image.device, generator))
        max_kernel = max(1, min(image.shape[-2:]))
        if max_kernel % 2 == 0:
            max_kernel -= 1
        kernel = min(max_kernel, max(3, 2 * int(math.ceil(3.0 * sigma)) + 1))
        image = TF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    else:
        ratio = _uniform(cfg.video_occlusion_ratio, image.device, generator)
        height, width = image.shape[-2:]
        side = min(height, width, max(1, int(round(math.sqrt(ratio) * min(height, width)))))
        top = int(torch.randint(0, height - side + 1, (), device=image.device, generator=generator).item())
        left = int(torch.randint(0, width - side + 1, (), device=image.device, generator=generator).item())
        image = image.clone()
        image[..., top : top + side, left : left + side] = 0.0
    return _normalize_video(image.clamp(0.0, 1.0))


def corrupt_bora_batch(
    waveforms: torch.Tensor,
    videos: torch.Tensor,
    cfg: BoraConfig,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mutually exclusive clean, single-modality corruption, or single-modality dropout per sample."""
    if waveforms.ndim != 2 or videos.ndim not in (4, 5) or waveforms.size(0) != videos.size(0):
        raise ValueError(
            "Expected waveforms [B, L] and videos [B, C, H, W] or [B, T, C, H, W], "
            f"got {waveforms.shape} and {videos.shape}."
        )
    corrupted_waveforms = waveforms.clone()
    corrupted_videos = videos.clone()
    action = torch.rand(waveforms.size(0), device=waveforms.device, generator=generator)
    modality = torch.randint(0, 2, (waveforms.size(0),), device=waveforms.device, generator=generator)
    dropout_limit = cfg.modality_dropout_probability
    corruption_limit = dropout_limit + cfg.corruption_probability

    for index in range(waveforms.size(0)):
        action_value = float(action[index].item())
        modality_value = int(modality[index].item())
        if action_value < dropout_limit:
            if modality_value == 0:
                corrupted_waveforms[index].zero_()
            else:
                corrupted_videos[index].zero_()
        elif action_value < corruption_limit:
            if modality_value == 0:
                corrupted_waveforms[index] = _corrupt_audio_sample(
                    corrupted_waveforms[index], cfg, generator
                )
            else:
                corrupted_videos[index] = _corrupt_video_sample(
                    corrupted_videos[index], cfg, generator
                )
    return corrupted_waveforms, corrupted_videos
