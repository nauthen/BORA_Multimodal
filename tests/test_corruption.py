import torch

from config import BoraConfig
from utils.corruption import corrupt_bora_batch


def test_modality_dropout_drops_exactly_one_modality_per_sample() -> None:
    config = BoraConfig(corruption_probability=0.0, modality_dropout_probability=1.0)
    waveform = torch.ones(12, 64)
    video = torch.ones(12, 3, 16, 16)
    corrupted_audio, corrupted_video = corrupt_bora_batch(
        waveform, video, config, generator=torch.Generator().manual_seed(7)
    )
    audio_dropped = corrupted_audio.abs().sum(dim=1) == 0
    video_dropped = corrupted_video.abs().sum(dim=(1, 2, 3)) == 0
    assert torch.all(torch.logical_xor(audio_dropped, video_dropped))
    assert torch.equal(waveform, torch.ones_like(waveform))
    assert torch.equal(video, torch.ones_like(video))


def test_corruption_preserves_shape_dtype_and_finite_values() -> None:
    config = BoraConfig(
        corruption_probability=1.0,
        modality_dropout_probability=0.0,
        audio_snr_db=(5.0, 5.0),
        video_blur_sigma=(0.5, 0.5),
    )
    waveform = torch.randn(16, 128, dtype=torch.float32)
    video = torch.randn(16, 3, 16, 16, dtype=torch.float32)
    corrupted_audio, corrupted_video = corrupt_bora_batch(
        waveform, video, config, generator=torch.Generator().manual_seed(11)
    )
    assert corrupted_audio.shape == waveform.shape and corrupted_audio.dtype == waveform.dtype
    assert corrupted_video.shape == video.shape and corrupted_video.dtype == video.dtype
    assert torch.isfinite(corrupted_audio).all()
    assert torch.isfinite(corrupted_video).all()
