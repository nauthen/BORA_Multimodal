import numpy as np
import pytest

import dataset.multimodal_dataset as multimodal_dataset
from dataset.multimodal_dataset import MultimodalFishDataset


def _entries(indexes):
    return [
        {"audio_path": f"/a/{i}.wav", "video_path": f"/v/{i}.mp4", "label": i % 4, "sample_key": f"k/{i}"}
        for i in indexes
    ]


@pytest.fixture
def decode_counter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(multimodal_dataset, "_AUDIO_RAM_CACHE", {})
    monkeypatch.setattr(multimodal_dataset, "_VIDEO_RAM_CACHE", {})
    calls = {"audio": [], "video": []}

    class FakeAudio:
        @staticmethod
        def load_audio(path, sr):
            calls["audio"].append(path)
            import torch

            return torch.full((1, 4), float(path.split("/")[-1].split(".")[0]))

    def fake_decode(self, video_path, label):
        calls["video"].append(video_path)
        return {"clip_form": np.full((2, 3, 4, 4), label, dtype=np.uint8), "target": label}

    monkeypatch.setattr(multimodal_dataset, "FishVoiceDataLoader", FakeAudio)
    monkeypatch.setattr(MultimodalFishDataset, "_decode_video", fake_decode)
    return calls


def _dataset(entries, split="train", num_frames=2):
    return MultimodalFishDataset(
        entries=entries, split=split, sample_rate=16, image_size=4,
        cache_audio=True, cache_video=True, num_workers=1, num_frames=num_frames,
    )


def test_folds_reuse_decoded_clips(decode_counter) -> None:
    # Fold 0 and fold 1 re-split the same six clips into different splits.
    fold0 = [_dataset(_entries([0, 1, 2, 3]), "train"), _dataset(_entries([4, 5]), "test")]
    fold1 = [_dataset(_entries([2, 3, 4, 5]), "train"), _dataset(_entries([0, 1]), "test")]

    assert sorted(decode_counter["audio"]) == sorted(f"/a/{i}.wav" for i in range(6))
    assert sorted(decode_counter["video"]) == sorted(f"/v/{i}.mp4" for i in range(6))
    # Every dataset keeps its own entry order over the shared arrays.
    assert [float(w[0]) for w in fold1[0].audio_cache] == [2.0, 3.0, 4.0, 5.0]
    assert fold1[1].video_cache[0] is fold0[0].video_cache[0]
    assert fold1[0][0]["sample_key"] == "k/2"


def test_different_decode_settings_are_cached_separately(decode_counter) -> None:
    _dataset(_entries([0]), num_frames=2)
    _dataset(_entries([0]), num_frames=8)
    assert decode_counter["video"] == ["/v/0.mp4", "/v/0.mp4"]
    assert decode_counter["audio"] == ["/a/0.wav"]
