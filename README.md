# Multimodal Deep Fusion — Temporal Reliability Fusion (PANNs Cnn6 + MobileViT-XXS)

Audio-video fish feeding intensity classification (AV-FFIA, 27K two-second
clips, 4 classes `none`, `weak`, `medium`, `strong`).

Architecture under study: **Temporal Reliability Fusion** — an 8-frame,
motion-aware temporal fusion with a single 4-class (nominal) decoder:

- **Temporal video evidence** — per-frame MobileViT features, adjacent-frame
  motion residuals, a 2-layer temporal Transformer, an audio-conditioned event
  mask, and three audio-conditioned attention-pooling queries.
- **Reliability gate** — auxiliary audio/video classifiers supply a label-free
  top-1/top-2 margin confidence; learned reliability heads (regressing the
  auxiliary probability of the true class) are modulated by that margin, and a
  per-query softmax gate mixes audio and video evidence.
- **Nominal decoder** — one classifier over the concatenated query evidence,
  global audio and pooled video, plus a learned teacher prior (log-probabilities
  of the reused single-modal classifiers, mixed with the gate weights).
- **Teacher preservation** — label-anchored cross-entropy keeps the reused
  single-modal classifier heads discriminative during fine-tuning.

Loss: `CE(decision) + 0.3·CE(aux heads) + 0.1·reliability + 0.1·motion + 0.3·teacher preservation`.
The architecture and shapes are documented in `docs/temporal_reliability_fusion.md`.
The earlier CORN/ordinal design (Dual-Decoder Temporal BORA) is described in the
historical notes under `docs/`. The protocol is deliberately academic: one
training run, best checkpoint selected on clean validation only, test evaluated
once by the trainer, and every result reported with accuracy together with rank
MAE, QWK, within-one accuracy, and severe-error rate.

## 0. Prerequisites

- Same seed-42 random holdout as all previous runs
  (21,467 / 2,800 / 2,800), generated automatically by `FishDataSplitter`.
- **Audio teacher checkpoint**:
  `/marimo/checkpoints/audio_run/DL_audio/checkpoint/panns_cnn6/audio_best.pt`
  with its `splits/` sidecar beside it.
- **Video teacher checkpoint (MobileViT-XXS, timm `mobilevit_xxs`)**: the
  single-modal U_FFIA27K_video run
  `MobileViTXXS_holdout_random_sample_end_20260826_030403` (test acc 0.931,
  seed-42 random holdout, trained on the *end* frame at 224 px with ImageNet
  normalization). Copy its whole `checkpoint/mobilevit_xxs/` folder to
  `/marimo/checkpoints/video_run/U_FFIA27K_video/checkpoint/mobilevit_xxs/` so
  `video_best.pt` keeps its `splits/` sidecar beside it. Identity/label
  equality across splits is enforced by `validate_checkpoint_split_integrity`.
  The previous EfficientNetB0 setup is kept in
  `config/train_config.efficientnet_b0.json`.
- Dataset root: `/marimo/Fish_Feeding_Intensity_Dataset`.

## 1. Install

```bash
pip install -r requirements.txt
```

MobileViT-XXS needs `timm>=0.9` (the checkpoint strict-loads with timm 1.0.x).

Note: `decord` has no wheel for Python 3.13; the loader automatically falls
back to OpenCV decoding when decord is unavailable (slower preload, identical
sampling logic).

## 2. Train + evaluate

Edit only `config/train_config.json` (already set to
`MobileViTXXS + temporal_reliability_fusion`, `num_frames=8`,
`teacher_preservation_weight=0.3`), then:

To replace the PANNs Cnn6 audio branch with the Tiny PANNs + ECA checkpoint,
change only the audio block (leave the audio-feature settings identical to the
single-modal training run):

```json
"audio": {
  "backbone": "TinyPANNS_ECA",
  "pretrained": false,
  "freeze": false,
  "checkpoint_path": "/path/to/tiny_panns_eca/audio_best.pt"
}
```

Temporal reliability fusion accepts `TinyPANNS_ECA`. Its audio checkpoint
loader remains strict and expects the single-modal wrapper keys
`frontend.*` and `backbone.*`, preventing a silently partial or wrong-model
load.

```bash
python main.py
```

The trainer fits, selects the best epoch by validation accuracy, reloads that
checkpoint, runs the held-out test once, and writes to
`outputs/PANNS_Cnn6_MobileViTXXS_temporal_reliability_fusion/holdout/`:

- `result.csv` — accuracy, mAP, rank MAE, QWK, within-one, severe-error rate;
- `history.csv`, `learning_curves.png`, confusion outputs;
- `checkpoint/multimodal_best.pt` plus `checkpoint/snapshots/*.pt`
  (every new validation-best) for reproducibility/analysis;
- `predictions.csv`, `gate_summary.csv` — per-sample audit of gates,
  reliabilities and probabilities.

## Ablation: margin confidence in the reliability gate

`fusion.bora.gate_confidence` selects what the per-query gate sees:
`"margin"` (default) uses `r * (0.25 + 0.75 * (p_top1 - p_top2))` of the auxiliary heads,
`"none"` uses the raw learned reliability `r`. Everything else is unchanged.
The ablation run is written to a separate `..._temporal_reliability_fusion_noconf/`
directory, so it never overwrites the baseline.

```bash
python main.py                                                   # baseline
python main.py --config config/train_config.ablation_no_confidence.json
python scripts/compare_gate_ablation.py \
  outputs/PANNS_Cnn6_MobileViTXXS_temporal_reliability_fusion/holdout \
  outputs/PANNS_Cnn6_MobileViTXXS_temporal_reliability_fusion_noconf/holdout \
  --out outputs/gate_confidence_ablation.csv
```

The report gives test outcome metrics, per-query gate mechanics (video-gate
mean/std, fraction of "dead" gates in [0.45, 0.55], gate separation between
correct and wrong predictions, reliability mean/std) and an exact McNemar test
on the paired test predictions. Use several seeds when the accuracy gap is
below ~0.5%.

## Tests

```bash
pytest -q
```

Covers config validation, temporal views/transforms, fusion forward shapes,
warmup + adaptive gates, margin confidence, the teacher prior, gradient flow to
both encoders, auxiliary and reliability heads, the fusion loss terms,
corruption augmentation, optimizer parameter groups, MobileViT loading, and
checkpoint/split integrity.
