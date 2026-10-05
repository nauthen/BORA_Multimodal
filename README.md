# Multimodal Deep Fusion — Dual-Decoder Temporal BORA (PANNs Cnn6 + MobileViT-XXS)

Audio-video fish feeding intensity classification (AV-FFIA, 27K two-second
clips, 4 ordinal classes `none < weak < medium < strong`).

Architecture under study: **Dual-Decoder Temporal BORA-Fuse** — an 8-frame,
motion-aware, boundary-conditioned temporal fusion with two coupled decoders:

- **Ordinal decoder (CORN)** — three conditional boundaries with per-boundary
  temporal queries, uncertainty-calibrated reliability gates, and teacher
  decision residuals; guarantees rank-consistent predictions.
- **Nominal decoder (exact-class branch)** — a parallel 4-class head over
  pooled boundary evidence, coupled into the final decision through a learned
  logit-space residual gate.
- **Teacher preservation** — label-anchored cross-entropy keeps the reused
  single-modal PANNs/Swin-style classifier heads discriminative during
  fine-tuning, preventing the documented teacher drift.

Rationale, failed post-hoc alternatives that motivated this design, and the
hypotheses H1–H3 are recorded in `docs/temporal_bora_research.md`. The protocol
is deliberately academic: one training run, best checkpoint selected on clean
validation only, test evaluated once by the trainer, and every result reported
with accuracy together with rank MAE, QWK, within-one accuracy, and severe-error
rate.

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

Edit only `config/train_config.json`, then run:

```bash
python main.py
```

The config ships with the setup of run
`MultimodalDL_TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion_holdout_20260924_172932`
(test acc 0.9682): `TinyPANNS_ECA + MobileViTXXS + temporal_bora_fusion`,
`num_frames=2`, `epochs=200`, `patience=200`, dual-decoder losses
`nominal_loss_weight=0.5`, `teacher_preservation_weight=0.3`.

To change the teacher pair, edit only the `"audio"` / `"video"` blocks: the
`"backbone"` name plus its `checkpoint_path` (and `cv_checkpoint_path` for
cross-validation). Every pair of these runs in holdout and cross-validation:

- audio: `PANNS_Cnn6`, `TinyPANNS_ECA`, `PANNS_Cnn6_DW_ECA`;
- video: `EfficientNetB0`, `MobileViTXXS`, `MobileNetV2`, `SwinTiny`.

The teacher loaders are strict and expect the single-modal wrapper keys
(`frontend.*` and `backbone.*` for audio, `backbone.*` for video), so a
checkpoint of a different backbone than the configured one fails immediately
instead of loading silently.

The trainer fits, selects the best epoch by validation accuracy, reloads that
checkpoint, runs the held-out test once, and writes to
`outputs/<audio>_<video>_temporal_bora_fusion/holdout/` (one directory per
pair):

- `result.csv` — accuracy, mAP, rank MAE, QWK, within-one, severe-error rate;
- `history.csv`, `learning_curves.png`, confusion outputs;
- `checkpoint/multimodal_best.pt` plus `checkpoint/snapshots/*.pt`
  (every new validation-best) for reproducibility/analysis;
- `predictions.csv`, `gate_summary.csv` — per-sample audit of gates,
  reliabilities and probabilities.

## Cross-validation

BORA cross-validation needs **one teacher per fold**: a holdout teacher was
trained on most of every CV test fold, which would leak test labels. Train the
single-modal audio and video teachers in cross-validation mode with the same
splitter settings (seed 42, `num_folds=5`, `cv_val_ratio=0.2`). Then, in
`config/train_config.json`, set `"evaluation_mode": "cross_validation"`, point
`cv_checkpoint_path` at them with a `{fold}` template (each checkpoint keeps its
`splits/` sidecar beside it), and run `python main.py`:

```json
"audio": { "cv_checkpoint_path": "/marimo/checkpoints/audio_cv/fold_{fold:02d}/checkpoint/audio_best.pt" },
"video": { "cv_checkpoint_path": "/marimo/checkpoints/video_cv/fold_{fold:02d}/checkpoint/video_best.pt" }
```

Before any training, the selected folds' teachers are checked against their
splits (`validate_checkpoint_split_integrity`), so a missing or mismatched
teacher fails immediately instead of at fold 3. Results go to
`outputs/<experiment>/cross_validation/fold_XX/` plus `fold_results.csv` and
`summary_mean_std.csv`. Holdout keeps using `checkpoint_path`.

**Per-fold upload.** As soon as a fold finishes, its directory is zipped and
pushed to the Hugging Face repo in `config/artifact_upload_config.json` as
`MultimodalDL_<audio>_<video>_<fusion>_cross_validation_fold_XX_<timestamp>.zip`
(retried 3 times; a failed upload is logged and training continues). The archive
holds `outputs/<experiment>/cross_validation/fold_XX/...`, so unzipping it at the
project root puts the fold back in place.

**Choosing folds / resuming.** `"dataset": {"cv_folds": [0, 1, 2, 3, 4]}` lists
the folds this run trains (`null` = all). If the server loses its data after,
say, folds 0–2 were uploaded:

1. unzip the uploaded `..._fold_00/01/02_*.zip` archives at the project root;
2. set `"cv_folds": [3, 4]` and run `python main.py`.

After every fold, `summary_mean_std.csv` (with `n_folds` and `folds` columns)
and `fold_results.csv` are rebuilt from every `fold_XX/result.csv` on disk, so
the final summary covers all five folds; a warning names any fold still
missing.

## Tests

```bash
pytest -q
```

Covers config validation, CORN ordinal utilities, temporal views/transforms,
fusion forward shapes, warmup + adaptive gates, gradient flow to both encoders
and teacher residual, the nominal decoder and its coupling gate, both new loss
terms, corruption augmentation, optimizer parameter groups, and
checkpoint/split integrity.

The `scripts/` directory retains optional analysis utilities (snapshot
probability caching, ensemble evaluation); none of them are part of the
reported result path.
