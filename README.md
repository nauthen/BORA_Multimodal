# Multimodal Deep Fusion — Dual-Decoder Temporal BORA (TinyPANNs-ECA + MobileViT-XXS)

Audio-video fish feeding intensity classification (AV-FFIA, 27K two-second
clips, 4 ordinal classes `none < weak < medium < strong`).

Architecture under study: **Dual-Decoder Temporal BORA-Fuse** — a multi-frame
(baseline: 2 frames), motion-aware, boundary-conditioned temporal fusion with
two coupled decoders:

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
- **Audio teacher checkpoint (TinyPANNS_ECA)**: the path in
  `config/train_config.json`
  (`/marimo/checkpoints/audio_run/DL_audio/checkpoint/panns_cnn6/audio_best.pt`,
  which held the TinyPANNS_ECA teacher for the baseline run) with its `splits/`
  sidecar beside it.
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

`config/train_config.json` is the baseline of run
`MultimodalDL_TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion_holdout_20260924_172932`
(test acc 0.9682): `TinyPANNS_ECA + MobileViTXXS + temporal_bora_fusion`,
`num_frames=2`, `epochs=200`, `patience=200`, dual-decoder losses
`nominal_loss_weight=0.5`, `teacher_preservation_weight=0.3`. Then:

```bash
python main.py                                       # holdout (evaluation_mode in the config)
python main.py --evaluation-mode cross_validation    # see "Cross-validation" below
```

To use the PANNs Cnn6 audio branch instead, change only the audio block
(`"backbone": "PANNS_Cnn6"` and its checkpoint). Both global and temporal BORA
accept `PANNS_Cnn6`, `PANNS_Cnn6_DW_ECA` and `TinyPANNS_ECA`; the audio
checkpoint loader is strict and expects the single-modal wrapper keys
`frontend.*` and `backbone.*`, preventing a silently partial or wrong-model
load.

The trainer fits, selects the best epoch by validation accuracy, reloads that
checkpoint, runs the held-out test once, and writes to
`outputs/TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion/holdout/`:

- `result.csv` — accuracy, mAP, rank MAE, QWK, within-one, severe-error rate;
- `history.csv`, `learning_curves.png`, confusion outputs;
- `checkpoint/multimodal_best.pt` plus `checkpoint/snapshots/*.pt`
  (every new validation-best) for reproducibility/analysis;
- `predictions.csv`, `gate_summary.csv` — per-sample audit of gates,
  reliabilities and probabilities.

## Ablations

Leave-one-component-out: `--ablation <preset>` applies a preset on top of the
config, so each variant differs from the full model only by the removed
component and the loss attached to it. Design rationale, what each row
measures, and how to interpret it: `docs/ablation_design.md`.

| `--ablation` | Config switches | Removes | Output suffix |
|---|---|---|---|
| `none` | — | nothing (full model) | — |
| `no_motion` | `temporal_motion="none"`, `motion_loss_weight=0` | frame-difference cue in tokens and event gate, motion regressor + loss | `_nomotion` |
| `no_confidence` | `gate_confidence="none"` | margin calibration of the reliability gate | `_noconf` |
| `ordinal_only` | `decoders="ordinal"`, `nominal_loss_weight=0` | nominal decoder (decoder 2) + coupling | `_ordinalonly` |
| `nominal_only` | `decoders="nominal"`, `nominal_loss_weight=0` | ordinal CORN decoder (decoder 1) + its loss | `_nominalonly` |

```bash
bash scripts/run_ablations.sh holdout                  # full model + 4 ablations
bash scripts/run_ablations.sh holdout no_motion        # selected variants only
python main.py --ablation ordinal_only                 # one variant
python scripts/summarize_ablations.py --out outputs/ablation_summary.csv
python scripts/compare_ablation.py \
  outputs/TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion/holdout \
  outputs/TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion_noconf/holdout \
  --out outputs/no_confidence_vs_full.csv
```

`summarize_ablations.py` prints every test metric (holdout and, when present,
cross-validation mean ± std) with its difference from the full model.
`compare_ablation.py` gives outcome metrics, per-boundary gate mechanics
(video-gate mean/std, fraction of "dead" gates in [0.45, 0.55], gate separation
between correct and wrong predictions, reliability mean/std) and an exact
McNemar test on the paired test predictions. Re-run the full model with the
same code (`none` is included in `run_ablations.sh`); this overwrites
`outputs/<baseline>/holdout`. Accuracy gaps below ~0.5% need McNemar and ideally
cross-validation or several seeds.

## Cross-validation

BORA cross-validation needs **one teacher per fold**: a holdout teacher was
trained on most of every CV test fold, which would leak test labels. Train the
single-modal audio and video teachers in cross-validation mode with the same
splitter settings (seed 42, `num_folds=5`, `cv_val_ratio=0.2`), then point the
config at them with a `{fold}` template (each checkpoint keeps its `splits/`
sidecar beside it):

```json
"audio": { "cv_checkpoint_path": "/marimo/checkpoints/audio_cv/fold_{fold:02d}/checkpoint/audio_best.pt" },
"video": { "cv_checkpoint_path": "/marimo/checkpoints/video_cv/fold_{fold:02d}/checkpoint/video_best.pt" }
```

```bash
python main.py --evaluation-mode cross_validation
bash scripts/run_ablations.sh cross_validation
```

Before any training, every fold's teachers are checked against that fold's
split (`validate_checkpoint_split_integrity`), so a missing or mismatched
teacher fails immediately instead of at fold 3. Results go to
`outputs/<experiment>/cross_validation/fold_XX/` plus `fold_results.csv` and
`summary_mean_std.csv`. Holdout keeps using `checkpoint_path`.

## Tests

```bash
pytest -q
```

Covers config validation, CORN ordinal utilities, temporal views/transforms,
fusion forward shapes, warmup + adaptive gates, gradient flow to both encoders
and teacher residual, the nominal decoder and its coupling gate, both new loss
terms, corruption augmentation, optimizer parameter groups,
checkpoint/split integrity, every ablation variant (modules, outputs, losses,
unchanged full-model construction order), ablation presets, and per-fold
teacher resolution in cross-validation.

Besides the ablation tools above, `scripts/` retains optional analysis
utilities (snapshot probability caching, ensemble evaluation); none of them are
part of the reported result path.
