> **Historical note:** this document describes the earlier CORN/ordinal Dual-Decoder Temporal BORA design, which has been removed from the code. The current architecture is described in `docs/temporal_reliability_fusion.md`.

# Temporal BORA-Fuse research note

## Evidence and protocol audit

- The original AV-FFIA work describes 27,000 synchronized two-second clips at
  25 FPS and reports 91.7% for its audio-visual cross-attention benchmark.
  It explicitly notes that video carries dynamic context unavailable to a
  single image: <https://arxiv.org/abs/2309.05058>.
- ACAF-Net uses an S3D video encoder to retain spatiotemporal movement and
  reports 95.4% on AV-FFIA:
  <https://doi.org/10.3390/ani15152245>.
- AquaMutual samples 16 frames from each two-second clip specifically to
  preserve temporal feeding dynamics:
  <https://doi.org/10.1016/j.biosystemseng.2026.104468>.
- The 2026 uncertainty-aware heterogeneous multi-level distillation work
  reports 97.47% on public AV-FFIA and identifies decision-, feature-, and
  relation-level teacher transfer plus entropy-based uncertainty defense as
  important: <https://doi.org/10.1016/j.inpa.2026.06.006>.
- LDMF reports 98.2%, but its paper states that preprocessing reduces AV-FFIA
  from 27,000 clips to 4,066 paired examples. That number is therefore not a
  like-for-like target for the unfiltered 2,800-sample holdout used here:
  <https://doi.org/10.21203/rs.3.rs-8304036/v1>.

## Finding from Global BORA

On the exact random holdout shared by both single-modal checkpoints:

| Model | Clean test accuracy | Rank MAE | QWK | Severe error rate |
|---|---:|---:|---:|---:|
| Gated baseline | 94.821% | 0.06286 | 0.96157 | 0.571% |
| Global BORA | 94.500% | 0.06036 | 0.96976 | 0.286% |

Global BORA improves ordinal error severity but not exact classification. Its
reliability estimates saturate: mean video reliability lies around
0.969–0.996 and the resulting video gates remain only 0.519–0.565. Gate means
are almost identical for correct and incorrect examples. The dominant errors
are adjacent boundaries: medium→strong (49), weak→medium (37),
medium→weak (34), and strong→medium (26), in dataset-label notation.

This indicates an input-information and calibration bottleneck rather than a
lack of capacity in the global fusion MLP.

## Implemented hypothesis: uncertainty-calibrated Temporal BORA

`temporal_bora_fusion` keeps the fixed PANNs Cnn6 and 2D Swin-Tiny checkpoint
contract but changes the evidence presented to the fusion head:

1. Uniformly decode eight frames over the complete two-second clip and apply
   one consistent augmentation to the whole clip.
2. Encode every frame with the checkpointed video encoder and form explicit
   adjacent-frame motion residuals.
3. Learn an audio-conditioned event mask to suppress static/background frames.
4. Use a distinct temporal query for each ordinal boundary, because evidence
   for `none|weak+`, `weak-|medium+`, and `medium-|strong` need not occur in the
   same frames.
5. Calibrate learned reliability with each auxiliary boundary margin so a
   saturated reliability MLP cannot dominate while its decision is uncertain.
6. Reuse the pretrained audio and per-frame video classifier logits as a
   learnable boundary-level decision residual. This preserves teacher
   knowledge while the temporal/ordinal representations are learned.
7. Optimize fused CORN, auxiliary CORN, reliability, categorical NLL, and
   ordinal motion-intensity losses jointly.

Temporal attention, reliability gating, and distillation are known ideas. The
specific research contribution to test is their **ordinal boundary-conditioned
combination**: each intensity threshold receives a different temporal evidence
distribution, uncertainty gate, and teacher residual. Claims of novelty should
be limited to this combination unless a broader systematic review is added.

## Dual-decoder extension (current architecture)

Post-hoc analysis of the first temporal run bounded what decoding can fix:
threshold calibration transferred negatively from validation to test, three-view
temporal consensus did not beat the plain snapshot ensemble, and direct blending
of teacher categorical outputs saturated below the ensemble ceiling — while mAP
reached 0.993 with errors concentrated at adjacent boundaries (146/154 Global
BORA errors). Three architecture-level changes follow directly:

1. **Dual decoder (exact-class branch).** The CORN head factorizes the class
   decision into sequential conditionals `P(rank>k | rank>k-1)`; exact-class
   probability is a product of conditionals and ranking quality does not imply
   exact-class accuracy. A parallel nominal 4-class head observes *pooled*
   boundary evidence: the mean of the three boundary-conditioned interaction
   embeddings together with global audio and pooled-video representations. It
   sees all boundary evidences simultaneously — information CORN's sequential
   factorization cannot use at any single conditional.
2. **Learned logit-space coupling.** The final class decision is
   `log_softmax(log p_ordinal + sigmoid(s) * nominal_logits)` with one learnable
   scalar gate initialized small (~0.25), so training starts ordinal-dominated
   and the network learns how much exact-class evidence to inject. This replaces
   post-hoc ensembling with a single trained decision rule.
3. **Label-anchored teacher preservation.** The reused single-modal classifiers
   previously drifted during fine-tuning (audio teacher fell from 85.5% to
   64–79%) because nothing anchored them. A cross-entropy term on both hooked
   teacher logits against ground truth keeps them discriminative while the
   encoders adapt, stabilizing the boundary-level teacher residual they feed.

New losses: `nominal_loss_weight * CE(nominal_logits)` and
`teacher_preservation_weight * CE(teacher_logits)`. Hypotheses: (H1) the dual
decoder converts ranking quality into exact-class gains on adjacent-boundary
errors; (H2) preservation removes teacher-residual degradation across epochs;
(H3) the coupled decision matches or beats any validation-frozen post-hoc
combination of the two heads without touching test more than once.

## Locked evaluation rule

- Same seed-42 random holdout and exact sidecar identities/labels.
- No use of `date/session/sample_id` as model input and no neighbor-label
  smoothing; both would exploit the clip-level random split.
- Best checkpoint selected only by clean validation accuracy.
- Test inspected after selection; target is strictly greater than 97%.
- Always report accuracy together with rank MAE, QWK, within-one accuracy, and
  severe-error rate.
