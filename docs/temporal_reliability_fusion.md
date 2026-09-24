# Temporal Reliability Fusion

Current fusion head: `TemporalReliabilityFusion` in `models/fusion/fusion_heads.py`,
selected with `fusion.type = "temporal_reliability_fusion"`. Loss: `utils/losses.py::fusion_loss`.

Notation: `B` batch, `T = 8` frames, `d = proj_dim = 256`, `K = 3` pooling queries,
`D_a` audio feature size (512 for PANNs Cnn6, 256 for TinyPANNS_ECA), `D_v = 320` for MobileViT-XXS.
Classes are in dataset order `[none, strong, medium, weak]` everywhere.

```
① Encoders (fine-tuned, lr × encoder_lr_scale)
   waveform [B, 128000] → AudioFrontend + PANNs → f_a [B, D_a]     teacher t_a = fc_audioset  [B, 4]
   video [B, 8, 3, 224, 224] → MobileViT-XXS per frame → f_v [B, 8, 320]   teacher t_v = head.fc [B, 8, 4]

② Projection + temporal
   a = proj_audio(f_a)                       Linear·LN·act·Dropout          [B, d]
   x = proj_video(f_v)                                                      [B, 8, d]
   m = motion_projection([0, |x_t − x_t−1|])  Linear·LN·GELU                [B, 8, d]
   H = TemporalTransformer(x + m + position)  2 pre-norm layers, 4 heads    [B, 8, d]

③ Audio-conditioned temporal pooling
   e = σ(event_gate([H ‖ a ‖ m]))                                           [B, 8]
   Q = audio_query(a).view(B, K, d) + pooling_queries                       [B, K, d]
   α = softmax_t( cos(H, Q) · min(e^s, 100) + log e )                       [B, K, 8]
   V_k = Σ_t α_k,t H_t                                                      [B, K, d]
   v̄ = mean_k V_k                                                           [B, d]

④ Reliability gate
   audio_aux = Linear(d→4)(a)          video_aux = Linear(d→4)(V_k)          [B, 4], [B, K, 4]
   margin c = p_top1 − p_top2 of each auxiliary softmax (label free)
   ρ_a = σ(MLP(a)) [B, 1]              ρ_v = σ(MLP(V_k)) [B, K]
   ρ̃ = ρ · (0.25 + 0.75 c)   (gate_confidence = "margin";  "none" → ρ̃ = ρ)
   w_k = softmax( [log ρ̃_a, log ρ̃_v,k] / gate_temperature )               [B, K, 2]
   first warmup_epochs: w = 0.5 / 0.5

⑤ Query interaction (one set of weights per query k)
   ab_k = P_a,k(a)     vb_k = P_v,k(V_k)
   z_k = MLP([w_k,a·ab_k + w_k,v·vb_k ‖ |ab_k − vb_k| ‖ ab_k ⊙ vb_k])        [B, d]

⑥ Nominal decoder
   ℓ  = NominalHead([z_0 ‖ z_1 ‖ z_2 ‖ a ‖ v̄])       Linear(5d→h)·LN·act·Dropout·Linear(h→4)
   ℓ* = ℓ + σ(β) · ( w̄_a · log softmax(t_a) + w̄_v · log mean_t softmax(t_v) ),   w̄ = mean_k w_k, β₀ = 0
   clipwise_output = log_softmax(ℓ*)
   motion_score = σ(MLP(mean_t m))                                          [B]
```

## Loss

```
L = CE(clipwise_output, y)                                   decision
  + aux_loss_weight            · [CE(audio_aux) + mean_k CE(video_aux_k)]
  + reliability_loss_weight    · ½ [SmoothL1(ρ_a, p_aux_a(y)) + SmoothL1(ρ_v,k, p_aux_v,k(y))]   (targets detached)
  + motion_loss_weight         · SmoothL1(motion_score, rank(y) / 3)
  + teacher_preservation_weight· ½ [CE(t_a, y) + CE(mean_t t_v, y)]
```

`rank(y)` maps `none, weak, medium, strong` to `0..3` and is used only for the motion target and
the ordinal evaluation metrics (rank MAE, QWK, within-one, severe error).

## Outputs

`predictions.csv`: per test sample `p_class_*`, `audio_reliability_q{k}` (the single audio value
repeated per query), `video_reliability_q{k}`, `audio_gate_q{k}`, `video_gate_q{k}`.
`gate_summary.csv`: gate and reliability mean/std per true class and query.
