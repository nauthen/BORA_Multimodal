# Sơ đồ đã đối chiếu với code

Bản sửa trực tiếp theo sơ đồ text người dùng gửi. Cấu hình: PANNS_Cnn6 + EfficientNetB0,
8 frame, D=256, Temporal BORA-Fuse với ordinal và nominal decoder.
Các tên tensor được giữ thống nhất giữa các trang; tensor ghi tên ở trang sau được truyền từ trang trước.

Các điểm đã sửa/bổ sung:

| Vị trí trong bản gửi | Kết quả đối chiếu |
| --- | --- |
| Video corruption trước spatial transform | Đảo lại: dataset transform/normalize -> batch -> BORA corruption -> model reshape. |
| Hai ô corruption độc lập | Một action chung cho mỗi sample; chỉ một modality bị corrupt/drop. |
| AudioFrontend | Đúng thứ tự: STFT -> log-Mel -> pad đầu trục time 2 hàng -> BN -> SpecAugment. |
| CNN6 chỉ ghi Conv 5x5 và shape sau giảm kích thước | Bổ sung BN, ReLU, AvgPool 2x2 và dropout; pooling mới làm giảm kích thước. |
| EfficientNet stages -> pooling -> feature 1280D | Bổ sung head Conv 1x1 từ 320 lên 1280 channels trước pooling. |
| Hook -> teacher logits | Hook giữ input/output; logits được tính bởi classifier Linear gốc. |
| Position [1,8,256], query [1,3,256] | Parameter thật lần lượt [1,16,256] và [3,256]; slice/unsqueeze khi forward. |
| Norm(Tokens) @ Norm(Q).T | Cho [B,8,3], ngược shape cần dùng. Sửa thành Norm(Q) @ Norm(H).transpose(1,2). |
| Teacher_Audio_Ord / Teacher_Video_Ord | Bổ sung softmax, mean video probabilities, reorder, conditional conversion và logit. |
| Input của auxiliary/reliability heads | Ghi rõ: audio dùng z_a; video ordinal dùng v_global; video reliability dùng từng v_k. |
| Teacher_Logit_k | Bổ sung weighted teacher logit, cùng w_k và beta_k dùng trong các nhánh. |
| CORN “Conditional Probabilities” | Tích sigmoid là survival probabilities; từng sigmoid mới là conditional. |
| log(P_dataset + 1e-8) | Code dùng log(clamp_min(P_dataset,1e-8)). |
| Final Prediction [B,4] | Đây là log-probabilities; class prediction sau argmax có shape [B]. |
| Motion nối từ final output, shape [B,1] | Nhánh lấy trực tiếp motion_context; squeeze cuối cho [B]. |
| Chưa có loss/nguồn khởi tạo | Bổ sung 7 nhóm loss, nhãn/mask, model mẹ fine-tune và các khối mới. |

## 01  Input, tiền xử lý và hai encoder

Transform của dataset chạy trước BORA corruption trong trainer. T = 8; N = B*T.

```text
AUDIO DATASET                                       VIDEO DATASET
WAV -> mono -> resample 64 kHz                       MP4 -> 8 uniformly spaced RGB frames
    -> crop/pad 2 s                                      -> resize 224 x 224; float / 255
    -> waveform [B,128000]                               -> train: shared flip + brightness
                                                        -> ImageNet normalize -> collate
                                                        -> video_form [B,8,3,224,224]
             |                                                       |
             +---------------------------+---------------------------+
                                         v
                BORA ACTION (TRAIN ONLY; one draw per sample)
                88% no additional corruption
                10% corrupt ONE randomly chosen modality
                 2% zero ONE randomly chosen modality
                Audio: noise / gain / time mask; video: brightness / blur / occlusion
                Video corruption denormalizes, corrupts, then renormalizes.
             +---------------------------+---------------------------+
             |                                                       |
             v                                                       v
AudioFrontend                                       Flatten time into batch
STFT: FFT/window 2048, hop 1024                      [N,3,224,224]
  -> [B,1,126,1025]                                       |
Log-Mel: 128 bins, 1-32000 Hz                              v
  -> [B,1,126,128]                                   EfficientNet-B0 (shared across frames)
ZeroPad: 2 rows at START of time                     Stem -> [N,32,112,112]
  -> [B,1,128,128]                                   Stage 1 -> [N,16,112,112]
BatchNorm on mel axis                               Stage 2 -> [N,24,56,56]
SpecAugment (train only)                             Stage 3 -> [N,40,28,28]
             |                                      Stage 4 -> [N,80,14,14]
             v                                      Stage 5 -> [N,112,14,14]
PANNS_Cnn6                                          Stage 6 -> [N,192,7,7]
Each block: Conv5x5/s1/p2 -> BN -> ReLU              Stage 7 -> [N,320,7,7]
            -> AvgPool2x2 -> Dropout(0.2)            Head Conv1x1/BN/SiLU -> [N,1280,7,7]
Block 1 -> [B,64,64,64]                              GlobalAvgPool -> Flatten -> DO(0.2)
Block 2 -> [B,128,32,32]                                           |
Block 3 -> [B,256,16,16]                                           v
Block 4 -> [B,512,8,8]                               Pre-classifier feature [N,1280]
Mean over frequency -> [B,512,8]                          |                  |
Max(time) + Mean(time) -> [B,512]                         |                  v
DO(0.2) -> Linear512->512 -> ReLU -> DO(0.2)              |          Original Linear1280->4
             |                                          |          -> logits [N,4]
             v                                          |                  |
Audio feature f_a [B,512]                                v                  v
     |                   |                          f_v [B,8,1280]   t_v [B,8,4]
     |                   v                              |              to page 3
     |          Original Linear512->4                    v
     |          -> t_a [B,4]                        Linear1280->256 + LN + GELU + DO(0.15)
     |             to page 3                        -> F [B,8,256]
     v                                                  to page 2
Linear512->256 + LN + GELU + DO(0.15)
-> z_a [B,256] -> pages 2-5

FeatureHook captures BOTH the original classifier input (f_a/f_v) and output (t_a/t_v).
DO = dropout (train only); LN = LayerNorm. The original classifiers remain in the model.
```

## 02  Temporal, event gate, boundary attention và motion

Query được học mới; audio điều chỉnh query theo từng mẫu. Motion head là một nhánh riêng.

```text
F [B,8,256] (page 1)
    |
    +--> m_0 = 0; m_t = abs(F_t - F_(t-1)) -> m [B,8,256]
    |                         |
    |                         v
    |    Linear256->256 -> LN -> GELU -> c = motion_context [B,8,256]
    |                         |
    |                         +----------------------------------------+
    |                         |                                        |
    v                         v                                        v
X = F + c + position[:, :8]                              mean_time(c) [B,256]
    position PARAMETER [1,16,256]                               |
    active slice [1,8,256]                                      v
    |                                                   Linear256->256 -> GELU
    v                                                   -> Linear256->1 -> Sigmoid
TransformerEncoder: 2 layers                                   -> [B,1] -> squeeze(-1)
4 heads; D=256; FFN=512; GELU; dropout=0.15                      -> motion_score [B]
norm_first=True; final LayerNorm                               -> motion loss (page 6)
    |
    v
H = Tokens [B,8,256] -> used by event gate, attention scores and temporal pooling below.

EVENT GATE INPUTS: H, z_a, c                     QUERY INPUT: z_a [B,256]
concat(H, broadcast_time(z_a), c)                Linear256->768
  -> [B,8,768]                                   -> reshape [B,3,256]
  -> Linear768->256 -> GELU -> DO(0.15)            + learned queries [3,256].unsqueeze(0)
  -> Linear256->1 -> squeeze -> Sigmoid           -> Q [B,3,256]
  -> e [B,8]                                     |
       |                                         |
       +-------------------+---------------------+
                           |
                           +<--- H [B,8,256] from Transformer
                           v
BOUNDARY TEMPORAL ATTENTION (inputs: Q, H, e)
  Hn = L2_normalize(H, dim=-1); Qn = L2_normalize(Q, dim=-1)
  scale = min(exp(attention_logit_scale),100); initial scale = 10
  scores = scale * (Qn @ Hn.transpose(1,2))                 [B,3,8]
         + log(clamp_min(e,1e-6)).unsqueeze(1)              [B,1,8] broadcast
  alpha = softmax(scores, dim=-1)                          [B,3,8]
  V = alpha @ H                                           [B,3,256]
  v_global = mean_boundary(V)                              [B,256]
                          |
                          +--> V[:,k] -> video reliability / boundary fusion
                          +--> v_global -> video ordinal head / nominal decoder

Qn @ Hn.T gives [B,3,8]. Hn @ Qn.T would give [B,8,3] and needs transposing.
Pooling uses H, not normalized Hn. Event gate supplies a shared prior to all 3 queries.
Motion head is computed in forward, but its score does not feed the class prediction.

INITIALIZATION
  Checkpoint-loaded: audio frontend/CNN6/classifier; video EfficientNet/classifier.
  Newly initialized: projections, temporal encoder, event gate, queries, fusion and heads.
  boundary_queries: truncated normal std=0.02; learned jointly from supervised losses.
  Both original encoders/classifiers continue fine-tuning; STFT/Mel weights stay frozen.
```

## 03  Teacher conversion, auxiliary heads và reliability

Teacher là classifier gốc được tái sử dụng. Audio heads nhận z_a; video heads có hai input khác nhau.

```text
t_a [B,4] (page 1)                                t_v [B,8,4] (page 1)
    |                                                |
    v                                                v
softmax over classes                             softmax over classes, EACH frame
    -> p_a [B,4]                                     -> mean_time -> p_v [B,4]
    |                                                |
    +-----------------------+------------------------+
                            v
APPLY THE SAME CONVERSION TO p_a AND p_v:
  Dataset order [none,strong,medium,weak] -> rank order via indices [0,3,2,1]
  s0 = p_weak + p_medium + p_strong
  s1 = p_medium + p_strong; s2 = p_strong
  q_teacher = [s0, s1/max(s0,1e-6), s2/max(s1,1e-6)]
  C = logit(clamp(q_teacher,1e-5,1-1e-5))
  Outputs: C_a [B,3], C_v [B,3]
  beta = sigmoid(teacher_residual_scale [3]); initialized at sigmoid(-1.1) ~ 0.25
                            |
              +-------------+------------------------+
              v                                      v
z_a [B,256] -> Linear256->3               v_global [B,256] -> Linear256->3
              + beta*C_a                              + beta*C_v
              -> u_a [B,3]                            -> u_v [B,3]
              |                                      |
              v                                      v
conf_a = 2*abs(sigmoid(u_a)-0.5)           conf_v = 2*abs(sigmoid(u_v)-0.5)

z_a [B,256]                              V [B,3,256], one vector per boundary
    |                                                |
    v                                                v
Linear256->256 -> GELU -> DO(0.15)        Shared Linear256->256 -> GELU -> DO(0.15)
-> Linear256->3 -> Sigmoid               -> Linear256->1 -> Sigmoid -> squeeze(-1)
-> r_a [B,3]                             -> r_v [B,3]
    |                                                |
    v                                                v
rho_a = r_a*(0.25+0.75*conf_a)            rho_v = r_v*(0.25+0.75*conf_v)
    |                                                |
    +-----------------------+------------------------+
                            v
MODALITY GATE (independent weights for each boundary k)
  reliability = clamp_min(stack([rho_a,rho_v],dim=-1),1e-6)  [B,3,2]
  w = softmax(log(reliability)/0.3, dim=-1)                  [B,3,2]
  w_a = w[:,:,0]; w_v = w[:,:,1]; w_a + w_v = 1

  If set_epoch(epoch) with epoch 0,1,2: replace w with [0.5,0.5].
  This also covers validation at those epochs in the current trainer.
  If epoch >= 3 or set_epoch(None): use learned weights; test uses None.
                            |
                            +--> w, C_a, C_v, beta -> boundary fusion (page 4)
                            +--> u_a, u_v -> auxiliary CORN loss (page 6)
                            +--> r_a, r_v -> reliability loss (page 6)

Teacher probabilities/logits are NOT detached in prediction paths.
Preservation loss uses mean_time(t_v) LOGITS, unlike conversion's mean of probabilities.
```

## 04  Boundary-specific fusion

Ba bộ projection, interaction MLP và classifier riêng; cùng một audio z_a đi vào cả ba.

```text
FOR k IN {0,1,2}:

z_a [B,256]                                    V[:,k] [B,256]
    |                                                |
    v                                                v
Audio Linear256->256 [k]                        Video Linear256->256 [k]
    -> A_k [B,256]                                  -> B_k [B,256]
    |                                                |
    +------------------------+-----------------------+
                             |
               +-------------+--------------------+
               |             |                    |
               v             v                    v
   w_a,k*A_k + w_v,k*B_k   abs(A_k-B_k)           A_k*B_k
   weighted [B,256]        difference [B,256]     product [B,256]
               |             |                    |
               +-------------+--------------------+
                             v
                concat -> [B,768]
                             |
                             v
                Linear768->256 [k]
                -> LayerNorm -> GELU -> Dropout(0.15)
                             |
                             v
                I_k [B,256] -------------------------------> keep for nominal decoder
                             |
                             v
                Linear256->1 [k]
                -> learned_logit_k [B,1]
                             |
                             |             C_a,k, C_v,k (page 3)
                             |                 |
                             |                 v
                             |             R_k = w_a,k*C_a,k + w_v,k*C_v,k
                             |                 |
                             |                 v
                             |             beta_k * R_k -> unsqueeze [B,1]
                             |                 |
                             +--------+--------+
                                      v
                          l_k = learned_logit_k + beta_k*R_k
                                      |
END LOOP                              v
                     l = concat(l_0,l_1,l_2) [B,3] -> ordinal decoder (page 5)
                     I = stack(I_0,I_1,I_2) [B,3,256] -> nominal decoder (page 5)

Inputs w, C_a, C_v and beta are the outputs of page 3.
The SAME beta_k is used in audio auxiliary, video auxiliary and fused teacher residual.
Difference/product are not multiplied by modality gate weights.
I_k is captured before its classifier and before teacher-logit addition.

ORDINAL TASKS (semantic meaning assigned by label-derived loss)
  k=0: P(rank>0)                   : none vs weak/medium/strong
  k=1: P(rank>1 | rank>=1)         : weak vs medium/strong, when active
  k=2: P(rank>2 | rank>=2)         : medium vs strong, when active
```

## 05  Dual decoder và output cuối

Nominal decoder còn nhận trực tiếp z_a và v_global. clipwise_output là log-probabilities.

```text
ORDINAL DECODER                                  NOMINAL DECODER
l [B,3] (page 4)                                 I [B,3,256] (page 4)
    |                                               |
    v                                               v
q = sigmoid(l) [B,3]                             mean_boundary(I) [B,256]
q0 = P(rank>0)                                      |
q1 = P(rank>1 | rank>=1)                             |     z_a [B,256] (page 1)
q2 = P(rank>2 | rank>=2)                             |     v_global [B,256] (page 2)
    |                                               |           |
    v                                               +-----+-----+
SURVIVAL PROBABILITIES                                     v
S0 = q0                                         concat(mean_boundary(I),z_a,v_global)
S1 = q0*q1                                      -> [B,768]
S2 = q0*q1*q2                                            |
    |                                                    v
    v                                           Linear768->256 -> LayerNorm
p_rank = [1-S0, S0-S1, S1-S2, S2]               -> GELU -> Dropout(0.15)
clamp_min(0) -> [B,4]                           -> Linear256->4
Rank order: none,weak,medium,strong              -> n [B,4], raw nominal logits
    |                                                    |
    v                                                    |
p_dataset = p_rank[:,[0,3,2,1]]                           |
Dataset order: none,strong,medium,weak                    |
    |                                                    |
    v                                                    v
o = log(clamp_min(p_dataset,1e-8))               gamma*n
    [B,4]                                      gamma = sigmoid(nominal_residual_scale)
    |                                          scalar init sigmoid(-1.1) ~ 0.25
    |                                                    |
    +-------------------------+--------------------------+
                              v
               combined_logits = o + gamma*n [B,4]
                              |
                              v
               clipwise_output = log_softmax(combined_logits,dim=-1)
               [B,4] FINAL LOG-PROBABILITIES
                              |
                  +-----------+----------------+
                  v                            v
               exp(output)                 argmax(output,dim=1)
               probabilities [B,4]         predicted dataset label [B]
                                           0=none, 1=strong, 2=medium, 3=weak

z_a and v_global bypass boundary projections/modality weighting on the way to nominal head.
rank_probabilities is the ordinal-only distribution in RANK order, before nominal correction.
There is NO edge from combined_logits/clipwise_output to the motion head (see page 2).
```

## 06  Nhãn, loss và đường học

Boundary học từ nhãn thật qua backpropagation; feature và teacher residual tận dụng model mẹ.

```text
Dataset target y [B]: none=0, strong=1, medium=2, weak=3
    |
    v
Ordinal rank r = [0,3,2,1][y]     (none=0, weak=1, medium=2, strong=3)
    |
    +--> for k=0,1,2: target t_k = 1[r>k]; active mask M_k = 1[r>=k]
    |    CORN(x,r) = mean(BCEWithLogits(x,t) over all active batch-boundary entries)
    |
    +--> normalized rank r/3 -> motion target

FORWARD OUTPUT / TARGETS                        LOSS TERM
------------------------------------------     -------------------------------------------
l, r ---------------------------------------> L_fused = CORN(l,r)

u_a, u_v, r --------------------------------> L_aux = CORN(u_a,r) + CORN(u_v,r)

u_a, t -> exp(-BCE(u_a,t)).detach() --+
                                     +------> SL1_a = SmoothL1(r_a,detached_target_a)
r_a ---------------------------------+

u_v, t -> exp(-BCE(u_v,t)).detach() --+
                                     +------> SL1_v = SmoothL1(r_v,detached_target_v)
r_v ---------------------------------+
                                               L_rel = 0.5*(masked_mean(SL1_a,M)
                                                          +masked_mean(SL1_v,M))

clipwise_output, y --------------------------> L_cat = NLL(clipwise_output,y)

motion_score, r/3 ---------------------------> L_motion = SmoothL1(motion_score,r/3)

n, y ---------------------------------------> L_nom = CrossEntropy(n,y)

t_a, mean_time(t_v), y ----------------------> L_pres = 0.5*(CE(t_a,y)+CE(mean_time(t_v),y))

                                                         |
                                                         v
TOTAL = L_fused + 0.3*L_aux + 0.1*L_rel + 1.0*L_cat
                 + 0.1*L_motion + 0.5*L_nom + 0.3*L_pres
    |
    v
backward() -> Adam step
    +--> original encoder branches + original classifiers: lr=1e-5
    +--> new fusion/projections/Transformer/queries/heads: lr=1e-4

Boundary queries/event gate/attention learn indirectly through downstream supervised losses.
No separate frame-event label, attention target, or copied boundary query from a parent model.
Raw r_a/r_v are supervised by reliability loss; confidence-modulated rho_a/rho_v feed the gate.
Teacher preservation uses real labels, not a frozen-teacher distillation target.

CURRENT TRAINING SETTINGS
  120 epochs maximum; batch_size=32; Adam; early stopping patience=20.
  Best checkpoint selected by validation accuracy.
  Frozen: STFT and Mel filter weights. Encoders and their original classifiers fine-tune.
```

Nguồn: `models/multimodal_model.py`, `models/fusion/fusion_heads.py`,
`features/audio_frontend.py`, `models/audio/panns_cnn6.py`, `models/video/efficientnet_b0.py`,
`dataset/multimodal_dataset.py`, `transforms/video_transform.py`, `tasks/trainer.py`,
`utils/ordinal.py`, `utils/corruption.py`, `config/train_config.json`.
