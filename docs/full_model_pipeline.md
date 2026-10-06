# Full pipeline — Dual-Decoder Temporal BORA-Fuse

Tài liệu này mô tả đúng đường thực thi bắt đầu từ `main.py` với cấu hình hiện tại trong
`config/train_config.json`. Các script trong `scripts/` không thuộc pipeline dưới đây.

> **Kiến trúc đang thực sự được chạy:** PANNs CNN6 cho audio + EfficientNet-B0 cho từng
> video frame + Temporal BORA-Fuse + hai decoder Ordinal/ Nominal.
>
> `main.py` không đọc `config/train_config.temporal_bora.json`. File đó dùng `SwinTiny`,
> còn config được `main.py` đọc hiện dùng `EfficientNetB0`.

## Pipeline methodology cho paper — đọc từ trái sang phải

Sơ đồ chính dưới đây nối đủ hai encoder, temporal evidence, classifier reuse,
boundary fusion, global bypass, hai decoder và output. Các module được phóng chi
tiết trong bản ghép (a–d); objective và công thức chính xác nằm trong tài liệu
methodology riêng.

[Methodology + caption tiếng Anh + công thức đối chiếu code](paper_methodology.md) ·
[Hình chính SVG](diagrams/paper_methodology_overview.svg) ·
[Bản đầy đủ (a–d) SVG](diagrams/paper_methodology.svg) ·
[PNG độ phân giải cao](diagrams/paper_methodology.png)

![Methodology: complete forward graph with classifier reuse and ungated global bypass](diagrams/paper_methodology_overview.svg)

Bản sơ đồ dọc trước đây vẫn được giữ để tham khảo từng bước:
[SVG](diagrams/full_model_pipeline.svg) · [PNG](diagrams/full_model_pipeline.png).

### Cách đọc các đường truyền

- Audio và video ở bên trái; ordinal và nominal decoder ở bên phải.
- `a` đi vào temporal module, fusion và global bypass. `V` đi vào fusion và phép
  mean để tạo `g`; `g` đi vào auxiliary video head và nominal decoder.
- Đường hồng mang classifier logits/conditional logits trong forward, không phải
  loss. `C_a/C_v` đi vào cả auxiliary heads và fused ordinal logits.
- Chấm tròn là nhánh rẽ; đường giao nhau không có chấm không nối tensor. Loss được
  giải thích riêng, label không phải input của model khi dự đoán.
- `B` là batch size; `k=0,1,2` là ba conditional tasks; `T=8`, `D=256`.

### Các điểm cần giữ đúng khi sử dụng sơ đồ

1. Audio được tổng hợp thành vector toàn clip. Transformer xử lý chuỗi **video**, rồi
   audio hướng dẫn việc chọn bằng chứng theo thời gian.
2. Motion được tính từ **projected frame features**, sau projection có dropout;
   đây là độ thay đổi đặc trưng, không phải optical flow.
3. `video_global` là trung bình của **ba boundary features**, sau temporal attention.
4. Auxiliary video ordinal head nhận `video_global`; video reliability head nhận từng
   `V[k]`. Hai head này có đầu vào khác nhau.
5. Teacher logits được lấy từ classifier của chính encoder đang fine-tune. Video
   teacher residual dùng `mean(softmax(frame_logits))`; preservation loss dùng
   `CE(mean(frame_logits), label)`.
6. Nominal head nhận `[mean(I_0,I_1,I_2), a, video_global]` có **768 chiều**.
   Teacher residual được cộng vào ordinal logits, sau khi lấy interaction features cho
   nominal head.
7. `clipwise_output` là **log-probability cuối** theo thứ tự dataset. Tensor
   `rank_probabilities` vẫn là xác suất ordinal trước nominal correction.

Nguồn đối chiếu chính: [`forward()` của multimodal wrapper](../models/multimodal_model.py),
[`TemporalBORAFusion.forward()`](../models/fusion/fusion_heads.py),
[`bora_loss()`](../utils/ordinal.py) và [`MultimodalTrainer`](../tasks/trainer.py).

Sơ đồ được xuất thành SVG có thể phóng to và PNG 1600 × 5472 px. Để tái tạo hai hình:

```bash
python docs/diagrams/render_pipeline.py
```



## 1. Cấu hình quyết định kiến trúc

| Thành phần | Giá trị hiện tại |
|---|---:|
| Số lớp | 4 |
| Thứ tự label dataset | `none=0, strong=1, medium=2, weak=3` |
| Thứ tự ordinal | `none=0, weak=1, medium=2, strong=3` |
| Audio | 2 giây, 64 kHz, 128.000 samples |
| Audio encoder | PANNs CNN6 |
| Video | 8 RGB frames, `224 x 224` |
| Video encoder | EfficientNet-B0 chạy độc lập trên từng frame |
| Fusion dimension | 256 |
| Temporal encoder | 2 Transformer layers, 4 heads |
| Fusion type | `temporal_bora_fusion` |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate fusion head | `1e-4` |
| Learning rate encoder | `1e-5` (`encoder_lr_scale=0.1`) |

## 2. Pipeline thực thi từ `main.py`

```mermaid
flowchart TD
    ENTRY["python main.py"] --> LOADCFG["Đọc config/train_config.json"]
    LOADCFG --> VALIDCFG["Pydantic kiểm tra ràng buộc Temporal BORA"]
    VALIDCFG --> SEED["Đặt seed = 42"]
    SEED --> ROOT["Tạo output directory<br/>outputs/PANNS_Cnn6_EfficientNetB0_temporal_bora_fusion"]
    ROOT --> SPLIT["FishDataSplitter<br/>random_sample holdout"]
    SPLIT --> S3["train / validation / test"]
    S3 --> CHECK["Kiểm tra split sidecar của<br/>audio checkpoint và video checkpoint"]
    CHECK -->|"identity + label khớp hoàn toàn"| LOADERS["Tạo 3 multimodal DataLoader"]
    CHECK -->|"không khớp"| STOP["Dừng để ngăn leakage hoặc checkpoint sai"]
    LOADERS --> TRAINER["Khởi tạo MultimodalTrainer"]
    TRAINER --> MODEL["Khởi tạo MultimodalDeepFusionModel"]
    MODEL --> FIT["Train epoch hiện tại"]
    FIT --> VAL["Đánh giá clean validation"]
    VAL --> SELECT{"Validation accuracy<br/>tốt hơn best hiện tại?"}
    SELECT -->|"Có"| SAVE["Ghi đè multimodal_best.pt"]
    SELECT -->|"Không"| EARLY["Tăng early-stopping counter"]
    SAVE --> NEXT{"Đã đủ 120 epochs?"}
    EARLY --> PATIENCE{"Counter đạt patience=20<br/>hoặc đã đủ 120 epochs?"}
    PATIENCE -->|"Chưa"| FIT
    NEXT -->|"Chưa"| FIT
    PATIENCE -->|"Rồi"| RELOAD["Load lại multimodal_best.pt"]
    NEXT -->|"Rồi"| RELOAD
    RELOAD --> TEST["Đánh giá test đúng một lần"]
    TEST --> ARTIFACTS["result.csv, predictions.csv,<br/>gate_summary.csv, confusion matrix,<br/>history và checkpoint"]
```

## 3. Tạo một multimodal sample

```mermaid
flowchart LR
    PAIR["Một cặp audio-video đồng bộ<br/>cùng label"]

    subgraph AUDIO_DATA["Audio data path"]
        WAV["Audio file"] --> MONO["Đổi về mono"]
        MONO --> RESAMPLE["Resample 64 kHz"]
        RESAMPLE --> FIXLEN["Cắt hoặc zero-pad đúng 2 giây"]
        FIXLEN --> AIN["waveform<br/>[128000]"]
    end

    subgraph VIDEO_DATA["Video data path"]
        VID["Video file"] --> UNIFORM["Lấy đều 8 vị trí<br/>trên toàn clip 2 giây"]
        UNIFORM --> RGB["8 RGB frames<br/>[8,3,224,224]"]
        RGB --> AUG["Một spatial/color transform<br/>nhất quán cho cả clip"]
        AUG --> VIN["video_form<br/>[8,3,224,224]"]
    end

    PAIR --> WAV
    PAIR --> VID
    AIN --> COLLATE["multimodal_collate_fn"]
    VIN --> COLLATE
    PAIR --> LABEL["target scalar"]
    LABEL --> COLLATE
    COLLATE --> BATCH["Batch<br/>waveform [B,128000]<br/>video [B,8,3,224,224]<br/>target [B]"]
```

Ở train split, video clip có thể được lật ngang và đổi brightness. Một phép biến đổi duy
nhất được áp dụng cho tất cả frame trong clip để không tạo chuyển động giả giữa các frame.
Validation và test chỉ resize/chuyển tensor/chuẩn hóa ImageNet.

## 4. Forward pass tổng thể

```mermaid
flowchart TB
    INPUT["Multimodal batch"]

    subgraph ABRANCH["Audio branch"]
        A0["waveform<br/>[B,128000]"] --> AF["AudioFrontend<br/>STFT + Log-Mel + BatchNorm<br/>+ SpecAugment khi train"]
        AF --> AMEL["log-Mel spectrogram<br/>[B,1,128,128]"]
        AMEL --> PANN["PANNs CNN6"]
        PANN --> AFEAT["classifier input hook<br/>audio feature [B,512]"]
        PANN --> ATEACH["fc_audioset<br/>audio teacher logits [B,4]"]
    end

    subgraph VBRANCH["Video branch"]
        V0["video clip<br/>[B,8,3,224,224]"] --> FLAT["Gộp batch và time<br/>[B*8,3,224,224]"]
        FLAT --> EFF["Cùng một EfficientNet-B0<br/>cho từng frame"]
        EFF --> VHOOK["classifier input hook<br/>[B*8,1280]"]
        VHOOK --> VFEAT["reshape<br/>video features [B,8,1280]"]
        EFF --> VLOGIT0["frame teacher logits<br/>[B*8,4]"]
        VLOGIT0 --> VTEACH["reshape<br/>video teacher logits [B,8,4]"]
    end

    INPUT --> A0
    INPUT --> V0
    AFEAT --> FUSION["Temporal BORA-Fuse"]
    ATEACH --> FUSION
    VFEAT --> FUSION
    VTEACH --> FUSION
    FUSION --> FINAL["clipwise_output<br/>log-probability [B,4]"]
    FINAL --> PRED["argmax<br/>none / strong / medium / weak"]
```

Hai classifier đơn phương thức không bị bỏ đi. Input của classifier được lấy làm feature,
còn output classifier được giữ làm teacher logits cho teacher residual và preservation loss.
Các teacher này tiếp tục được fine-tune; chúng không phải teacher frozen.

## 5. Audio encoder chi tiết

```mermaid
flowchart LR
    A0["[B,128000]"] --> STFT["STFT<br/>n_fft=2048, hop=1024"]
    STFT --> SPEC["[B,1,126,1025]"]
    SPEC --> MEL["128-bin Log-Mel<br/>1 Hz đến 32 kHz"]
    MEL --> M0["[B,1,126,128]"]
    M0 --> PAD["Pad 2 bước thời gian"]
    PAD --> M1["[B,1,128,128]"]
    M1 --> BN["BatchNorm theo mel bins"]
    BN --> SA["SpecAugment khi train"]
    SA --> C1["Conv 5x5, 1→64<br/>AvgPool 2x2"]
    C1 --> X1["[B,64,64,64]"]
    X1 --> C2["Conv 5x5, 64→128<br/>AvgPool 2x2"]
    C2 --> X2["[B,128,32,32]"]
    X2 --> C3["Conv 5x5, 128→256<br/>AvgPool 2x2"]
    C3 --> X3["[B,256,16,16]"]
    X3 --> C4["Conv 5x5, 256→512<br/>AvgPool 2x2"]
    C4 --> X4["[B,512,8,8]"]
    X4 --> FMEAN["Mean theo frequency"]
    FMEAN --> XT["[B,512,8]"]
    XT --> POOL["Max theo time + Mean theo time"]
    POOL --> FC1["Linear 512→512 + ReLU"]
    FC1 --> EMB["audio feature [B,512]"]
    EMB --> FCOUT["fc_audioset 512→4"]
    FCOUT --> LOGITS["audio teacher logits [B,4]"]
```

Max pooling giữ sự kiện âm thanh nổi bật; mean pooling giữ mức hoạt động kéo dài. Tổng của
hai pooling cung cấp cả bằng chứng cục bộ mạnh và thống kê toàn clip.

## 6. Video encoder chi tiết

```mermaid
flowchart LR
    CLIP["[B,8,3,224,224]"] --> FLAT2["[B*8,3,224,224]"]
    FLAT2 --> STEM["EfficientNet-B0 stem"]
    STEM --> MB["MBConv stages<br/>+ squeeze-excitation<br/>+ residual/stochastic depth"]
    MB --> LAST["Final 1x1 convolution<br/>1280 channels"]
    LAST --> GAP["Global average pooling"]
    GAP --> DROP["Dropout"]
    DROP --> VF["frame feature [B*8,1280]"]
    VF --> CLS["Linear 1280→4"]
    CLS --> VL["frame teacher logits [B*8,4]"]
    VF --> R1["reshape [B,8,1280]"]
    VL --> R2["reshape [B,8,4]"]
```

EfficientNet chỉ học đặc trưng không gian trong từng frame. Quan hệ thời gian được học sau
khi mỗi frame đã được nén thành vector 1280 chiều.

## 7. Temporal BORA-Fuse chi tiết

### 7.1 Projection, motion và temporal context

```mermaid
flowchart TB
    AF2["audio feature [B,512]"] --> AP["Linear 512→256<br/>LayerNorm + GELU + Dropout"]
    AP --> A256["audio [B,256]"]

    VF2["video features [B,8,1280]"] --> VP["Linear 1280→256<br/>LayerNorm + GELU + Dropout"]
    VP --> FRAMES["frames [B,8,256]"]

    FRAMES --> DIFF["motion[0]=0<br/>motion[t]=abs(frame[t]-frame[t-1])"]
    DIFF --> MP["Linear 256→256<br/>LayerNorm + GELU"]
    MP --> MC["motion context [B,8,256]"]

    FRAMES --> SUM["Cộng theo từng timestep"]
    MC --> SUM
    POS["Learned position embedding<br/>[1,16,256], lấy 8 vị trí đầu"] --> SUM
    SUM --> TE["TransformerEncoder<br/>2 layers, 4 heads<br/>FFN dimension 512, pre-norm"]
    TE --> TOK["contextual video tokens<br/>[B,8,256]"]

    TOK --> EG["Event gate"]
    A256 --> EXP["Lặp trên 8 timestep<br/>[B,8,256]"]
    EXP --> EG
    MC --> EG
    EG --> EP["sigmoid event probabilities<br/>[B,8]"]
```

Event gate nhận phép nối `[temporal token, global audio, motion context]` có shape
`[B,8,768]`, rồi sinh một xác suất sự kiện cho mỗi frame. Audio ở đây là thông tin global
cho cả clip, không phải một chuỗi audio token theo thời gian.

### 7.2 Ba boundary-conditioned temporal attention

```mermaid
flowchart TB
    A3["audio [B,256]"] --> AQ["Linear 256→768<br/>reshape [B,3,256]"]
    BQ["3 learned boundary queries<br/>[3,256]"] --> ADDQ["Cộng query"]
    AQ --> ADDQ
    ADDQ --> Q["queries [B,3,256]"]

    T3["temporal tokens [B,8,256]"] --> COS["Cosine attention<br/>query k với frame t"]
    Q --> COS
    SCALE["Learned logit scale<br/>khởi tạo exp(log 10)=10"] --> COS
    EVENT["log event probability<br/>[B,1,8]"] --> COS
    COS --> SOFT["Softmax theo 8 frames"]
    SOFT --> ATT["temporal attention [B,3,8]"]
    ATT --> WSUM["Weighted sum video tokens"]
    T3 --> WSUM
    WSUM --> VB["3 video boundary features<br/>[B,3,256]"]
    VB --> VGLOBAL["Mean theo 3 boundaries<br/>video_global [B,256]"]
```

Ba query tương ứng ba quyết định ordinal:

1. `none` so với `weak+`;
2. `weak-` so với `medium+`;
3. `medium-` so với `strong`.

Vì bằng chứng cho mỗi ranh giới có thể xuất hiện ở frame khác nhau, mỗi boundary có một
phân phối attention riêng thay vì dùng một video vector chung.

### 7.3 Auxiliary ordinal heads, teacher residual và reliability

```mermaid
flowchart TB
    A4["audio [B,256]"] --> AO["Audio ordinal head<br/>Linear 256→3"]
    VG4["video_global [B,256]"] --> VO["Video ordinal head<br/>Linear 256→3"]

    ATL["audio teacher logits [B,4]"] --> ATP["Softmax"]
    VTL["video teacher logits [B,8,4]"] --> VTP["Softmax từng frame<br/>rồi mean theo 8 frames"]
    ATP --> ATCORN["Đổi audio probability<br/>thành 3 CORN conditional logits"]
    VTP --> VTCORN["Đổi video probability<br/>thành 3 CORN conditional logits"]
    TG["3 learned teacher gates<br/>sigmoid(-1.1) ≈ 0.25 lúc đầu"] --> AADD["Teacher residual"]
    TG --> VADD["Teacher residual"]
    ATCORN --> AADD
    VTCORN --> VADD
    AO --> AADD
    VO --> VADD
    AADD --> AOL["audio ordinal logits [B,3]"]
    VADD --> VOL["video ordinal logits [B,3]"]

    A4 --> ARH["Audio reliability MLP + sigmoid"]
    VB4["video boundaries [B,3,256]"] --> VRH["Video reliability MLP + sigmoid"]
    ARH --> AR["raw audio reliability [B,3]"]
    VRH --> VR["raw video reliability [B,3]"]

    AOL --> AC["confidence = 2*abs(sigmoid(logit)-0.5)"]
    VOL --> VC["confidence = 2*abs(sigmoid(logit)-0.5)"]
    AR --> ACR["audio gate reliability<br/>r*(0.25+0.75*confidence)"]
    AC --> ACR
    VR --> VCR["video gate reliability<br/>r*(0.25+0.75*confidence)"]
    VC --> VCR
    ACR --> GATE["Boundary-wise softmax<br/>softmax(log reliability / 0.3)"]
    VCR --> GATE
    GATE --> GW["audio/video gate weights<br/>[B,3,2]"]
```

Trong epoch 0, 1 và 2, `set_epoch()` bật warm-up và thay toàn bộ gate bằng `[0.5, 0.5]`.
Từ epoch 3, gate mới sử dụng reliability học được. Confidence calibration ngăn một
reliability MLP bị bão hòa gần 1 thống trị fusion dù auxiliary decision đang không chắc chắn.

### 7.4 Fusion riêng cho từng boundary

Quy trình sau được lặp độc lập cho `k = 0, 1, 2`:

```mermaid
flowchart LR
    A5["audio [B,256]"] --> ABP["Audio boundary projection k<br/>Linear 256→256"]
    V5["video_boundary k [B,256]"] --> VBP["Video boundary projection k<br/>Linear 256→256"]
    ABP --> WEIGHT["w_audio*A_k + w_video*V_k"]
    VBP --> WEIGHT
    GW5["gate weights k"] --> WEIGHT

    ABP --> DIFF5["abs(A_k - V_k)"]
    VBP --> DIFF5
    ABP --> PROD["A_k * V_k"]
    VBP --> PROD
    WEIGHT --> CAT5["Concatenate<br/>[weighted, abs difference, product]"]
    DIFF5 --> CAT5
    PROD --> CAT5
    CAT5 --> INT["Linear 768→256<br/>LayerNorm + GELU + Dropout"]
    INT --> BF["boundary interaction feature<br/>[B,256]"]
    BF --> BCLS["Boundary classifier<br/>Linear 256→1"]
    TRES["Reliability-weighted<br/>teacher conditional logit"] --> ADD5["Cộng teacher residual<br/>với learned gate"]
    BCLS --> ADD5
    ADD5 --> BLOGIT["ordinal boundary logit k<br/>[B,1]"]
```

- Weighted feature mang thông tin đã cân bằng theo độ tin cậy.
- Absolute difference biểu diễn mức bất đồng giữa hai modality.
- Element-wise product biểu diễn mức đồng thuận/tương tác.

Ghép ba scalar boundary logits tạo `ordinal_logits [B,3]`.

## 8. Dual decoder và output cuối

```mermaid
flowchart TB
    OL["ordinal logits [B,3]"] --> SIG["sigmoid 3 conditional probabilities"]
    SIG --> CUM["Cumulative products<br/>q0, q0*q1, q0*q1*q2"]
    CUM --> RP["rank probabilities [B,4]<br/>none, weak, medium, strong"]
    RP --> REORDER["Đổi về dataset order"]
    REORDER --> DP["ordinal dataset probabilities [B,4]<br/>none, strong, medium, weak"]
    DP --> LOGP["log ordinal probabilities"]

    B0["boundary feature 0 [B,256]"] --> BMEAN["Mean 3 boundary features"]
    B1["boundary feature 1 [B,256]"] --> BMEAN
    B2["boundary feature 2 [B,256]"] --> BMEAN
    BMEAN --> CATN["Concat pooled boundary evidence,<br/>audio và video_global<br/>[B,768]"]
    AN["audio [B,256]"] --> CATN
    VN["video_global [B,256]"] --> CATN
    CATN --> NH["Nominal head<br/>Linear 768→256→4"]
    NH --> NL["nominal logits [B,4]"]

    LOGP --> COMB["combined logits = log p_ordinal<br/>+ sigmoid(s)*nominal_logits"]
    NL --> COMB
    NG["Learned nominal gate<br/>sigmoid(-1.1) ≈ 0.25 lúc đầu"] --> COMB
    COMB --> LSM["log_softmax"]
    LSM --> OUT["clipwise_output [B,4]<br/>LOG-PROBABILITIES"]
    OUT --> ARG["argmax"]
    ARG --> CLASS["Predicted dataset label"]
```

### CORN probability

Với `q0`, `q1`, `q2` là ba conditional probabilities:

```text
P(rank=0) = 1 - q0
P(rank=1) = q0 - q0*q1
P(rank=2) = q0*q1 - q0*q1*q2
P(rank=3) = q0*q1*q2
```

Ordinal decoder giữ cấu trúc thứ tự và giảm lỗi xa. Nominal decoder nhìn đồng thời cả ba
boundary evidence để sửa lỗi exact-class, nhất là các cặp kề nhau. Learned coupling tương
đương với việc hiệu chỉnh `p_ordinal` bằng bằng chứng nominal trong logit space.

## 9. Nhánh motion auxiliary

```mermaid
flowchart LR
    MC9["motion context [B,8,256]"] --> AVG9["Mean theo 8 frames"]
    AVG9 --> MR["Motion regressor<br/>Linear 256→256→1 + sigmoid"]
    MR --> MS["motion_score [B]"]
    RANK9["ordinal rank / 3<br/>[B]"] --> HUBER["Smooth L1 motion loss"]
    MS --> HUBER
```

Nhánh này ép temporal representation giữ thông tin liên quan đến mức chuyển động và cường
độ ăn. Nó là auxiliary output, không trực tiếp được dùng để chọn class ở inference.

## 10. Training-only corruption

Trước forward pass, mỗi training sample chọn đúng một trong ba trạng thái loại trừ nhau:

```mermaid
flowchart TD
    SAMPLE["Training sample"] --> DRAW["Random action"]
    DRAW -->|"88%"| CLEAN["Giữ sạch cả hai modality"]
    DRAW -->|"2%"| DROP["Chọn ngẫu nhiên audio hoặc video<br/>rồi zero toàn modality"]
    DRAW -->|"10%"| CORRUPT["Chọn ngẫu nhiên một modality<br/>rồi làm hỏng modality đó"]
    CORRUPT --> ACORR["Audio: noise theo SNR<br/>hoặc gain hoặc temporal mask"]
    CORRUPT --> VCORR["Video: brightness<br/>hoặc Gaussian blur hoặc occlusion"]
```

Tính theo toàn bộ batch, kỳ vọng là 1% audio dropout, 1% video dropout, 5% audio corruption,
5% video corruption và 88% clean. Mục đích là dạy reliability head phản ứng với modality
kém chất lượng thay vì luôn dự đoán cả hai đều đáng tin.

## 11. Toàn bộ loss graph

```mermaid
flowchart TB
    TARGET["Dataset target [B]"] --> RANK["Map sang ordinal rank [B]"]

    FOL["fused ordinal logits [B,3]"] --> LF["Fused CORN loss<br/>weight 1.0"]
    AOL11["audio ordinal logits [B,3]"] --> LA["Audio auxiliary CORN"]
    VOL11["video ordinal logits [B,3]"] --> LV["Video auxiliary CORN"]
    RANK --> LF
    RANK --> LA
    RANK --> LV
    LA --> LAUX["Auxiliary ordinal loss<br/>0.3*(audio + video)"]
    LV --> LAUX

    AOL11 --> RT["Detached reliability targets<br/>exp(-boundary BCE)"]
    VOL11 --> RT
    AR11["raw audio reliability"] --> LREL["Smooth L1 reliability loss<br/>weight 0.1"]
    VR11["raw video reliability"] --> LREL
    RT --> LREL

    FINAL11["clipwise_output [B,4]"] --> LCAT["Final categorical NLL<br/>weight 1.0"]
    TARGET --> LCAT

    NOM11["nominal logits [B,4]"] --> LNOM["Nominal cross-entropy<br/>weight 0.5"]
    TARGET --> LNOM

    MOT11["motion score [B]"] --> LMOT["Smooth L1 với rank/3<br/>weight 0.1"]
    RANK --> LMOT

    AT11["audio teacher logits [B,4]"] --> LPRES["Teacher preservation CE"]
    VT11["mean video teacher logits [B,4]"] --> LPRES
    TARGET --> LPRES
    LPRES --> WPRES["Preservation loss<br/>weight 0.3"]

    LF --> TOTAL["TOTAL LOSS"]
    LAUX --> TOTAL
    LREL --> TOTAL
    LCAT --> TOTAL
    LNOM --> TOTAL
    LMOT --> TOTAL
    WPRES --> TOTAL
    TOTAL --> BACK["backward()"]
    BACK --> OPT["Adam step<br/>encoder lr=1e-5<br/>fusion/head lr=1e-4"]
```

Loss tổng:

```text
L = L_fused_CORN
  + 0.3 * (L_audio_CORN + L_video_CORN)
  + 0.1 * L_reliability
  + 1.0 * L_final_categorical
  + 0.1 * L_motion
  + 0.5 * L_nominal
  + 0.3 * L_teacher_preservation
```

## 12. Tensor shape ledger

| Stage | Tensor shape |
|---|---:|
| Raw audio batch | `[B,128000]` |
| STFT | `[B,1,126,1025]` |
| Padded log-Mel | `[B,1,128,128]` |
| Audio embedding | `[B,512]` |
| Audio teacher logits | `[B,4]` |
| Raw video batch | `[B,8,3,224,224]` |
| Flattened frame batch | `[B*8,3,224,224]` |
| EfficientNet frame embedding | `[B*8,1280]` |
| Video embeddings | `[B,8,1280]` |
| Video teacher logits | `[B,8,4]` |
| Projected audio | `[B,256]` |
| Projected video frames | `[B,8,256]` |
| Motion context | `[B,8,256]` |
| Temporal tokens | `[B,8,256]` |
| Event probabilities | `[B,8]` |
| Boundary queries | `[B,3,256]` |
| Temporal attention | `[B,3,8]` |
| Video boundary features | `[B,3,256]` |
| Audio/video reliability | `[B,3]` mỗi modality |
| Audio/video gate weights | `[B,3]` mỗi modality |
| Boundary interaction features | ba tensor `[B,256]` |
| Ordinal logits | `[B,3]` |
| Rank probabilities | `[B,4]` |
| Nominal logits | `[B,4]` |
| Motion score | `[B]` |
| Final `clipwise_output` | `[B,4]` log-probabilities |

## 13. Output dictionary

| Key | Vai trò |
|---|---|
| `clipwise_output` | Log-probability cuối theo dataset order; dùng để tính NLL và `argmax` |
| `rank_probabilities` | Probability thuần ordinal theo rank order |
| `ordinal_logits` | Ba fused CORN conditional logits |
| `audio_ordinal_logits` | Auxiliary audio CORN logits |
| `video_ordinal_logits` | Auxiliary video CORN logits |
| `audio_reliability`, `video_reliability` | Reliability thô dùng trong reliability loss |
| `audio_gate_reliability`, `video_gate_reliability` | Reliability đã hiệu chỉnh bằng confidence |
| `audio_gate_weights`, `video_gate_weights` | Trọng số fusion cho từng boundary |
| `temporal_attention` | Phân phối attention trên 8 frame cho từng boundary |
| `event_probabilities` | Xác suất frame chứa sự kiện hữu ích |
| `motion_score` | Dự đoán normalized intensity từ motion context |
| `nominal_logits` | Raw output của exact-class decoder |
| `teacher_residual_scales` | Ba gate điều khiển teacher residual |
| `nominal_residual_gate` | Gate coupling nominal decoder vào output cuối |
| `audio_teacher_logits` | Classifier logits của audio branch |
| `video_teacher_logits_mean` | Trung bình frame logits của video branch |
| `audio_features` | PANNs feature `[B,512]` |
| `video_features` | EfficientNet features `[B,8,1280]` |

`rank_probabilities` phản ánh ordinal decoder trước nominal correction. Vì prediction cuối lấy
từ `clipwise_output`, argmax của hai tensor này không bắt buộc giống nhau.

## 14. Tóm tắt ý đồ kiến trúc

```mermaid
flowchart LR
    P1["Một frame thiếu motion"] --> S1["8 frames + explicit motion<br/>+ temporal Transformer"]
    P2["Audio/video có thể nhiễu<br/>khác nhau theo sample"] --> S2["Corruption training +<br/>boundary reliability gates"]
    P3["Bằng chứng cho ba mức chuyển tiếp<br/>không nhất thiết cùng thời điểm"] --> S3["3 boundary-conditioned<br/>temporal queries"]
    P4["Fine-tuning làm classifier cũ drift"] --> S4["Teacher residual +<br/>label-anchored preservation"]
    P5["Ordinal tốt nhưng dễ nhầm<br/>hai lớp liền kề"] --> S5["Parallel nominal decoder +<br/>learned logit coupling"]
```

Mô hình vì vậy không phải phép nối feature đơn giản. Nó thực hiện lần lượt:

```text
single-modal representation
→ temporal motion reasoning
→ audio-conditioned event selection
→ boundary-specific evidence extraction
→ reliability-aware multimodal interaction
→ ordinal decoding
→ exact-class correction
→ final four-class log-probability
```

## 15. Các file nguồn tương ứng

- Entry point: [`main.py`](../main.py)
- Config được entry point đọc: [`config/train_config.json`](../config/train_config.json)
- Dataset/collate: [`dataset/multimodal_dataset.py`](../dataset/multimodal_dataset.py)
- Audio loading: [`dataset/audio_loader.py`](../dataset/audio_loader.py)
- Video decoding: [`dataset/video_loader.py`](../dataset/video_loader.py)
- Clip transforms: [`transforms/video_transform.py`](../transforms/video_transform.py)
- Multimodal wrapper và feature hooks: [`models/multimodal_model.py`](../models/multimodal_model.py)
- Audio frontend: [`features/audio_frontend.py`](../features/audio_frontend.py)
- PANNs CNN6: [`models/audio/panns_cnn6.py`](../models/audio/panns_cnn6.py)
- EfficientNet-B0 wrapper: [`models/video/efficientnet_b0.py`](../models/video/efficientnet_b0.py)
- Temporal BORA-Fuse: [`models/fusion/fusion_heads.py`](../models/fusion/fusion_heads.py)
- CORN mapping và toàn bộ BORA loss: [`utils/ordinal.py`](../utils/ordinal.py)
- Corruption: [`utils/corruption.py`](../utils/corruption.py)
- Training/checkpoint/test loop: [`tasks/trainer.py`](../tasks/trainer.py)

</details>
