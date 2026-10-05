# Thiết kế ablation — Dual-Decoder Temporal BORA-Fuse

Tài liệu này giải thích mỗi ablation đo cái gì, vì sao cài đặt như vậy và cách diễn giải
kết quả. Cách chạy: đặt trường `"ablation"` trong `config/train_config.json` thành một
trong các tên ở tiêu đề bên dưới (hoặc `"none"` cho Full), rồi chạy `python main.py`.
Chi tiết xem README, mục *Ablations* và *Cross-validation*.

Baseline (Full): run `MultimodalDL_TinyPANNS_ECA_MobileViTXXS_temporal_bora_fusion_holdout_20260924_172932`,
tức TinyPANNS_ECA + MobileViT-XXS, 2 frame, 200 epoch (test acc 0.9682, QWK 0.978,
severe error 0.29%). Cấu hình nằm trong `config/train_config.json`.

## Nguyên tắc: leave-one-component-out

Mỗi biến thể bằng Full trừ đúng một thành phần, cùng với loss chỉ phục vụ thành phần đó.
Mọi thứ khác giữ nguyên: seed 42, split, teacher, epoch, learning rate, corruption,
warm-up và các trọng số loss còn lại. Khi nạp config, preset của trường `"ablation"`
được ghi đè lên chính config baseline, nên các biến thể không thể lệch cấu hình so với
Full. Mỗi hàng trong bảng đo đóng góp *biên* của một thành phần khi các thành phần còn
lại vẫn giữ nguyên.

Với cấu hình mặc định, Full giữ nguyên thứ tự tạo module nên khởi tạo theo seed giống
hệt code trước khi thêm ablation. Đã kiểm tra: tham số, output và loss khớp từng bit.

## 1. Bỏ temporal motion (`no_motion`)

**Bỏ:** đặc trưng `m_t = |v_t − v_{t−1}|` và `motion_projection`, nên token chỉ còn
`frame + position`, còn event gate chỉ nhận `[token, audio]`. Bỏ luôn `motion_regressor`
và `L_motion`.

**Giữ:** temporal Transformer, position embedding, event gate, ba boundary query và
attention theo thời gian.

**Câu hỏi:** cue thay đổi giữa các frame, khi đưa vào tường minh và có giám sát theo
`rank/3`, có giúp gì thêm ngoài khả năng tự so sánh frame của Transformer không?

**Vì sao bỏ cả ba chỗ cùng lúc:** motion context `c` được dùng ở đầu vào token, ở event
gate và ở regressor phụ. Regressor chỉ tồn tại để định hình `c`. Nếu chỉ bỏ một chỗ thì
phần còn lại vẫn mang tín hiệu motion, và ablation không còn trả lời được câu hỏi trên.

**Lưu ý khi diễn giải:**
- Với 2 frame, `_uniform_frame_indices` lấy frame đầu và frame cuối của clip 2 giây.
  Khi đó motion chỉ là một hiệu thay đổi thô ở cấp clip, nên hiệu ứng có thể nhỏ.
- Muốn tách "cue" khỏi "loss phụ", chạy thêm Full với `motion_loss_weight=0`. Biến thể
  này không cần thêm code.
- Bỏ cả Transformer là một câu hỏi khác ("mô hình hoá thời gian có cần không"), không
  phải ablation motion.

## 2. Bỏ confidence calibration (`no_confidence`)

**Bỏ:** điều chỉnh reliability theo margin
`r̃ = r · (0.25 + 0.75 · 2|σ(o_aux) − 0.5|)`. Gate dùng `r` thô:
`softmax(log r / T)`.

**Giữ:** reliability head, `L_rel` (`r` học theo `exp(−BCE_aux)`), warm-up và
temperature.

**Câu hỏi:** điều chỉnh reliability bằng độ tự tin của auxiliary head có ngăn được việc
một reliability MLP bão hòa gần 1 chiếm gate, dù quyết định phụ của nó đang không chắc
chắn hay không?

**Lưu ý khi diễn giải:**
- Không bỏ `L_rel`. Bỏ nó là ablation "học reliability", một thành phần khác.
- Gate chỉ phụ thuộc tỉ lệ `r̃_a / r̃_v`. Vì vậy ablation này loại đúng thừa số tỉ lệ
  độ tự tin giữa hai modality.
- Logit phụ có cộng teacher residual, nên ablation này cũng cắt đường đưa độ tự tin của
  teacher vào gate.
- Đo trực tiếp cơ chế bằng `scripts/compare_ablation.py`: tỉ lệ dead gate trong
  [0.45, 0.55], gate separation giữa mẫu đúng và mẫu sai, cùng mean/std của reliability.

## 3. Chỉ giữ decoder 1 — ordinal/CORN (`ordinal_only`)

**Bỏ:** nominal head, coupling gate `σ(s)` và `L_nom`. Dự đoán cuối là
`log_softmax(log p_ord)`.

**Giữ:** CORN decoder, `L_fused` (CORN) và `L_cat` (NLL trên phân phối CORN).

**Câu hỏi:** nominal decoder có biến chất lượng xếp hạng thành accuracy đúng lớp không?

**Kỳ vọng:** rank MAE và QWK gần như không đổi, accuracy giảm, nhất là ở các cặp lớp kề
nhau.

**Lưu ý khi diễn giải:** biến thể này gần với Temporal BORA trước khi có dual decoder.
Nó cũng mất "global bypass", tức `a` và `g` không đi qua gate, vì bypass này chỉ dẫn
vào nominal head.

## 4. Chỉ giữ decoder 2 — nominal (`nominal_only`)

**Bỏ:** boundary classifier, teacher residual vào boundary logit, chuyển CORN sang rank
probability, `L_fused` và coupling. Dự đoán cuối là `log_softmax(nominal_logits)`.
`rank_probabilities` được suy ra từ softmax nominal (sắp theo thứ tự rank) để file
audit giữ cùng định dạng.

**Giữ:** toàn bộ phần fusion theo boundary (gate, query, `I_k`), nominal head, các
auxiliary CORN head và `L_rel`, vì gate cần chúng.

**Câu hỏi:** CORN decoder có giảm lỗi xa và giữ được thứ tự không?

**Kỳ vọng:** accuracy gần như giữ nguyên hoặc giảm nhẹ, severe error tăng, QWK giảm.

**Lưu ý khi diễn giải:**
- Đặt `nominal_loss_weight=0`. Lúc này `L_cat` và `L_nom` cùng tác động lên một logit,
  nên giữ cả hai là đếm loss hai lần.
- Mô hình vẫn còn thiên kiến thứ tự: ba boundary query, các aux CORN head và `L_motion`.
  Vì vậy kết luận đúng là "giá trị của CORN *decoder*", không phải "giá trị của
  ordinal modeling nói chung".
- Biến thể này cũng mất đường teacher residual vào dự đoán cuối. Cần ghi rõ điều này
  khi báo cáo.

## Thống kê và báo cáo

- Báo cáo đủ accuracy, F1-macro, rank MAE, QWK, within-one và severe error. Trainer đã
  ghi các chỉ số này vào `result.csv`; `scripts/summarize_ablations.py` gom chúng thành
  một bảng kèm chênh lệch so với Full.
- Test holdout có 2.800 mẫu, nên 1 mẫu tương ứng 0,036%. Với acc ≈ 0,968, sai số chuẩn
  khoảng 0,33%. Chênh lệch dưới ~0,5% cần kiểm định McNemar trên từng cặp dự đoán
  (`scripts/compare_ablation.py`), tốt nhất kèm cross-validation (mean ± std) hoặc
  nhiều seed.
- Chạy lại Full (`"ablation": "none"`) bằng cùng code và môi trường với bốn biến thể.
  GPU không tất định, nên số của Full có thể lệch nhẹ so với 0,9682.
- Cross-validation chỉ hợp lệ khi mỗi fold có teacher riêng, train bằng cùng splitter.
  Dùng chung teacher holdout sẽ rò rỉ nhãn test (xem README, mục *Cross-validation*).
