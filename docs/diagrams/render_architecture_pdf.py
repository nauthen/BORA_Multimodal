"""Code-faithful, seven-page architecture atlas; no model weights are loaded.

Run: python docs/diagrams/render_architecture_pdf.py
Output: output/pdf/temporal_bora_architecture.pdf and seven separate PDFs.
Layout follows the supplied vertical, two-branch architecture reference.
"""
from __future__ import annotations

import math
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf"
PARTS = OUT / "architecture_parts"
W, H = 840, 1188
INK, MUTED, LINE = "#172B42", "#526579", "#8191A2"
PALETTE = {
    "audio": ("#EBF4FE", "#2265A5"),
    "video": ("#EAF7F3", "#137C69"),
    "fusion": ("#F2EEFC", "#7054A2"),
    "teacher": ("#FFF0F3", "#AB4D6C"),
    "loss": ("#FFF5E7", "#A66B22"),
    "plain": ("#F5F7FA", "#60758A"),
}


def fonts():
    candidates = [
        (Path("C:/Windows/Fonts"), ("arial.ttf", "arialbd.ttf", "consola.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu"),
         ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf")),
    ]
    for folder, names in candidates:
        if all((folder / n).exists() for n in names):
            for alias, filename in zip(("Body", "Bold", "Mono"), names):
                pdfmetrics.registerFont(TTFont(alias, str(folder / filename)))
            return
    raise RuntimeError("Install Arial or DejaVu fonts before rendering.")


class Page:
    def __init__(self, index, slug, title, subtitle, sources):
        self.path = PARTS / f"{index:02d}_{slug}.pdf"
        self.c = canvas.Canvas(str(self.path), pagesize=(W, H), pageCompression=1)
        self.c.setTitle(f"{index:02d} | {title} | Temporal BORA-Fuse")
        self.c.setAuthor("Architecture traced from local implementation")
        self.c.setSubject("PANNS_Cnn6 + EfficientNetB0; 8 frames; dual decoder")
        self.text(42, 35, "KIẾN TRÚC TỪ CODE  /  DUAL-DECODER TEMPORAL BORA-FUSE", 10, "Bold", MUTED)
        self.text(42, 66, title, 24, "Bold")
        self.text(42, 91, subtitle, 11.5, color=MUTED)
        self.rule(42, 108, 798, 108)
        self.text(42, 1136, sources, 9.1, color=MUTED)
        self.rule(42, 1150, 798, 1150)
        self.text(42, 1169, "B = batch  |  T = 8 frames  |  D = 256  |  K = 3 ordinal boundaries", 9, color=MUTED)
        self.text(798, 1169, f"{index:02d} / 07", 9, "Bold", MUTED, "right")
        self.boxes = []

    def text(self, x, y, text, size=11, font="Body", color=INK, align="left"):
        self.c.setFont(font, size)
        self.c.setFillColor(HexColor(color))
        width = pdfmetrics.stringWidth(text, font, size)
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        assert x >= 20 and x + width <= W - 20, (self.path.name, text, x, width)
        self.c.drawString(x, H - y, text)

    def rule(self, x1, y1, x2, y2, color=LINE, dashed=False):
        self.c.setStrokeColor(HexColor(color))
        self.c.setLineWidth(1)
        self.c.setDash(4, 3) if dashed else self.c.setDash()
        self.c.line(x1, H-y1, x2, H-y2)
        self.c.setDash()

    def arrow(self, points, color=LINE, dashed=False):
        self.c.setStrokeColor(HexColor(color))
        self.c.setFillColor(HexColor(color))
        self.c.setLineWidth(1.2)
        self.c.setDash(4, 3) if dashed else self.c.setDash()
        path = self.c.beginPath()
        path.moveTo(points[0][0], H-points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, H-y)
        self.c.drawPath(path)
        self.c.setDash()
        x, y = points[-1]
        xp, yp = points[-2]
        a = math.atan2(y-yp, x-xp)
        tip = self.c.beginPath()
        tip.moveTo(x, H-y)
        for delta in (-0.43, 0.43):
            tip.lineTo(x-7*math.cos(a+delta), H-(y-7*math.sin(a+delta)))
        tip.close()
        self.c.drawPath(tip, stroke=0, fill=1)

    def box(self, x, y, w, h, title, lines=(), kind="plain", mono=False):
        fill, stroke = PALETTE[kind]
        self.c.setFillColor(HexColor(fill))
        self.c.setStrokeColor(HexColor(stroke))
        self.c.setLineWidth(0.85)
        self.c.roundRect(x, H-y-h, w, h, 7, fill=1, stroke=1)
        self.c.setFillColor(HexColor(stroke))
        self.c.roundRect(x, H-y-h, 4, h, 2, fill=1, stroke=0)
        assert pdfmetrics.stringWidth(title, "Bold", 12) <= w-28, (self.path.name, title)
        self.text(x+14, y+21, title, 12, "Bold", stroke)
        font = "Mono" if mono else "Body"
        for i, line in enumerate(lines):
            assert pdfmetrics.stringWidth(line, font, 10.5) <= w-28, (self.path.name, title, line)
            self.text(x+14, y+41+i*15, line, 10.5, font)
        if lines:
            assert 41+(len(lines)-1)*15 <= h-9, (title, h)
        self.boxes.append((x,y,w,h))
        return (x, y, w, h)

    def down(self, a, b, color=LINE, dashed=False):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        p = [(ax+aw/2, ay+ah), (bx+bw/2, by)]
        if ax+aw/2 != bx+bw/2:
            mid = (ay+ah+by)/2
            p = [p[0], (p[0][0], mid), (p[-1][0], mid), p[-1]]
        self.arrow(p, color, dashed)

    def label(self, x, y, s, kind="plain"):
        self.text(x, y, s, 10, "Bold", PALETTE[kind][1])

    def note(self, y, lines):
        for i, s in enumerate(lines):
            self.text(42, y+i*16, s, 10.5, color=MUTED)

    def save(self):
        # Boxes must never overlap; all connection routing is specified explicitly.
        for i, (x,y,w,h) in enumerate(self.boxes):
            assert y >= 115 and y+h <= 1112
            for x2,y2,w2,h2 in self.boxes[i+1:]:
                assert min(x+w,x2+w2)-max(x,x2) <= 0 or min(y+h,y2+h2)-max(y,y2) <= 0
        self.c.showPage()
        self.c.save()
        return self.path


def overview():
    p = Page(1, "overview", "01  Tổng quan kiến trúc", "Hai encoder độc lập, chọn bằng chứng theo thời gian, fusion theo ranh giới và hai decoder.",
             "Nguồn: models/multimodal_model.py | models/fusion/fusion_heads.py | config/train_config.json")
    p.label(42, 136, "AUDIO BRANCH", "audio")
    p.label(444, 136, "VIDEO BRANCH", "video")
    a = p.box(42,150,354,68,"Audio input",["Mono, 64 kHz, crop/pad 2 giây", "waveform [B,128000]"],"audio")
    v = p.box(444,150,354,68,"Video input",["8 RGB frames lấy đều; resize + normalize", "video_form [B,8,3,224,224]"],"video")
    ae = p.box(42,244,354,85,"AudioFrontend + PANNS_Cnn6  [trang 2]",["Log-Mel [B,1,128,128] -> CNN6 -> f_a [B,512]", "Classifier gốc: t_a [B,4]", "FeatureHook giữ cả input và output classifier."],"audio")
    ve = p.box(444,244,354,85,"EfficientNet-B0 dùng chung  [trang 2]",["[B*8,3,224,224] -> f_v [B,8,1280]", "Classifier từng frame: t_v [B,8,4]", "Cùng một encoder cho cả 8 frames."],"video")
    p.down(a,ae); p.down(v,ve)
    ap = p.box(42,355,354,68,"Audio projection",["Linear 512 -> 256 + LN + GELU + DO(0.15)", "a [B,256]"],"audio")
    vt = p.box(444,355,354,98,"Temporal evidence  [trang 3]",["Linear 1280 -> 256; motion + position", "Transformer: 2 layers, 4 heads, FFN 512", "Audio-conditioned event gate + 3 queries", "V [B,3,256]; g = mean_k(V) [B,256]"],"video")
    p.down(ae,ap); p.down(ve,vt)
    p.arrow([(396,389),(420,389),(420,405),(444,405)],PALETTE["audio"][1])
    rel = p.box(130,490,580,100,"Teacher residual + reliability gates  [trang 4]",["a, V, g + classifier logits t_a, t_v", "Auxiliary ordinal logits -> confidence-modulated reliability", "Boundary-wise audio/video weights w [B,3,2]; temperature = 0.3", "Teacher conditional logits C_a, C_v [B,3] remain in the forward path."],"teacher")
    p.down(ap,rel); p.down(vt,rel)
    fus = p.box(130,620,580,85,"Ba khối boundary fusion riêng biệt  [trang 5]",["Inputs: a, V_k, w_k, C_a,k, C_v,k; k = 0,1,2", "Weighted features + absolute difference + element-wise product", "Outputs: I [B,3,256] and fused ordinal logits l [B,3]"],"fusion")
    p.down(rel,fus)
    o = p.box(42,746,354,83,"Ordinal decoder  [trang 6]",["l -> sigmoid -> cumulative products", "4 rank probabilities -> dataset order", "p_ord [B,4]"],"fusion")
    n = p.box(444,746,354,83,"Nominal decoder  [trang 6]",["concat(mean_k(I), a, g) [B,768]", "Linear 768 -> 256 -> 4", "nominal logits n [B,4]"],"fusion")
    p.down(fus,o); p.down(fus,n)
    out = p.box(130,874,580,85,"Learned coupling -> output cuối",["clipwise_output = log_softmax(log(p_ord) + sigmoid(s_nom) * n)", "[B,4] LOG-PROBABILITIES; prediction = argmax", "Dataset order: [none, strong, medium, weak]"],"fusion")
    p.down(o,out); p.down(n,out)
    p.note(994,["Các đầu vào ghi tên trong khối là tensor được truyền tiếp; xem trang chi tiết để theo dõi mọi nhánh.",
                "a và g đi thẳng vào nominal decoder; nhánh này không bị nhân với reliability weights.",
                "Motion regressor và 7 thành phần loss được tách ở trang 7.",
                "Các classifier gọi là 'teacher' được tái sử dụng và fine-tune cùng encoder.",
                "Sơ đồ mô tả một forward pass 8 frame; không thêm ensemble/TTA hay số đo accuracy từ PDF mẫu."])
    return p.save()


def encoders():
    p = Page(2,"encoders","02  Chi tiết hai encoder","Shape giữ đầy đủ batch axis; N = B*8 là batch frame của EfficientNet-B0.",
             "Nguồn: features/audio_frontend.py | models/audio/panns_cnn6.py | models/video/efficientnet_b0.py")
    p.label(42,133,"AUDIO: FRONTEND + PANNS_CNN6","audio")
    p.label(444,133,"VIDEO: SHARED EFFICIENTNET-B0","video")
    audio = [
        ("Load waveform",["Mono + resample 64 kHz + crop/pad 2 s", "[B,128000]"]),
        ("STFT -> log-Mel",["FFT/window 2048; hop 1024; Hann; center", "[B,1,126,1025] -> [B,1,126,128]"]),
        ("Time pad + mel BatchNorm",["Pad 2 rows at start -> [B,1,128,128]", "BN on 128 mel bins; SpecAugment in train"]),
        ("ConvBlock 1: 1 -> 64",["Conv 5x5 / s1 / p2 + BN + ReLU", "AvgPool 2x2 -> [B,64,64,64]; DO(0.2)"]),
        ("ConvBlock 2: 64 -> 128",["Conv 5x5 / s1 / p2 + BN + ReLU", "AvgPool 2x2 -> [B,128,32,32]; DO(0.2)"]),
        ("ConvBlock 3: 128 -> 256",["Conv 5x5 / s1 / p2 + BN + ReLU", "AvgPool 2x2 -> [B,256,16,16]; DO(0.2)"]),
        ("ConvBlock 4: 256 -> 512",["Conv 5x5 / s1 / p2 + BN + ReLU", "AvgPool 2x2 -> [B,512,8,8]; DO(0.2)"]),
        ("Frequency mean -> temporal pooling",["Mean over F -> [B,512,8]", "max over time + mean over time -> [B,512]"]),
        ("Audio feature hook",["DO(0.2) -> Linear 512 -> 512 -> ReLU", "DO(0.2) -> f_a [B,512] (classifier input)"]),
        ("Original classifier + fusion projection",["fc_audioset: Linear 512 -> 4 -> t_a [B,4]", "f_a -> Linear 512 -> 256 + LN/GELU/DO"]),
    ]
    video = [
        ("Load clip + reshape",["8 evenly spaced RGB frames; normalize", "[B,8,3,224,224] -> [N,3,224,224]"]),
        ("Stem: Conv 3x3 / stride 2",["3 -> 32; BatchNorm + SiLU", "[N,32,112,112]"]),
        ("Stage 1: MBConv1, k3, x1",["32 -> 16; first stride 1", "[N,16,112,112]"]),
        ("Stage 2: MBConv6, k3, x2",["16 -> 24; first stride 2", "[N,24,56,56]"]),
        ("Stage 3: MBConv6, k5, x2",["24 -> 40; first stride 2", "[N,40,28,28]"]),
        ("Stage 4: MBConv6, k3, x3",["40 -> 80; first stride 2", "[N,80,14,14]"]),
        ("Stage 5: MBConv6, k5, x3",["80 -> 112; first stride 1", "[N,112,14,14]"]),
        ("Stage 6 + Stage 7",["MBConv6 k5 x4 / s2 -> [N,192,7,7]", "MBConv6 k3 x1 / s1 -> [N,320,7,7]"]),
        ("Head + video feature hook",["Conv 1x1 -> [N,1280,7,7]; global avg pool", "Flatten -> DO(0.2) -> [N,1280]"]),
        ("Original classifier + reshape",["Linear 1280 -> 4 -> t_v [B,8,4]", "Hook input f_v [B,8,1280] -> projection"]),
    ]
    for x, items, kind in [(42,audio,"audio"),(444,video,"video")]:
        prev = None
        for i,(title,lines) in enumerate(items):
            b = p.box(x,148+i*87,354,69,title,lines,kind)
            if prev: p.down(prev,b)
            prev=b
    p.note(1047,["MBConv: expansion -> depthwise conv -> squeeze-excitation -> projection; residual khi shape cho phép.",
                "Video train augmentation dùng cùng flip/brightness cho toàn clip; eval chỉ resize + normalize.",
                "LN = LayerNorm; DO = dropout (chỉ train). STFT/Mel weights frozen; encoder và classifier fine-tune.",
                "Không có fusion giữa các CNN stages; fusion nhận feature sau pooling và classifier-input hooks."])
    return p.save()


def temporal():
    p = Page(3,"temporal_attention","03  Temporal evidence & attention","Audio toàn clip điều kiện hóa event gate và ba query; attention được chuẩn hóa theo 8 frame.",
             "Nguồn: models/fusion/fusion_heads.py | TemporalBORAFusion.__init__ / forward")
    a=p.box(42,142,264,84,"Audio projection",["f_a [B,512]", "Linear 512 -> 256 + LN/GELU/DO", "a [B,256]"],"audio")
    v=p.box(344,142,454,84,"Frame projection",["f_v [B,8,1280]", "Linear 1280 -> 256 + LN + GELU + DO(0.15)", "v [B,8,256]"],"video")
    m=p.box(344,252,454,84,"Explicit motion -> motion context",["m_0 = 0; m_t = abs(v_t - v_(t-1))", "c = GELU(LN(Linear 256 -> 256(m)))", "m, c [B,8,256]"],"video",True); p.down(v,m)
    t=p.box(344,362,454,114,"Temporal Transformer",["Input = v + c + position[:, :8]", "Learned position table [1,16,256]", "2 pre-norm encoder layers; 4 self-attention heads", "FFN 256 -> 512 -> 256; GELU; dropout 0.15", "Final LayerNorm -> h [B,8,256]"],"video"); p.down(m,t)
    q=p.box(42,362,264,114,"Audio-conditioned queries",["Linear 256 -> 768(a)", "Reshape [B,3,256]", "+ learned boundary_queries [3,256]", "Q [B,3,256]", "Một query cho mỗi k = 0,1,2."],"audio"); p.down(a,q)
    e=p.box(344,503,454,84,"Audio-conditioned event gate",["concat(h_t, a, c_t) [B,8,768]", "Linear 768 -> 256 -> GELU -> DO -> Linear -> 1", "e = sigmoid(event_logits) [B,8]"],"video"); p.down(t,e)
    att=p.box(130,628,580,115,"Boundary-conditioned temporal attention",["score_k,t = scale * cosine(h_t, Q_k) + log(max(e_t, 1e-6))", "scale = min(exp(attention_logit_scale), 100); init = 10", "alpha_k,: = softmax_t(score_k,:)", "alpha [B,3,8]; sum_t(alpha_k,t) = 1", "Normalize h và Q chỉ khi tính score; pooling dùng h chưa normalize."],"fusion")
    p.arrow([(174,476),(174,609),(290,609),(290,628)],PALETTE["audio"][1])
    p.down(e,att)
    pooled=p.box(130,774,580,83,"Weighted temporal pooling",["V_k = sum_t(alpha_k,t * h_t)", "V = stack(V_0,V_1,V_2) [B,3,256]", "g = mean_k(V_k) [B,256]"],"fusion",True); p.down(att,pooled)
    end=p.box(130,888,580,83,"Truyền bằng chứng sang fusion và decoder",["V_k -> video reliability head và video boundary projection k", "g -> auxiliary video ordinal head và nominal decoder", "a -> audio reliability, audio ordinal, boundary projections và nominal"],"fusion"); p.down(pooled,end)
    p.note(1011,["Ba boundary tương ứng các conditional tasks: r > 0; r > 1 | r >= 1; r > 2 | r >= 2.",
                "Temporal attention có ba phân phối frame riêng; g là mean theo boundary, không phải mean thẳng f_v.",
                "Event gate được học qua objective chung; code không có event-label loss riêng.",
                "Motion được tính trên projected frame embeddings, không phải optical flow hoặc pixel difference.",
                "a được broadcast trên 8 frame tại event gate; c cũng đi tới motion regressor [trang 7]."])
    return p.save()


def reliability():
    p=Page(4,"teacher_reliability","04  Teacher residual & reliability","Classifier gốc vẫn nằm trong forward pass; các đầu ra auxiliary đồng thời điều khiển confidence.",
           "Nguồn: models/fusion/fusion_heads.py | _teacher_probabilities_to_conditional_logits / _gate_weights")
    a=p.box(42,142,354,83,"Audio classifier probabilities",["t_a [B,4] -> softmax", "p_a [B,4]", "Dataset order: [none,strong,medium,weak]"],"teacher")
    v=p.box(444,142,354,83,"Video classifier probabilities",["t_v [B,8,4] -> softmax từng frame", "p_v = mean_t(softmax(t_v)) [B,4]", "Average probabilities trước conditional conversion."],"teacher")
    conv=p.box(130,256,580,129,"Đổi xác suất 4 lớp thành 3 conditional logits",["Reorder -> p_rank = [p_none,p_weak,p_medium,p_strong]", "s0 = p_weak+p_medium+p_strong; s1 = p_medium+p_strong; s2 = p_strong", "q = [s0, s1/max(s0,1e-6), s2/max(s1,1e-6)]", "C = logit(clamp(q,1e-5,1-1e-5))", "Thực hiện cho cả audio và video -> C_a, C_v [B,3]", "beta = sigmoid(teacher_residual_scale) [3]; init sigmoid(-1.1) ~ 0.25"],"teacher"); p.down(a,conv); p.down(v,conv)
    ao=p.box(42,416,354,83,"Audio auxiliary ordinal logits",["u_a = Linear 256 -> 3(a) + beta * C_a", "u_a [B,3]", "a [B,256] từ trang 3."],"audio")
    vo=p.box(444,416,354,83,"Video auxiliary ordinal logits",["u_v = Linear 256 -> 3(g) + beta * C_v", "u_v [B,3]", "g = mean_k(V_k) [B,256]."],"video"); p.down(conv,ao); p.down(conv,vo)
    ar=p.box(42,531,354,99,"Audio reliability + confidence",["r_a = sigmoid(MLP 256 -> 256 -> 3(a))", "conf_a = 2*abs(sigmoid(u_a)-0.5)", "rho_a = r_a * (0.25 + 0.75*conf_a)", "r_a, rho_a [B,3]"],"audio")
    vr=p.box(444,531,354,99,"Video reliability + confidence",["r_v,k = sigmoid(MLP 256 -> 256 -> 1(V_k))", "conf_v = 2*abs(sigmoid(u_v)-0.5)", "rho_v = r_v * (0.25 + 0.75*conf_v)", "r_v, rho_v [B,3]"],"video"); p.down(ao,ar); p.down(vo,vr)
    gate=p.box(130,665,580,114,"Boundary-wise modality gate",["w_k = softmax_modalities(log(max([rho_a,k,rho_v,k],1e-6)) / 0.3)", "w [B,3,2]; w_a,k + w_v,k = 1", "set_epoch(epoch) với epoch < 3: w_k = [0.5,0.5]", "set_epoch(None): dùng learned gate; trainer test gọi với epoch=None.", "Warm-up cũng áp dụng cho validation khi trainer truyền epoch 0,1,2."],"fusion"); p.down(ar,gate); p.down(vr,gate)
    dest=p.box(130,813,580,98,"Đầu ra của cụm này",["w -> weighted feature fusion và weighted teacher residual [trang 5]", "u_a, u_v -> auxiliary CORN losses; r_a, r_v -> reliability loss [trang 7]", "C_a, C_v, beta -> fused boundary logits [trang 5]", "Reliability MLPs dùng GELU + DO(0.15) giữa hai Linear layers."],"fusion"); p.down(gate,dest)
    p.note(954,["Teacher residual dùng cùng beta_k ở auxiliary audio, auxiliary video và fused logit của boundary k.",
                "Ba beta_k là tham số học được; nominal decoder có một scalar gate riêng.",
                "Teacher logits không detach trong đường dự đoán; hai encoder và classifier tiếp tục nhận gradient.",
                "Chỉ target exp(-BCE) của reliability loss được detach; xem trang 7.",
                "Preservation loss dùng mean video LOGITS; bước teacher conversion ở trên dùng mean PROBABILITIES."])
    return p.save()


def fusion():
    p=Page(5,"boundary_fusion","05  Fusion theo từng ranh giới","Khối dưới được lặp độc lập cho k = 0,1,2 với projection, interaction và classifier riêng.",
           "Nguồn: models/fusion/fusion_heads.py | TemporalBORAFusion.forward (boundary loop)")
    a=p.box(42,146,354,83,"Audio boundary projection k",["a [B,256] -> Linear 256 -> 256", "A_k [B,256]", "audio_boundary_projections[k]"],"audio")
    v=p.box(444,146,354,83,"Video boundary projection k",["V_k [B,256] -> Linear 256 -> 256", "B_k [B,256]", "video_boundary_projections[k]"],"video")
    both=p.box(130,265,580,68,"Ba thành phần tương tác",["Inputs: A_k, B_k và w_a,k / w_v,k từ reliability gate", "Mỗi thành phần có shape [B,256]."],"fusion"); p.down(a,both); p.down(v,both)
    wt=p.box(42,373,238,83,"Weighted sum",["w_a,k * A_k", "+ w_v,k * B_k", "Giữ bằng chứng theo reliability."],"fusion")
    dif=p.box(301,373,238,83,"Absolute difference",["abs(A_k - B_k)", "[B,256]", "Giữ độ khác biệt hai modality."],"fusion")
    prod=p.box(560,373,238,83,"Element-wise product",["A_k * B_k", "[B,256]", "Giữ tương tác từng feature."],"fusion")
    for b in (wt,dif,prod): p.down(both,b)
    cat=p.box(130,501,580,68,"Concatenate theo feature dimension",["concat(weighted_sum, abs_difference, elementwise_product)", "[B,768]"],"fusion",True)
    for b in (wt,dif,prod): p.down(b,cat)
    inter=p.box(130,603,580,83,"Boundary interaction k",["Linear 768 -> 256 -> LayerNorm -> GELU -> DO(0.15)", "I_k [B,256]", "Giữ I_k để mean qua 3 boundaries và đưa vào nominal decoder."],"fusion"); p.down(cat,inter)
    head=p.box(42,727,354,83,"Learned boundary logit",["Linear 256 -> 1(I_k)", "learned_logit_k [B,1]", "boundary_classifiers[k]"],"fusion")
    tea=p.box(444,727,354,83,"Teacher residual k",["R_k = w_a,k*C_a,k + w_v,k*C_v,k", "beta_k * R_k [B,1]", "C_a, C_v, beta, w từ trang 4."],"teacher"); p.down(inter,head)
    final=p.box(130,850,580,83,"Cộng residual -> ghép ba boundary logits",["l_k = learned_logit_k + beta_k * R_k", "l = concat(l_0,l_1,l_2) [B,3] -> ordinal decoder", "I = stack(I_0,I_1,I_2) [B,3,256] -> nominal decoder"],"fusion",True); p.down(head,final); p.down(tea,final)
    p.note(977,["w chỉ nhân weighted-sum features và teacher logits; abs-difference/product không được gate trực tiếp.",
                "I_k được lấy trước classifier và trước teacher-logit residual.",
                "Fusion làm việc trên vector 256D; không cập nhật các spatial feature maps trong hai CNN.",
                "CORN dùng ba conditional decisions để biểu diễn bốn mức: none < weak < medium < strong."])
    return p.save()


def decoder():
    p=Page(6,"dual_decoder","06  Hai decoder & dự đoán cuối","Ordinal distribution được hiệu chỉnh bằng nominal logits với một coupling gate học được.",
           "Nguồn: models/fusion/fusion_heads.py | utils/ordinal.py | tasks/trainer.py")
    o=p.box(42,147,354,84,"ORDINAL: conditional logits",["l [B,3] từ trang 5", "q = sigmoid(l) = [q0,q1,q2]", "q_k là conditional probability."],"fusion")
    n=p.box(444,147,354,84,"NOMINAL: pooled evidence + bypass",["I [B,3,256] -> mean_k -> [B,256]", "concat(mean_k(I), a, g)", "[B,768]"],"fusion")
    probs=p.box(42,263,354,129,"CORN -> rank probabilities",["P(none)   = 1 - q0", "P(weak)   = q0 - q0*q1", "P(medium) = q0*q1 - q0*q1*q2", "P(strong) = q0*q1*q2", "Clamp minimum 0; shape [B,4].", "Rank order: [none,weak,medium,strong]"],"fusion",True); p.down(o,probs)
    nh=p.box(444,263,354,129,"Nominal head",["Linear 768 -> 256", "LayerNorm(256)", "GELU", "Dropout(0.15)", "Linear 256 -> 4", "n [B,4], raw dataset-class logits"],"fusion",True); p.down(n,nh)
    re=p.box(42,427,354,84,"Reorder + log",["p_ord = p_rank[:, [0,3,2,1]]", "o = log(max(p_ord,1e-8))", "o [B,4], dataset order"],"fusion",True); p.down(probs,re)
    ng=p.box(444,427,354,84,"Learned nominal gate",["gamma = sigmoid(s_nom)", "s_nom initialized at -1.1; gamma ~ 0.25", "nominal correction = gamma * n [B,4]"],"fusion"); p.down(nh,ng)
    out=p.box(130,551,580,99,"Coupling trong logit space",["z = o + gamma * n", "clipwise_output = log_softmax(z, dim=-1) [B,4]", "P_final = exp(clipwise_output)", "P_final,c proportional to max(p_ord,c,1e-8) * exp(gamma*n_c)."],"fusion"); p.down(re,out); p.down(ng,out)
    pred=p.box(130,686,580,83,"Prediction của training/evaluation loop",["pred = argmax(clipwise_output, dim=1) [B]", "Dataset labels: 0=none, 1=strong, 2=medium, 3=weak", "rank_probabilities là output riêng trước nominal correction."],"fusion"); p.down(out,pred)
    p.label(42,817,"LABEL MAPPING DÙNG XUYÊN SUỐT MÔ HÌNH")
    headers=[("Dataset ID",0), ("Tên lớp",1), ("Ordinal rank",2)]
    for label,col in headers: p.text(60+col*245,847,label,12,"Bold")
    p.rule(42,860,798,860)
    for i,(label,name,rank) in enumerate([(0,"none",0),(1,"strong",3),(2,"medium",2),(3,"weak",1)]):
        yy=886+i*32
        for col,value in enumerate((label,name,rank)): p.text(60+col*245,yy,str(value),12)
        p.rule(42,yy+11,798,yy+11,color="#E0E7EE")
    p.note(1030,["a và g ở nominal head là projected audio và pooled temporal video, chưa qua modality weighting.",
                "argmax(rank_probabilities) có thể khác prediction cuối do nominal correction và thứ tự label.",
                "clipwise_output là log-probability; khi hiển thị xác suất dùng exp, không hiển thị trực tiếp số âm."])
    return p.save()


def training():
    p=Page(7,"training_objective","07  Huấn luyện & toàn bộ loss","Cấu hình thực tế: 120 epochs, batch 32, Adam; checkpoint chọn theo validation accuracy.",
           "Nguồn: utils/ordinal.py::bora_loss | utils/corruption.py | tasks/trainer.py | config/train_config.json")
    aug=p.box(42,144,756,99,"Training inputs -> augmentation/corruption -> forward [trang 1-6]",["BORA action cho từng sample: 88% không thêm corruption | 10% corrupt một modality | 2% zero một modality.", "Modality được chọn đều audio/video. Audio: noise SNR 5-20 dB, gain 0.5-1.0 hoặc time mask 10-20%.", "Video: brightness 0.7-1.3, blur sigma 0.1-1.0 hoặc occlusion 5-20%; cùng phép biến đổi trên clip.", "SpecAugment + clip flip/brightness là augmentation train riêng, vẫn có thể áp dụng ở nhóm 88%."],"loss")
    target=p.box(42,272,756,84,"Supervision: dataset label y -> ordinal rank r = [0,3,2,1][y]",["Boundary k = 0,1,2: target t_k = 1[r > k]; active mask M_k = 1[r >= k].", "CORN(x,r) = mean của BCEWithLogits(x_k,t_k) trên tất cả phần tử active trong batch.", "Loss reliability cũng dùng mask M; các loss categorical/nominal/preservation dùng dataset labels y."],"loss")
    p.down(aug,target,dashed=True)
    rows=[
        ("1.00", "Fused ordinal", "l [B,3]", "L_fused = CORN(l,r)"),
        ("0.30", "Auxiliary ordinal", "u_a, u_v [B,3]", "L_aux = CORN(u_a,r) + CORN(u_v,r)"),
        ("0.10", "Reliability", "r_a, r_v [B,3]", "L_rel = 0.5 * (masked SL1_a + masked SL1_v)"),
        ("1.00", "Final categorical", "clipwise_output [B,4]", "L_cat = NLL(clipwise_output,y)"),
        ("0.10", "Motion auxiliary", "motion_score [B]", "L_motion = SmoothL1(motion_score,r/3)"),
        ("0.50", "Nominal", "n [B,4]", "L_nom = CrossEntropy(n,y)"),
        ("0.30", "Teacher preservation", "t_a, mean_t(t_v) [B,4]", "L_pres = 0.5 * (CE(t_a,y) + CE(mean_t(t_v),y))"),
    ]
    p.label(42,389,"WEIGHT / LOSS TERM")
    p.label(291,389,"TENSOR VÀ CÔNG THỨC")
    for i,(weight,title,tensor,formula) in enumerate(rows):
        y=405+i*59
        p.box(42,y,224,51,f"{weight}   {title}",[tensor],"loss")
        p.box(291,y,507,51,"Loss definition",[formula],"loss")
        p.arrow([(266,y+25),(291,y+25)],PALETTE["loss"][1],True)
    p.note(839,["Reliability target từng modality: exp(-BCEWithLogits(u,t)).detach(); SL1 = SmoothL1(reliability,target).",
                "Motion head: mean_t(c) -> Linear 256->256 -> GELU -> Linear 256->1 -> sigmoid.",
                "Motion head chỉ hỗ trợ training; motion context c vẫn nằm trong đường prediction."])
    total=p.box(42,898,756,83,"Total objective -> backward -> Adam step",["L = L_fused + 0.3*L_aux + 0.1*L_rel + L_cat + 0.1*L_motion + 0.5*L_nom + 0.3*L_pres", "Encoder branches (gồm original classifiers, BN): lr = 1e-5; fusion/head parameters: lr = 1e-4.", "Fine-tune cả hai branches; STFT/Mel filter weights frozen. Early stopping patience = 20."],"loss")
    p.note(1017,["Nguồn đối chiếu: notebook Multimodal_Deep_Fusion_dual_decoder_effnet.ipynb, cell 0 và code module.",
                "Cấu hình _build_config() trong notebook trùng config/train_config.json tại thời điểm dựng sơ đồ.",
                "Core encoder/fusion/decoder/loss được kiểm tra từ code; PDF mẫu chỉ định hướng cách trình bày.",
                "Số đo 97.82% trên tài liệu mẫu không được gán cho mô hình này."])
    return p.save()


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    fonts()
    paths=[f() for f in (overview,encoders,temporal,reliability,fusion,decoder,training)]
    writer=PdfWriter()
    for path in paths:
        writer.append(PdfReader(path), outline_item=path.stem)
    writer.add_metadata({"/Title":"Dual-Decoder Temporal BORA-Fuse - Code Architecture Atlas",
                         "/Subject":"7 diagrams based on notebook and repository code"})
    merged=OUT / "temporal_bora_architecture.pdf"
    with merged.open("wb") as f: writer.write(f)
    print(merged)
    for path in paths: print(path)


if __name__ == "__main__":
    main()
