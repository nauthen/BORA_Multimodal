"""Render the main.py architecture as one narrow, readable vector diagram.

Run: python docs/diagrams/render_pipeline.py
Only writes the two generated architecture assets beside this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WIDTH, HEIGHT = 1000, 3420
fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.axis("off")
fig.patch.set_facecolor("white")
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"
checks = []


def box(x, y, w, h, title, lines, fill="#edf4fb", size=16):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                              facecolor=fill, edgecolor="#718096", linewidth=1.1))
    heading = ax.text(x + w / 2, y + 27, title, ha="center", va="center",
                      fontsize=17, weight="bold", color="#142b40")
    checks.append((heading, x + 8, x + w - 8, y, y + h))
    for i, line in enumerate(lines):
        txt = ax.text(x + w / 2, y + 55 + 28 * i, line, ha="center", va="center",
                      fontsize=size, color="#142b40")
        checks.append((txt, x + 8, x + w - 8, y, y + h))


def arrow(x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=17, linewidth=1.7,
                                color="#42556b", linestyle="--" if dashed else "-"))


def note(y, text):
    ax.text(500, y, text, ha="center", va="center", fontsize=14, color="#42556b")


ax.text(500, 35, "DUAL-DECODER TEMPORAL BORA-FUSE", ha="center",
        fontsize=22, weight="bold", color="#142b40")
note(68, "main.py → config/train_config.json | B = batch size, T = 8, D = 256")
box(65, 95, 870, 100, "1  INPUT: PAIRED AUDIO + VIDEO", [
    "Audio: mono, 64 kHz, crop/pad to 2 s → [B,128000]",
    "Video: 8 uniform RGB frames, normalize → [B,8,3,224,224]",
], fill="#f4f6f8", size=16)
arrow(285, 229, 285, 245)
arrow(715, 229, 715, 245)
note(218, "Train only: single-modality corruption / dropout before model.forward()")

box(65, 245, 420, 210, "2a  AUDIO ENCODER", [
    "STFT → Log-Mel → pad → BN",
    "[B,1,128,128] → PANNs CNN6",
    "Feature hook: f_a [B,512]",
    "Classifier: t_a [B,4]",
    "SpecAugment / dropout in train",
], size=15)
box(515, 245, 420, 210, "2b  VIDEO ENCODER", [
    "Reshape to [B×8,3,224,224]",
    "Shared EfficientNet-B0 per frame",
    "Feature hook: f_v [B,8,1280]",
    "Classifier: t_v [B,8,4]",
    "Both encoders load checkpoints",
], size=15)
arrow(285, 455, 285, 495)
arrow(715, 455, 715, 495)
box(65, 495, 420, 130, "3a  AUDIO PROJECTION", [
    "f_a → Linear 512→256",
    "LayerNorm → GELU → Dropout",
    "a [B,256] — global audio",
], size=15)
box(515, 495, 420, 130, "3b  VIDEO PROJECTION", [
    "f_v → Linear 1280→256",
    "LayerNorm → GELU → Dropout",
    "v [B,8,256] — frame sequence",
], size=15)
arrow(715, 666, 715, 690)
note(655, "Named tensors a, t_a and t_v are reused at the input ports below.")

box(65, 690, 870, 220, "4  MOTION + TEMPORAL CONTEXT", [
    "INPUT: projected frames v",
    "m[0] = 0; m[t] = |v[t] − v[t−1]|",
    "c = GELU(LayerNorm(Linear(m)))                 [B,8,256]",
    "h = Transformer(v + c + learned position)      [B,8,256]",
    "2 encoder layers · 4 heads · FFN 512 · pre-norm",
    "Motion auxiliary: sigmoid(MLP(mean_time(c))) → score [B]",
], size=16)
arrow(500, 910, 500, 950)
box(65, 950, 870, 250, "5  EVENT GATE + BOUNDARY TEMPORAL ATTENTION", [
    "INPUT: temporal tokens h, motion c, global audio a",
    "e[t] = sigmoid(MLP([h[t], a, c[t]]))                    [B,8]",
    "Q = reshape(Linear(a), [B,3,256]) + learned queries",
    "α[k,t] = softmax_time(scale × cosine(Q[k],h[t]) + log e[t])",
    "V[k] = Σ_t α[k,t] h[t]                                  [B,3,256]",
    "g = mean_boundary(V)                                    [B,256]",
    "Each boundary k selects a different temporal evidence distribution.",
], size=15)
arrow(500, 1200, 500, 1240)
box(65, 1240, 870, 220, "6  REUSED TEACHER DECISIONS + AUXILIARY HEADS", [
    "INPUT: a, g, original classifier logits t_a and t_v",
    "p_a = softmax(t_a); p_v = mean_time(softmax(t_v))",
    "C_a, C_v = categorical probabilities → CORN logits [B,3]",
    "s_k = sigmoid(teacher_residual_scale[k]); initially ≈ 0.25",
    "u_a = Linear(a) + s ⊙ C_a; u_v = Linear(g) + s ⊙ C_v",
    "OUTPUT: teacher conditionals C_a/C_v and auxiliary logits u_a/u_v",
], fill="#fff4df", size=15)
arrow(500, 1460, 500, 1500)
box(65, 1500, 870, 220, "7  RELIABILITY GATES — ONE PAIR PER BOUNDARY", [
    "INPUT: a, V[k], auxiliary logits u_a/u_v",
    "r_a = sigmoid(MLP(a)); r_v[k] = sigmoid(MLP(V[k]))",
    "confidence = 2 × |sigmoid(u) − 0.5|",
    "r_eff = r × (0.25 + 0.75 × confidence)",
    "[w_a,w_v] = softmax_modality(log(r_eff) / 0.3)       [B,3,2]",
    "Epochs 0–2: override with [0.5,0.5]; then use learned gates",
], size=16)
arrow(500, 1720, 500, 1760)
box(65, 1760, 870, 190, "8  BOUNDARY INTERACTION — REPEAT FOR k = 0, 1, 2", [
    "INPUT: global audio a, boundary video V[k], gates w_a/w_v",
    "A_k = Linear_k(a); B_k = Linear_k(V[k])",
    "F_k = w_a[k] × A_k + w_v[k] × B_k",
    "I_k = MLP_k([F_k, |A_k − B_k|, A_k ⊙ B_k])        [B,256]",
    "Each MLP: Linear 768→256 → LayerNorm → GELU → Dropout",
], size=16)
arrow(285, 1990, 285, 2010)
arrow(715, 1990, 715, 2010)
note(1978, "The same interaction features I_0, I_1, I_2 feed both decoders.")
box(65, 2010, 420, 280, "9a  ORDINAL DECODER", [
    "INPUT: I_k, gates, C_a/C_v",
    "l_k = Linear_k(I_k)",
    "+ s_k × (w_a[k] C_a[k]",
    "+ w_v[k] C_v[k])",
    "q = sigmoid(l) → CORN products",
    "p_rank [B,4] → dataset order",
    "OUTPUT: log(p_ord) [B,4]",
], fill="#eaf5ee", size=15)
box(515, 2010, 420, 280, "9b  NOMINAL DECODER", [
    "INPUT: I_0, I_1, I_2, a, g",
    "I_pool = mean_boundary(I)",
    "Concat [I_pool, a, g] → [B,768]",
    "Linear 768→256 → LayerNorm",
    "GELU → Dropout",
    "Linear 256→4",
    "OUTPUT: n [B,4] raw logits",
], fill="#eaf5ee", size=15)
arrow(285, 2290, 390, 2340)
arrow(715, 2290, 610, 2340)
box(65, 2340, 870, 160, "10  FINAL COUPLING + PREDICTION", [
    "z = log(p_ord) + sigmoid(nominal_residual_scale) × n",
    "clipwise_output = log_softmax(z)                         [B,4]",
    "prediction = argmax(clipwise_output)",
    "DATASET ORDER: 0 none · 1 strong · 2 medium · 3 weak",
], fill="#eaf5ee", size=16)

arrow(500, 2548, 500, 2570, dashed=True)
note(2535, "Dashed arrow: supervision during training; no labels needed at inference.")
box(65, 2570, 870, 340, "11  TRAINING OBJECTIVE — OUTPUTS + GROUND TRUTH", [
    "Dataset label → rank: none 0, weak 1, medium 2, strong 3",
    "1.0 × CORN(l, rank)",
    "+ 0.3 × [CORN(u_a, rank) + CORN(u_v, rank)]",
    "+ 0.1 × reliability loss: raw r vs detached exp(−BCE(u))",
    "+ 1.0 × NLL(clipwise_output, label)",
    "+ 0.1 × SmoothL1(motion_score, rank/3)",
    "+ 0.5 × CE(nominal logits n, label)",
    "+ 0.3 × mean[CE(t_a, label), CE(mean_time(t_v), label)]",
    "CORN and reliability losses use the active conditional-task mask.",
    "Adam: encoder/classifiers lr 1e−5; fusion heads lr 1e−4",
], fill="#fff4df", size=15)
arrow(500, 2910, 500, 2960, dashed=True)
box(65, 2960, 870, 160, "12  RUN CONTROL FROM main.py", [
    "Verify checkpoint split identities → initialize model → train",
    "Clean validation each epoch → save highest-accuracy checkpoint",
    "Up to 120 epochs; early stopping patience = 20",
    "Reload best → test → metrics / predictions / gate summaries",
], fill="#f4f6f8", size=16)
box(65, 3160, 870, 180, "READING THE DATA FLOW", [
    "a, c, g, t_a/t_v, C_a/C_v are named connections reused downstream.",
    "Every INPUT line lists the tensors actually consumed by that block.",
    "Teacher logits remain trainable and participate in inference.",
    "Video teacher residual uses mean of probabilities; preservation uses mean logits.",
    "Rank probabilities are BEFORE nominal correction; final output is AFTER it.",
], fill="#f4f6f8", size=14)

# Validate all node text against its real rendered box, not estimated character widths.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for txt, xlo, xhi, ylo, yhi in checks:
    extent = txt.get_window_extent(renderer).transformed(ax.transData.inverted())
    if extent.x0 < xlo or extent.x1 > xhi or min(extent.y0, extent.y1) < ylo or max(extent.y0, extent.y1) > yhi:
        raise RuntimeError(f"Text does not fit: {txt.get_text()}")

out = Path(__file__).resolve().parent
fig.savefig(out / "full_model_pipeline.svg", facecolor="white")
fig.savefig(out / "full_model_pipeline.png", dpi=160, facecolor="white")
print(f"Rendered {out / 'full_model_pipeline.svg'}; {len(checks)} text bounds verified.")
