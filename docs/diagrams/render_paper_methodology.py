"""Render the code-faithful methodology figure (SVG and 300-dpi PNG).

Run from any directory: python docs/diagrams/render_paper_methodology.py
Manual diagram specification, not an automatic trace of the PyTorch graph.
Source of truth: TemporalBORAFusion.forward + utils.ordinal.bora_loss.
All dimensions omit the batch axis. See ../paper_methodology.md for equations,
numerical clamps, code references, and the exact supervised objective.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox


W, H = 2000, 1530
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "mathtext.fontset": "dejavusans"})
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
fig.patch.set_facecolor("white")

INK = "#233142"
AUDIO, VIDEO, FUSION, TEACHER = "#197184", "#5264a3", "#99711b", "#995379"
BLUE, VIOLET, GOLD, PINK = "#edf7fa", "#f0f1fa", "#fff8e8", "#fbf0f6"
GREEN, GRAY = "#eef7ef", "#f5f6f8"
text_checks = []


def label(x, y, value, size=21, color=INK, weight="normal", ha="center", bg=False):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", zorder=6,
                   bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.4} if bg else None)


def node(x, y, w, h, title, lines=(), fill=GRAY, edge=INK, size=21):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                              linewidth=1.25, edgecolor=edge, facecolor=fill, zorder=3))
    positions = [y + 24] + [y + 55 + 31 * i for i in range(len(lines))]
    for i, (value, yy) in enumerate(zip([title, *lines], positions)):
        txt = label(x + w / 2, yy, value, size=size if i else size + 1,
                    weight="bold" if i == 0 else "normal")
        text_checks.append((txt, (x + 7, y + 5, x + w - 7, y + h - 5)))


def wire(points, color=INK, arrow=True, dashed=False):
    # White under-stroke is a crossover bridge, not a tensor junction.
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle="-|>" if arrow else "-",
                            mutation_scale=18, linewidth=1.8, color=color,
                            linestyle=(0, (5, 4)) if dashed else "-", zorder=4)
    patch.set_path_effects([pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])
    ax.add_patch(patch)


def dot(x, y, color=INK):
    ax.plot(x, y, "o", markersize=4.5, color=color, zorder=5)


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                              linewidth=1, edgecolor="#aeb8c3", facecolor="white", zorder=0))
    label(x + 18, y + 27, title, size=22, weight="bold", ha="left")


# (a) Complete prediction graph. Encoder boxes include their input preparation.
label(30, 32, "(a) Overall architecture", size=25, weight="bold", ha="left")
label(1970, 32, r"$T=8$ frames  |  $D=256$  |  3 conditional boundaries", size=22, ha="right")

node(30, 230, 295, 155, "PANNs CNN6", [
    "2 s mono · 64 kHz", "Log-Mel frontend", r"$f_a\in\mathbb{R}^{512}$"], BLUE, AUDIO)
node(30, 450, 295, 170, "EfficientNet-B0", [
    r"8 RGB frames, $224^2$", "Frame-wise encoder", r"$f_v\in\mathbb{R}^{8\times1280}$"], VIOLET, VIDEO, size=20)
node(385, 265, 225, 120, "Projection", [r"$512\rightarrow256:\ a$", "LN/GELU/DO"], BLUE, AUDIO, size=20)
node(385, 485, 225, 120, "Projection", [r"$1280\rightarrow256:\ v$", "LN/GELU/DO"], VIOLET, VIDEO, size=20)
node(735, 90, 390, 150, "Classifier reuse", [
    r"$p_a=\mathrm{softmax}(t_a)$",
    r"$p_v=\mathrm{mean}_t\,\mathrm{softmax}(t_{v,t})$",
    r"$p_a,p_v\rightarrow C_a,C_v\in\mathbb{R}^{3}$"], PINK, TEACHER, size=20)
node(735, 450, 390, 170, "Temporal module (b)", [
    "Motion + Transformer", "Audio-guided pooling",
    r"$V=[V_0,V_1,V_2]\in\mathbb{R}^{3\times256}$"], VIOLET, VIDEO, size=21)
node(1220, 270, 305, 265, "Fusion (c)", [
    "Auxiliary heads", "Reliability gates",
    r"Weights $w\in\mathbb{R}^{3\times2}$",
    "3 interactions", r"$I\in\mathbb{R}^{3\times256}$;  $\ell\in\mathbb{R}^{3}$"], GOLD, FUSION, size=20)
node(1220, 605, 305, 100, "Global bypass", [r"$a,\quad g=\mathrm{mean}_k V_k$"], GRAY, size=20)
node(1650, 280, 320, 110, "Ordinal (d)", [r"$\ell\rightarrow q\rightarrow p_{\rm ord}$"], GOLD, FUSION)
node(1650, 440, 320, 110, "Nominal (d)", [r"$[\mathrm{mean}_k I_k,a,g]\rightarrow n$"], GREEN, size=20)
node(1650, 615, 320, 110, "Class prediction", [
    r"$\log p=\mathrm{logsoftmax}$", r"$\left(\log p_{\rm ord}+\beta n\right)$"], GREEN, size=20)

# Teacher logits originate in the original, trainable classifier heads.
wire([(175, 230), (175, 145), (735, 145)], TEACHER)
label(420, 128, r"$t_a$", color=TEACHER, bg=True)
wire([(175, 450), (175, 415), (690, 415), (690, 200), (735, 200)], TEACHER)
label(445, 415, r"$t_v$", color=TEACHER, bg=True)
wire([(1125, 165), (1372, 165), (1372, 270)], TEACHER)
label(1268, 143, r"$C_a,C_v$", color=TEACHER, bg=True)

wire([(325, 315), (385, 315)], AUDIO)
wire([(325, 535), (385, 535)], VIDEO)
wire([(610, 315), (1220, 315)], AUDIO)
label(1030, 295, r"Global audio $a$", color=AUDIO, bg=True)
wire([(880, 315), (880, 450)], AUDIO)
dot(880, 315, AUDIO)
wire([(650, 315), (650, 638), (1220, 638)], AUDIO)
dot(650, 315, AUDIO)
label(1070, 638, r"$a$", color=AUDIO, bg=True)
wire([(610, 535), (735, 535)], VIDEO)
label(715, 516, r"$v$", color=VIDEO, bg=True)
wire([(1125, 500), (1160, 500), (1160, 395), (1220, 395)], VIDEO)
label(1189, 376, r"$V$", color=VIDEO, bg=True)
wire([(1125, 580), (1175, 580), (1175, 674), (1220, 674)], VIDEO)
label(1196, 559, r"$V$", color=VIDEO, bg=True)
wire([(1320, 605), (1320, 535)], VIDEO)
label(1296, 571, r"$g$", color=VIDEO, bg=True)
wire([(1525, 335), (1650, 335)], FUSION)
label(1586, 315, r"$\ell$", color=FUSION, bg=True)
wire([(1525, 485), (1650, 485)], FUSION)
label(1586, 465, r"$I$", color=FUSION, bg=True)
wire([(1525, 655), (1580, 655), (1580, 526), (1650, 526)])
label(1580, 589, r"$a,g$", bg=True)
wire([(1970, 335), (1986, 335), (1986, 670), (1970, 670)], FUSION)
wire([(1808, 550), (1808, 615)])
label(1832, 582, r"$n$", bg=True)

label(30, 679, "Encoders + original classifiers are fine-tuned.", size=21, ha="left")
label(30, 714, "Feature hooks capture classifier inputs, after backbone dropout.", size=20, ha="left")
label(1810, 752, r"$\hat y=\arg\max_j\log p_j$", size=22)
label(30, 756, "Solid arrows: forward tensors. Dots: branches. Crossovers without dots: no connection.", size=20, ha="left")

# Enlarged module definitions: keep the top-level prediction graph readable.
panel(30, 800, 630, 635, "(b) Temporal evidence")
panel(685, 800, 630, 635, "(c) Boundary fusion")
panel(1340, 800, 630, 635, "(d) Dual decoding")

node(50, 850, 590, 108, "Projected feature differences", [
    r"$m_0=0;\quad m_t=|v_t-v_{t-1}|$", r"$c=\mathrm{GELU}(\mathrm{LN}(W_m m+b_m))$"], VIOLET, VIDEO)
node(50, 986, 590, 108, "Video temporal encoder", [
    r"$h=\mathrm{Transformer}(v+c+P)$", "2 pre-norm layers, 4 heads, FFN 512"], VIOLET, VIDEO)
wire([(345, 958), (345, 986)], VIDEO)
node(50, 1126, 295, 108, "Event gate", [
    r"$e_t=\sigma(\mathrm{MLP}([h_t,a,c_t]))$", r"$e\in(0,1)^8$"], VIOLET, VIDEO, size=18)
node(365, 1126, 275, 108, "Audio queries", [
    r"$Q=\mathrm{reshape}(L_q(a))$", r"$+\;Q_{\rm learned}\quad (3\times256)$"], BLUE, AUDIO, size=18)
wire([(200, 1094), (200, 1126)], VIDEO)
node(50, 1270, 590, 140, "Boundary-specific pooling", [
    r"$\alpha_{k,t}=\mathrm{softmax}_t[\gamma\cos(Q_k,h_t)+\log e_t]$",
    r"$V_k=\alpha_k h;\quad g=\mathrm{mean}_k V_k$",
    r"$\gamma=\min(\exp(\eta),100)$"], VIOLET, VIDEO, size=20)
wire([(200, 1234), (200, 1270)], VIDEO)
wire([(500, 1234), (500, 1270)], AUDIO)

node(705, 850, 590, 170, "Auxiliary logits and raw reliability", [
    r"$u_a=H_a(a)+s\odot C_a;\quad u_v=H_v(g)+s\odot C_v$",
    r"$r_a=R_a(a);\quad r_{v,k}=R_v(V_k)$",
    r"$s_k=\sigma(\theta_k),\quad k\in\{0,1,2\}$",
    r"$H$: linear heads;  $R$: sigmoid MLPs"], GOLD, FUSION, size=20)
node(705, 1050, 590, 140, "Confidence-modulated weights", [
    r"$\rho_m=r_m\odot(0.25+0.75\cdot2|\sigma(u_m)-0.5|)$",
    r"$w_k=\mathrm{softmax}_{m\in\{a,v\}}(\log\rho_{m,k}/0.3)$",
    "Epoch state 0–2: weights fixed at 0.5"], GOLD, FUSION, size=20)
wire([(1000, 1020), (1000, 1050)], FUSION)
node(705, 1220, 590, 190, "Per-boundary interactions", [
    r"$A_k=P_{a,k}(a),\quad B_k=P_{v,k}(V_k)$",
    r"$F_k=w_{a,k}A_k+w_{v,k}B_k$",
    r"$I_k=\mathrm{MLP}_k([F_k,|A_k-B_k|,A_k\odot B_k])$",
    r"$\ell_k=h_k(I_k)+s_k(w_{a,k}C_{a,k}+w_{v,k}C_{v,k})$"], GOLD, FUSION, size=20)
wire([(1000, 1190), (1000, 1220)], FUSION)

node(1360, 850, 590, 202, "Ordinal conditional factorization", [
    r"$q_k=\sigma(\ell_k),\quad S_k=q_0\cdots q_k$",
    r"$p_{\rm rank}=[1-S_0,\ S_0-S_1,\ S_1-S_2,\ S_2]$",
    "Rank order: none, weak, medium, strong",
    r"$p_{\rm ord}=p_{\rm rank}[0,3,2,1]$",
    "Dataset: none, strong, medium, weak"], GOLD, FUSION, size=20)
node(1360, 1074, 590, 107, "Nominal exact-class logits", [
    r"$n=\mathrm{MLP}_{\rm nom}([\mathrm{mean}_k I_k,a,g])\in\mathbb{R}^{4}$",
    r"$768\rightarrow256\rightarrow4$"], GREEN, size=21)
node(1360, 1205, 590, 107, "Learned logit-space coupling", [
    r"$\log p=\mathrm{logsoftmax}(\log p_{\rm ord}+\beta n)$",
    r"$\beta=\sigma(\theta_{\rm nom})$;  output: 4 log-probabilities"], GREEN, size=21)
wire([(1655, 1181), (1655, 1205)])
label(1655, 1345, "Auxiliary head (loss use only):", size=21, weight="bold")
label(1655, 1380, r"$\mathrm{mean}_t(c)\rightarrow\mathrm{MLP}+\sigma\rightarrow\hat m$;  target $r/3$", size=21)

label(30, 1474, "Batch axis omitted. [ , ] = concatenation; ⊙ = elementwise product. Stabilizing clamps are specified in the companion methodology.",
      size=20, ha="left")
label(30, 1507, "Motion is feature change, not optical flow. The event gate has no direct event labels. No frozen external teacher is used.",
      size=20, ha="left")


def validate_text():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt, (x0, y0, x1, y1) in text_checks:
        bounds = txt.get_window_extent(renderer).transformed(ax.transData.inverted())
        if (bounds.x0 < x0 or bounds.x1 > x1
                or min(bounds.y0, bounds.y1) < y0 or max(bounds.y0, bounds.y1) > y1):
            raise RuntimeError(f"Text outside its node: {txt.get_text()}")
    # Also check all standalone labels against the page, including math glyphs.
    for txt in ax.texts:
        bounds = txt.get_window_extent(renderer).transformed(ax.transData.inverted())
        if (bounds.x0 < 0 or bounds.x1 > W
                or min(bounds.y0, bounds.y1) < 0 or max(bounds.y0, bounds.y1) > H):
            raise RuntimeError(f"Text outside canvas: {txt.get_text()}")


if __name__ == "__main__":
    validate_text()
    output = Path(__file__).resolve().parent
    fig.savefig(output / "paper_methodology.svg", facecolor="white")
    fig.savefig(output / "paper_methodology.png", dpi=300, facecolor="white")
    # The overview can be used as the main paper figure, with details separately.
    overview_bounds = Bbox.from_extents(0, (H - 788) / 100, W / 100, H / 100)
    fig.savefig(output / "paper_methodology_overview.svg", bbox_inches=overview_bounds,
                pad_inches=0, facecolor="white")
    fig.savefig(output / "paper_methodology_overview.png", dpi=300,
                bbox_inches=overview_bounds, pad_inches=0, facecolor="white")
    # A compact preview is convenient for visual QA; same figure, no content changes.
    fig.savefig(output / "paper_methodology_preview.png", dpi=100, facecolor="white")
    print(f"Rendered composite + overview SVG/PNG and preview; {len(text_checks)} node labels verified.")
