"""The encoder proposed for the Kruger (ideal / not ideal) architecture: native-resolution
aligned stacks, a five-stage convolutional encoder with global pooling, trained with a
contrastive objective, then four regional heads. figures/kruger_encoder.png."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import patches

INK, MUTED, BLUE, ORANGE, GREEN, GREY, RED = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#e6e5e1", "#c9357a"
fig = plt.figure(figsize=(16, 10.4), facecolor="white"); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 104); ax.axis("off")


def box(x, y, w, h, text, fc="#f4f4f2", ec=INK, fs=8.5, bold=False, color=INK):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(x0, y0, x1, y1, color=INK, text=None, dy=1.2, fs=7.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
    if text: ax.text((x0 + x1) / 2, y0 + dy, text, ha="center", va="bottom", fontsize=fs, color=MUTED, linespacing=1.15)


def fmap(x, y, w, h, label, color, above=False):
    for i in range(4):
        ax.add_patch(patches.Rectangle((x + 0.9 * i, y + 0.6 * i), w, h, fc=color, ec=INK, lw=0.6, alpha=0.9 - 0.15 * i))
    ax.text(x + w / 2 + 1.5, (y + h + 2.6) if above else (y - 1.6), label, ha="center", va="bottom" if above else "top", fontsize=7.2, color=INK)


ax.text(3, 100.5, "Encoder for the Kruger architecture: native resolution, global pooling, contrastive objective", fontsize=13, fontweight="bold", color=INK)
ax.text(3, 97.3, "Changes from the current encoder are marked.  The U-Net front end and the virtual stains are unchanged; the sperm gate has already selected single, in-focus sperm.", fontsize=9, color=MUTED)

# ---------------- phase 1: the encoder
ax.add_patch(patches.FancyBboxPatch((2, 50), 156, 44, boxstyle="round,pad=0.4", fc="#fbfbfa", ec=GREY, lw=1.2))
ax.text(4, 91.4, "PHASE 1   representation learning (contrastive, no labels)", fontsize=9.5, fontweight="bold", color=BLUE)
Y = 72
box(4, Y - 7, 14, 14, "virtual stains\n+ inputs\n(from the U-Net,\nnative size)", fc="#eef9f3", fs=7.6)
arrow(18.3, Y, 23.7, Y, text="align,\nhead left")
fmap(24.5, Y - 7, 8, 14, "input stack  7 × 96 × 96\nat 0.333 µm/px (native)\nhead ≈ 12 px wide", "#dfe9f7", above=True)
ax.text(29, Y - 13.2, "CHANGE 1: 96 px at native pitch,\nwas 48 px at 0.667 µm", fontsize=6.8, color=RED, ha="center", va="top", fontweight="bold")
arrow(36.5, Y, 40.5, Y, color=BLUE)
fmap(41, Y - 6, 6.5, 12, "conv 3×3 s2\n32 × 48 × 48", "#c9dcf5")
arrow(50, Y, 53, Y, color=BLUE)
fmap(53.5, Y - 4.8, 5.5, 9.5, "conv s2\n64 × 24 × 24", "#b3cdf2")
arrow(61.5, Y, 64.5, Y, color=BLUE)
fmap(65, Y - 3.6, 4.5, 7, "conv s2\n128 × 12 × 12", "#9dbeef")
arrow(72, Y, 75, Y, color=BLUE)
fmap(75.5, Y - 2.6, 3.5, 5, "conv s2\n256 × 6 × 6", "#87afec")
arrow(81.5, Y, 84.5, Y, color=BLUE)
fmap(85, Y - 1.8, 2.8, 3.4, "conv s2\n256 × 3 × 3", "#71a0e9")
ax.text(70, Y - 13.2, "CHANGE 2: five stride-2 stages (was four) to bring 96 px to a 3 × 3 map", fontsize=6.8, color=RED, ha="center", va="top", fontweight="bold")
arrow(90.3, Y, 96.5, Y, color=BLUE, text="global\navg+max\npool", dy=3.0, fs=6.6)
box(97, Y - 4, 10, 8, "512\npooled", fc="#dfe9f7", fs=8)
ax.text(102, Y - 13.2, "CHANGE 3: pooling, not flatten + dense:\nposition-tolerant, 10× fewer parameters", fontsize=6.8, color=RED, ha="center", va="top", fontweight="bold")
arrow(107.3, Y, 113.5, Y, color=BLUE, text="linear", dy=1.0)
box(114, Y - 4.5, 8, 9, "z\n32-d", fc=BLUE, ec=BLUE, fs=9, bold=True, color="white")
arrow(122.3, Y, 128.5, Y, color=ORANGE, text="projection\nMLP", dy=1.0)
box(129, Y - 4, 10, 8, "h\n64-d\n(training only)", fc="#fff4e8", ec=ORANGE, fs=7)
box(141, Y - 8, 16, 16, "contrastive loss\n\ntwo augmented views\nof the same cell pulled\ntogether, all other\ncells in the batch\npushed apart\n(NT-Xent, temperature 0.1)", fc="#fff4e8", ec=ORANGE, fs=6.6)
arrow(139.3, Y, 140.7, Y, color=ORANGE)
ax.text(149, Y - 13.2, "CHANGE 4: contrastive objective,\nwas pixel reconstruction;\nno decoder", fontsize=6.8, color=RED, ha="center", va="top", fontweight="bold")
ax.text(80, 54.6, "augmentations that define 'the same cell': rotation (any angle, then re-aligned head-left), ±4 px shift, flips, intensity ×0.8–1.2 per channel, Gaussian blur σ ≤ 1 px, random crop 88–96 px", fontsize=6.8, color=MUTED, ha="center")
ax.text(80, 52.2, "BatchNorm + ReLU after every convolution   ·   Adam, cosine decay, batch 256, ~200 epochs on the training half (10,000 objects), never the held-out half", fontsize=6.8, color=MUTED, ha="center")

# ---------------- phase 2: the heads
ax.add_patch(patches.FancyBboxPatch((2, 3), 156, 44, boxstyle="round,pad=0.4", fc="#fbfbfa", ec=GREY, lw=1.2))
ax.text(4, 44.4, "PHASE 2   frozen encoder, four regional heads (supervised, small), WHO combination", fontsize=9.5, fontweight="bold", color=GREEN)
Y2 = 24
fmap(5, Y2 - 7, 8, 14, "every sperm's\naligned stack", "#dfe9f7")
arrow(17.5, Y2, 22.5, Y2, color=BLUE, text="frozen\nencoder")
box(23, Y2 - 4.5, 8, 9, "z\n32-d", fc=BLUE, ec=BLUE, fs=9, bold=True, color="white")
box(23, Y2 - 15, 8, 8, "features\n(µm)", fc="#dfe9f7", fs=7.6)
ax.text(27, Y2 - 16.4, "from the feature engine", fontsize=6.2, color=MUTED, ha="center", va="top")
for yy in (Y2, Y2 - 11): arrow(31.3, yy, 37.7, Y2 - 2, color=INK)
box(38, Y2 - 7, 12, 14, "concatenate\n+ standardise\n(32 + ≈400)", fc="#f4f4f2", fs=7.4)
heads = [("HEAD", "length, width, L/W, acrosome fraction,\novality, symmetry; vacuoles if resolvable"),
         ("MIDPIECE", "length vs head, calibre proxy,\ninsertion angle and offset, sharp bend"),
         ("TAIL", "length, uniform calibre, sharp bends,\ncoiled (> 360°), short, multiple"),
         ("RESIDUE", "cytoplasm outside every stain\n> ⅓ of head area")]
ys = [Y2 + 10.5, Y2 + 3.5, Y2 - 3.5, Y2 - 10.5]
for (name, what), yy in zip(heads, ys):
    arrow(50.3, Y2, 56.7, yy + 2.4)
    box(57, yy, 36, 5.4, "%s head:  p(defect)\n%s" % (name, what), fc="#fff4e8", ec=ORANGE, fs=6.4)
ax.text(75, Y2 - 14.2, "each head = logistic regression or a 2-layer MLP (32 → 1), trained on its own labels; thresholds are learned, not WHO's numbers", fontsize=6.6, color=MUTED, ha="center", va="top")
for yy in ys: arrow(93.3, yy + 2.7, 99.7, Y2 + (yy + 2.7 - Y2) * 0.45)
box(100, Y2 - 7, 20, 14, "WHO combination\nresidue → tail →\nmidpiece → head;\nideal = no defect;\nborderline → defect", fc="#eef9f3", ec=GREEN, fs=7.2)
arrow(120.3, Y2, 126.5, Y2)
box(127, Y2 - 9, 14, 18, "per sperm\n1 + 4 record\n\nper sample\n% ideal, %H %NM %T %C\nTZI, MAI, SDI", fc="#eef9f3", ec=GREEN, fs=7)
box(143, Y2 - 9, 14, 18, "labels per head\n\n1. measured stains\n(geometry, no judgement)\n2. technician tags\nin WHO order\n3. reviewed clusters", fc="#f4f4f2", fs=6.6)
ax.text(80, 5.6, "What is kept from the current encoder: the convolution-BatchNorm-ReLU block, stride-2 downsampling to a 3 × 3 map, a 32-d latent, Gaussian-mixture clustering of z for the unsupervised view and the galleries.   "
                 "Parameter count ≈ 1.2 M in the convolutions, ≈ 20 k after pooling (was ≈ 1.2 M in the two dense layers alone).", fontsize=6.8, color=MUTED, ha="center")
fig.savefig("figures/kruger_encoder.png", dpi=180, facecolor="white"); plt.close(fig); print("wrote figures/kruger_encoder.png")
