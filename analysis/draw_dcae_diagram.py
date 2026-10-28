"""Draw the autoencoder architecture and two-phase pipeline to figures/dcae_architecture.png."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import patches

INK, MUTED, BLUE, ORANGE, GREEN, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#e6e5e1"
fig = plt.figure(figsize=(16, 9.6), facecolor="white"); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 96); ax.axis("off")


def box(x, y, w, h, text, fc="#f4f4f2", ec=INK, fs=8.5, bold=False, color=INK):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1, color=INK, text=None, above=True):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
    if text: ax.text((x0 + x1) / 2, y0 + (1.2 if above else -1.2), text, ha="center", va="bottom" if above else "top", fontsize=7.2, color=MUTED, linespacing=1.15)


def fmap(x, y, w, h, label, color, label_above=False):
    for i in range(4):
        ax.add_patch(patches.Rectangle((x + 0.9 * i, y + 0.6 * i), w, h, fc=color, ec=INK, lw=0.6, alpha=0.9 - 0.15 * i))
    if label_above: ax.text(x + w / 2 + 1.5, y + h + 2.6, label, ha="center", va="bottom", fontsize=7.4, color=INK)
    else: ax.text(x + w / 2 + 1.5, y - 1.6, label, ha="center", va="top", fontsize=7.4, color=INK)


ax.text(3, 92.5, "Convolutional autoencoder on the virtual-stain stacks", fontsize=13, fontweight="bold", color=INK)
ax.text(3, 89.3, "Phase 1 trains the encoder and decoder with a reconstruction loss and no labels.  Phase 2 keeps the encoder, clusters its latent vectors, and only then looks at the measured stains.", fontsize=9, color=MUTED)

# ---------------- phase 1
ax.add_patch(patches.FancyBboxPatch((2, 42.8), 156, 42.2, boxstyle="round,pad=0.4", fc="#fbfbfa", ec=GREY, lw=1.2))
ax.text(4, 82.6, "PHASE 1   representation learning (unsupervised)", fontsize=9.5, fontweight="bold", color=BLUE)
Y = 66
box(4, Y - 7, 14, 14, "label-free\nimages\n(2 brightfield,\ndarkfield)", fc="#eef3fb", fs=7.8)
arrow(18.3, Y, 23.7, Y, text="U-Net")
box(24, Y - 7, 14, 14, "virtual stains\nLDHC/AKAP4\nTOMM20\nDAPI, ACRV1", fc="#eef9f3", fs=7.8)
arrow(38.3, Y, 44.2, Y, text="align,\nhead left")
fmap(45, Y - 6, 7, 12, "input stack  7 × 48 × 48\n(4 stains + 3 inputs)", "#dfe9f7", label_above=True)
arrow(55.5, Y, 59.5, Y, color=BLUE)
fmap(60, Y - 4.5, 5.5, 9, "conv 3×3, stride 2\n32 × 24 × 24", "#c9dcf5")
arrow(68, Y, 71, Y, color=BLUE)
fmap(71.5, Y - 3.3, 4.5, 6.5, "conv, s2\n64 × 12 × 12", "#b3cdf2")
arrow(78.5, Y, 81.5, Y, color=BLUE)
fmap(82, Y - 2.3, 3.5, 4.5, "conv, s2\n128 × 6 × 6", "#9dbeef")
arrow(88, Y, 91, Y, color=BLUE)
fmap(91.5, Y - 1.5, 2.6, 3, "conv, s2\n256 × 3 × 3", "#87afec")
arrow(96.8, Y, 101.7, Y, color=BLUE, text="FC")
box(102, Y - 4.5, 7, 9, "latent\nz\n32-d", fc=BLUE, ec=BLUE, fs=8.5, bold=True, color="white")
arrow(109.3, Y, 114.2, Y, color=ORANGE, text="FC")
fmap(114.5, Y - 1.5, 2.6, 3, "256 × 3 × 3", "#f8d3c3")
arrow(119.8, Y, 122.8, Y, color=ORANGE)
fmap(123.3, Y - 2.3, 3.5, 4.5, "deconv ×2\n128 × 6 × 6", "#f6c3ac")
arrow(129.3, Y, 132.3, Y, color=ORANGE)
fmap(132.8, Y - 3.3, 4.5, 6.5, "deconv\n64 × 12 × 12", "#f4b395")
arrow(139.8, Y, 142.5, Y, color=ORANGE)
fmap(143, Y - 6, 7, 12, "reconstruction\n7 × 48 × 48", "#f2a37e", label_above=True)
ax.text(60, 53.4, "ENCODER   BatchNorm + ReLU after each convolution; FC = fully connected layer", fontsize=8, color=BLUE, fontweight="bold")
ax.text(114, 53.4, "DECODER   mirror of the encoder; discarded after training", fontsize=8, color=ORANGE, fontweight="bold")
# loss bracket, below the labels
for x in (49.5, 147.5): ax.plot([x, x], [Y - 9.5, 50.6], color=INK, lw=1)
ax.plot([49.5, 147.5], [50.6, 50.6], color=INK, lw=1)
box(84, 46.2, 29, 4.4, "loss = mean squared error (input − reconstruction)", fc="#fff4e8", ec=ORANGE, fs=8)
ax.text(80, 44.0, "augmentation each batch: ±15° rotation, ±3 px shift, vertical flip (head stays left), ×0.85–1.15 intensity   ·   Adam with cosine decay, 60 epochs on 4,000 stacks", fontsize=7.2, color=MUTED, ha="center")

# ---------------- phase 2
ax.add_patch(patches.FancyBboxPatch((2, 4), 156, 36, boxstyle="round,pad=0.4", fc="#fbfbfa", ec=GREY, lw=1.2))
ax.text(4, 38.2, "PHASE 2   clustering, then reading the clusters against measurements they never saw", fontsize=9.5, fontweight="bold", color=GREEN)
Y2 = 25
fmap(6, Y2 - 6, 7, 12, "every object's\naligned stack", "#dfe9f7")
arrow(16.5, Y2, 21.5, Y2, color=BLUE, text="frozen\nencoder")
box(22, Y2 - 4.5, 7, 9, "z\n32-d", fc=BLUE, ec=BLUE, fs=8.5, bold=True, color="white")
arrow(29.3, Y2, 34.7, Y2, text="scale")
box(35, Y2 - 5, 17, 10, "Gaussian mixture\n(k-means initialised)\nk = 3 on the ImageStream\nk = 2 on the A8", fc="#eef9f3", ec=GREEN, fs=7.8)
arrow(52.3, Y2, 57.2, Y2)
box(57.5, Y2 - 5, 15, 10, "populations\n(cluster label\nper object)", fc="#eef9f3", ec=GREEN, fs=8)
arrow(72.8, Y2 + 2.5, 77.2, Y2 + 6); arrow(72.8, Y2, 77.2, Y2); arrow(72.8, Y2 - 2.5, 77.2, Y2 - 6)
box(77.5, Y2 + 3.8, 25, 6.4, "stability: refit on acquisition halves\nand bootstraps, adjusted Rand index", fc="#f4f4f2", fs=7.6)
box(77.5, Y2 - 3.2, 25, 6.4, "galleries: nearest-to-centre and\nperiphery, brightfield + virtual stains", fc="#f4f4f2", fs=7.6)
box(77.5, Y2 - 10.2, 25, 6.4, "post-hoc truth: measured stains (ISX),\nwells S / P / SP (A8) → ARI, contingency", fc="#fff4e8", ec=ORANGE, fs=7.6)
box(107, Y2 - 10.5, 49, 21, "", fc="#f8f8f6", ec=GREY)
ax.text(131.5, Y2 + 8.2, "what came out (2026-10-28)", ha="center", fontsize=8.5, fontweight="bold", color=INK)
ax.text(109, Y2 + 4.8, "ImageStream, 3,611 gated objects, k = 3:", fontsize=7.8, color=INK)
ax.text(111, Y2 + 2.2, "clusters = sperm · weakly stained objects · bright round bodies", fontsize=7.4, color=MUTED)
ax.text(111, Y2 - 0.4, "agreement with the measured side  ARI 0.68   (features 0.78, raw pixels 0.42)", fontsize=7.4, color=MUTED)
ax.text(109, Y2 - 4, "A8, 600 objects, k = 2:", fontsize=7.8, color=INK)
ax.text(111, Y2 - 6.6, "sperm 200 / 169 / 1  vs  PBMC 0 / 31 / 199  (S / SP / P wells): exact", fontsize=7.4, color=MUTED)
ax.text(4, 9.4, "Inputs are the virtual stains the U-Net paints from the label-free images, plus those images, aligned on the cell axis at 0.667 µm/px with the head to the left before the stack is formed.", fontsize=7.4, color=MUTED)
ax.text(4, 7.0, "Representation learning and clustering are separate steps and the decoder is only a training device. Latent vectors are scaled by median and IQR and clipped at ±5 before the mixture is fitted.", fontsize=7.4, color=MUTED)
fig.savefig("figures/dcae_architecture.png", dpi=180, facecolor="white"); print("wrote figures/dcae_architecture.png")
