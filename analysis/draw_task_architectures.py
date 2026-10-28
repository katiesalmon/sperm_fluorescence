"""Two proposed end-to-end architectures, drawn: (1) sperm versus not sperm, for sorting on
the A8 / S8; (2) ideal versus not ideal sperm under Kruger strict criteria, for whole-ejaculate
morphology on the ImageStream. figures/arch_sperm_vs_not.png, figures/arch_kruger.png."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import patches

INK, MUTED, BLUE, ORANGE, GREEN, GREY, RED = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#e6e5e1", "#c9357a"


def canvas(w=160, h=100, figsize=(16, 10)):
    fig = plt.figure(figsize=figsize, facecolor="white"); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="#f4f4f2", ec=INK, fs=8.5, bold=False, color=INK, lw=1.0):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK, text=None, dy=1.2, fs=7.2, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=1.3))
    if text: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=MUTED, linespacing=1.15)


def band(ax, x, y, w, h, title, color):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc="#fbfbfa", ec=GREY, lw=1.2))
    ax.text(x + 2, y + h - 2.6, title, fontsize=9.5, fontweight="bold", color=color)


# ------------------------------------------------------------------ 1. sperm vs not
fig, ax = canvas()
ax.text(3, 96, "Proposed architecture 1:  sperm versus not sperm, for sorting (BD FACSDiscover A8 / S8)", fontsize=13, fontweight="bold", color=INK)
ax.text(3, 92.8, "Goal: a per-event decision fast enough to drive a sort gate, trained from stains but running without them. Three stages; the first two exist.", fontsize=9, color=MUTED)

band(ax, 2, 56, 156, 33, "STAGE A   virtual staining (exists)", BLUE)
Y = 72
box(ax, 5, Y - 7, 16, 14, "CellView images\nlight loss\nFSC\nSSC", fc="#eef3fb", fs=8)
arrow(ax, 21.3, Y, 27.7, Y, text="per event,\nnative size")
box(ax, 28, Y - 7, 22, 14, "stems U-Net\none encoder stem per image,\nshared trunk\n(trained on real stains,\nreplicate 2 → 3)", fc="#dfe9f7", fs=7.6)
arrow(ax, 50.3, Y, 56.7, Y)
box(ax, 57, Y - 7, 18, 14, "virtual stains\nLDHC/AKAP4\nCD45\nACRV1", fc="#eef9f3", fs=8)
ax.text(66, 57.8, "event r within population: LDHC 0.75 (sperm), CD45 0.61 (PBMC), ACRV1 0.37 (sperm)", fontsize=7, color=MUTED, ha="center")
arrow(ax, 75.3, Y, 81.7, Y, text="align on the\ncell axis")
box(ax, 82, Y - 7, 18, 14, "aligned stack\n6 × 48 × 48\n(3 stains + 3 inputs)", fc="#eef3fb", fs=8)
arrow(ax, 100.3, Y, 106.7, Y)
box(ax, 107, Y - 7, 22, 14, "convolutional encoder\n(autoencoder-trained,\nreconstruction loss,\ndecoder discarded)", fc="#dfe9f7", fs=7.6)
arrow(ax, 129.3, Y, 135.7, Y)
box(ax, 136, Y - 5, 9, 10, "z\n16-d", fc=BLUE, ec=BLUE, fs=9, bold=True, color="white")
ax.text(80, 60.6, "optional in parallel: the feature engine (≈360 measurements on the predicted compartments) → PCA → 20 components", fontsize=7.4, color=MUTED, ha="center")

band(ax, 2, 20, 156, 33, "STAGE B   population decision (exists unsupervised; proposed supervised)", GREEN)
Y2 = 36
box(ax, 5, Y2 - 5, 9, 10, "z", fc=BLUE, ec=BLUE, fs=9, bold=True, color="white")
arrow(ax, 14.3, Y2 + 2.5, 22.7, Y2 + 7.5); arrow(ax, 14.3, Y2 - 2.5, 22.7, Y2 - 7.5)
box(ax, 23, Y2 + 3, 30, 10, "unsupervised (exists)\nGaussian mixture, k = 2\n→ sperm 370 / PBMC 230, exact on the wells", fc="#eef9f3", ec=GREEN, fs=7.4)
box(ax, 23, Y2 - 13, 30, 10, "supervised (proposed)\nsmall classifier head on z (and/or features):\nlogistic or 2-layer MLP, output p(sperm)", fc="#fff4e8", ec=ORANGE, fs=7.4)
arrow(ax, 53.3, Y2 + 8, 61.7, Y2 + 8); arrow(ax, 53.3, Y2 - 8, 61.7, Y2 - 8)
box(ax, 62, Y2 - 13, 26, 26, "labels, in order of trust\n\n1. measured CD45 and well identity\n   (S / P / SP), free on stained runs\n2. unsupervised cluster membership\n3. technician tags on a few hundred\n   light-loss images", fc="#f4f4f2", fs=7.2)
arrow(ax, 88.3, Y2, 96.7, Y2, text="held-out\nreplicate", dy=1.0)
box(ax, 97, Y2 - 9, 26, 18, "validation\n\nsperm-well vs PBMC-well AUC\n(predicted CD45 alone: 0.95;\nfull z: exact at k = 2)\n+ mixed-well leukocyte recovery\n+ a second patient", fc="#fff4e8", ec=ORANGE, fs=7.2)
arrow(ax, 123.3, Y2, 131.7, Y2)
box(ax, 132, Y2 - 9, 24, 18, "decision\n\np(sperm) > threshold\nset for the sort's purpose:\nhigh purity for ICSI\n(no leukocytes), with the\nunsure band rejected", fc="#eef9f3", ec=GREEN, fs=7.2)

band(ax, 2, 3, 156, 14, "STAGE C   deployment on the sorter (proposed)", RED)
ax.text(5, 9.2, "S8 shares the A8's CellView imaging, so stages A and B run unchanged on its images. Latency budget: U-Net + encoder + head per event; at 64 × 64 px and width 64 the U-Net is the cost, and a distilled\n"
              "classifier trained to imitate p(sperm) directly from the three input images (bypassing the stains at run time, supervised by them at training time) is the fallback if the full chain is too slow for the sort rate.", fontsize=7.6, color=INK, va="center")
fig.savefig("figures/arch_sperm_vs_not.png", dpi=180, facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ 2. Kruger ideal vs not
fig, ax = canvas()
ax.text(3, 96, "Proposed architecture 2:  ideal versus not ideal sperm under Kruger strict criteria (ImageStream, whole ejaculate)", fontsize=13, fontweight="bold", color=INK)
ax.text(3, 92.8, "Goal: a per-sperm 1 + 4 record (ideal; head / midpiece / tail / residual-cytoplasm defects) and the sample's % ideal and TZI, from label-free images. Kruger / WHO strict: borderline counts as abnormal.", fontsize=9, color=MUTED)

band(ax, 2, 61, 156, 28, "STAGE A   virtual staining and the sperm gate (exists)", BLUE)
Y = 74
box(ax, 5, Y - 6, 15, 12, "ImageStream\nbrightfield ×2\ndarkfield (60×)", fc="#eef3fb", fs=7.8)
arrow(ax, 20.3, Y, 25.7, Y)
box(ax, 26, Y - 6, 17, 12, "stems U-Net\n(trained on the\nstained tube)", fc="#dfe9f7", fs=7.8)
arrow(ax, 43.3, Y, 48.7, Y)
box(ax, 49, Y - 6, 20, 12, "virtual stains\nDAPI = nucleus\nACRV1 = acrosome\nTOMM20 = midpiece\nLDHC/AKAP4 = principal piece", fc="#eef9f3", fs=7.2)
arrow(ax, 69.3, Y, 74.7, Y)
box(ax, 75, Y - 6, 18, 12, "sperm gate\nstained vs unstained,\nthen not a large body\n(k = 2, then k = 3)", fc="#eef9f3", ec=GREEN, fs=7.4)
arrow(ax, 93.3, Y, 98.7, Y, text="single sperm,\nin frame, in focus")
box(ax, 99, Y - 6, 26, 12, "aligned stack, higher resolution\n7 × 96 × 96 at 0.333 µm/px\n(the head is 12 px wide here;\n48 px / 0.667 µm cannot see head shape)", fc="#eef3fb", fs=7.2)
arrow(ax, 125.3, Y, 130.7, Y)
box(ax, 131, Y - 6, 24, 12, "two parallel descriptions\nfeature engine → WHO-named\nmeasurements (µm)\nencoder → z (contrastive, 32-d)", fc="#dfe9f7", fs=7.2)

band(ax, 2, 22, 156, 37, "STAGE B   the four regional heads and the mapping (proposed)", GREEN)
Y2 = 40
box(ax, 5, Y2 - 8, 20, 16, "per sperm:\nfeatures + z", fc="#dfe9f7", fs=8)
ys = [Y2 + 8.4, Y2 + 2.2, Y2 - 4.0, Y2 - 10.2]
heads = [("HEAD defect", "length 3.7–4.7 µm, width 2.5–3.2, L/W 1.3–1.8,\nacrosome 40–70 % of head, oval, no large vacuole"),
         ("MIDPIECE defect", "≈ head length, slender, axial insertion,\nnot sharply bent (TOMM20 mask vs head axis)"),
         ("TAIL defect", "≈ 45 µm, uniform, no sharp bend, not coiled,\nnot short, single (LDHC skeleton)"),
         ("RESIDUE defect", "cytoplasm > ⅓ head area\n(brightfield object outside every stain)")]
for (name, crit), y in zip(heads, ys):
    arrow(ax, 25.3, Y2, 31.7, y + 2.6, style="-|>")
    box(ax, 32, y, 40, 5.6, "%s\n%s" % (name, crit), fc="#fff4e8", ec=ORANGE, fs=6.6)
ax.text(52, Y2 - 12.6, "each head: a small classifier (logistic / tree on the named features, or a linear probe on z), trained per region; its learned threshold replaces the hand rule", fontsize=6.8, color=MUTED, ha="center")
for y in ys: arrow(ax, 72.3, y + 2.8, 79.7, Y2 + (y + 2.8 - Y2) * 0.5)
box(ax, 80, Y2 - 7, 22, 14, "WHO decision order\nresidue → tail → midpiece\n→ head; ideal = no defect;\nborderline → defect", fc="#eef9f3", ec=GREEN, fs=7.2)
arrow(ax, 102.3, Y2, 109.7, Y2)
box(ax, 110, Y2 - 9, 20, 18, "per sperm\n1 + 4 record\n\nper sample\n% ideal, %H %NM %T %C,\nTZI (1.00–4.00), MAI, SDI,\nfree heads / tails apart", fc="#eef9f3", ec=GREEN, fs=7)
arrow(ax, 130.3, Y2, 137.7, Y2)
box(ax, 138, Y2 - 9, 18, 18, "report\n\n+ gate losses\n+ label provenance\n(stain / technician /\ncluster) per flag\n+ sampling error\n(n ≥ 200; 1,500 for\n3 % vs 5 %)", fc="#f4f4f2", fs=6.8)

band(ax, 2, 2.5, 156, 18, "LABELS AND VALIDATION   (proposed; nothing here exists yet)", RED)
ax.text(5, 16.2, "Labels, per region, in order of trust:  (1) measured stains on the stained tube give head, midpiece and tail geometry free of judgement, calibrated against IDEAS Length / Width (r 0.92–0.99);", fontsize=7.4, color=INK)
ax.text(5, 13.4, "(2) a technician tags a few hundred ImageStream images in the WHO order (residue, tail, midpiece, head) so each tag is one regional flag;  (3) named clusters, gallery-reviewed, with the reviewer recorded.", fontsize=7.4, color=INK)
ax.text(5, 10.2, "Validation:  per-feature Bland–Altman against the measured stains on the same cells  →  per-flag precision against the technician tags  →  % ideal and TZI against the andrology lab's manual result on", fontsize=7.4, color=INK)
ax.text(5, 7.4, "ten normospermic and ten teratozoospermic patients, judged against the known spread between technicians  →  domain shift: the same sample run unstained.", fontsize=7.4, color=INK)
ax.text(5, 4.4, "Known limits: midpiece width (0.6 µm) is under 2 px even at 0.333 µm/px and is proxied by TOMM20 area per length; head vacuoles are probably out of reach; WHO is defined on smears, so the target is agreement, not identity.", fontsize=7.2, color=MUTED)
fig.savefig("figures/arch_kruger.png", dpi=180, facecolor="white"); plt.close(fig)
print("wrote figures/arch_sperm_vs_not.png figures/arch_kruger.png")
