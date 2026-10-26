"""Draw the architecture diagrams in docs/diagrams/ (PNG + SVG). Matplotlib only.

    .venv/bin/python docs/diagrams/make_diagrams.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
INK = "#222222"; GREY = "#666666"
C_INPUT = "#dbe9f6"; C_STEM = "#fde9c9"; C_TRUNK = "#e2f0d9"; C_HEAD = "#f4d7e7"; C_POST = "#ececec"; C_DATA = "#fff2cc"; C_CFG = "#e6e0f8"
MARKER = {"LDHC/AKAP4": "#59c959", "ACRV1": "#d4c94a", "CD45": "#ff66d9", "DAPI": "#a659f2", "TOMM20": "#ff4d4d"}


def box(ax, x, y, w, h, text, fc, fs=10, bold=False, ec=INK, lw=1.0, style="round,pad=0.02,rounding_size=0.12", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color, linespacing=1.3)


def arrow(ax, p, q, text=None, lw=1.4, color=INK, fs=8.5, dashed=False, offset=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=lw, color=color, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text, ha="center", va="bottom", fontsize=fs, color=GREY)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ------------------------------------------------------------------ 1. the model ----
def model_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.8), facecolor="white"); ax.set_xlim(0, 13); ax.set_ylim(-0.2, 6.8); ax.axis("off")
    ax.text(0.2, 6.6, "One marker model: modular label-free input, fixed stain output", fontsize=14, fontweight="bold", va="top")
    ax.text(0.2, 6.2, "sperm_fluorescence --arch stems    (the trunk is the same 4-level U-Net every earlier run used)", fontsize=9.5, color=GREY, va="top")
    roles = [("light loss", "A8 / S8"), ("FSC", "A8 / S8"), ("SSC", "A8 / S8"), ("brightfield", "ImageStream / CytPix"), ("brightfield 2", "ImageStream"), ("darkfield", "ImageStream")]
    ys = [4.6, 3.9, 3.2, 2.5, 1.8, 1.1]
    ax.text(1.05, 5.05, "input roles\n(any subset)", ha="center", fontsize=9, color=GREY)
    ax.text(3.55, 5.05, "one stem per role\n(1 → w features)", ha="center", fontsize=9, color=GREY)
    for (name, inst), y in zip(roles, ys):
        present = y >= 3.0
        box(ax, 0.3, y - 0.25, 1.5, 0.5, "%s\n%s" % (name, inst), C_INPUT if present else "#f7f7f7", fs=8, ec=INK if present else "#aaaaaa", color=INK if present else "#999999")
        box(ax, 2.7, y - 0.22, 1.7, 0.44, "stem  +  role bias", C_STEM if present else "#f7f7f7", fs=8.5, ec=INK if present else "#aaaaaa", color=INK if present else "#999999")
        arrow(ax, (1.8, y), (2.7, y), lw=1.2, color=INK if present else "#bbbbbb")
        arrow(ax, (4.4, y), (5.3, 2.85), lw=1.2, color=INK if present else "#bbbbbb", dashed=not present)
    ax.text(0.3, 0.45, "greyed roles: absent on this instrument (or dropped in training) — they contribute nothing", fontsize=8, color=GREY)
    box(ax, 5.3, 2.45, 1.0, 0.8, "Σ / n\nmean over\npresent roles", "#ffffff", fs=8)
    arrow(ax, (6.3, 2.85), (7.0, 2.85))
    # trunk
    box(ax, 7.0, 1.3, 2.6, 3.1, "", C_TRUNK, style="round,pad=0.02,rounding_size=0.15")
    ax.text(8.3, 4.15, "U-Net trunk", ha="center", fontsize=11, fontweight="bold")
    ax.text(8.3, 3.85, "4 levels, w → 8w, reflect padding\nno global pooling: runs at any size", ha="center", fontsize=8, color=GREY, va="top")
    for i, (yy, ww) in enumerate(zip([3.05, 2.65, 2.25, 1.85], [2.2, 1.7, 1.2, 0.7])):
        box(ax, 8.3 - ww / 2, yy - 0.14, ww, 0.28, "", "#c9e3b8", ec="#7aa96a", lw=0.8, style="round,pad=0.01,rounding_size=0.05")
    ax.text(8.3, 1.48, "skip connections at every level", ha="center", fontsize=7.5, color=GREY)
    arrow(ax, (9.6, 2.85), (10.1, 2.85))
    box(ax, 10.1, 2.5, 1.1, 0.7, "head\n1×1 conv → 1", C_HEAD, fs=8.5)
    arrow(ax, (11.2, 2.85), (11.6, 2.85))
    box(ax, 11.6, 2.5, 1.2, 0.7, "stain image\n(fixed output)", "#ffffff", fs=8.5, bold=True)
    box(ax, 11.6, 1.5, 1.2, 0.5, "per-event\nintensity", C_POST, fs=8)
    box(ax, 11.6, 0.75, 1.2, 0.5, "mask\n(calibrated)", C_POST, fs=8)
    arrow(ax, (12.2, 2.5), (12.2, 2.0), lw=1.0); arrow(ax, (12.2, 1.5), (12.2, 1.25), lw=1.0)
    ax.text(12.2, 0.6, "post-processing,\noutside the network", ha="center", va="top", fontsize=7.5, color=GREY)
    ax.text(0.3, 0.1, "training: masked L1 on the scaled target; role dropout p per present role (never all); per-marker checkpoint selected on within-population event r",
            ha="left", fontsize=8, color=GREY)
    save(fig, "model")


# -------------------------------------------------------------- 2. the pipeline ----
def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.4), facecolor="white"); ax.set_xlim(0, 13); ax.set_ylim(-0.4, 6.4); ax.axis("off")
    ax.text(0.2, 6.2, "From instrument files to a deployable marker model", fontsize=14, fontweight="bold", va="top")
    # sources
    srcs = [("A8 / S8 bundle\n6-page TIFFs + FCS", 5.0), ("ImageStream .cif + .daf\n12 channels + IDEAS masks", 4.0), ("CytPix sample\nbrightfield only", 3.0)]
    for t, y in srcs: box(ax, 0.2, y - 0.35, 2.2, 0.7, t, C_DATA, fs=8.5)
    box(ax, 0.2, 1.6, 2.2, 0.9, "configs/instruments/*.toml\nroles → pages\nstains → pages", C_CFG, fs=8)
    box(ax, 3.1, 3.5, 1.6, 1.0, "sperm_fluorescence\nprepare", "#ffffff", fs=10, bold=True)
    for _, y in srcs: arrow(ax, (2.4, y), (3.1, 4.0), lw=1.1)
    arrow(ax, (2.4, 2.05), (3.4, 3.5), lw=1.1, dashed=True)
    box(ax, 5.2, 3.1, 2.7, 1.8, "", C_DATA)
    ax.text(6.55, 4.65, "ragged cache", ha="center", fontsize=10, fontweight="bold")
    ax.text(6.55, 4.35, "objects at native size\nx: named roles,  y: named markers\npresent / avail tables\nmeta: roles, markers, scales", ha="center", va="top", fontsize=7.5, color=GREY)
    arrow(ax, (4.7, 4.0), (5.2, 4.0))
    box(ax, 9.0, 3.5, 1.5, 1.0, "sperm_fluorescence\ntrain", "#ffffff", fs=10, bold=True)
    arrow(ax, (7.9, 4.0), (9.0, 4.0))
    ax.text(8.45, 4.12, "--roles  --target\n--arch stems", ha="center", va="bottom", fontsize=7.5, color=GREY)
    box(ax, 8.75, 1.6, 2.0, 0.9, "configs/markers/*.toml\nname, colour,\nselection population", C_CFG, fs=8)
    arrow(ax, (9.75, 2.5), (9.75, 3.5), lw=1.1, dashed=True)
    box(ax, 11.0, 3.05, 1.9, 1.9, "", "#ffffff")
    ax.text(11.95, 4.7, "marker model", ha="center", fontsize=10, fontweight="bold")
    ax.text(11.95, 4.4, "best.pt + results.json\narch, roles trained,\nscale, mask threshold", ha="center", va="top", fontsize=7.5, color=GREY)
    arrow(ax, (10.5, 4.0), (11.0, 4.0))
    # evaluation row
    evals = ["sheet", "masks", "accuracy", "slides", "ablate", "subset"]
    for i, e in enumerate(evals):
        box(ax, 3.1 + i * 1.6, 0.3, 1.4, 0.5, e, C_POST, fs=8.5)
    ax.text(3.1, 0.95, "sperm_fluorescence <eval command>  — each loads a run by its results.json and scores it on the held-out replicate", fontsize=8, color=GREY, va="bottom")
    arrow(ax, (11.95, 3.05), (11.95, 0.8), lw=1.1)
    ax.text(0.2, -0.2, "server (stdlib scripts/): survey, bundle, FCS, spectral export    →    laptop / GPU box: everything above", fontsize=8, color=GREY)
    save(fig, "pipeline")


# -------------------------------------------- 3. models per marker and family ----
def matrix_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="white"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.text(0.2, 5.0, "One model per marker and input family; new markers never touch old models", fontsize=14, fontweight="bold", va="top")
    fams = [("CellView family\nA8 (train)  →  S8 (deploy)\nroles: light loss, FSC, SSC", 1.4), ("camera family\nImageStream (train)  →  CytPix (deploy)\nroles: brightfield, brightfield 2, darkfield", 6.0)]
    markers_f = {0: ["LDHC/AKAP4", "ACRV1", "CD45"], 1: ["LDHC/AKAP4", "ACRV1", "DAPI", "TOMM20"]}
    for f, (title, x0) in enumerate(fams):
        box(ax, x0, 3.6, 4.0, 0.9, title, C_INPUT, fs=8.5)
        for j, m in enumerate(markers_f[f]):
            x = x0 + 0.1 + j * 0.98
            box(ax, x, 2.3, 0.9, 0.85, "model\n%s" % m, "#ffffff", fs=7.5, ec=MARKER[m], lw=2.0)
            arrow(ax, (x + 0.45, 3.6), (x + 0.45, 3.15), lw=1.0)
            box(ax, x, 1.55, 0.9, 0.5, "stain image", C_POST, fs=7)
            arrow(ax, (x + 0.45, 2.3), (x + 0.45, 2.05), lw=1.0)
        box(ax, x0 + 0.1 + len(markers_f[f]) * 0.98, 2.3, 0.9, 0.85, "model\nnext marker", "#fafafa", fs=7.5, ec="#aaaaaa", lw=1.2, color="#888888")
    ax.text(0.2, 0.95, "Each model: same trunk, same training recipe, its own weights and calibrated mask threshold.  Adding a marker = one configs/markers/*.toml + one training run.",
            fontsize=8.5, color=GREY)
    ax.text(0.2, 0.55, "Across families nothing is shared by default (the generalist run showed instruments should not share weights at our scale); a cross-family warm start is an option, not a requirement.",
            fontsize=8.5, color=GREY)
    save(fig, "models_per_marker")


if __name__ == "__main__":
    model_diagram(); pipeline_diagram(); matrix_diagram()
