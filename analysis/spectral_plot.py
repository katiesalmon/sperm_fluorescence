"""Plot the raw spectral detector signal exported by scripts/spectral_export.py.

    python analysis/spectral_plot.py spectral.json --out figures/spectral

signature.png  one line per FCS file: median per detector with the p10-p90 band, detectors
               grouped by laser in wavelength order, asinh y axis.
heatmap.png    the subsampled events (rows, grouped by file) x detectors (columns), asinh.
"""
import argparse, json, os, re

import numpy as np

LASER = {"UV": "UV 349 nm", "V": "violet 405 nm", "B": "blue 488 nm", "YG": "yellow-green 561 nm", "R": "red 637 nm", "ImgB": "488 nm imaging",
         "VL": "violet 405 nm", "BL": "blue 488 nm", "YL": "yellow 561 nm", "RL": "red 637 nm"}
ORDER = ["UV", "V", "B", "ImgB", "YG", "R", "VL", "BL", "YL", "RL"]
WELL_COLOUR = {"S": "#1f77b4", "P": "#d62728", "SP": "#2ca02c"}
REP_STYLE = {"1": ":", "2": "-", "3": "--"}


def split(name):
    m = re.match(r"^([A-Za-z]+)(\d+)-A$", name.strip())
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def well_of(fname):
    m = re.search(r"_(\d)(SP|S|P)\b", fname) or re.search(r"(\d)(SP|S|P)\.", fname)
    return (m.group(1), m.group(2)) if m else ("?", "?")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="A8 spectral detectors")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    data = json.load(open(a.json)); os.makedirs(a.out, exist_ok=True)
    files = sorted(data, key=lambda f: (well_of(f)[1], well_of(f)[0]))
    dets = data[files[0]]["detectors"]
    dets = sorted(dets, key=lambda d: (ORDER.index(split(d)[0]) if split(d)[0] in ORDER else 99, split(d)[1]))
    groups = []
    for d in dets:
        g = split(d)[0]
        if not groups or groups[-1][0] != g: groups.append([g, 0])
        groups[-1][1] += 1
    x = np.arange(len(dets)); asinh = lambda v: np.arcsinh(np.asarray(v, float) / 1000.0)

    # ---- signature ----
    fig, ax = plt.subplots(figsize=(max(12, 0.22 * len(dets) + 3), 5.2), facecolor="white")
    start = 0
    for i, (g, n) in enumerate(groups):
        ax.axvspan(start - 0.5, start + n - 0.5, color="#f2f2f2" if i % 2 else "#ffffff", zorder=0)
        ax.text(start + n / 2 - 0.5, 1.01, LASER.get(g, g), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="#444")
        start += n
    for f in files:
        rep, well = well_of(f); P = data[f]["percentiles"]
        med = asinh([P[d]["p50"] for d in dets]); lo = asinh([P[d]["p10"] for d in dets]); hi = asinh([P[d]["p90"] for d in dets])
        c = WELL_COLOUR.get(well, "#555"); ls = REP_STYLE.get(rep, "-")
        ax.plot(x, med, color=c, ls=ls, lw=1.6, label="%s well, replicate %s%s" % (well, rep, " (unstained)" if rep == "1" else ""))
        if rep != "1": ax.fill_between(x, lo, hi, color=c, alpha=0.08, lw=0)
    ax.set_xticks(x); ax.set_xticklabels([d[:-2] for d in dets], rotation=90, fontsize=7)
    ax.set_xlim(-0.5, len(dets) - 0.5); ax.set_ylabel("per-event signal, asinh(x / 1000)", fontsize=10)
    ax.set_title(a.title + " — median per detector (band = 10th to 90th percentile of events)", fontsize=12, fontweight="bold", pad=22)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "signature.png"), dpi=200, facecolor="white"); plt.close(fig)

    # ---- heatmap ----
    blocks, labels, bounds = [], [], []
    for f in files:
        cols = data[f]["sample_columns"]; S = np.array(data[f]["sample"], float)
        idx = [cols.index(d) for d in dets]; M = asinh(S[:, idx])
        order = np.argsort(-M.sum(axis=1)); blocks.append(M[order]); rep, well = well_of(f)
        labels.append("%s, rep %s" % (well, rep)); bounds.append(len(M))
    H = np.vstack(blocks); vmax = np.percentile(H, 99.5)
    fig, ax = plt.subplots(figsize=(max(12, 0.22 * len(dets) + 3), 0.012 * len(H) + 2.5), facecolor="white")
    im = ax.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    y = 0; ticks, tl = [], []
    for n, lab in zip(bounds, labels):
        ticks.append(y + n / 2); tl.append(lab); y += n
        ax.axhline(y - 0.5, color="white", lw=0.8)
    start = 0
    for g, n in groups:
        if start: ax.axvline(start - 0.5, color="white", lw=0.8)
        ax.text(start + n / 2 - 0.5, -0.01, LASER.get(g, g), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#444")
        start += n
    ax.set_yticks(ticks); ax.set_yticklabels(tl, fontsize=8); ax.set_xticks(x); ax.set_xticklabels([d[:-2] for d in dets], rotation=90, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("asinh(signal / 1000)", fontsize=9)
    ax.set_title(a.title + " — every subsampled event (rows, brightest first within each file) x every detector", fontsize=12, fontweight="bold", pad=18)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "heatmap.png"), dpi=200, facecolor="white"); plt.close(fig)
    print("wrote", a.out, "| files:", ", ".join(files), "| detectors:", len(dets))


if __name__ == "__main__":
    main()
