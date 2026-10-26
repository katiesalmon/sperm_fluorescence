"""Score predictions against DENOISED measurements: how much of the pixel-level gap is the
truth's own noise.

    python -m sperm_fluorescence denoised runs/abl_dropout --cache data/a8_roles --test 3 --sample 300 --out figures/denoised

For each marker and each filter applied to the measured stain image (none, Gaussian at
several widths, median, non-local means, total variation), the pooled pixel r between the
prediction and the filtered measurement, over valid pixels of the held-out objects. Also
r between the filtered and the raw measurement, which says how much of the raw image the
filter treats as noise. Event r is reported once: summing over the cell already averages
the noise, so filtering barely moves it.

Writes r_vs_filter.png (one panel per marker), samples.png (raw, filtered, predicted),
and denoised.json.
"""
import argparse
import json
import os

import numpy as np
from scipy import ndimage as ndi

from . import render
from .load import load_run
from .metrics import evaluate


def filters():
    from skimage.restoration import denoise_nl_means, denoise_tv_chambolle
    def noise_sigma(a):
        # robust noise level from the median absolute deviation; the background dominates the image
        return 1.4826 * float(np.median(np.abs(a - np.median(a))))
    def nlm(a):
        s = noise_sigma(a); return denoise_nl_means(a, h=0.8 * s, sigma=s, patch_size=3, patch_distance=4, fast_mode=True) if s > 0 else a
    def tv(a):
        s = float(np.std(a)) or 1.0; return denoise_tv_chambolle(a / s, weight=0.3) * s
    return [("none", lambda a: a),
            ("gaussian σ=0.7", lambda a: ndi.gaussian_filter(a, 0.7)), ("gaussian σ=1.0", lambda a: ndi.gaussian_filter(a, 1.0)),
            ("gaussian σ=1.5", lambda a: ndi.gaussian_filter(a, 1.5)), ("gaussian σ=2.0", lambda a: ndi.gaussian_filter(a, 2.0)),
            ("median 3×3", lambda a: ndi.median_filter(a, 3)), ("median 5×5", lambda a: ndi.median_filter(a, 5)),
            ("non-local means", nlm), ("total variation", tv)]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence denoised", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--include", default=None); ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--sample", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-well", type=int, default=1, help="objects per well on the sample sheet")
    ap.add_argument("--crop", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint); te, wte, names = R.te, R.wte, R.names
    rng = np.random.default_rng(a.seed); idx = np.sort(rng.choice(len(te), min(a.sample, len(te)), replace=False))
    res, P = evaluate(R.model, te, wte[idx], R.scale, R.dev, names, idx=idx)
    av = te.avail[idx] if te.avail is not None else np.ones((len(idx), te.cout), bool)
    ks = [k for k in range(te.cout) if av[:, k].any()]
    objs = []
    for q, i in enumerate(idx):
        x, y, m = te.item(i); h, w = te.shape(i); H, W = m.shape; t, l = (H - h) // 2, (W - w) // 2
        objs.append((x[:, t:t + h, l:l + w], y[:, t:t + h, l:l + w], m[t:t + h, l:l + w], P[q]))
    F = filters(); table = {}
    for k in ks:
        table[names[k]] = {"event_r": res[names[k]]["event_r"], "filters": {}}
        for fname, fn in F:
            pp, tt, rr = [], [], []
            for q, (x, y, m, p) in enumerate(objs):
                if not av[q, k]: continue
                d = fn(y[k].astype(np.float64)); sel = m > 0
                pp.append(p[k][sel]); tt.append(d[sel]); rr.append(y[k][sel])
            pp, tt, rr = np.concatenate(pp), np.concatenate(tt), np.concatenate(rr)
            table[names[k]]["filters"][fname] = {"pred_vs_filtered": float(np.corrcoef(pp, tt)[0, 1]), "filtered_vs_raw": float(np.corrcoef(tt, rr)[0, 1])}
    os.makedirs(a.out, exist_ok=True)
    json.dump(table, open(os.path.join(a.out, "denoised.json"), "w"), indent=1)
    print("%-12s %-18s %22s %18s" % ("marker", "filter", "r pred vs filtered", "r filtered vs raw"))
    for n, d in table.items():
        for fname, v in d["filters"].items():
            print("%-12s %-18s %22.3f %18.3f" % (n, fname, v["pred_vs_filtered"], v["filtered_vs_raw"]))
        print("%-12s event r (unfiltered) %.3f" % (n, d["event_r"]))

    # ---- graph ----
    fnames = [f for f, _ in F]
    fig, axes = plt.subplots(1, len(ks), figsize=(4.4 * len(ks) + 0.5, 4.6), facecolor="white", squeeze=False)
    for ax, k in zip(axes[0], ks):
        d = table[names[k]]["filters"]; x = np.arange(len(fnames))
        c = render.color_for(names[k], k)
        ax.plot(x, [d[f]["pred_vs_filtered"] for f in fnames], "o-", color=c, lw=2, label="prediction vs filtered measurement")
        ax.plot(x, [d[f]["filtered_vs_raw"] for f in fnames], "s--", color="#555555", lw=1.2, label="filtered vs raw measurement")
        ax.axhline(table[names[k]]["event_r"], color="#999999", lw=0.8, ls=":", label="event r (per-cell sums, unfiltered)")
        ax.set_xticks(x); ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=8); ax.set_ylim(0, 1.0)
        ax.set_title(names[k], fontsize=12, fontweight="bold", color=tuple(0.7 * v for v in c)); ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("pixel r over valid pixels", fontsize=10); axes[0][0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Pixel correlation against a denoised measurement (%d held-out objects)" % len(idx), fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "r_vs_filter.png"), dpi=200, facecolor="white"); plt.close(fig)

    # ---- sample sheet ----
    show_f = [("none", F[0][1]), ("gaussian σ=1.0", F[2][1]), ("median 3×3", F[5][1]), ("non-local means", F[7][1])]
    pick = []
    for w in sorted(set(wte[idx])):
        cand = np.flatnonzero(wte[idx] == w)
        strong = [q for q in cand if all((objs[q][1][k] * objs[q][2]).sum() >= np.median([(objs[c][1][k] * objs[c][2]).sum() for c in cand]) for k in ks if av[q, k])]
        pick += [int(q) for q in rng.choice(strong or cand, min(a.per_well, len(strong or cand)), replace=False)]
    cols = 1 + len(ks) * (len(show_f) + 1)
    fig, axes = plt.subplots(len(pick), cols, figsize=(1.5 * cols, 1.7 * len(pick) + 0.6), facecolor="black", squeeze=False)
    for i, q in enumerate(pick):
        x, y, m, p = objs[q]
        if a.crop:
            from .slides import window
            x, y, m, p = window(x, y, m, p, a.crop)
        rng_ = render.ranges([y], [m]); c_ = 0
        ax = axes[i, c_]; ax.imshow(render.grey(x[0]), interpolation="nearest"); c_ += 1
        if i == 0: ax.set_title("input", fontsize=7.5, color="white")
        for k in ks:
            c = render.color_for(names[k], k); lo, hi = rng_[k]
            for fname, fn in show_f:
                ax = axes[i, c_]; ax.imshow(render.colorize(fn(y[k].astype(np.float64)), c, lo, hi), interpolation="nearest"); c_ += 1
                if i == 0: ax.set_title("%s\nmeasured, %s" % (names[k], fname), fontsize=6.5, color="white")
            ax = axes[i, c_]; ax.imshow(render.colorize(np.clip(p[k], 0, None), c, lo, hi), interpolation="nearest"); c_ += 1
            if i == 0: ax.set_title("%s\npredicted" % names[k], fontsize=6.5, color="white")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
        axes[i, 0].set_ylabel("%s well" % wte[idx][q], fontsize=7, color="white")
    fig.suptitle("Measured stain, raw and denoised, beside the prediction", fontsize=9, color="white")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "samples.png"), dpi=150, facecolor="black"); plt.close(fig)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
