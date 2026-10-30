"""Four scatters in a row: one measurement per marker, virtual against measured on the same cells,
for an abstract attachment. Each panel: points shaded by local density, the identity line, r and
n in the title, bias and 95 % limits of agreement as a text line (the Bland-Altman numbers without
the second plot).

    python analysis/agreement_snapshot.py data/isx_vstain/features.csv --within runs/pop_isx_all_k2/populations.csv:1 \\
        --measures ACRV1.sigma_major_um DAPI.sigma_major_um TOMM20.sigma_major_um LDHC_AKAP4.sigma_major_um \\
        --names "ACRV1=Acrosome (ACRV1)" "DAPI=Nucleus (DAPI)" "TOMM20=Midpiece (TOMM20)" "LDHC_AKAP4=Flagellum (LDHC)" \\
        --quantity "spread along the axis, σ (µm)" --out figures/aua_isx_agreement_snapshot.png

The virtual side is preds.<measure>, the measured side refs.<measure> (the stain-image features,
computed above each frame's noise floor within the cell region; features/stain.py).
"""
import argparse
import os

import numpy as np

INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table"); ap.add_argument("--within", default=None, help="populations.csv:cluster to restrict to (e.g. the stained / sperm cluster)")
    ap.add_argument("--measures", nargs="+", required=True, help="<channel>.<stat> suffixes, left to right")
    ap.add_argument("--names", nargs="*", default=[], help="channel=panel title")
    ap.add_argument("--prefixes", nargs=2, default=["preds", "refs"])
    ap.add_argument("--quantity", default="", help="what the axes measure, printed once under the row")
    ap.add_argument("--title", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pandas as pd
    from scipy.stats import gaussian_kde
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pd.read_csv(a.table, low_memory=False)
    if a.within:
        path, c = a.within.rsplit(":", 1); p = pd.read_csv(path, usecols=["id.index", "cluster"])
        keep = set(p[p.cluster == int(c)]["id.index"]); df = df[df["id.index"].isin(keep)]
    names = dict(s.split("=", 1) for s in a.names)
    n_panels = len(a.measures)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.1 * n_panels + 0.4, 3.75), facecolor="white")
    stats = []
    for ax, m in zip(np.atleast_1d(axes), a.measures):
        ch = m.split(".")[0]; v = df["%s.%s" % (a.prefixes[0], m)].to_numpy(float); t = df["%s.%s" % (a.prefixes[1], m)].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(t); v, t = v[ok], t[ok]
        lo, hi = np.percentile(np.concatenate([v, t]), [0.5, 99.5]); pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
        r = float(np.corrcoef(t, v)[0, 1]); d = v - t; bias = float(d.mean()); sd = float(d.std(ddof=1))
        stats.append({"measure": m, "n": int(ok.sum()), "r": r, "bias": bias, "loa": [bias - 1.96 * sd, bias + 1.96 * sd], "median_measured": float(np.median(t)), "median_virtual": float(np.median(v))})
        # density shading: darker where the points pile up, so the ridge along the identity reads through the cloud
        sub = np.random.default_rng(0).choice(len(t), min(len(t), 3000), replace=False)
        dens = gaussian_kde(np.vstack([t[sub], v[sub]]))(np.vstack([t, v])); order = np.argsort(dens)
        ax.scatter(t[order], v[order], c=dens[order], cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#b9d3f3", BLUE, "#0d3f80"]), s=6, linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--", zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title("%s\nr = %.2f, n = %s" % (names.get(ch, ch), r, "{:,}".format(int(ok.sum()))), fontsize=9.5, fontweight="bold", color=INK, pad=6)
        ax.text(0.04, 0.96, "bias %+.2f\n95%% limits %+.2f to %+.2f" % (bias, bias - 1.96 * sd, bias + 1.96 * sd), transform=ax.transAxes, fontsize=7.2, color=MUTED, va="top", ha="left", linespacing=1.3)
        ax.set_xlabel("measured stain", fontsize=8.5, color=MUTED); ax.set_ylabel("virtual stain", fontsize=8.5, color=MUTED)
        ax.tick_params(labelsize=7.5, colors=MUTED, length=2.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.grid(color=GRID, lw=0.5, zorder=0)
    if a.title: fig.suptitle(a.title, fontsize=10.5, fontweight="bold", color=INK, y=0.995)
    foot = "Same held-out cells on both axes; %sdashed line = identity. r = Pearson correlation across cells; bias and 95 %% limits of agreement = mean and ±1.96 SD of virtual − measured." % (("each axis: %s. " % a.quantity) if a.quantity else "")
    fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=7, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97 if a.title else 1.0)); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300, facecolor="white"); plt.close(fig)
    import json; json.dump({"table": a.table, "within": a.within, "panels": stats}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    for s_ in stats: print("%-28s n %5d  r %.2f  bias %+.2f  limits %+.2f..%+.2f" % (s_["measure"], s_["n"], s_["r"], s_["bias"], *s_["loa"]))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
