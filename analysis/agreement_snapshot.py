"""Four scatters in a row: one measurement per marker, virtual against measured on the same cells,
for an abstract attachment. Each panel: points shaded by local density, the identity line, r and
n in the title, bias and 95 % limits of agreement as a text line (the Bland-Altman numbers without
the second plot).

    python analysis/agreement_snapshot.py data/isx_vstain/features.csv --within runs/pop_isx_all_k2/populations.csv:1 \\
        --measures ACRV1.sigma_major_um DAPI.sigma_major_um TOMM20.sigma_major_um LDHC_AKAP4.sigma_major_um \\
        --names "ACRV1=Acrosome (ACRV1)" "DAPI=Nucleus (DAPI)" "TOMM20=Midpiece (TOMM20)" "LDHC_AKAP4=Flagellum (LDHC)" \\
        --out figures/aua_isx_agreement_snapshot.png
    python analysis/agreement_snapshot.py data/isx_vstain/features.csv --query "\`id.pop_scalar\` == 'sperm'" --measures ... # sperm by the DAPI scalar, no clustering needed

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
    ap.add_argument("--query", default=None, help="pandas query on the feature table to select the cells, e.g. \"`id.pop_scalar` == 'sperm'\" (sperm by the DAPI scalar); an alternative to --within")
    ap.add_argument("--population", default=None, help="keep the cells whose id.pop_scalar equals this (e.g. sperm: sperm by the DAPI scalar); the quoting-free alternative to --query")
    ap.add_argument("--population-column", default="id.pop_scalar")
    ap.add_argument("--footer", action="store_true", help="print the one-line explanation of the statistics under the row (off by default)")
    ap.add_argument("--quantity", default="", help="what the axes measure, for the footer")
    ap.add_argument("--title", default=None, help="default: the covered-length title; pass \"\" for none"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pandas as pd
    from scipy.stats import gaussian_kde
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pd.read_csv(a.table, low_memory=False)
    if a.within:
        path, c = a.within.rsplit(":", 1); p = pd.read_csv(path, usecols=["id.index", "cluster"])
        keep = set(p[p.cluster == int(c)]["id.index"]); df = df[df["id.index"].isin(keep)]
    if a.query: df = df.query(a.query)
    if a.population: df = df[df[a.population_column].astype(str) == a.population]
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
        ax.text(0.04, 0.96, "95%% CI: (%.2f, %.2f)" % (bias - 1.96 * sd, bias + 1.96 * sd), transform=ax.transAxes, fontsize=7.6, color=MUTED, va="top", ha="left")
        ax.set_xlabel("Measured Stain", fontsize=8.5, color=MUTED); ax.set_ylabel("Virtual Stain", fontsize=8.5, color=MUTED)
        ax.tick_params(labelsize=7.5, colors=MUTED, length=2.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.grid(color=GRID, lw=0.5, zorder=0)
    title = a.title if a.title is not None else "Length of Covered Area Per Marker Calculated from Virtual Stain Compared to Measured Stain (\u03bcm)"
    if title: fig.suptitle(title, fontsize=10.5, fontweight="bold", color=INK, y=0.995)
    foot = "Same held-out cells on both axes; %sdashed line = identity. r = Pearson correlation across cells; bias and 95 %% limits of agreement = mean and ±1.96 SD of virtual − measured." % (("each axis: %s. " % a.quantity) if a.quantity else "")
    if a.footer: fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=7, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.07 if a.footer else 0.03, 1, 0.94 if title else 1.0)); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15); plt.close(fig)     # tight bbox: axis labels never fall off the frame
    import json; json.dump({"table": a.table, "within": a.within, "query": a.query, "population": a.population, "panels": stats}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    for s_ in stats: print("%-28s n %5d  r %.2f  bias %+.2f  limits %+.2f..%+.2f" % (s_["measure"], s_["n"], s_["r"], s_["bias"], *s_["loa"]))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
