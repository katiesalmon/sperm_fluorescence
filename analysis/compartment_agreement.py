"""Do compartment measurements from the virtual stains agree with the same measurements from
the real stains, on the same cells? One row per measurement: scatter (virtual vs measured,
with r), Bland-Altman (difference vs mean, with bias and 95 % limits), and the two
sample-level distributions overlaid (the census view). Restricted to a population when
--within and a populations.csv are given.

    python analysis/compartment_agreement.py data/isx_vstain/features.csv --within runs/isx_all_latent_k2/populations.csv:1 \\
        --out figures/isx_compartment_agreement.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEASURES = [
    ("nucleus.feret_max_um", "nucleus length (µm)"), ("nucleus.feret_min_um", "nucleus width (µm)"),
    ("head.feret_max_um", "head length (µm)"), ("head.feret_min_um", "head width (µm)"),
    ("x.acrosome_frac", "acrosome / head area"), ("midpiece.skel.path_um", "midpiece length (µm)"),
    ("flagellum.skel.path_um", "flagellum path length (µm)"), ("tail.skel.turning_total_deg", "tail total turning (°)"),
    ("x.d_nucleus_midpiece_um", "nucleus to midpiece distance (µm)"), ("x.head_midpiece_angle_deg", "head–midpiece angle (°)"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table"); ap.add_argument("--within", default=None, help="populations.csv:cluster to restrict to")
    ap.add_argument("--measures", nargs="*", default=None); ap.add_argument("--out", required=True); ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    import pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pd.read_csv(a.table); n_all = len(df); label = "all objects"
    if a.within:
        path, c = a.within.rsplit(":", 1); p = pd.read_csv(path, usecols=["id.index", "cluster", "gate.passed"])
        keep = p[(p["gate.passed"]) & (p["cluster"] == int(c))]["id.index"]; df = df[df["id.index"].isin(keep)]; label = "cluster %s of %s" % (c, os.path.basename(os.path.dirname(path)))
    measures = [(m, n) for m, n in MEASURES if (a.measures is None or m in a.measures) and "pred." + m in df and "ref." + m in df]
    fig, axes = plt.subplots(len(measures), 3, figsize=(13, 2.6 * len(measures)), facecolor="white")
    rows = []
    for i, (m, name) in enumerate(measures):
        x = df["ref." + m].to_numpy(float); y = df["pred." + m].to_numpy(float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan; d = y - x; bias, sd = d.mean(), d.std()
        lo, hi = np.nanpercentile(np.concatenate([x, y]), [0.5, 99.5])
        ax = axes[i, 0]; ax.scatter(x, y, s=3, alpha=0.3, color="#2a78d6", rasterized=True); ax.plot([lo, hi], [lo, hi], color="#52514e", lw=0.8, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("measured stain", fontsize=8); ax.set_ylabel("virtual stain", fontsize=8)
        ax.set_title("%s   r = %.2f, n = %d" % (name, r, len(x)), fontsize=9, fontweight="bold", loc="left")
        ax = axes[i, 1]; mean = (x + y) / 2; ax.scatter(mean, d, s=3, alpha=0.3, color="#eb6834", rasterized=True)
        for v, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")): ax.axhline(v, color="#52514e", lw=0.8, ls=ls)
        ax.set_xlim(lo, hi); ax.set_ylim(bias - 3.5 * sd, bias + 3.5 * sd); ax.set_xlabel("mean of the two", fontsize=8); ax.set_ylabel("virtual − measured", fontsize=8)
        ax.set_title("bias %+.2f, 95 %% limits %+.2f to %+.2f" % (bias, bias - 1.96 * sd, bias + 1.96 * sd), fontsize=8.5, loc="left")
        ax = axes[i, 2]; bins = np.linspace(lo, hi, 40)
        ax.hist(x, bins, color="#52514e", alpha=0.45, label="measured (median %.2f)" % np.median(x)); ax.hist(y, bins, color="#2a78d6", alpha=0.45, label="virtual (median %.2f)" % np.median(y))
        ax.set_xlabel(name, fontsize=8); ax.set_ylabel("objects", fontsize=8); ax.legend(fontsize=7, frameon=False)
        ax.set_title("sample distribution", fontsize=8.5, loc="left")
        for ax in axes[i]:
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7)
        rows.append((name, len(x), r, bias, sd, float(np.median(x)), float(np.median(y))))
    fig.suptitle(a.title or "Compartment measurements: virtual stain against measured stain, same cells (%s, n = %d of %d)" % (label, len(df), n_all), fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=150, facecolor="white"); plt.close(fig)
    print("%-36s %6s %6s %8s %8s %10s %10s" % ("measure", "n", "r", "bias", "sd", "med meas", "med virt"))
    for row in rows: print("%-36s %6d %6.2f %+8.2f %8.2f %10.2f %10.2f" % row)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
