"""Large virtual-stain-only galleries: for one run, every cluster as a grid of its members
drawn as the predicted stains alone (no input image), ordered from the cluster centre
outward, so a whole population can be seen at once.

    python analysis/stain_gallery.py runs/pop_a8_sperm_feat --store data/a8_vstain --cols 16 --rows 6 --out figures/a8_sperm_feat_stains.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--store", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=16); ap.add_argument("--rows", type=int, default=6); ap.add_argument("--size", type=int, default=56)
    ap.add_argument("--names", nargs="*", default=[], help="cluster=name"); ap.add_argument("--title", default=None)
    ap.add_argument("--order", default="centre", choices=["centre", "random"], help="members from the cluster centre outward, or a random draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measured", action="store_true", help="draw the MEASURED stains (from the cache) instead of the virtual ones; the clusters are unchanged")
    a = ap.parse_args(argv)
    import pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from sperm_fluorescence.data.cache import open_cache
    from sperm_fluorescence.data.store import open_store
    from sperm_fluorescence.eval import render
    from sperm_fluorescence.eval.slides import window
    from sperm_fluorescence.population.gallery import INSTRUMENT
    st = open_store(a.store); ds, _, _ = open_cache(st.meta["source_cache"], st.classes)
    inst = INSTRUMENT.get(st.meta["classes"][st.classes[0]]["instrument"], "")
    names = dict(s.split("=", 1) for s in a.names)
    t = pd.read_csv(os.path.join(a.run, "populations.csv")); s = json.load(open(os.path.join(a.run, "summary.json")))
    t = t[t["gate.passed"]].reset_index(drop=True); pcs = [c for c in t.columns if c.startswith("pc") and c[2:].isdigit()]
    ks = sorted(c for c in t["cluster"].unique() if c >= 0); per = a.cols * a.rows; rng = np.random.default_rng(a.seed)
    sample = t["id.index"].to_numpy()[:: max(1, len(t) // 300)]
    def stain_of(i):
        if a.measured:
            _, y, mm = ds.item(int(i)); return y, mm
        p, _, mm, _ = st.item(int(i)); return p, mm
    hi = [max(float(np.percentile(np.concatenate([np.clip(stain_of(i)[0][k][stain_of(i)[1] > 0], 0, None) for i in sample]), 99.5)), 1e-4) for k in range(st.cout)]
    legend = "   ".join("%s %s" % (nm, {"LDHC/AKAP4": "green", "TOMM20": "red", "DAPI": "violet", "ACRV1": "yellow", "CD45": "magenta"}.get(nm, "")) for nm in st.names)
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6"]
    import textwrap
    gates = ", ".join("%s (%s)" % (g["gate"], g["query"].replace("`", "")) for g in s["gates"]) or "none"
    what = s["set"] if not s.get("pixels") else "pixels"
    header = "%d of %d objects after gates: %s  ·  features: %s (%d columns, %d PCs)  ·  %s, k = %d  ·  run %s" % (
        len(t), s["n"], gates, what, len(s["columns"]), s["pca"]["n_components"], s["method"], s["k"], os.path.basename(a.run.rstrip("/")))
    lines = textwrap.wrap(header, int(11 * a.cols)) + ["%s: %s  ·  one display range per marker across the figure  ·  store %s, classes %s" % ("MEASURED (true) stains" if a.measured else "virtual stains only", legend, a.store, ",".join(st.classes))]
    head_in = 0.55 + 0.16 * len(lines)
    fig = plt.figure(figsize=(0.62 * a.cols + 0.3, (0.62 * a.rows + 0.55) * len(ks) + head_in + 0.3), facecolor="white")
    outer = gridspec.GridSpec(len(ks), 1, figure=fig, hspace=0.18, left=0.01, right=0.99, top=1 - (head_in + 0.25) / fig.get_figheight(), bottom=0.01)
    for row, c in enumerate(ks):
        sel = t[t["cluster"] == c]
        if a.order == "centre":
            S = sel[pcs].to_numpy(float); S = S[:, ~np.isnan(S).all(0)]; d = np.linalg.norm(S - np.nanmean(S, 0), axis=1); order = np.argsort(d)
        else:
            order = rng.permutation(len(sel))
        members = sel["id.index"].to_numpy(int)[order][:per]
        inner = gridspec.GridSpecFromSubplotSpec(a.rows, a.cols, subplot_spec=outer[row], wspace=0.03, hspace=0.03)
        for j in range(per):
            ax = fig.add_subplot(inner[j // a.cols, j % a.cols]); ax.set_xticks([]); ax.set_yticks([])
            if j >= len(members): ax.axis("off"); continue
            i = int(members[j]); x, y, m = ds.item(i); p, _ = stain_of(i)
            _, _, _, pw = window(x, y, m, p, a.size)
            ax.imshow(render.composite([render.colorize(np.clip(pw[k], 0, None), render.color_for(st.names[k], k), 0, hi[k]) for k in range(st.cout)]), interpolation="nearest")
            for sp in ax.spines.values(): sp.set_edgecolor("#333333"); sp.set_linewidth(0.3)
        head = fig.add_subplot(outer[row]); head.axis("off")
        name = names.get(str(c), "")
        head.set_title("cluster %d  (n = %d, showing %d%s)%s" % (c, len(sel), min(per, len(sel)), " from the centre outward" if a.order == "centre" else ", random", ("   " + name) if name else ""),
                       fontsize=10, fontweight="bold", color=colours[c % len(colours)], loc="left", pad=6)
    fig.suptitle(a.title or "%s: virtual stains by cluster" % inst, fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 1 - 0.42 / fig.get_figheight(), "\n".join(lines), ha="center", va="top", fontsize=7.5, color="#52514e", linespacing=1.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=150, facecolor="white"); plt.close(fig); print("wrote", a.out)


if __name__ == "__main__":
    main()
