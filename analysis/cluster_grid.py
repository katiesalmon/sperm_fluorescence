"""One page, 2 x 2: two clusterings (rows, e.g. k = 2 and k = 3) in two views (columns, e.g. the
virtual stains and the side-scatter image). Each cell shows every cluster of that run as a strip
of its members from the cluster centre outward, so the same cells can be read in both views.

    python analysis/cluster_grid.py runs/pop_srv_a8_feat_k2 runs/pop_srv_a8_feat_k3 --store <store> --views stains ssc \\
        --names2 "0=leukocytes" "1=sperm" --names3 "0=leukocytes" "1=sperm, straight tail" "2=sperm, looped tail" \\
        --title "A8, full held-out replicate: feature clustering at k = 2 and k = 3" --out figures/srv_a8_pop_grid_k2_k3.png

Views: stains (virtual), measured, or an input role (ssc, lightloss, fsc, brightfield, darkfield).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6", "#c0392b"]
VIEW_LABEL = {"stains": "virtual stains", "measured": "measured stains", "ssc": "side scatter (SSC)", "lightloss": "light loss", "fsc": "forward scatter (FSC)", "brightfield": "brightfield", "brightfield2": "brightfield 2", "darkfield": "darkfield"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="population runs, one grid row each (top to bottom)")
    ap.add_argument("--store", required=True); ap.add_argument("--views", nargs="+", default=["stains", "ssc"])
    ap.add_argument("--rows", type=int, default=3, help="tile rows per cluster"); ap.add_argument("--cols", type=int, default=12); ap.add_argument("--size", type=int, default=56)
    ap.add_argument("--names2", nargs="*", default=[], help="cluster=name for the first run"); ap.add_argument("--names3", nargs="*", default=[], help="cluster=name for the second run")
    ap.add_argument("--names", nargs="*", action="append", default=None, help="repeat once per run (alternative to --names2/--names3)")
    ap.add_argument("--title", default=None); ap.add_argument("--out", required=True); ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)
    import pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from sperm_fluorescence.data.cache import open_cache
    from sperm_fluorescence.data.store import open_store
    from sperm_fluorescence.eval import render
    from sperm_fluorescence.eval.slides import window
    st = open_store(a.store); ds, _, _ = open_cache(st.meta["source_cache"], st.classes)
    name_sets = a.names if a.names else [a.names2, a.names3] + [[]] * (len(a.runs) - 2)
    runs = []
    for r in a.runs:
        t = pd.read_csv(os.path.join(r, "populations.csv"), low_memory=False); s = json.load(open(os.path.join(r, "summary.json")))
        t = t[t["gate.passed"]].reset_index(drop=True); pcs = [c for c in t.columns if c.startswith("pc") and c[2:].isdigit()]
        runs.append((r, t, s, pcs, sorted(c for c in t["cluster"].unique() if c >= 0)))
    per = a.rows * a.cols
    def members_of(t, pcs, c):
        sel = t[t["cluster"] == c]; S = sel[pcs].to_numpy(float); S = S[:, ~np.isnan(S).all(0)]
        order = np.argsort(np.linalg.norm(S - np.nanmean(S, 0), axis=1))
        return sel["id.index"].to_numpy(int)[order][:per], len(sel)
    def image_of(i, view):
        x, y, m = ds.item(int(i))
        if view == "stains": p, _, _, _ = st.item(int(i)); return p, m, x, y
        if view == "measured": return y, m, x, y
        k = list(ds.roles).index(view); return x[k:k + 1], m, x, y
    # one display range per marker across the whole figure, from a spread of objects
    sample = runs[0][1]["id.index"].to_numpy()[:: max(1, len(runs[0][1]) // 300)]
    hi = {}
    for view in a.views:
        if view in ("stains", "measured"):
            hi[view] = [max(float(np.percentile(np.concatenate([np.clip(image_of(i, view)[0][k][image_of(i, view)[1] > 0], 0, None) for i in sample]), 99.5)), 1e-4) for k in range(st.cout)]
    # layout: outer grid rows = runs (height by number of clusters), cols = views
    heights = [len(ks) * (a.rows + 0.55) for _, _, _, _, ks in runs]
    tile = 0.5
    figw = tile * a.cols * len(a.views) + 0.6 * (len(a.views) + 1) + 0.7; figh = tile * sum(heights) + 1.6 + 0.6 * (len(runs) - 1)
    fig = plt.figure(figsize=(figw, figh), facecolor="white")
    outer = gridspec.GridSpec(len(runs), len(a.views), figure=fig, height_ratios=heights, hspace=0.3, wspace=0.08, left=0.75 / figw, right=0.99, top=1 - 1.15 / figh, bottom=0.01)
    first_cell = {}
    for ri, (r, t, s, pcs, ks) in enumerate(runs):
        names = dict(n.split("=", 1) for n in (name_sets[ri] or []))
        for vi, view in enumerate(a.views):
            cell = gridspec.GridSpecFromSubplotSpec(len(ks), 1, subplot_spec=outer[ri, vi], hspace=0.35)
            for ci, c in enumerate(ks):
                members, n = members_of(t, pcs, c)
                inner = gridspec.GridSpecFromSubplotSpec(a.rows, a.cols, subplot_spec=cell[ci], wspace=0.03, hspace=0.03)
                for j in range(per):
                    ax = fig.add_subplot(inner[j // a.cols, j % a.cols]); ax.set_xticks([]); ax.set_yticks([])
                    if j >= len(members): ax.axis("off"); continue
                    img, m, x, y = image_of(members[j], view); _, _, _, pw = window(x, y, m, img, a.size)
                    if view in ("stains", "measured"):
                        ax.imshow(render.composite([render.colorize(np.clip(pw[k], 0, None), render.color_for(st.names[k], k), 0, hi[view][k]) for k in range(st.cout)]), interpolation="nearest")
                    else:
                        ax.imshow(render.grey(pw[0]), interpolation="nearest")
                    for sp in ax.spines.values(): sp.set_edgecolor("#333333"); sp.set_linewidth(0.3)
                head = fig.add_subplot(cell[ci]); head.axis("off")
                label = "Cluster %d" % c + (": %s" % names[str(c)] if str(c) in names else "")
                head.set_title(label, fontsize=9.5, fontweight="bold", color=COLOURS[c % len(COLOURS)], loc="left", pad=4)
                if ci == 0 and vi == 0:
                    first_cell[ri] = head
    # view headers along the top, k labels down the left margin
    fig.canvas.draw()
    for vi, view in enumerate(a.views):
        b = fig.add_subplot(outer[0, vi]); b.axis("off"); pos = b.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 1 - 0.85 / figh, VIEW_LABEL.get(view, view)[0].upper() + VIEW_LABEL.get(view, view)[1:], fontsize=12, fontweight="bold", ha="center", va="bottom")
    for ri, (r, t, s, pcs, ks) in enumerate(runs):
        b = fig.add_subplot(outer[ri, 0]); b.axis("off"); pos = b.get_position()
        fig.text(0.22 / figw, (pos.y0 + pos.y1) / 2, "k = %d" % s["k"], fontsize=13, fontweight="bold", ha="center", va="center", rotation=90)
    import textwrap
    title = a.title or "Clusterings side by side"
    fig.suptitle(title.replace(" | ", "\n") if " | " in title else "\n".join(textwrap.wrap(title, 110)), fontsize=13, fontweight="bold", y=1 - 0.1 / figh, va="top")      # " | " in the title = a line break where you want it
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=a.dpi, facecolor="white"); plt.close(fig); print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
