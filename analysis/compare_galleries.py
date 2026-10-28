"""Side-by-side galleries of two clusterings of the same objects, one figure: left column the
first run's clusters, right column the second's, each cluster a strip of its members nearest
the centre (brightfield-like channel above, virtual stains below), with the contingency
between the two clusterings at the bottom.

    python analysis/compare_galleries.py runs/pop_a8_sperm_feat runs/pop_a8_sperm_latent --store data/a8_vstain \\
        --labels "features (358 columns)" "autoencoder latent (16-d)" --out figures/a8_sperm_feature_vs_latent.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a"); ap.add_argument("run_b"); ap.add_argument("--store", required=True)
    ap.add_argument("--labels", nargs=2, default=None); ap.add_argument("--n", type=int, default=8); ap.add_argument("--size", type=int, default=56)
    ap.add_argument("--names", nargs="*", default=[], help="optional cluster names as run:cluster=name, e.g. a:1=straight")
    ap.add_argument("--title", default=None); ap.add_argument("--out", required=True)
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
    bf_i = next((ds.roles.index(r) for r in ("brightfield", "lightloss") if r in ds.roles), 0)
    bf_name = {"brightfield": "brightfield", "lightloss": "light loss"}.get(ds.roles[bf_i], ds.roles[bf_i])
    inst = INSTRUMENT.get(st.meta["classes"][st.classes[0]]["instrument"], "")
    names = {}
    for spec in a.names:
        k, v = spec.split("=", 1); names[k] = v
    runs = []
    for key, run in (("a", a.run_a), ("b", a.run_b)):
        t = pd.read_csv(os.path.join(run, "populations.csv")); s = json.load(open(os.path.join(run, "summary.json")))
        t = t[t["gate.passed"]].reset_index(drop=True)
        pcs = [c for c in t.columns if c.startswith("pc") and c[2:].isdigit()]
        runs.append(dict(key=key, run=run, tab=t, summ=s, pcs=pcs))
    la, lb = a.labels or (os.path.basename(a.run_a), os.path.basename(a.run_b))
    # shared set and contingency
    ta, tb = runs[0]["tab"], runs[1]["tab"]
    m = ta[["id.index", "cluster"]].merge(tb[["id.index", "cluster"]], on="id.index", suffixes=("_a", "_b"))
    ct = pd.crosstab(m["cluster_a"], m["cluster_b"])
    ka = sorted(ta["cluster"].unique())
    # order the right-hand clusters so each row faces the left cluster it overlaps most (greedy on the contingency)
    kb, left = [], list(sorted(tb["cluster"].unique()))
    for i in ka:
        if not left: break
        j = max(left, key=lambda j: ct.loc[i, j] if j in ct.columns else 0); kb.append(j); left.remove(j)
    kb += left
    # display ranges per marker, shared across the figure
    sample = ta["id.index"].to_numpy()[:: max(1, len(ta) // 300)]
    hi = []
    for k in range(st.cout):
        vals = np.concatenate([np.clip(st.item(int(i))[0][k][st.item(int(i))[2] > 0], 0, None) for i in sample]); hi.append(max(float(np.percentile(vals, 99.5)), 1e-4))

    def strip_members(r, c):
        t = r["tab"]; sel = t[t["cluster"] == c]; S = sel[r["pcs"]].to_numpy(float); S = S[:, ~np.isnan(S).all(0)]
        d = np.linalg.norm(S - np.nanmean(S, 0), axis=1); order = np.argsort(d)[: a.n]
        return sel["id.index"].to_numpy(int)[order]

    def draw_cell(ax_bf, ax_st, i):
        x, y, mm = ds.item(i); p, pm, _, _ = st.item(i)
        xw, _, mw, pw = window(x, y, mm, p, a.size)
        ax_bf.imshow(render.grey(xw[bf_i]), interpolation="nearest")
        layers = [render.colorize(np.clip(pw[k], 0, None), render.color_for(st.names[k], k), 0, hi[k]) for k in range(st.cout)]
        ax_st.imshow(render.composite(layers), interpolation="nearest")
        for ax in (ax_bf, ax_st):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.5)

    nrows = max(len(ka), len(kb))
    fig = plt.figure(figsize=(2 * (1.0 * a.n + 1.2), 2.25 * nrows + 3.2), facecolor="white")
    outer = gridspec.GridSpec(nrows + 1, 2, figure=fig, height_ratios=[1] * nrows + [0.75], wspace=0.08, hspace=0.42, left=0.04, right=0.99, top=0.86, bottom=0.02)
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6"]
    for col, (r, label, ks) in enumerate(((runs[0], la, ka), (runs[1], lb, kb))):
        for row, c in enumerate(ks):
            inner = gridspec.GridSpecFromSubplotSpec(2, a.n, subplot_spec=outer[row, col], wspace=0.04, hspace=0.04)
            members = strip_members(r, c)
            for j in range(a.n):
                ax_bf = fig.add_subplot(inner[0, j]); ax_st = fig.add_subplot(inner[1, j])
                if j < len(members): draw_cell(ax_bf, ax_st, int(members[j]))
                else: ax_bf.axis("off"); ax_st.axis("off")
                if j == 0:
                    ax_bf.set_ylabel(bf_name, fontsize=6.5); ax_st.set_ylabel("virtual stains", fontsize=6.5)
            n_c = int((r["tab"]["cluster"] == c).sum())
            name = names.get("%s:%d" % (r["key"], c), "")
            pos = fig.add_subplot(outer[row, col]); pos.axis("off")
            pos.set_title("%s   cluster %d  (n = %d)%s" % (label.upper() if False else label, c, n_c, ("   " + name) if name else ""), fontsize=9, fontweight="bold", color=colours[c % len(colours)], pad=14, loc="left")
    # contingency table
    axc = fig.add_subplot(outer[nrows, :]); axc.axis("off")
    cell_text = [[str(ct.loc[i, j]) if j in ct.columns else "0" for j in kb] for i in ka]
    tbl = axc.table(cellText=cell_text, rowLabels=["%s %d" % (la, i) for i in ka], colLabels=["%s %d" % (lb, j) for j in kb], loc="center", cellLoc="center", bbox=[0.3, 0.05, 0.4, 0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(m["cluster_a"], m["cluster_b"])
    axc.set_title("the same %d objects under both clusterings, rows = %s clusters, columns = %s clusters (adjusted Rand index %.2f); each strip above shows the %d members nearest its cluster centre; rows are paired by overlap" % (len(m), la, lb, ari, a.n), fontsize=8.5, color="#52514e", pad=2)
    ga = runs[0]["summ"]["gates"]; gates = ", ".join("%s (%s)" % (g["gate"], g["query"].replace("`", "")) for g in ga) or "none"
    title = a.title or "%s: the same population clustered two ways" % inst
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, "%d of %d objects after gates: %s   ·   left: %s, %s k = %d   ·   right: %s, %s k = %d\nstore %s · classes %s · runs %s, %s" % (
        len(ta), runs[0]["summ"]["n"], gates, la, runs[0]["summ"]["method"], runs[0]["summ"]["k"], lb, runs[1]["summ"]["method"], runs[1]["summ"]["k"],
        a.store, ",".join(st.classes), os.path.basename(a.run_a), os.path.basename(a.run_b)), ha="center", va="top", fontsize=7.5, color="#52514e", linespacing=1.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=200, facecolor="white"); plt.close(fig)
    print("wrote", a.out); print(ct)


if __name__ == "__main__":
    main()
