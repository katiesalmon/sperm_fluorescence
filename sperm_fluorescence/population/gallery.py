"""Pictures for populations: the 2-D embedding coloured by cluster and by the DAPI
population, and one gallery per cluster of its members nearest the centre and farthest from
it, drawn as brightfield with the predicted compartment masks outlined."""
import os

import numpy as np

from ..eval import render

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6", "#c9357a", "#4fb8c9", "#8c8c8c"]
MASK_COLOUR = {"nucleus": (0.65, 0.35, 0.95), "acrosome": (1.0, 1.0, 0.3), "midpiece": (1.0, 0.3, 0.3), "flagellum": (0.35, 1.0, 0.35)}
SET_MASKS = {"head": ("nucleus", "acrosome"), "tail": ("midpiece", "flagellum"), "morph": ("nucleus", "acrosome", "midpiece", "flagellum"),
             "intensity": ("nucleus", "acrosome", "midpiece", "flagellum"), "all": ("nucleus", "acrosome", "midpiece", "flagellum")}


def embedding_plot(tab, labels, kind, path, title=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor="#fcfcfb")
    x, y = tab["emb_x"].to_numpy(), tab["emb_y"].to_numpy()
    for c in sorted(set(labels)):
        sel = labels == c; col = "#bbbbbb" if c < 0 else COLOURS[c % len(COLOURS)]
        axes[0].scatter(x[sel], y[sel], s=4, color=col, alpha=0.6, label=("noise" if c < 0 else "cluster %d" % c) + " (%d)" % sel.sum(), rasterized=True)
    pops = tab["id.pop_pred"].to_numpy().astype(str)
    for j, p in enumerate(("sperm", "round", "dim", "none")):
        sel = pops == p
        if sel.any(): axes[1].scatter(x[sel], y[sel], s=4, color=COLOURS[j % len(COLOURS)], alpha=0.6, label="%s (%d)" % (p, sel.sum()), rasterized=True)
    axes[0].set_title("by cluster", fontsize=10); axes[1].set_title("by predicted-DAPI population", fontsize=10)
    for ax in axes:
        ax.set_xlabel("%s 1" % kind, fontsize=9); ax.set_ylabel("%s 2" % kind, fontsize=9); ax.legend(fontsize=7.5, frameon=False, markerscale=3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    fig.suptitle(title, fontsize=11, fontweight="bold"); fig.tight_layout(); fig.savefig(path, dpi=170, facecolor="#fcfcfb"); plt.close(fig)


def galleries(store_path, tab, scores, labels, feature_set, out_dir, n=8, size=72):
    """One PNG per cluster: top row the n members nearest the cluster centre, bottom row the n
    farthest (its periphery), each as brightfield with compartment outlines."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ..data.cache import open_cache
    from ..data.store import open_store
    from ..eval.masks import outline
    from ..eval.slides import window
    from ..features.engine import COMPARTMENT
    st = open_store(store_path); ds, _, _ = open_cache(st.meta["source_cache"], st.classes)
    bf_i = ds.roles.index("brightfield") if "brightfield" in ds.roles else 0
    comp_of = {COMPARTMENT[nm]: k for k, nm in enumerate(st.names) if nm in COMPARTMENT}
    show = [c for c in SET_MASKS.get(feature_set, SET_MASKS["morph"]) if c in comp_of]
    idx_all = tab["id.index"].to_numpy(dtype=int)
    for c in sorted(set(labels)):
        if c < 0: continue
        sel = np.flatnonzero(labels == c); centre = scores[sel].mean(0); d = np.hypot(*(scores[sel] - centre).T) if scores.shape[1] == 2 else np.linalg.norm(scores[sel] - centre, axis=1)
        near = sel[np.argsort(d)[:n]]; far = sel[np.argsort(-d)[:n]]
        fig, axes = plt.subplots(2, n, figsize=(1.15 * n, 2.7), facecolor="white", squeeze=False)
        for r, (group, label) in enumerate(((near, "nearest the centre"), (far, "periphery"))):
            for j in range(n):
                ax = axes[r, j]; ax.set_xticks([]); ax.set_yticks([])
                if j >= len(group): ax.axis("off"); continue
                i = int(idx_all[group[j]]); x, y, m = ds.item(i); p, pm, _, _ = st.item(i)
                xw, _, mw, pw = window(x, y, m, pm.astype(np.float32), size)
                img = render.grey(xw[bf_i])
                for comp in show:
                    o = outline(pw[comp_of[comp]] > 0.5); img[o] = MASK_COLOUR[comp]
                ax.imshow(img, interpolation="nearest")
                ax.set_title("#%d" % i, fontsize=6, pad=2)
                for sp in ax.spines.values(): sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.5)
            axes[r, 0].set_ylabel(label, fontsize=7)
        fig.suptitle("cluster %d (n = %d): %s outlined" % (c, len(sel), ", ".join(show)), fontsize=9, fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "gallery_%d.png" % c), dpi=200, facecolor="white"); plt.close(fig)
