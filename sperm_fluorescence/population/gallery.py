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


def measured_columns(tab):
    """The measured stain sums in their own compartment masks, whatever markers the instrument has."""
    out = []
    for c in tab.columns:
        parts = c.split(".")
        if len(parts) == 4 and parts[0] == "ref" and parts[3] == "sum" and parts[2] != "cell": out.append((c, "measured " + parts[1].replace("_", "/")))
    return out[:4]


def embedding_plot(tab, labels, kind, path, title="", measured_labels=None, measured=None):
    """Top row: the embedding by cluster, and by the measured-side clustering when the table
    has reference features (else by the DAPI-scalar population rule). Bottom row, when the
    measured stain sums are available: the same points coloured by each measured marker."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    MEASURED = measured_columns(measured) if measured is not None else []
    have_meas = bool(MEASURED)
    fig, axes = plt.subplots(2 if have_meas else 1, 4 if have_meas else 2, figsize=(16 if have_meas else 10, 8.6 if have_meas else 4.6), facecolor="#fcfcfb", squeeze=False)
    x, y = tab["emb_x"].to_numpy(), tab["emb_y"].to_numpy()
    ax = axes[0, 0]
    for c in sorted(set(labels)):
        sel = labels == c; col = "#bbbbbb" if c < 0 else COLOURS[c % len(COLOURS)]
        ax.scatter(x[sel], y[sel], s=4, color=col, alpha=0.6, label=("noise" if c < 0 else "cluster %d" % c) + " (%d)" % sel.sum(), rasterized=True)
    ax.set_title("label-free clusters", fontsize=10)
    ax = axes[0, 1]
    if measured_labels is not None:
        for c in sorted(set(measured_labels)):
            sel = measured_labels == c
            ax.scatter(x[sel], y[sel], s=4, color=COLOURS[c % len(COLOURS)], alpha=0.6, label="measured-side cluster %d (%d)" % (c, sel.sum()), rasterized=True)
        ax.set_title("same recipe on the measured stains (IDEAS masks)", fontsize=10)
    else:
        pops = tab["id.pop_pred"].to_numpy().astype(str)
        for j, p in enumerate(("sperm", "round", "dim", "none")):
            sel = pops == p
            if sel.any(): ax.scatter(x[sel], y[sel], s=4, color=COLOURS[j % len(COLOURS)], alpha=0.6, label="%s (%d)" % (p, sel.sum()), rasterized=True)
        ax.set_title("by predicted-DAPI population rule", fontsize=10)
    if have_meas:
        for j in range(2, 4): axes[0, j].axis("off")
        for j, (col, name) in enumerate(MEASURED):
            ax = axes[1, j]
            if col not in measured: ax.axis("off"); continue
            v = measured[col].to_numpy(float); v = np.log10(np.clip(v, 1, None))
            sc = ax.scatter(x, y, s=4, c=v, cmap="viridis", alpha=0.8, vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98), rasterized=True)
            ax.set_title(name + " (log10 sum)", fontsize=10); fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    for ax in axes.ravel():
        if not ax.axison: continue
        ax.set_xlabel("%s 1" % kind, fontsize=9); ax.set_ylabel("%s 2" % kind, fontsize=9)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7.5, frameon=False, markerscale=3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    fig.suptitle(title, fontsize=11, fontweight="bold"); fig.tight_layout(); fig.savefig(path, dpi=170, facecolor="#fcfcfb"); plt.close(fig)


INSTRUMENT = {"isx": "ImageStream (60x)", "a8": "BD FACSDiscover A8"}


def galleries(store_path, tab, scores, labels, feature_set, out_dir, n=8, size=72, context=""):
    """One PNG per cluster: the n members nearest the cluster centre and the n farthest from
    it, each as brightfield with compartment outlines and, beneath it, the predicted stains
    overlaid in the IDEAS colours (one display range per marker across the gallery)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ..data.cache import open_cache
    from ..data.store import open_store
    from ..eval.masks import outline
    from ..eval.slides import window
    from ..features.engine import COMPARTMENT
    st = open_store(store_path); ds, _, _ = open_cache(st.meta["source_cache"], st.classes)
    bf_i = next((ds.roles.index(r) for r in ("brightfield", "lightloss") if r in ds.roles), 0)
    comp_of = {COMPARTMENT[nm]: k for k, nm in enumerate(st.names) if nm in COMPARTMENT}
    show = [c for c in SET_MASKS.get(feature_set, SET_MASKS["morph"]) if c in comp_of]
    idx_all = tab["id.index"].to_numpy(dtype=int)
    bf_name = {"brightfield": "brightfield", "lightloss": "light loss"}.get(ds.roles[bf_i], ds.roles[bf_i])
    for c in sorted(set(labels)):
        if c < 0: continue
        sel = np.flatnonzero(labels == c); centre = scores[sel].mean(0); d = np.linalg.norm(scores[sel] - centre, axis=1)
        near = sel[np.argsort(d)[:n]]; far = sel[np.argsort(-d)[:n]]
        members = [(int(idx_all[q]), g) for g, grp in (("nearest the centre", near), ("periphery", far)) for q in grp]
        items = {}
        for i, _ in members:
            x, y, m = ds.item(i); p, pm, _, _ = st.item(i)
            xw, _, mw, pw = window(x, y, m, np.concatenate([p, pm.astype(np.float32)]), size)
            items[i] = (xw, mw, pw[:st.cout], pw[st.cout:] > 0.5)
        hi = [max(float(np.percentile(np.concatenate([np.clip(items[i][2][k][items[i][1] > 0], 0, None) for i, _ in members]), 99.5)), 1e-4) for k in range(st.cout)]
        fig, axes = plt.subplots(4, n, figsize=(1.15 * n, 5.9), facecolor="white", squeeze=False)
        for r, (group, label) in enumerate(((near, "nearest the centre"), (far, "periphery"))):
            for j in range(n):
                ax_bf, ax_st = axes[2 * r, j], axes[2 * r + 1, j]
                for ax in (ax_bf, ax_st): ax.set_xticks([]); ax.set_yticks([])
                if j >= len(group): ax_bf.axis("off"); ax_st.axis("off"); continue
                i = int(idx_all[group[j]]); xw, mw, pw, masks = items[i]
                img = render.grey(xw[bf_i])
                for comp in show:
                    o = outline(masks[comp_of[comp]]); img[o] = MASK_COLOUR[comp]
                ax_bf.imshow(img, interpolation="nearest"); ax_bf.set_title("#%d" % i, fontsize=6, pad=2)
                layers = [render.colorize(np.clip(pw[k], 0, None), render.color_for(st.names[k], k), 0, hi[k]) for k in range(st.cout)]
                ax_st.imshow(render.composite(layers), interpolation="nearest")
                for ax in (ax_bf, ax_st):
                    for sp in ax.spines.values(): sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.5)
            axes[2 * r, 0].set_ylabel(label + "\n" + bf_name, fontsize=6.5); axes[2 * r + 1, 0].set_ylabel("virtual stains", fontsize=6.5)
        legend = "   ".join("%s %s" % (nm, {"LDHC/AKAP4": "green", "TOMM20": "red", "DAPI": "violet", "ACRV1": "yellow", "CD45": "magenta"}.get(nm, "")) for nm in st.names)
        inst = INSTRUMENT.get(st.meta["classes"][st.classes[0]]["instrument"], st.meta["classes"][st.classes[0]]["instrument"])
        fig.suptitle("%s  ·  cluster %d of %d  (n = %d of %d clustered)" % (inst, c, len([k for k in set(labels) if k >= 0]), len(sel), int((labels >= 0).sum())), fontsize=9.5, fontweight="bold", y=0.995)
        import textwrap
        width = int(18 * n)
        lines = textwrap.wrap(context, width) + textwrap.wrap("store %s  ·  classes %s  ·  run %s" % (store_path, ",".join(st.classes), os.path.basename(out_dir.rstrip("/"))), width)
        lines += textwrap.wrap("rows: %s with compartment outlines, then the virtual stains (%s); top pair nearest the cluster centre, bottom pair its periphery" % (bf_name, legend), width)
        fig.text(0.5, 0.962, "\n".join(lines), ha="center", va="top", fontsize=6.8, color="#52514e", linespacing=1.35)
        top = 0.962 - 0.028 * len(lines) - 0.01
        fig.tight_layout(rect=(0, 0, 1, top)); fig.savefig(os.path.join(out_dir, "gallery_%d.png" % c), dpi=200, facecolor="white"); plt.close(fig)
