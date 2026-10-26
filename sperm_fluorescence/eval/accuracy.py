"""Accuracy figures for a finished run, from its checkpoint, on held-out objects.

    python -m sperm_fluorescence accuracy runs/gen_isx_only --cache data/gen_cache --test 3 --include 2I,3I \\
        --title "ImageStream, held-out half of the tube" --sample 3000 --out figures/acc_isx
    python -m sperm_fluorescence accuracy runs/gen_a8_scratch --cache data/gen_cache --test 3 \\
        --include 2S,2P,2SP,3S,3P,3SP --title "BD FACSDiscover A8, held-out replicate" --out figures/acc_a8

Writes
  scatter.png   per marker: predicted vs measured per-event intensity, one point per object,
                coloured by group (ISX: sperm / round / dim by the DAPI scalar; A8: well type),
                with Pearson r over all objects and within each group.
  event_r.png   the same r values as grouped bars.
  masks.png     mask IoU per marker from the run's masks*.json, if present, with the
                thresholded-truth-vs-IDEAS ceiling where it exists.
"""
import argparse, glob, json, os

import numpy as np

from . import render
from .load import load_run
from .metrics import evaluate

GROUP_COLOURS = {"sperm": "#1f77b4", "round": "#d62728", "dim": "#9e9e9e", "S": "#1f77b4", "P": "#d62728", "SP": "#2ca02c", "I": "#1f77b4"}
GROUP_NAMES = {"sperm": "sperm", "round": "round cells (DAPI-bright)", "dim": "DAPI-dim fragments",
               "S": "sperm well (S)", "P": "PBMC well (P)", "SP": "mixed well (SP)", "I": "all objects"}


def groups_for(ds, wells, idx):
    """A label per object: ISX by the DAPI scalar, A8 by well type."""
    g = np.array(wells[idx], dtype=object)
    scal = np.asarray(ds.scalars).reshape(len(ds.scalars), -1)[:, 0]
    isx = np.isfinite(scal[idx])
    if isx.any():
        d = scal[idx]; sperm_med = np.nanmedian(d[isx & (d > np.nanpercentile(d[isx], 45))])
        g[isx & (d < 0.25 * sperm_med)] = "dim"; g[isx & (d >= 0.25 * sperm_med) & (d <= 4 * sperm_med)] = "sperm"; g[isx & (d > 4 * sperm_med)] = "round"
    return g


def r_of(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence accuracy")
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--include", default=None); ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--title", required=True); ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint); model, dev, scale, te, wte, res, names = R.model, R.dev, R.scale, R.te, R.wte, R.res, R.names
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(te), min(a.sample, len(te)), replace=False))
    _, P = evaluate(model, te, wte[idx], scale, dev, names, idx=idx)
    av = te.avail[idx] if te.avail is not None else np.ones((len(idx), te.cout), bool)
    ks = [k for k in range(te.cout) if av[:, k].any()]
    ev_t = np.full((len(idx), te.cout), np.nan); ev_p = np.full((len(idx), te.cout), np.nan)
    for q, i in enumerate(idx):
        _, y, m = te.item(i)
        for k in ks:
            if av[q, k]: ev_t[q, k] = (y[k] * m).sum(); ev_p[q, k] = (P[q][k] * m).sum()
    g = groups_for(te, wte, idx)
    order = [x for x in ["sperm", "round", "dim", "S", "P", "SP", "I"] if x in set(g)]
    os.makedirs(a.out, exist_ok=True)

    # ---- scatter ----
    n = len(ks); fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 0.4, 5.3), facecolor="white", squeeze=False)
    rtab = {}
    for ax, k in zip(axes[0], ks):
        ok = np.isfinite(ev_t[:, k]); t, p = ev_t[ok, k], ev_p[ok, k]; gg = g[ok]
        main = gg == ("sperm" if "sperm" in order else order[0])
        lim = 1.15 * np.percentile(t[main] if main.sum() > 10 else t, 99.5); lo = min(0.0, np.percentile(t, 0.5))   # axis on the main population; outliers run off
        for grp in order:
            sel = gg == grp
            if sel.any(): ax.scatter(t[sel], p[sel], s=5, alpha=0.45, color=GROUP_COLOURS[grp], label=GROUP_NAMES[grp], linewidths=0, rasterized=True)
        ax.plot([lo, lim], [lo, lim], color="black", lw=0.8, ls="--")
        ax.set_xlim(lo, lim); ax.set_ylim(lo, lim); ax.set_aspect("equal")
        c = render.color_for(names[k], k); ax.set_title(names[k], fontsize=14, fontweight="bold", color=tuple(0.75 * v for v in c))
        ax.set_xlabel("measured intensity per event", fontsize=10); ax.set_ylabel("predicted intensity per event", fontsize=10)
        ax.tick_params(labelsize=8); ax.ticklabel_format(style="sci", scilimits=(-2, 3))
        lines = ["r = %.2f  (all, n=%d)" % (r_of(t, p), len(t))]; rtab[names[k]] = {"all": r_of(t, p)}
        for grp in order:
            sel = gg == grp
            if sel.sum() > 10:
                rr = r_of(t[sel], p[sel]); rtab[names[k]][grp] = rr; lines.append("r = %.2f  within %s" % (rr, GROUP_NAMES[grp]))
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc", pad=4))
        for sp in ax.spines.values(): sp.set_color("#888888")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=10, frameon=False, markerscale=4, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(a.title + " — per-event intensity, predicted vs measured", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.9)); fig.savefig(os.path.join(a.out, "scatter.png"), dpi=220, facecolor="white"); plt.close(fig)

    # ---- event r bars ----
    cols = ["all"] + [x for x in order if any(x in rtab[names[k]] for k in ks)]
    fig, ax = plt.subplots(figsize=(1.6 * len(ks) * len(cols) / 2 + 3, 4.4), facecolor="white")
    w = 0.8 / len(cols); xs = np.arange(len(ks))
    for j, grp in enumerate(cols):
        vals = [rtab[names[k]].get(grp, np.nan) for k in ks]
        colour = "#444444" if grp == "all" else GROUP_COLOURS[grp]
        bars = ax.bar(xs + (j - (len(cols) - 1) / 2) * w, vals, w, color=colour, label="all objects" if grp == "all" else "within " + GROUP_NAMES[grp], edgecolor="white")
        for b, v in zip(bars, vals):
            if np.isfinite(v): ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.015, "%.2f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([names[k] for k in ks], fontsize=11)
    for lab, k in zip(ax.get_xticklabels(), ks): lab.set_color(tuple(0.7 * v for v in render.color_for(names[k], k))); lab.set_fontweight("bold")
    ax.set_ylim(min(0, np.nanmin([v for d in rtab.values() for v in d.values()]) - 0.05), 1.05); ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Pearson r, per-event intensity", fontsize=11); ax.legend(fontsize=9, frameon=False, loc="upper right" if len(cols) < 4 else "lower right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(a.title + " — event-level accuracy", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "event_r.png"), dpi=220, facecolor="white"); plt.close(fig)

    # ---- mask IoU bars ----
    mfiles = sorted(glob.glob(os.path.join(a.run, "masks*.json")))
    if mfiles:
        rows = {}
        for f in mfiles:
            for key, v in json.load(open(f))["held_out"].items():
                inst, nm = key.split(":"); rows.setdefault(inst, {})[nm] = v
        inst = next(i for i in rows if any(names[k] in rows[i] for k in ks))
        names = [names[k] for k in ks if names[k] in rows[inst]]
        fig, ax = plt.subplots(figsize=(1.5 * len(names) + 3.5, 4.6), facecolor="white")
        xs = np.arange(len(names)); iou = [rows[inst][n]["iou"] for n in names]; dc = [rows[inst][n]["dice"] for n in names]
        b1 = ax.bar(xs - 0.2, iou, 0.38, color="#1f77b4", label="IoU, predicted mask vs reference"); b2 = ax.bar(xs + 0.2, dc, 0.38, color="#9ecae1", label="Dice")
        for bb, vv in list(zip(b1, iou)) + list(zip(b2, dc)): ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.01, "%.2f" % vv, ha="center", va="bottom", fontsize=8)
        ceil = [rows[inst][n].get("truth_rule_vs_ideas_iou") for n in names]
        if any(c is not None for c in ceil):
            ax.scatter(xs - 0.2, [c if c is not None else np.nan for c in ceil], marker="_", s=600, color="black", lw=2, zorder=5, label="ceiling: a threshold of the measured stain vs IDEAS")
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=11)
        for lab, n in zip(ax.get_xticklabels(), names): lab.set_color(tuple(0.7 * v for v in render.color_for(n))); lab.set_fontweight("bold")
        ax.set_ylim(0, 1.0); ax.set_ylabel("overlap with reference mask", fontsize=11)
        ref = "the IDEAS per-channel mask" if inst == "isx" else "a threshold of the measured stain\n(the A8 stores no mask)"
        ax.set_title(a.title + " — stain masks\nreference = " + ref, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, frameon=False, loc="lower left"); [ax.spines[sp].set_visible(False) for sp in ("top", "right")]
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "masks.png"), dpi=220, facecolor="white"); plt.close(fig)
    json.dump(rtab, open(os.path.join(a.out, "event_r.json"), "w"), indent=1)
    print("wrote", a.out, "| event r:", {k: {g_: round(v, 3) for g_, v in d.items()} for k, d in rtab.items()})


if __name__ == "__main__":
    main()
