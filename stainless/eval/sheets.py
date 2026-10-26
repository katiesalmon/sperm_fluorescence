"""IDEAS-style sheets: per-channel colour on black, true beside predicted, additive overlays."""
import os

import numpy as np

from . import render


def sheet(path, title, x, y, p, m, labels, names, avail=None):
    """IDEAS-style sheet. Rows = objects. Columns: brightfield input; for each marker its
    true and predicted image in that marker's colour on the truth's display range; then
    the all-channel overlay, true and predicted, composited additively.
    `avail` (n, C) bool: channels an object's instrument did not stain are left out of
    its overlay and drawn blank; a channel no row carries gets no column at all."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n, cout = len(x), y.shape[1]
    if avail is None: avail = np.ones((n, cout), bool)
    avail = np.asarray(avail, bool)
    shown = [k for k in range(cout) if avail[:, k].any()]
    rng_ = render.ranges([y[i] for i in range(n)], [m[i] for i in range(n)])
    cols = 1 + 2 * len(shown) + 2
    fig, axes = plt.subplots(n, cols, figsize=(1.55 * cols, 1.9 * n), facecolor="black", squeeze=False)
    blank = np.zeros(y.shape[2:] + (3,), np.float32)
    for i in range(n):
        tl, pl = [], []
        panels = [render.grey(x[i, 0])]
        for k in shown:
            nm = names[k]
            if not avail[i, k]:
                panels += [blank, blank]; continue
            c = render.color_for(nm, k); lo, hi = rng_[k]
            t = render.colorize(y[i, k], c, lo, hi); q = render.colorize(np.clip(p[i, k], 0, None), c, lo, hi)
            panels += [t, q]; tl.append(t); pl.append(q)
        panels += [render.composite(tl) if tl else blank, render.composite(pl) if pl else blank]
        for c_, (ax, img) in enumerate(zip(axes[i], panels)):
            ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
            if i == 0:
                heads = ["BF"] + [h % names[k] for k in shown for h in ("%s", "%s pred")] + ["overlay", "overlay pred"]
                ax.set_title(heads[c_], fontsize=7, color="white")
        axes[i, 0].set_ylabel(str(labels[i]), fontsize=7, color="white")
    fig.suptitle(title, fontsize=9, color="white"); fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black"); plt.close(fig)


def snapshot_sheet(out, ep, x, y, p, m, wells, names, avail=None):
    sheet(os.path.join(out, "epoch_%02d.png" % ep), "epoch %d" % ep, x, y, p, m, wells, names, avail)


def progression_sheet(out, x, y, preds, m, names):
    """Per marker: rows = objects, columns = BF, truth, prediction at each snapshot epoch,
    in the marker's colour on the truth's range. Plus one sheet for the all-channel
    overlay: truth, then the predicted overlay at each epoch."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    eps = sorted(preds); n, cout = len(x), y.shape[1]
    rng_ = render.ranges([y[i] for i in range(n)], [m[i] for i in range(n)])
    def draw(path, title, true_fn, pred_fn):
        fig, axes = plt.subplots(n, 2 + len(eps), figsize=(1.5 * (2 + len(eps)), 1.8 * n), facecolor="black")
        for i in range(n):
            axes[i, 0].imshow(render.grey(x[i, 0])); axes[i, 1].imshow(true_fn(i))
            for j, ep in enumerate(eps):
                axes[i, 2 + j].imshow(pred_fn(i, ep))
                if i == 0: axes[i, 2 + j].set_title("ep %d" % ep, fontsize=8, color="white")
            if i == 0: axes[i, 0].set_title("BF", fontsize=8, color="white"); axes[i, 1].set_title("true", fontsize=8, color="white")
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_edgecolor("#444")
        fig.suptitle(title, fontsize=9, color="white"); fig.tight_layout(); fig.savefig(path, dpi=100, facecolor="black"); plt.close(fig)
    for k, nm in enumerate(names):
        c = render.color_for(nm, k); lo, hi = rng_[k]
        draw(os.path.join(out, "progression_%s.png" % nm.replace("/", "_")), "%s -- prediction by epoch" % nm,
             lambda i: render.colorize(y[i, k], c, lo, hi),
             lambda i, ep: render.colorize(np.clip(preds[ep][i, k], 0, None), c, lo, hi))
    if cout > 1:
        draw(os.path.join(out, "progression_overlay.png"), "all-channel overlay -- prediction by epoch",
             lambda i: render.composite([render.colorize(y[i, k], render.color_for(nm, k), *rng_[k]) for k, nm in enumerate(names)]),
             lambda i, ep: render.composite([render.colorize(np.clip(preds[ep][i, k], 0, None), render.color_for(nm, k), *rng_[k]) for k, nm in enumerate(names)]))

