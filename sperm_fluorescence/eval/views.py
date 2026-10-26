"""Inputs beside outputs for a modular (stems) model: every label-free view the object has,
then for each marker the measured stain and the prediction under each input subset.

    python -m sperm_fluorescence views runs/abl_dropout --cache data/a8_roles --test 3 \\
        --subsets "lightloss,fsc,ssc; lightloss; ssc" --per-well 2 --crop 56 --out figures/views_dropout.png
"""
import argparse
import os

import numpy as np

from . import render
from .load import load_run
from .metrics import evaluate
from .slides import crop as crop_obj, window

ROLE_LABEL = {"lightloss": "light loss", "fsc": "FSC", "ssc": "SSC", "brightfield": "brightfield", "brightfield2": "brightfield 2", "darkfield": "darkfield"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence views", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--include", default=None); ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--subsets", required=True, help="semicolon-separated role lists; the first is drawn as the main prediction")
    ap.add_argument("--per-well", type=int, default=2); ap.add_argument("--crop", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint); te, wte, names, roles = R.te, R.wte, R.names, list(R.te.roles)
    subsets = [[r.strip() for r in s.split(",")] for s in a.subsets.split(";") if s.strip()]
    rng = np.random.default_rng(a.seed)
    pick = []
    for w in sorted(set(wte)):
        idx = np.flatnonzero(wte == w)
        # objects with a clear stain on every available marker: above the median per-event sum
        sums = np.array([[(te.item(i)[1][k] * te.item(i)[2]).sum() for k in range(te.cout)] for i in idx[:400]])
        ok = np.all(sums >= np.median(sums, axis=0), axis=1); cand = idx[:400][ok]
        pick += [int(i) for i in rng.choice(cand, min(a.per_well, len(cand)), replace=False)]
    preds = {}
    for sub in subsets:
        mask = np.array([r in sub for r in roles], bool)
        _, P = evaluate(R.model, te, wte[pick], R.scale, R.dev, names, idx=np.array(pick), roles_mask=mask)
        preds[",".join(sub)] = P
    av = te.avail[pick] if te.avail is not None else np.ones((len(pick), te.cout), bool)
    ks = [k for k in range(te.cout) if av[:, k].any()]
    cols = len(roles) + len(ks) * (1 + len(subsets))
    fig, axes = plt.subplots(len(pick), cols, figsize=(1.5 * cols, 1.7 * len(pick) + 0.7), facecolor="black", squeeze=False)
    for i, obj in enumerate(pick):
        x, y, m = crop_obj(te, obj)
        ps = {s: P[i] for s, P in preds.items()}
        if a.crop:
            x, y, m, p0 = window(x, y, m, ps[",".join(subsets[0])], a.crop)
            # the same window for every subset's prediction
            ref = crop_obj(te, obj); full = ref[2].shape
            for s in ps:
                xx, yy, mm, ps[s] = window(ref[0], ref[1], ref[2], ps[s], a.crop)
        rng_ = render.ranges([y], [m]); c_ = 0
        for r_i, r in enumerate(roles):
            ax = axes[i, c_]; ax.imshow(render.grey(x[r_i]), interpolation="nearest"); c_ += 1
            if i == 0: ax.set_title("input: " + ROLE_LABEL.get(r, r), fontsize=7.5, color="white")
        for k in ks:
            c = render.color_for(names[k], k); lo, hi = rng_[k]
            ax = axes[i, c_]; ax.imshow(render.colorize(y[k], c, lo, hi), interpolation="nearest"); c_ += 1
            if i == 0: ax.set_title("%s\nmeasured" % names[k], fontsize=7.5, color="white")
            for sub in subsets:
                key = ",".join(sub); ax = axes[i, c_]; ax.imshow(render.colorize(np.clip(ps[key][k], 0, None), c, lo, hi), interpolation="nearest"); c_ += 1
                if i == 0: ax.set_title("predicted from\n" + " + ".join(ROLE_LABEL.get(r, r) for r in sub), fontsize=7, color="white")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
        axes[i, 0].set_ylabel("%s well" % wte[obj], fontsize=7, color="white")
    fig.suptitle("%s: the same model, the same objects, fewer input views" % os.path.basename(a.run.rstrip("/")), fontsize=9, color="white")
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=150, facecolor="black"); plt.close(fig)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
