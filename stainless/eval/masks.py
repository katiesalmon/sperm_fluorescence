"""Stain masks from a trained model: threshold each predicted channel into a binary mask
and score it against a reference mask, the way IDEAS draws a per-channel mask.

    python -m stainless masks runs/gen_joint --cache data/gen_cache --test 3 --isx data/isx/ALL_1_400_1.cif

Reference mask per object and channel:
  ISX  the IDEAS mask stored in the .cif for that channel (M02, M03, M07, M11).
  A8   no instrument mask exists, so the truth image is thresholded (mask_from_image).
       The same rule applied to ISX truth is scored against the IDEAS mask, which says how
       close that stand-in is to what IDEAS would draw.
Predicted mask: the prediction thresholded at one value per channel and instrument, chosen
on a sample of training objects to maximise mean IoU against the reference. Reports IoU,
Dice and detection agreement on the held-out set, and draws an outline sheet.
"""
import argparse, json, os

import numpy as np
from scipy import ndimage as ndi

from ..data.cache import open_cache
from ..data.prepare_generalist import ISX_TGT
from ..io import isx as isx_io
from . import render
from .load import load_run
from .metrics import evaluate

FRACS = np.array([0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
REL = np.array([0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])


def pred_mask(p, rule, valid=None):
    """Threshold one predicted channel. rule = ("abs", t): fixed value in cache units;
    ("rel", f): f x the object's own 99.5th percentile, which follows IDEAS in scaling the
    mask to each cell rather than to the population (a faint halo on a dim sperm head is
    below 20% of its peak, but above a population-wide cut)."""
    kind, v = rule
    thr = v if kind == "abs" else v * max(float(np.percentile(p[valid > 0] if valid is not None else p, 99.5)), 1e-6)
    m = p > thr
    return m & (valid > 0) if valid is not None else m


def mask_from_image(img, valid, sigma=1.0, k=5.0, rel=0.15, min_px=4, peak_frac=0.0):
    """Binary stain mask from one truth channel: smooth, threshold at the larger of k x the
    background noise and `rel` x the object's 99.5th percentile, drop specks, and drop any
    component whose peak is below `peak_frac` of the brightest component's (A8 noise)."""
    sm = ndi.gaussian_filter(img.astype(np.float32), sigma)
    v = sm[valid > 0]
    med = np.median(v); noise = 1.4826 * np.median(np.abs(v[v <= med] - med)) + 1e-6
    top = np.percentile(v, 99.5) - med
    thr = med + max(k * noise, rel * top)
    m = (sm > thr) & (valid > 0)
    lab, n = ndi.label(m)
    if n:
        ids = range(1, n + 1)
        sizes = ndi.sum(m, lab, ids); peaks = ndi.maximum(sm, lab, ids) - med
        keep = (sizes >= min_px) & (peaks >= peak_frac * peaks.max())
        m = np.isin(lab, 1 + np.flatnonzero(keep))
    return m


# per-instrument stand-in rule when no instrument mask exists (A8 is noisy, ISX is clean)
RULE = {"isx": dict(sigma=1.0, k=5.0, rel=0.15, min_px=4, peak_frac=0.0),
        "a8": dict(sigma=1.5, k=5.0, rel=0.25, min_px=12, peak_frac=0.4)}


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0   # both empty: agree


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def ideas_mask(cif, idx, ch, shape):
    """IDEAS mask for channel `ch` of object `idx`, placed in the cache frame (to_multiple pads
    the object centrally up to a multiple of 8)."""
    mk = cif.mask(idx)[ch] > 0
    H, W = shape; h, w = mk.shape
    out = np.zeros((H, W), bool); bh, bw = (H - h) // 2, (W - w) // 2
    out[bh:bh + h, bw:bw + w] = mk
    return out


def outline(m):
    return m & ~ndi.binary_erosion(m)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stainless masks")
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--include", default=None); ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--isx", default=None, help=".cif the ISX classes were cut from, for the IDEAS masks")
    ap.add_argument("--limit", type=int, default=600, help="held-out objects per class to score")
    ap.add_argument("--calib", type=int, default=200, help="training objects per class to pick thresholds on")
    ap.add_argument("--per-well", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint); model, dev, scale, te, wte, te_cls, res, names = R.model, R.dev, R.scale, R.te, R.wte, R.te_cls, R.res, R.names
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    inst_of = {c: v["instrument"] for c, v in meta["classes"].items()}
    cif = isx_io.CIF(a.isx) if a.isx else None
    rng = np.random.default_rng(a.seed)

    def source_index(cls, row):
        # prepare_generalist cuts the ISX into first/second half; A8 rows are the bundle order
        return int(row) if cls == "2I" else len(cif) // 2 + int(row)

    def reference(ds, cls, i, k, y, valid):
        if inst_of[cls] == "isx" and cif is not None:
            return ideas_mask(cif, source_index(cls, i - ds_offset(ds, cls)), ISX_TGT[k], y.shape[1:])
        return mask_from_image(y[k], valid, **RULE[inst_of[cls]])

    def ds_offset(ds, cls):
        return int(np.flatnonzero(ds.classes == cls)[0])

    def collect(ds, wells, pick):
        """Predictions and per-channel reference masks for the objects `pick`."""
        _, P = evaluate(model, ds, wells, scale, dev, names, idx=np.array(pick))
        out = []
        for j, i in enumerate(pick):
            x, y, m = ds.item(i); cls = str(ds.classes[i]); av = ds.avail[i] if ds.avail is not None else np.ones(ds.cout, bool)
            refs = {k: reference(ds, cls, i, k, y, m) for k in range(ds.cout) if av[k]}
            alt = {k: mask_from_image(y[k], m, **RULE["a8"]) for k in refs} if inst_of[cls] == "isx" and cif is not None else None
            out.append((cls, x, y, m, P[j], refs, alt))
        return out

    # ---- thresholds, from training objects ----
    tr_cls = [c for c in res["train"]]
    tr, wtr, _ = open_cache(a.cache, tr_cls)
    pick = [int(i) for c in tr_cls for i in rng.choice(np.flatnonzero(tr.classes == c), min(a.calib, (tr.classes == c).sum()), replace=False)]
    cal = collect(tr, wtr[pick], pick)
    unit = scale.view(-1).numpy() / 10.0             # the channel's mean |y| in cache units
    thr = {}
    for inst in sorted(set(inst_of[c] for c in tr_cls)):
        rows = [r for r in cal if inst_of[r[0]] == inst]
        for k in range(tr.cout):
            if not any(k in r[5] for r in rows): continue
            rk = [r for r in rows if k in r[5]]
            cands = [("abs", float(f * unit[k])) for f in FRACS] + [("rel", float(f)) for f in REL]
            scored = [(np.mean([iou(pred_mask(r[4][k], c, r[3]), r[5][k]) for r in rk]), c) for c in cands]
            thr[(inst, k)] = max(scored, key=lambda t: t[0])[1]
    print("thresholds: abs = fraction of the channel's mean |y|, rel = fraction of the object's own peak")
    for (inst, k), (kind, v) in sorted(thr.items()):
        print("  %-4s %-11s %s %.2f" % (inst, names[k], kind, v / unit[k] if kind == "abs" else v))

    # ---- held-out score ----
    pick = [int(i) for c in te_cls for i in rng.choice(np.flatnonzero(te.classes == c), min(a.limit, (te.classes == c).sum()), replace=False)]
    held = collect(te, wte[pick], pick)
    print("\nheld-out stain masks (IoU / Dice are per-object means; detect = both say present or both say absent):")
    print("%-4s %-11s %6s %6s %6s %7s   %s" % ("inst", "marker", "n", "IoU", "Dice", "detect", "truth-rule vs IDEAS IoU (ISX only)"))
    table = {}
    for inst in sorted(set(inst_of[c] for c in te_cls)):
        rows = [r for r in held if inst_of[r[0]] == inst]
        for k in range(te.cout):
            rk = [r for r in rows if k in r[5]]
            if not rk: continue
            pm = [pred_mask(r[4][k], thr[(inst, k)], r[3]) for r in rk]; rm = [r[5][k] for r in rk]
            ious = [iou(p, t) for p, t in zip(pm, rm)]; dcs = [dice(p, t) for p, t in zip(pm, rm)]
            det = float(np.mean([(p.any() == t.any()) for p, t in zip(pm, rm)]))
            alt = [iou(r[6][k], r[5][k]) for r in rk if r[6] is not None]
            table[(inst, names[k])] = {"n": len(rk), "iou": float(np.mean(ious)), "dice": float(np.mean(dcs)), "detect": det,
                                         "truth_rule_vs_ideas_iou": float(np.mean(alt)) if alt else None, "threshold": list(thr[(inst, k)])}
            print("%-4s %-11s %6d %6.3f %6.3f %7.3f   %s" % (inst, names[k], len(rk), np.mean(ious), np.mean(dcs), det, ("%.3f" % np.mean(alt)) if alt else ""))
    json.dump({"thresholds": {"%s:%s" % (i, names[k]): list(t) for (i, k), t in thr.items()}, "held_out": {"%s:%s" % k: v for k, v in table.items()}},
              open(os.path.join(a.run, "masks.json"), "w"), indent=1)

    # ---- sheet: BF with outlines, then per marker the reference and predicted masks ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = []
    for w in sorted(set(wte[pick])):
        cand = [r for r, ww in zip(held, wte[pick]) if ww == w]
        show += cand[:a.per_well]
    shown_k = sorted({k for r in show for k in r[5]})
    cols = 1 + 2 * len(shown_k) + 1
    fig, axes = plt.subplots(len(show), cols, figsize=(1.55 * cols, 1.9 * len(show)), facecolor="black", squeeze=False)
    for i, (cls, x, y, m, p, refs, _) in enumerate(show):
        inst = inst_of[cls]; bf = render.grey(x[0]).copy()
        panels = []
        for k in shown_k:
            c = np.asarray(render.color_for(names[k], k), np.float32)
            if k not in refs:
                panels += [np.zeros(bf.shape, np.float32)] * 2; continue
            t = refs[k]; q = pred_mask(p[k], thr[(inst, k)], m)
            lo, hi = 0.0, max(float(np.percentile(y[k][m > 0], 99.5)), 1e-4)
            # same convention everywhere: white = reference mask, bright colour = predicted mask
            tp = render.colorize(y[k], c, lo, hi) * 0.6; tp[outline(t)] = 1.0
            pp = render.colorize(np.clip(p[k], 0, None), c, lo, hi) * 0.6; pp[outline(t)] = 1.0; pp[outline(q)] = np.minimum(c + 0.25, 1.0)
            panels += [tp, pp]
            bf[outline(t)] = 1.0; bf[outline(q)] = c
        panels = [bf] + panels + [render.grey(x[0])]
        for c_, (ax, img) in enumerate(zip(axes[i], panels)):
            ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
            if i == 0:
                heads = ["BF + outlines"] + [h % names[k] for k in shown_k for h in ("%s mask", "%s mask pred")] + ["BF"]
                ax.set_title(heads[c_], fontsize=7, color="white")
        axes[i, 0].set_ylabel("%s (%s)" % (cls.lstrip("123"), inst), fontsize=7, color="white")
    fig.suptitle("%s: stain masks. white = reference (IDEAS on ISX, thresholded truth on A8), colour = predicted" % os.path.basename(a.run.rstrip("/")),
                 fontsize=9, color="white")
    fig.tight_layout(); out = os.path.join(a.run, "masks.png"); fig.savefig(out, dpi=110, facecolor="black"); plt.close(fig)
    print("wrote", os.path.join(a.run, "masks.json"), "and", out)


if __name__ == "__main__":
    main()
