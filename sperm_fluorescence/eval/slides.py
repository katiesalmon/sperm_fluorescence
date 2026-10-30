"""Presentation figures: one object, its label-free input, and for every marker the true
stain beside the prediction, with the all-channel overlay at the end.

    python -m sperm_fluorescence slides runs/gen_isx_only --cache data/gen_cache --test 3 --include 2I,3I \\
        --title "ImageStream (60x)" --input-name "Brightfield" --um-per-px 0.333 --out figures/isx
    python -m sperm_fluorescence slides runs/gen_a8_scratch --cache data/gen_cache --test 3 --include 2S,2P,2SP,3S,3P,3SP \\
        --title "BD FACSDiscover A8" --input-name "Light loss" --um-per-px 0.333 --approx-scale --out figures/a8

Candidates are held-out objects whose truth is clearly stained on every available marker
(per-channel event intensity above the median of the sample) and whose per-object pixel r,
averaged over markers, is in the top of that set -- representative good cases, chosen by
a stated rule rather than by eye. Writes one PNG per candidate and a stacked version.
"""
import argparse, os

import numpy as np

from . import render
from .load import load_run
from .metrics import evaluate


def crop(ds, i):
    """The object at its own size: the cache pads centrally up to a multiple of 8."""
    x, y, m = ds.item(i); h, w = ds.shape(i); H, W = m.shape
    t, l = (H - h) // 2, (W - w) // 2
    return x[:, t:t + h, l:l + w], y[:, t:t + h, l:l + w], m[t:t + h, l:l + w]


def window(x, y, m, p, size):
    """Crop every panel to a size x size window centred on the stained object (centre of
    mass of the summed truth), kept inside the frame. For the A8, whose frame is a wide
    strip of stream with a small cell in it."""
    from scipy import ndimage as ndi
    tot = ndi.gaussian_filter(np.abs(y).sum(axis=0) * m, 2.0); tot = np.clip(tot - np.median(tot), 0, None)
    h, w = m.shape; cy, cx = ndi.center_of_mass(tot) if tot.sum() > 0 else (h / 2, w / 2)
    sh, sw = min(size, h), min(size, w)
    t = int(np.clip(round(cy - sh / 2), 0, h - sh)); l = int(np.clip(round(cx - sw / 2), 0, w - sw))
    return x[:, t:t + sh, l:l + sw], y[:, t:t + sh, l:l + sw], m[t:t + sh, l:l + sw], p[:, t:t + sh, l:l + sw]


def pick(ds, wells, P, idx, n, want_well=None, population="any"):
    """Indices into `idx` of the best candidates by the rule in the module docstring."""
    av = ds.avail if ds.avail is not None else np.ones((len(ds), ds.cout), bool)
    ev = np.full((len(idx), ds.cout), np.nan); r = np.full((len(idx), ds.cout), np.nan)
    for q, i in enumerate(idx):
        _, y, m = crop(ds, i)
        for k in range(ds.cout):
            if not av[i, k]: continue
            ev[q, k] = (y[k] * m).sum()
            a, b = P[q][k][m > 0], y[k][m > 0]
            r[q, k] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
    ok = np.ones(len(idx), bool)
    if want_well is not None: ok &= wells == want_well
    if population != "any":
        # ISX only: the DAPI scalar splits the tube into DAPI-low fragments (~38%), sperm,
        # and a DAPI-bright round population (4-6%, > 4x the sperm median; likely immature
        # germ cells). See results.md.
        d = np.asarray(ds.scalars).reshape(len(ds.scalars), -1)[idx, 0]; sperm_med = np.nanmedian(d[d > np.nanpercentile(d, 45)])
        if population == "round": ok &= d > 4 * sperm_med
        elif population == "sperm": ok &= (d > 0.25 * sperm_med) & (d < 4 * sperm_med)
        elif population == "dim": ok &= d < 0.25 * sperm_med
    # the stained object must sit inside the frame: a truth that runs off the edge makes a
    # confusing slide (a tail cut at the border looks like a prediction error)
    for q, i in enumerate(idx):
        _, y, m = crop(ds, i); tot = np.abs(y).sum(axis=0) * m; b = 3
        edge = tot[:b].sum() + tot[-b:].sum() + tot[b:-b, :b].sum() + tot[b:-b, -b:].sum()
        if edge > 0.04 * tot.sum(): ok[q] = False
    med = np.nanmedian(ev[ok], axis=0)
    for k in range(ds.cout):
        if np.isfinite(med[k]): ok &= np.isnan(ev[:, k]) | (ev[:, k] > med[k])
    score = np.nanmean(r, axis=1); score[~ok] = -np.inf
    order = np.argsort(-score)
    return [int(q) for q in order[:n] if np.isfinite(score[q])], r


def draw(path, objs, names, avail_k, title, input_name, um_per_px, approx, rs):
    """objs: list of (x, y, m, p). One block of rows per object: top row true, bottom row
    predicted; columns = input, markers, overlay."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import patches
    ks = [k for k in range(len(names)) if avail_k[k]]
    ncol = 2 + len(ks); nrow = 2 * len(objs)
    pw = 2.2; aspect = max(x.shape[1] / x.shape[2] for x, _, _, _ in objs); ph = pw * min(max(aspect, 0.4), 1.6)
    head, foot, left = 1.25, 0.12, 0.6
    figw, figh = pw * ncol + left + 0.1, ph * nrow + head + foot
    fig = plt.figure(figsize=(figw, figh), facecolor="white")
    gs = fig.add_gridspec(nrow, ncol, left=left / figw, right=1 - 0.1 / figw, top=1 - head / figh, bottom=foot / figh, wspace=0.06, hspace=0.08)
    for o, (x, y, m, p) in enumerate(objs):
        rng_ = render.ranges([y], [m])
        bf = render.grey(x[0])
        tl, pl = [], []
        for k in ks:
            c = render.color_for(names[k], k); lo, hi = rng_[k]
            tl.append(render.colorize(y[k], c, lo, hi)); pl.append(render.colorize(np.clip(p[k], 0, None), c, lo, hi))
        rows = [[bf] + tl + [render.composite(tl)], [bf] + pl + [render.composite(pl)]]
        for rr, row in enumerate(rows):
            for cc, img in enumerate(row):
                ax = fig.add_subplot(gs[2 * o + rr, cc]); ax.set_facecolor("black")
                ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.8)
                if o == 0 and rr == 0:
                    if cc == 0: head, col = "Input: " + input_name, "black"
                    elif cc == ncol - 1: head, col = "All markers", "black"
                    else: head, col = names[ks[cc - 1]], tuple(0.75 * v for v in render.color_for(names[ks[cc - 1]], ks[cc - 1]))
                    ax.set_title(head, fontsize=13, fontweight="bold", color=col, pad=6)
                if cc == 0:
                    ax.set_ylabel("True stain" if rr == 0 else "Predicted", fontsize=13, fontweight="bold", labelpad=8)
                    if rr == 0 and um_per_px:
                        px = 5.0 / um_per_px; h, w = x.shape[1:]
                        ax.add_patch(patches.Rectangle((w - px - 4, h - 6), px, max(h / 60.0, 1.2), color="white"))
                        ax.text(w - px / 2 - 4, h - 8, ("≈" if approx else "") + "5 µm", color="white", fontsize=8, ha="center", va="bottom",
                                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
                if cc > 0 and cc < ncol - 1 and rr == 1 and rs is not None:
                    ax.text(0.03, 0.04, "r = %.2f" % rs[o][ks[cc - 1]], transform=ax.transAxes, color="white", fontsize=8, va="bottom")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1 - 0.12 / figh, va="top")
    fig.text(left / figw, 1 - 0.50 / figh, "Top row: measured fluorescence.   Bottom row: predicted from the label-free input alone.", fontsize=10, color="#444444", va="top")
    fig.text(left / figw, 1 - 0.70 / figh, "Each marker on its own measured intensity range.   r = per-pixel correlation with the measurement, within the cell.", fontsize=10, color="#444444", va="top")
    fig.savefig(path, dpi=250, facecolor="white"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence slides")
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--include", default=None); ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--denoise-measured", default="none", choices=["none", "floor", "smooth"], help="read the measured stains denoised (noise floor removed; smooth = 1-px Gaussian first), so the prediction is scored and shown against the stain without its read noise")
    ap.add_argument("--title", required=True); ap.add_argument("--input-name", required=True)
    ap.add_argument("--um-per-px", type=float, default=0.0); ap.add_argument("--approx-scale", action="store_true")
    ap.add_argument("--well", default=None, help="restrict candidates to one well type (e.g. S)")
    ap.add_argument("--crop", type=int, default=0, help="show a square window of this many px around the cell")
    ap.add_argument("--population", default="any", choices=["any", "sperm", "round", "dim"], help="ISX: pick by the DAPI scalar")
    ap.add_argument("--sample", type=int, default=600); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint, denoise=a.denoise_measured); model, dev, scale, te, wte, names = R.model, R.dev, R.scale, R.te, R.wte, R.names
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(te), min(a.sample, len(te)), replace=False))
    _, P = evaluate(model, te, wte[idx], scale, dev, names, idx=idx)
    best, r = pick(te, wte[idx], P, idx, a.n, a.well, a.population)
    os.makedirs(a.out, exist_ok=True)
    avail_k = te.avail[idx[best[0]]] if te.avail is not None else np.ones(te.cout, bool)
    objs, rs = [], []
    for q in best:
        x, y, m = crop(te, idx[q]); pq = P[q]
        if a.crop: x, y, m, pq = window(x, y, m, pq, a.crop)
        objs.append((x, y, m, pq)); rs.append(r[q])
        print("candidate %d: object %d well %s  per-marker pixel r %s" % (len(objs), idx[q], wte[idx[q]],
              "  ".join("%s %.2f" % (names[k], r[q][k]) for k in range(te.cout) if avail_k[k])))
    for n_, (ob, rr) in enumerate(zip(objs, rs), 1):
        draw(os.path.join(a.out, "candidate_%d.png" % n_), [ob], names, avail_k, a.title, a.input_name, a.um_per_px, a.approx_scale, [rr])
    draw(os.path.join(a.out, "stacked.png"), objs, names, avail_k, a.title, a.input_name, a.um_per_px, a.approx_scale, rs)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
