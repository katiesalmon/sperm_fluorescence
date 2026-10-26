#!/usr/bin/env python
"""Per-pixel linear floor for A8 virtual staining: do the label-free pixels carry the stain?

Before a network earns any credit, ask the cheapest possible version of the question. For
each fluorescence page, fit a ridge regression from a 5x5 patch of the three label-free
pages (75 features) to the unmixed fluorescence value at the centre pixel. Train on
replicate 2, test on replicate 3. Report, per marker:

  pixel r    Pearson between predicted and true pixels, pooled over test events
  event r    Pearson between predicted and true per-event sums -- the scalar view
  P:S ratio  median predicted page sum on PBMC-well events divided by sperm-well events,
             in the stained test replicate. ACRV1 and LDHC should be LOW on PBMCs (no
             acrosome, no flagellum) and CD45 should be HIGH -- so the ratio should be
             well below 1 for the sperm markers and well above 1 for CD45.

Replicate 1 (unstained) is deliberately NOT a control here: a correct virtual-stain model
should predict the stain on an unstained cell, since the label-free pages cannot know the
tube was not stained. Cell type is the control; staining status is not.

A linear model on local patches cannot know where it is on the cell, so this is a floor
that any convolutional model must clear. If it is already high, the stain is locally
legible in the label-free pages; if it is near zero, the information is non-local or
absent, and that is what the network has to find.

Example
-------
  python analysis/pilot_floor.py data/a8_bundle_seed0.zip
"""

import argparse
import sys
import os

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a8_io  # noqa: E402

K = 5  # patch side


def patches(st, M, max_px):
    """Feature rows (75) and targets (3) for a random subset of pixels in one event."""
    um = a8_io.unmix(st, M)
    lf = um[a8_io.LABEL_FREE]
    fl = um[a8_io.FLUOR]
    # normalise label-free pages per event so acquisition gain is not a feature
    lf = (lf - lf.mean(axis=(1, 2), keepdims=True)) / (lf.std(axis=(1, 2), keepdims=True) + 1e-6)
    h, w = lf.shape[1:]
    r = K // 2
    pad = np.pad(lf, ((0, 0), (r, r), (r, r)), mode="reflect")
    ys, xs = np.mgrid[0:h, 0:w]
    ys, xs = ys.ravel(), xs.ravel()
    if len(ys) > max_px:
        pick = np.random.default_rng(0).choice(len(ys), max_px, replace=False)
        ys, xs = ys[pick], xs[pick]
    X = np.stack([pad[:, y:y + K, x:x + K].ravel() for y, x in zip(ys, xs)])
    Y = fl[:, ys, xs].T
    return X, Y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", default="data/a8_bundle_seed0.zip")
    ap.add_argument("--train", default="2"); ap.add_argument("--test", default="3")
    ap.add_argument("--px-per-event", type=int, default=400)
    args = ap.parse_args(argv)

    b = a8_io.Bundle(args.bundle)
    M, _ = a8_io.spillover(b)
    tr = [c for c in b.classes if c[0] in args.train]
    te = [c for c in b.classes if c[0] in args.test]
    print("train %s   test %s" % (tr, te))

    Xtr, Ytr = zip(*[patches(b.image(c, r), M, args.px_per_event) for c, r in b.events(tr)])
    Xtr, Ytr = np.vstack(Xtr), np.vstack(Ytr)
    model = Ridge(alpha=1.0).fit(Xtr, Ytr)
    print("fit on %d pixels from %d events\n" % (len(Xtr), sum(len(b.rows(c)) for c in tr)))

    names = ["LDHC/AKAP4 (AF488)", "CD45 (PE)", "ACRV1 (PerCP)"]
    pix_p, pix_t, ev_p, ev_t, well = [], [], [], [], []
    for c, r in b.events(te):
        X, Y = patches(b.image(c, r), M, 10 ** 9)            # every pixel
        P = model.predict(X)
        pix_p.append(P); pix_t.append(Y)
        ev_p.append(P.sum(axis=0)); ev_t.append(Y.sum(axis=0)); well.append(c.lstrip("123"))
    pix_p, pix_t = np.vstack(pix_p), np.vstack(pix_t)
    ev_p, ev_t, well = np.array(ev_p), np.array(ev_t), np.array(well)
    expect = ["<< 1", ">> 1", "<< 1"]
    print("%-20s %9s %9s %12s %12s   %s" % ("marker", "pixel r", "event r", "pred P:S", "true P:S", "want"))
    for i, n in enumerate(names):
        pr = np.median(ev_p[well == "P", i]) / max(np.median(ev_p[well == "S", i]), 1e-9)
        tr_ = np.median(ev_t[well == "P", i]) / max(np.median(ev_t[well == "S", i]), 1e-9)
        print("%-20s %9.3f %9.3f %12.2f %12.2f   %s" % (
            n, np.corrcoef(pix_p[:, i], pix_t[:, i])[0, 1], np.corrcoef(ev_p[:, i], ev_t[:, i])[0, 1], pr, tr_, expect[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
