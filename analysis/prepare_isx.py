#!/usr/bin/env python
"""Cache an ImageStream .cif in the layout analysis/train_a8.py trains from.

Inputs are the three label-free channels -- Ch01 and Ch09 (brightfield, two cameras)
and Ch06 (side scatter) -- normalised per event. Targets are the four stained channels,
Ch02 LDHC/AKAP4, Ch03 ACRV1, Ch07 DAPI, Ch11 TOMM20, with each image's background
(median outside the combined mask) subtracted, in 12-bit counts. Frames are 112 x 80:
larger objects are centre-cropped, smaller ones padded with their own background and a validity mask.

There is one sample, so the cache is split into two pseudo-replicates by acquisition
order: class `1H` is the first half of the objects, `2H` the second. train_a8.py's
`--train 1 --test 2` then holds out the second half. That is a weaker split than a true
replicate and results from it should say so.

Example
-------
  python analysis/prepare_isx.py data/isx/ALL_1_400_1.cif data/isx/ALL_1_400_1.daf --cache data/isx_cache
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isx_daf  # noqa: E402
import isx_io  # noqa: E402

H, W = 112, 80
INPUTS = ["Ch01", "Ch09", "Ch06"]
TARGETS = ["Ch02", "Ch03", "Ch07", "Ch11"]
TARGET_NAMES = ["LDHC/AKAP4", "ACRV1", "DAPI", "TOMM20"]


def fit_frame(a, valid=None):
    """Centre-crop or reflect-pad (C, h, w) to (C, H, W); returns the frame and a mask."""
    C, h, w = a.shape
    out = np.zeros((C, H, W), np.float32); m = np.zeros((H, W), bool)
    # crop first
    if h > H: top = (h - H) // 2; a = a[:, top:top + H]; h = H
    if w > W: left = (w - W) // 2; a = a[:, :, left:left + W]; w = W
    ph, pw = H - h, W - w
    bh, bw = ph // 2, pw // 2
    # Pad with each channel's border median -- its background -- rather than reflecting.
    # Reflection tiles a small object into copies of itself across the frame, which the
    # mask keeps out of the loss but the network still sees as input.
    border = np.concatenate([a[:, 0, :], a[:, -1, :], a[:, :, 0], a[:, :, -1]], axis=1)
    fill = np.median(border, axis=1)
    out[:] = fill[:, None, None]
    out[:, bh:bh + h, bw:bw + w] = a; m[bh:bh + h, bw:bw + w] = True
    return out, m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cif"); ap.add_argument("daf"); ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    c = isx_io.CIF(args.cif)
    names, F, _ = isx_daf.read_features(args.daf)
    col = {n: i for i, n in enumerate(names)}
    n = len(c) if not args.limit else min(args.limit, len(c))
    os.makedirs(args.cache, exist_ok=True)
    ci = [int(x[2:]) - 1 for x in INPUTS]; ct = [int(x[2:]) - 1 for x in TARGETS]
    half = n // 2
    for cls, rows in (("1H", range(0, half)), ("2H", range(half, n))):
        N = len(rows)
        X = np.lib.format.open_memmap(os.path.join(args.cache, cls + "_x.npy"), "w+", np.float16, (N, len(ci), H, W))
        Y = np.lib.format.open_memmap(os.path.join(args.cache, cls + "_y.npy"), "w+", np.float16, (N, len(ct), H, W))
        Mk = np.lib.format.open_memmap(os.path.join(args.cache, cls + "_m.npy"), "w+", bool, (N, H, W))
        S = np.zeros((N, len(ct)), np.float32); t0 = time.time()
        for j, i in enumerate(rows):
            img = c.image(i).astype(np.float32); mk = c.mask(i)
            mc = mk.any(axis=0)
            bg = np.array([np.median(img[k][~mc]) if (~mc).any() else np.median(img[k]) for k in range(12)])
            x = img[ci]; x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)
            y = img[ct] - bg[ct][:, None, None]
            fx, m = fit_frame(x); fy, _ = fit_frame(y)
            X[j], Y[j], Mk[j] = fx, fy, m
            S[j] = [F[i, col["Intensity_MC_%s" % t]] for t in TARGETS]
            if (j + 1) % 2000 == 0: print("%s %5d / %d  %.0fs" % (cls, j + 1, N, time.time() - t0))
        X.flush(); Y.flush(); Mk.flush()
        np.save(os.path.join(args.cache, cls + "_rows.npy"), np.array(list(rows)))
        np.save(os.path.join(args.cache, cls + "_scalars.npy"), S)
        print("%s: %d objects cached (%.0fs)" % (cls, N, time.time() - t0))
    json.dump({"instrument": "ImageStream", "inputs": INPUTS, "targets": TARGETS, "target_names": TARGET_NAMES,
               "frame": [H, W], "split": "acquisition order halves: 1H first, 2H second", "pixel_um": 0.333},
              open(os.path.join(args.cache, "meta.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
