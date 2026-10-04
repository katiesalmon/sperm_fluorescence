#!/usr/bin/env python
"""One cache holding both instruments, for the generalist experiments.

Channels are mapped to ROLES, not instruments, and both are resampled to the ISX pitch:

    input roles   [brightfield, second brightfield / FSC, side scatter] (+ instrument one-hot)
    targets       [LDHC/AKAP4, ACRV1, DAPI, TOMM20, CD45]

    ISX  Ch01, Ch09, Ch06  ->  Ch02, Ch03, Ch07, Ch11, (none)          classes 2I (first half), 3I (second)
    A8   LightLoss, FSC, SSC (x1.25)  ->  AF488, PerCP, (none), (none), PE   classes 2S 2P 2SP / 3S 3P 3SP

Each class carries an availability vector per target: the loss and the metrics only
touch channels the instrument actually stained. Targets are scaled per instrument and
channel to unit mean magnitude, so the two instruments' very different units meet on
equal terms; correlations are unit-free so nothing is lost. A8 classes carry a `repeat`
so 600 A8 events are not drowned by 10,000 ISX ones.

Example
-------
  python analysis/prepare_generalist.py --isx data/isx/ALL_1_400_1.cif --isx-daf data/isx/ALL_1_400_1.daf \\
      --a8 data/a8_bundle_seed0.zip --cache data/gen_cache
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a8_io  # noqa: E402
import isx_daf  # noqa: E402
import isx_io  # noqa: E402
from prepare_isx import to_multiple, write_ragged  # noqa: E402

TARGETS = ["LDHC/AKAP4", "ACRV1", "DAPI", "TOMM20", "CD45"]
ISX_IN = [0, 8, 5]; ISX_TGT = {0: 1, 1: 2, 2: 6, 3: 10}             # target k -> ISX channel index
A8_IN = [0, 1, 2]; A8_TGT = {0: 3, 1: 5, 4: 4}                        # target k -> A8 page (AF488, PerCP, PE)


def standardise(x):
    return (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)


def finish(items, avail, cls, cache, onehot):
    """Add the instrument one-hot planes, pad to multiples of 8, write."""
    out = []
    for x, y in items:
        if onehot:
            planes = np.zeros((2,) + x.shape[1:], np.float32); planes[0 if avail[2] else 1] = 1.0
            x = np.concatenate([x, planes])
        fx, m = to_multiple(x); fy, _ = to_multiple(y)
        out.append((fx, fy, m))
    write_ragged(cls, out, cache)
    np.save(os.path.join(cache, cls + "_avail.npy"), np.tile(np.array(avail, bool), (len(out), 1)))
    np.save(os.path.join(cache, cls + "_rows.npy"), np.arange(len(out)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isx", required=True); ap.add_argument("--isx-daf", required=True)
    ap.add_argument("--a8", required=True, help="an a8_bundle zip"); ap.add_argument("--cache", required=True)
    ap.add_argument("--a8-zoom", type=float, default=1.25); ap.add_argument("--isx-limit", type=int, default=0)
    ap.add_argument("--a8-repeat", type=int, default=6); ap.add_argument("--no-onehot", dest="onehot", action="store_false")
    args = ap.parse_args(argv)
    os.makedirs(args.cache, exist_ok=True)
    meta = {"layout": "ragged", "target_names": TARGETS, "inputs": ["BF", "BF2/FSC", "SSC"] + (["isx", "a8"] if args.onehot else []),
            "classes": {}, "scales": {}}

    # ---- ISX ----
    c = isx_io.CIF(args.isx); names, F, _ = isx_daf.read_features(args.isx_daf); col = {n: i for i, n in enumerate(names)}
    n = len(c) if not args.isx_limit else min(args.isx_limit, len(c)); half = n // 2
    raw = {}
    t0 = time.time()
    for cls, rows in (("2I", range(0, half)), ("3I", range(half, n))):
        items = []; dapi = []
        for i in rows:
            img = c.image(i).astype(np.float32); mk = c.mask(i); mc = mk.any(axis=0)
            bg = np.array([np.median(img[k][~mc]) if (~mc).any() else np.median(img[k]) for k in range(12)])
            x = standardise(img[ISX_IN])
            y = np.zeros((len(TARGETS),) + img.shape[1:], np.float32)
            for k, ch in ISX_TGT.items(): y[k] = img[ch] - bg[ch]
            items.append((x, y)); dapi.append(F[i, col["Intensity_MC_Ch07"]])
        raw[cls] = (items, [True, True, True, True, False], np.array(dapi, np.float32))
        print("ISX %s: %d objects decoded (%.0fs)" % (cls, len(items), time.time() - t0))

    # ---- A8 ----
    b = a8_io.Bundle(args.a8); M, _ = a8_io.spillover(b)
    for cls in b.classes:
        if cls[0] == "1": continue                       # unstained control
        items = []
        for r in b.rows(cls):
            st = a8_io.unmix(b.image(cls, r), M)
            st = ndi.zoom(st, (1, args.a8_zoom, args.a8_zoom), order=1).astype(np.float32)
            x = standardise(st[A8_IN])
            y = np.zeros((len(TARGETS),) + st.shape[1:], np.float32)
            for k, pg in A8_TGT.items(): y[k] = st[pg]
            items.append((x, y))
        raw[cls] = (items, [True, True, False, False, True], np.full(len(items), np.nan, np.float32))
        print("A8 %s: %d objects (zoom %.2f)" % (cls, len(items), args.a8_zoom))

    # ---- per-instrument, per-channel target scale -> unit mean |y| ----
    for inst, classes in (("isx", ["2I", "3I"]), ("a8", [k for k in raw if k not in ("2I", "3I")])):
        tr_cls = [k for k in classes if k[0] == "2"]
        avail = raw[tr_cls[0]][1]
        scale = np.ones(len(TARGETS), np.float32)
        for k in range(len(TARGETS)):
            if avail[k]:
                vals = np.concatenate([np.abs(y[k]).ravel()[::7] for cl in tr_cls for _, y in raw[cl][0][:300]])
                scale[k] = max(float(vals.mean()), 1e-6)
        meta["scales"][inst] = scale.tolist()
        for cl in classes:
            items, av, dapi = raw[cl]
            raw[cl] = ([(x, y / scale[:, None, None]) for x, y in items], av, dapi)
            meta["classes"][cl] = {"instrument": inst, "avail": av, "repeat": (1 if inst == "isx" else args.a8_repeat)}

    for cl, (items, av, dapi) in raw.items():
        finish(items, av, cl, args.cache, args.onehot)
        np.save(os.path.join(args.cache, cl + "_scalars.npy"), dapi[:, None])
        print("wrote %s: %d objects, avail %s" % (cl, len(items), av))
    json.dump(meta, open(os.path.join(args.cache, "meta.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
