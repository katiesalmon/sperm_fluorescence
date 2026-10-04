#!/usr/bin/env python
"""Re-evaluate a trained run on a subset of the test objects -- e.g. nucleated sperm only.

The ISX sample is 61.5% nucleated sperm and 38.5% DAPI-low fragments, and a model earns
most of its whole-sample correlation by telling those apart. This reloads a run's
checkpoint and scores it on test objects whose IDEAS DAPI intensity clears a threshold,
which is the ImageStream analogue of the A8's within-well-type number.

Example
-------
  python analysis/eval_subset.py runs/isx_pilot --cache data/isx_cache --dapi-min 25119
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_a8 as T  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True)
    ap.add_argument("--dapi-min", type=float, default=25119.0, help="IDEAS Intensity_MC_Ch07 threshold (default: the trough between the two modes)")
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args(argv)

    res = json.load(open(os.path.join(args.run, "results.json")))
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    names = [res["target"]] if res.get("target") else meta["target_names"]
    T.NAMES = names
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    xs, ys, ms, wtr, _ = T.load_cache(args.cache, res["train"])
    xs_te, ys_te, ms_te, wte, scal = T.load_cache(args.cache, res["test"])
    if res.get("target"):
        k = meta["target_names"].index(res["target"])
        ys = [y[:, k:k + 1] for y in ys]; ys_te = [y[:, k:k + 1] for y in ys_te]
    tr = T.Cached(xs, ys, ms); te = T.Cached(xs_te, ys_te, ms_te)

    # the training scale, recomputed exactly as train() did
    ysample = np.concatenate([np.asarray(y[:200]) for y in tr.ys]).astype(np.float32)
    if res.get("channel_scale"):
        scale_vec = np.abs(ysample).mean(axis=(0, 2, 3)) * 10
    else:
        scale_vec = np.full(ysample.shape[1], np.abs(ysample).mean() * 10, np.float32)
    scale = torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1)

    model = T.build_model(res["width"], tr.xs[0].shape[1], tr.ys[0].shape[1]).to(dev)
    model.load_state_dict(torch.load(os.path.join(args.run, args.checkpoint), map_location=dev))

    dapi_col = meta["targets"].index("Ch07") if "Ch07" in meta["targets"] else None
    dapi = scal[:, dapi_col] if dapi_col is not None else None
    subsets = {"all": np.arange(len(te))}
    if dapi is not None:
        subsets["DAPI-high (nucleated)"] = np.flatnonzero(dapi >= args.dapi_min)
        subsets["DAPI-low"] = np.flatnonzero(dapi < args.dapi_min)
    out = {}
    print("%s  checkpoint %s  (%d test objects)\n" % (args.run, args.checkpoint, len(te)))
    print("%-24s %7s  " % ("subset", "n") + "  ".join("%18s" % n for n in names))
    for label, idx in subsets.items():
        r, _ = T.evaluate(model, te, wte[idx], scale, dev, idx=idx)
        out[label] = {n: {"pixel_r": r[n]["pixel_r"], "event_r": r[n]["event_r"]} for n in names}
        print("%-24s %7d  " % (label, len(idx)) + "  ".join("%7.3f px %7.3f ev" % (r[n]["pixel_r"], r[n]["event_r"]) for n in names))
    json.dump(out, open(os.path.join(args.run, "subset_results.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
