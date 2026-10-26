#!/usr/bin/env python
"""Re-evaluate a trained run on a subset of the test objects -- e.g. nucleated sperm only.

The ISX sample is 61.5% nucleated sperm and 38.5% DAPI-low fragments, and a model earns
most of its whole-sample correlation by telling those apart. This reloads a run's
checkpoint and scores it on test objects whose IDEAS DAPI intensity clears a threshold,
which is the ImageStream analogue of the A8's within-well-type number.

Example
-------
  python python -m sperm_fluorescence subset runs/isx_pilot --cache data/isx_cache --dapi-min 25119
"""

import argparse
import json
import os

import numpy as np
import torch

from ..data.cache import open_cache
from ..models.unet import build_model
from ..train.device import pick_device
from ..train.losses import target_scale
from .metrics import evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence subset", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True)
    ap.add_argument("--dapi-min", type=float, default=25119.0, help="IDEAS Intensity_MC_Ch07 threshold (default: the trough between the two modes)")
    ap.add_argument("--dapi-max", type=float, default=None,
                    help="Also score 'sperm only': DAPI between --dapi-min and this (excludes round cells, ~4x the sperm median and up)")
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args(argv)

    res = json.load(open(os.path.join(args.run, "results.json")))
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    names = [res["target"]] if res.get("target") else meta["target_names"]
    dev = pick_device()

    target_k = meta["target_names"].index(res["target"]) if res.get("target") else None
    tr, wtr, _ = open_cache(args.cache, res["train"], target_k)
    te, wte, scal = open_cache(args.cache, res["test"], target_k)

    # the training scale: stored by newer runs, else recomputed exactly as train() did
    scale_vec = np.array(res["scale"], np.float32) if res.get("scale") else target_scale(tr, res.get("channel_scale"))
    scale = torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1)

    model = build_model(res["width"], tr.cin, tr.cout).to(dev)
    model.load_state_dict(torch.load(os.path.join(args.run, args.checkpoint), map_location=dev))

    dapi_col = meta["targets"].index("Ch07") if "Ch07" in meta["targets"] else None
    dapi = scal[:, dapi_col] if dapi_col is not None else None
    subsets = {"all": np.arange(len(te))}
    if dapi is not None:
        subsets["DAPI-high (nucleated)"] = np.flatnonzero(dapi >= args.dapi_min)
        subsets["DAPI-low"] = np.flatnonzero(dapi < args.dapi_min)
        if args.dapi_max:
            subsets["sperm only (no round cells)"] = np.flatnonzero((dapi >= args.dapi_min) & (dapi < args.dapi_max))
            subsets["round cells (DAPI >= max)"] = np.flatnonzero(dapi >= args.dapi_max)
    out = {}
    print("%s  checkpoint %s  (%d test objects)\n" % (args.run, args.checkpoint, len(te)))
    print("%-24s %7s  " % ("subset", "n") + "  ".join("%18s" % n for n in names))
    for label, idx in subsets.items():
        r, _ = evaluate(model, te, wte[idx], scale, dev, names, idx=idx)
        out[label] = {n: {"pixel_r": r[n]["pixel_r"], "event_r": r[n]["event_r"]} for n in names}
        print("%-24s %7d  " % (label, len(idx)) + "  ".join("%7.3f px %7.3f ev" % (r[n]["pixel_r"], r[n]["event_r"]) for n in names))
    json.dump(out, open(os.path.join(args.run, "subset_results.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
