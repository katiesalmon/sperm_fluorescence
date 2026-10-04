#!/usr/bin/env python
"""Run an ISX-trained model on CytPix brightfield crops, with no ground truth: a look.

Three honest mismatches, handled explicitly rather than hidden:
  - the model takes three label-free pages (two brightfield cameras and side scatter);
    the CytPix has one brightfield image. It goes into both brightfield slots and the
    scatter slot is left at zero (--ssc mirror puts brightfield there too).
  - pixel pitch: a sperm head spans ~10 px on the CytPix and ~9 px on the ISX at 60x, so
    the scale is close; --zoom resamples if wanted.
  - contrast: CytPix brightfield is extinction with a phase-like edge; ISX brightfield
    has a stronger halo. Cells are dark on a bright field in both.
Replicate 1 of the CytPix run is the unstained control, so there is nothing to score
against; the output is a sheet, predictions only, on the ISX display ranges.

Example
-------
  .venv/bin/python analysis/predict_cytpix.py runs/isx_pilot --cache data/isx_cache --sample <dir with 1S/ 1P/ 1SP/> --n 6
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402
import train_a8 as T  # noqa: E402


def load_model(run, cache, dev):
    res = json.load(open(os.path.join(run, "results.json")))
    meta = json.load(open(os.path.join(cache, "meta.json")))
    names = [res["target"]] if res.get("target") else meta["target_names"]
    T.NAMES = names
    tr, _, _ = T.open_cache(cache, res["train"])
    ysample = tr.sample_targets(200)
    scale_vec = np.abs(ysample).mean(axis=(0, 2, 3)) * 10 if res.get("channel_scale") else np.full(ysample.shape[1], np.abs(ysample).mean() * 10, np.float32)
    model = T.build_model(res["width"], tr.cin, tr.cout).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=dev)); model.eval()
    # display ranges from real ISX truth, so a dim prediction reads as dim
    ranges = render.ranges([np.asarray(ysample[0, :, :, 0])[:, :, None]], None)
    ranges = [(0.0, float(np.percentile(np.abs(ysample[0, k, :, 0]), 99.5))) for k in range(ysample.shape[1])]
    return model, names, torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1), ranges


def prep(path, crop, zoom, ssc, blur=0.0):
    g = np.array(Image.open(path))
    g = g[..., 0].astype(np.float32) if g.ndim == 3 else g.astype(np.float32)
    if blur > 0: g = ndi.gaussian_filter(g, blur)   # the CytPix background is noise; the ISX's is smooth
    if zoom != 1.0: g = ndi.zoom(g, zoom, order=1)
    H, W = g.shape; c = crop; top, left = (H - c) // 2, (W - c) // 2
    g = g[top:top + c, left:left + c]
    z = (g - g.mean()) / (g.std() + 1e-6)
    s = z if ssc == "mirror" else np.zeros_like(z)
    return g, np.stack([z, z, s]).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--sample", required=True)
    ap.add_argument("--classes", nargs="+", default=["1S", "1P", "1SP"]); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--crop", type=int, default=128); ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--ssc", choices=["zero", "mirror"], default="zero")
    ap.add_argument("--blur", type=float, default=0.0, help="Gaussian sigma applied to the brightfield first")
    ap.add_argument("--out", default="analysis/out/cytpix_predictions.png")
    args = ap.parse_args(argv)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, names, scale, ranges = load_model(args.run, args.cache, dev)

    rows = []
    for cls in args.classes:
        files = sorted(glob.glob(os.path.join(args.sample, cls, "*.tif")))
        step = max(1, len(files) // args.n)
        for f in files[::step][:args.n]:
            g, x = prep(f, args.crop, args.zoom, args.ssc, args.blur)
            with torch.no_grad():
                p = (model(torch.tensor(x)[None].to(dev)).float().cpu() * scale)[0].numpy()
            rows.append((cls, os.path.basename(f), g, np.clip(p, 0, None)))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = 1 + len(names) + 1
    fig, axes = plt.subplots(len(rows), cols, figsize=(1.6 * cols, 1.7 * len(rows)), facecolor="black")
    for i, (cls, nm, g, p) in enumerate(rows):
        layers = [render.colorize(p[k], render.color_for(n, k), *ranges[k]) for k, n in enumerate(names)]
        panels = [render.grey(g)] + layers + [render.composite(layers)]
        for c, (ax, img) in enumerate(zip(axes[i], panels)):
            ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
            if i == 0: ax.set_title((["CytPix BF"] + ["pred %s" % n for n in names] + ["overlay"])[c], fontsize=7, color="white")
        axes[i, 0].set_ylabel("%s\n%s" % (cls, nm.replace(".tif", "")), fontsize=6, color="white")
    fig.suptitle("ISX-trained model on CytPix brightfield (no truth; ISX ranges; crop %d, blur %.1f, SSC = %s)" % (args.crop, args.blur, args.ssc), fontsize=8, color="white")
    fig.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True); fig.savefig(args.out, dpi=110, facecolor="black")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
