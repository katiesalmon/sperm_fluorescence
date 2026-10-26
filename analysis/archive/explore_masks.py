#!/usr/bin/env python
"""Render events from an .acs with the instrument's own segmentation drawn on them.

The archive ships a `*.masks.zip` holding one JSON per event, keyed by the same event id
as the image. Each file is

    [{"version": "1.0"}, {"masks": [{"pixelIndexes": {"indexes": [...]}}]}, {"masks": [...]}]

with flat pixel indices into the 248x248 frame. Layer 1 matches the instrument's
`NumPixels` column exactly, so it is the object mask. Layer 2 is a much smaller internal
region -- this script exists largely to see what it is.

Needs requirements-dev.txt (matplotlib) on top of requirements.txt.

Example
-------
  python analysis/explore_masks.py data/..._3SP.acs --targets data/targets --out analysis/out
"""

import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

SIDE = 248


def open_parts(acs_path):
    z = zipfile.ZipFile(acs_path)
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    mask_name = next((n for n in names if n.lower().endswith(".masks.zip")), None)
    masks = zipfile.ZipFile(io.BytesIO(z.read(mask_name))) if mask_name else None
    return z, masks


def mask_layers(masks, event):
    """Return a list of boolean frames, one per layer present for this event."""
    try:
        raw = masks.read("%d.json" % event)
    except KeyError:
        return []
    out = []
    for part in json.loads(raw)[1:]:
        frame = np.zeros(SIDE * SIDE, dtype=bool)
        for entry in part.get("masks") or []:
            idx = np.asarray(entry["pixelIndexes"]["indexes"], dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < frame.size)]
            frame[idx] = True
        out.append(frame.reshape(SIDE, SIDE))
    return out


def stretch(a, lo=1, hi=99):
    """Percentile stretch for display. The raw frames are ~10-bit in a uint16 container."""
    p0, p1 = np.percentile(a, [lo, hi])
    if p1 <= p0:
        p1 = p0 + 1
    return np.clip((a.astype(np.float32) - p0) / (p1 - p0), 0, 1)


def outline(mask):
    """Boundary pixels of a boolean mask, without pulling in scipy."""
    if not mask.any():
        return mask
    pad = np.pad(mask, 1)
    inner = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    return mask & ~inner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acs", help="Path to an .acs archive")
    ap.add_argument("--targets", default="data/targets", help="Directory of target CSVs, for the labels")
    ap.add_argument("--out", default="analysis/out", help="Where to write the contact sheet")
    ap.add_argument("--events", type=int, default=12, help="Events to draw (default: 12)")
    ap.add_argument("--sort-by", default=None, help="Marker column to sort by, e.g. CD45-PE-A")
    ap.add_argument("--top", action="store_true", help="With --sort-by, take the highest instead of a spread")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, masks = open_parts(args.acs)
    if masks is None:
        print("no masks.zip inside this archive")
        return 1

    stem = os.path.basename(args.acs).replace(".acs", "")
    csv = os.path.join(args.targets, stem + ".csv")
    table = pd.read_csv(csv).set_index("Event") if os.path.exists(csv) else None

    available = sorted(int(n[:-4]) for n in z.namelist() if n.endswith(".tif") and n[:-4].isdigit())
    if table is not None and args.sort_by and args.sort_by in table.columns:
        ranked = table.loc[table.index.intersection(available)].sort_values(args.sort_by, ascending=False)
        chosen = list(ranked.index[: args.events]) if args.top else \
            list(ranked.index[:: max(1, len(ranked) // args.events)][: args.events])
    else:
        chosen = available[:: max(1, len(available) // args.events)][: args.events]

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    for ax, event in zip(np.ravel(axes), chosen):
        img = np.array(Image.open(io.BytesIO(z.read("%d.tif" % event))))
        rgb = np.dstack([stretch(img)] * 3)
        layers = mask_layers(masks, event)
        if len(layers) > 0:
            rgb[outline(layers[0])] = [1.0, 0.25, 0.15]     # object mask, red
        if len(layers) > 1:
            rgb[layers[1]] = [0.15, 0.85, 1.0]              # inner region, cyan fill
        ax.imshow(rgb, interpolation="nearest")
        title = "event %d" % event
        if table is not None and event in table.index:
            r = table.loc[event]
            title += "\nDAPI %.0f  CD45 %.0f\nACRV1 %.0f  LDHC %.0f" % (
                r.get("DAPI-A", np.nan), r.get("CD45-PE-A", np.nan),
                r.get("ACRV-1-PerCP-ef710-A", np.nan), r.get("LDHC_AKAP4-AF488-A", np.nan))
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axes)[len(chosen):]:
        ax.axis("off")

    fig.suptitle("%s  --  red = object mask (layer 1), cyan = inner region (layer 2)" % stem, fontsize=9)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    suffix = ("_by_" + args.sort_by.split("-")[0]) if args.sort_by else ""
    path = os.path.join(args.out, "masks_%s%s.png" % (stem.split("_")[-1], suffix))
    fig.savefig(path, dpi=130)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
