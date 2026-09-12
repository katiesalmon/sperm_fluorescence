#!/usr/bin/env python
"""Characterise the per-event marker measurements before any modeling.

Answers, from the target tables alone:

  1. Which acquisitions were actually stained -- the FCS panel metadata is the operator's
     instrument configuration and carries over to every tube, so it says nothing about
     what was in the tube. Only the values do.
  2. Whether each channel reports the biology it is supposed to.
  3. Whether the "sperm" channels are tracking CD45 rather than their own antigen, which
     is what uncompensated spectral spillover looks like.
  4. How far replicate 2 and replicate 3 disagree.

Needs requirements.txt (numpy, pandas).

Example
-------
  python analysis/explore_targets.py data/targets
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

MARKERS = {
    "DAPI": "DAPI-A",
    "ACRV1": "ACRV-1-PerCP-ef710-A",
    "LDHC": "LDHC_AKAP4-AF488-A",
    "CD45": "CD45-PE-A",
}
ORDER = ["1S", "2S", "3S", "1P", "2P", "3P", "1SP", "2SP", "3SP"]


def load(directory):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        well = os.path.basename(path).split("_")[-1].replace(".csv", "")
        out[well] = pd.read_csv(path)
    if not out:
        raise IOError("No target CSVs under %s" % directory)
    return out


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random `pos` scores above a random `neg`)."""
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ranks = pd.Series(x).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def medians(data, wells):
    print("Median per-event intensity, uncompensated\n")
    print("%-6s %8s %10s %10s %10s %10s" % ("well", "n", *MARKERS))
    for well in wells:
        d = data[well]
        print("%-6s %8d %10.0f %10.0f %10.0f %10.0f"
              % (well, len(d), *[d[c].median() for c in MARKERS.values()]))


def staining(data, wells):
    """Replicate 1 is the reference: if 2/3 sit on top of it, that tube was not stained."""
    print("\n\nFold over the replicate-1 acquisition of the same well type")
    print("(replicate 1 turns out to be the unstained-except-DAPI control; ~1x means unstained)\n")
    print("%-6s %10s %10s %10s %10s" % ("well", *MARKERS))
    for well in wells:
        control = data.get("1" + well.lstrip("123"))
        if control is None or well.startswith("1"):
            continue
        d = data[well]
        folds = []
        for col in MARKERS.values():
            base = control[col].median()
            folds.append(d[col].median() / base if base else float("nan"))
        print("%-6s %9.1fx %9.1fx %9.1fx %9.1fx" % (well, *folds))


def separation(data):
    print("\n\nSeparating the PBMC well from the sperm well by one channel alone (AUC)")
    print("AUC below 0.5 means the PBMC well is *lower* -- for a sperm marker that is the")
    print("biologically correct direction\n")
    print("%-10s %8s %8s %8s %8s" % ("replicate", *MARKERS))
    for rep in ("1", "2", "3"):
        if rep + "S" not in data or rep + "P" not in data:
            continue
        s, p = data[rep + "S"], data[rep + "P"]
        print("%-10s %8.3f %8.3f %8.3f %8.3f"
              % (rep, *[auc(p[c].values, s[c].values) for c in MARKERS.values()]))


def spillover_check(data, wells, sample=20000):
    print("\n\nWithin-acquisition Spearman correlation with CD45")
    print("In a PBMC well, a sperm-marker channel should NOT track CD45. If it does, the")
    print("channel is reporting spectral spillover or non-specific binding, not antigen\n")
    print("%-6s %12s %12s %12s" % ("well", "ACRV1~CD45", "LDHC~CD45", "DAPI~CD45"))
    for well in wells:
        d = data[well]
        if len(d) > sample:
            d = d.sample(sample, random_state=0)
        cd45 = d[MARKERS["CD45"]]
        print("%-6s %12.3f %12.3f %12.3f"
              % (well, *[d[MARKERS[k]].corr(cd45, method="spearman") for k in ("ACRV1", "LDHC", "DAPI")]))


def replicate_gap(data):
    print("\n\nHow far apart are replicate 2 and replicate 3, same well type")
    print("(ratio of medians; 1.0 would mean the two stained replicates agree)\n")
    print("%-6s %10s %10s %10s %10s" % ("well", *MARKERS))
    for kind in ("S", "P", "SP"):
        if "2" + kind not in data or "3" + kind not in data:
            continue
        a, b = data["2" + kind], data["3" + kind]
        ratios = []
        for col in MARKERS.values():
            denom = b[col].median()
            ratios.append(a[col].median() / denom if denom else float("nan"))
        print("%-6s %10.2f %10.2f %10.2f %10.2f" % (kind, *ratios))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="?", default="data/targets", help="Directory of target CSVs")
    args = ap.parse_args(argv)

    data = load(args.targets)
    wells = [w for w in ORDER if w in data] + [w for w in sorted(data) if w not in ORDER]

    medians(data, wells)
    staining(data, wells)
    separation(data)
    spillover_check(data, wells)
    replicate_gap(data)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
