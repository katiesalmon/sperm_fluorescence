#!/usr/bin/env python
"""How much of each marker is predictable from image-derived morphology alone?

This is step 3 of docs/approach.md, and it runs without a single image: the CytPix
already computed shape, intensity and texture measurements per event, so the question
"does what the cell looks like predict what it stains for" can be asked directly from
the target tables.

Treat the result as a *floor*, not a ceiling. These are a few dozen instrument features,
not a learned representation, so a CNN on the images should beat this. If it does not,
the CNN is not working.

Held out by replicate, because that is the only split that is not secretly measuring
one acquisition against itself. Two controls run alongside every fit:

  shuffled   the same features against a permuted target -- anything above zero here
             means leakage, not signal
  intensity  brightfield intensity features included vs excluded. Intensity tracks the
             acquisition's noise floor (sperm_pbmc Finding 1), so a marker that is only
             predictable with them is suspect

Needs requirements.txt plus scikit-learn.

Example
-------
  python analysis/feature_baseline.py data/targets --train 2 --test 3
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

MARKERS = {
    "DAPI": "DAPI-A",
    "ACRV1": "ACRV-1-PerCP-ef710-A",
    "LDHC": "LDHC_AKAP4-AF488-A",
    "CD45": "CD45-PE-A",
}

SHAPE = [
    "NumPixels", "AreaSquareMicrons", "PerimeterMicrons",
    "MajorDiameterMicrons", "MinorDiameterMicrons",
    "MinorMajorRatioPercent", "EccentricityPercent", "CircularityPercent",
    "ObjectCount", "ParticleCount",
]
INTENSITY = ["TotalIntensity", "AverageIntensity", "StandardDeviationIntensity"]


def load(directory, replicates):
    frames = []
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        well = os.path.basename(path).split("_")[-1].replace(".csv", "")
        if well[0] not in replicates:
            continue
        d = pd.read_csv(path)
        d["well"] = well
        d["well_type"] = well.lstrip("123")
        frames.append(d)
    if not frames:
        raise IOError("No target CSVs for replicate(s) %s under %s" % (replicates, directory))
    return pd.concat(frames, ignore_index=True)


def _prep(frame, features, target_col):
    keep = [c for c in features if c in frame.columns]
    # Fluorescence is heavy-tailed and can be slightly negative; log1p on the clipped
    # value keeps the loss from being dominated by the brightest few percent of events.
    return frame[keep].values, np.log1p(np.clip(frame[target_col].values, 0, None))


def _score(xtr, ytr, xte, yte, seed=0):
    model = HistGradientBoostingRegressor(max_iter=150, random_state=seed)
    model.fit(xtr, ytr)
    return spearmanr(model.predict(xte), yte).correlation


def fit_and_score(train, test, features, target_col, permutations=3, seed=0):
    """Across-replicate score, a within-replicate score for comparison, and a real null.

    The null matters more than it looks. A permuted-label model does not predict a
    constant -- it still emits some arbitrary function of the features, and when the
    features carry this much information about the target, an arbitrary function of them
    can correlate with the truth by luck. So the null is run several times and reported as
    a range: a result is only meaningful if it clears the top of that range.
    """
    xtr, ytr = _prep(train, features, target_col)
    xte, yte = _prep(test, features, target_col)

    across = _score(xtr, ytr, xte, yte, seed)

    # Same model, same size, but trained and tested inside the training replicate. The
    # gap between this and `across` is the price of generalising to another acquisition.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xtr))
    cut = int(0.7 * len(idx))
    within = _score(xtr[idx[:cut]], ytr[idx[:cut]], xtr[idx[cut:]], ytr[idx[cut:]], seed)

    nulls = []
    for k in range(permutations):
        r = np.random.default_rng(seed + 100 + k)
        nulls.append(_score(xtr, r.permutation(ytr), xte, yte, seed + k))
    return across, within, nulls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="?", default="data/targets")
    ap.add_argument("--train", default="2", help="Replicate digit(s) to train on (default: 2)")
    ap.add_argument("--test", default="3", help="Replicate digit(s) to test on (default: 3)")
    ap.add_argument("--wells", nargs="+", default=None, help="Restrict to these well types, e.g. S P")
    ap.add_argument("--permutations", type=int, default=3, help="Permuted-target runs for the null (default: 3)")
    args = ap.parse_args(argv)

    train = load(args.targets, set(args.train))
    test = load(args.targets, set(args.test))
    if args.wells:
        wanted = {w.upper() for w in args.wells}
        train = train[train.well_type.isin(wanted)]
        test = test[test.well_type.isin(wanted)]

    print("train: replicate %s, %d events (%s)"
          % (args.train, len(train), ", ".join(sorted(train.well.unique()))))
    print("test:  replicate %s, %d events (%s)\n"
          % (args.test, len(test), ", ".join(sorted(test.well.unique()))))

    sets = [("shape only", SHAPE), ("shape + intensity", SHAPE + INTENSITY)]
    print("Spearman of predicted vs measured intensity.")
    print("  within  = random split inside replicate %s (what a naive split would report)" % args.train)
    print("  across  = trained on replicate %s, tested on replicate %s (the honest number)"
          % (args.train, args.test))
    print("  null    = permuted-target models, %d runs, reported as a range" % args.permutations)
    print("A result counts only if `across` clears the top of `null` by a clear margin.\n")
    print("%-8s %-20s %8s %8s %18s %s" % ("marker", "features", "within", "across", "null range", "verdict"))
    print("-" * 88)
    for name, col in MARKERS.items():
        for label, feats in sets:
            across, within, nulls = fit_and_score(train, test, feats, col, args.permutations)
            lo, hi = min(nulls), max(nulls)
            margin = across - hi
            verdict = "solid" if margin > 0.25 else ("weak" if margin > 0.08 else "NOT ABOVE NULL")
            print("%-8s %-20s %8.3f %8.3f   %6.3f..%6.3f  %s"
                  % (name, label, within, across, lo, hi, verdict))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
