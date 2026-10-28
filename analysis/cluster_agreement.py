"""A test of the features, with no rules: do populations found on the label-free side
(predicted compartments + inputs) match the populations found, by the same unsupervised
recipe, on the measured side (IDEAS masks of the real stains)? And what does each
label-free cluster group, described by measured quantities it never saw?

    python analysis/cluster_agreement.py data/isx_vstain/features.csv --set head --k 3

For each k in --ks: gate identically, cluster pred.* (+ in.*) and ref.* feature sets with the
population engine's recipe (robust standardise, PCA by Kaiser, k-means), report the
adjusted Rand index and the contingency table between the two clusterings, and the ARI of
the label-free clustering against the measured-DAPI population label the cache carries. Then
describe each label-free cluster (at --k) by the median measured DAPI scalar, measured stain
sums and IDEAS mask sizes.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sperm_fluorescence.population.engine import SETS, DEFAULT_GATES, select_columns, standardise, reduce_pca  # noqa: E402

DESCRIBE = [("id.scalar", "measured DAPI scalar"), ("ref.DAPI.nucleus.sum", "measured DAPI in its mask"), ("ref.LDHC_AKAP4.flagellum.sum", "measured LDHC/AKAP4 in its mask"),
            ("ref.TOMM20.midpiece.sum", "measured TOMM20 in its mask"), ("ref.ACRV1.acrosome.sum", "measured ACRV1 in its mask"),
            ("ref.nucleus.feret_max_um", "IDEAS nucleus Feret max, um"), ("ref.nucleus.n_components", "IDEAS nucleus components"),
            ("ref.flagellum.area_um2", "IDEAS flagellum area, um2"), ("ref.flagellum.skel.path_um", "IDEAS flagellum path, um"), ("ref.head.feret_max_um", "IDEAS head Feret max, um")]


def cluster(df, patterns, k, seed):
    from sklearn.cluster import KMeans
    cols, _ = select_columns(df, patterns, 0.3)
    Z, _, _ = standardise(df[cols].to_numpy(float)); scores, pca = reduce_pca(Z)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(scores), len(cols), pca["n_components"]


def main(argv=None):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table"); ap.add_argument("--set", default="head", choices=sorted(SETS)); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ks", default="2,3,4,5,6"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--inputs", action="store_true", help="add the in.* columns to the label-free side (the measured side has no counterpart)")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.table)
    if not a.no_gates:
        focus_p5 = float(np.nanpercentile(df["qc.focus_gradient_rms"], 5)); keep = np.ones(len(df), bool)
        for name, q in DEFAULT_GATES: keep &= np.asarray(df.eval(q, local_dict={"focus_p5": focus_p5}).fillna(False), bool)
        df = df[keep].reset_index(drop=True)
    print("%d objects after the quality gates; feature set %s" % (len(df), a.set))
    pred_pats = SETS[a.set] + (["in.*"] if a.inputs else []); ref_pats = [p.replace("pred.", "ref.") for p in SETS[a.set]]
    pop = df["id.pop_scalar"].astype(str).to_numpy()
    print("\n%4s %8s %8s %10s %10s %12s" % ("k", "n cols", "PCs", "ARI p~r", "NMI p~r", "ARI p~DAPIpop"))
    labs = {}
    for k in [int(v) for v in a.ks.split(",")]:
        lp, ncp, pcp = cluster(df, pred_pats, k, a.seed); lr, ncr, pcr = cluster(df, ref_pats, k, a.seed); labs[k] = (lp, lr)
        print("%4d %8s %8s %10.3f %10.3f %12.3f" % (k, "%d/%d" % (ncp, ncr), "%d/%d" % (pcp, pcr), adjusted_rand_score(lp, lr), normalized_mutual_info_score(lp, lr), adjusted_rand_score(lp, pop)))
    lp, lr = labs[a.k]
    print("\ncontingency at k = %d: rows = label-free clusters, columns = measured-side clusters" % a.k)
    print(pd.crosstab(pd.Series(lp, name="label-free"), pd.Series(lr, name="measured")))
    print("\nlabel-free clusters vs the measured-DAPI population label:")
    print(pd.crosstab(pd.Series(lp, name="label-free"), pd.Series(pop, name="DAPI pop")))
    print("\nwhat each label-free cluster groups (medians of measured quantities it never saw):")
    rows = []
    for c in sorted(set(lp)):
        sel = lp == c; r = {"cluster": c, "n": int(sel.sum())}
        for col, name in DESCRIBE:
            if col in df: r[name] = float(np.nanmedian(df.loc[sel, col].to_numpy(float)))
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("cluster")
    with pd.option_context("display.width", 220, "display.float_format", lambda v: "%.3g" % v): print(tab.T)


if __name__ == "__main__":
    main()
