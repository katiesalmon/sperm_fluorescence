"""`populate`: find populations in a feature table the way a cytometrist would, in feature
space: quality gates first, then a chosen feature set standardised and reduced (PCA by the
Kaiser criterion), clustered (k-means / Gaussian mixture with a given k, or HDBSCAN with
the count left open), tested for stability across acquisition halves and bootstraps, and
described by the features that separate each population from the rest. With the store
given, a gallery per population is drawn.

    python -m sperm_fluorescence populate data/isx_vstain/features.csv --set head --k 3 --out runs/pop_isx_head
    python -m sperm_fluorescence populate data/isx_vstain/features.csv --set tail --method hdbscan --store data/isx_vstain --out runs/pop_isx_tail

Feature sets (--set): head (shape of nucleus, acrosome, head and their ratios), tail
(shape and skeleton of midpiece, principal piece, tail), morph (both), intensity (predicted
channel sums, maxima and similarities), all (every pred.* and in.* column). --features adds
glob patterns of your own. Columns with more than --max-nan missing or no variance are
dropped; the rest are median-imputed and standardised (median / IQR, robust to outliers).

Gates (applied in order, each reporting its loss) default to: within = id.pop_pred == sperm
(when --within is given), single = qc.n_objects <= 1 and qc.n_nuclei <= 1, in_frame =
qc.flagellum_edge_frac < 0.05, focus = qc.focus_gradient_rms above its 5th percentile.
--gate adds or replaces one as name=pandas-query. Objects failing a gate are kept in the
output table with their gate flags but excluded from reduction and clustering. --join
name=populations.csv merges an earlier run's cluster labels as `name.cluster`, so a run can be
gated on another's population (tails within the head run's sperm cluster, say).

Outputs under --out: populations.csv (ids, gate flags, PC scores, embedding, cluster),
summary.json (gates, PCA variance, cluster sizes, stability, discriminating features,
provenance), embedding.png, and gallery_<cluster>.png when --store is given.
"""
import argparse
import fnmatch
import json
import os
import time

import numpy as np

SETS = {
    "head": ["pred.nucleus.*", "pred.acrosome.*", "pred.head.*", "pred.x.acrosome_*", "pred.x.nucleus_frac", "pred.x.d_nucleus_acrosome_um"],
    "tail": ["pred.midpiece.*", "pred.principal.*", "pred.tail.*", "pred.flagellum.*", "pred.x.d_nucleus_midpiece_um", "pred.x.d_nucleus_flagellum_um",
             "pred.x.head_midpiece_angle_deg", "pred.x.insertion_offset_um", "pred.x.midpiece_*", "pred.x.tail_len_over_head"],
    "intensity": ["pred.*.cell.sum", "pred.*.cell.max", "pred.*.cell.mean", "pred.*.cell.frac_in_mask", "pred.sim.*", "in.*.cell.sum", "in.*.cell.gradient_rms", "in.*.cell.std"],
    "all": ["pred.*", "in.*"],
}
SETS["morph"] = SETS["head"] + SETS["tail"]
EXCLUDE = ["*.centroid_y", "*.centroid_x", "*.orientation_deg", "*.end_y", "*.end_x", "*.n_px"]      # frame coordinates and raw pixel counts carry no shape
DEFAULT_GATES = [("single", "`qc.n_objects` <= 1 and `qc.n_nuclei` <= 1"), ("in_frame", "`qc.flagellum_edge_frac` < 0.05"), ("focus", "`qc.focus_gradient_rms` >= @focus_p5")]


def select_columns(df, patterns, max_nan):
    cols = [c for c in df.columns if any(fnmatch.fnmatch(c, p) for p in patterns) and not any(fnmatch.fnmatch(c, e) for e in EXCLUDE)]
    keep, dropped = [], {}
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        nan = float(np.isnan(v).mean())
        if nan > max_nan: dropped[c] = "nan %.0f%%" % (100 * nan); continue
        if np.nanstd(v) == 0 or not np.isfinite(np.nanstd(v)): dropped[c] = "no variance"; continue
        keep.append(c)
    return keep, dropped


CLIP = 5.0


def standardise(X):
    """Median / IQR per column after median imputation, clipped at +-CLIP so that a handful
    of extreme objects cannot own the principal components; IQR of 0 falls back to the std."""
    X = X.copy(); med = np.nanmedian(X, axis=0); X = np.where(np.isnan(X), med, X)
    q1, q3 = np.percentile(X, [25, 75], axis=0); scale = q3 - q1
    sd = X.std(axis=0); scale = np.where(scale > 0, scale, np.where(sd > 0, sd, 1.0))
    return np.clip((X - med) / scale, -CLIP, CLIP), med, scale


def reduce_pca(Z, min_components=2, max_components=20, fixed=None):
    from sklearn.decomposition import PCA
    p = PCA().fit(Z); ev = p.explained_variance_
    k = int(fixed) if fixed else int(max(min_components, min(max_components, (ev > 1.0).sum())))     # Kaiser criterion, the morphometry convention
    scores = p.transform(Z)[:, :k]
    return scores, {"n_components": k, "explained_variance_ratio": [float(v) for v in p.explained_variance_ratio_[:k]],
                    "cumulative": float(p.explained_variance_ratio_[:k].sum()), "loadings": p.components_[:k].tolist()}


def embed_2d(scores, seed):
    """2-D view for plots: UMAP when installed, else t-SNE on the PCA scores."""
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(scores), "umap"
    except ImportError:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", perplexity=min(30, max(5, len(scores) // 50)), random_state=seed).fit_transform(scores), "tsne"


MIN_SAMPLES = 10


def fit_cluster(scores, method, k, seed, min_cluster_size):
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.mixture import GaussianMixture
    if method == "kmeans":
        m = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scores); return m.labels_, m
    if method == "gmm":
        m = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed).fit(scores); return m.predict(scores), m
    if method == "hdbscan":
        m = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min(MIN_SAMPLES, min_cluster_size), cluster_selection_method="eom").fit(scores); return m.labels_, m
    raise SystemExit("unknown method %s" % method)


def assign(model, scores, method):
    if method == "kmeans" or method == "gmm": return model.predict(scores)
    from sklearn.neighbors import KNeighborsClassifier                    # HDBSCAN has no predict for new points
    lab = model.labels_; ok = lab >= 0
    if ok.sum() == 0: return np.full(len(scores), -1)
    return KNeighborsClassifier(n_neighbors=min(15, ok.sum())).fit(model._fit_X[ok], lab[ok]).predict(scores)


def stability(scores, labels, order, method, k, seed, min_cluster_size, n_boot=10):
    """Adjusted Rand index between: the clustering refit on each acquisition half and applied
    to the other (vs the full clustering there), and bootstrap refits (vs the full clustering
    on the resampled objects). Mean ARI near 1 = the populations survive resampling."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed); out = {}
    half = order < np.median(order); halves = []
    for sel, other in ((half, ~half), (~half, half)):
        if sel.sum() < 2 * max(k, min_cluster_size) or other.sum() < 10: continue
        lab, model = fit_cluster(scores[sel], method, k, seed, min_cluster_size); model._fit_X = scores[sel]
        pred = assign(model, scores[other], method); ok = (pred >= 0) & (labels[other] >= 0)
        halves.append(float(adjusted_rand_score(labels[other][ok], pred[ok])) if ok.sum() > 10 else np.nan)
    out["halves_ari"] = halves
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(scores), len(scores), replace=True)
        lab, _ = fit_cluster(scores[idx], method, k, seed + 1, min_cluster_size); ok = (lab >= 0) & (labels[idx] >= 0)
        boots.append(float(adjusted_rand_score(labels[idx][ok], lab[ok])) if ok.sum() > 10 else np.nan)
    out["bootstrap_ari"] = boots; out["bootstrap_ari_mean"] = float(np.nanmean(boots)) if boots else None
    return out


def discriminators(df, cols, labels, top=10):
    """Per population: the features with the largest Fisher discriminant ratio against the
    rest, with the population's and the rest's medians (the Feature Finder idea)."""
    out = {}
    X = df[cols].to_numpy(dtype=float)
    for c in sorted(set(labels)):
        if c < 0: continue
        a, b = X[labels == c], X[labels != c]
        if len(a) < 3 or len(b) < 3: continue
        fdr = (np.nanmean(a, 0) - np.nanmean(b, 0)) ** 2 / (np.nanvar(a, 0) + np.nanvar(b, 0) + 1e-12)
        order = np.argsort(-np.nan_to_num(fdr))[:top]
        out[str(c)] = [{"feature": cols[i], "fdr": float(fdr[i]), "median_in": float(np.nanmedian(a[:, i])), "median_rest": float(np.nanmedian(b[:, i]))} for i in order]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence populate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table"); ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="morph", choices=sorted(SETS)); ap.add_argument("--features", action="append", default=[], help="extra glob patterns of columns")
    ap.add_argument("--within", default=None, help="restrict to one id.pop_pred population before clustering, e.g. sperm")
    ap.add_argument("--gate", action="append", default=[], metavar="name=query", help="add or replace a gate (pandas query; backtick column names)")
    ap.add_argument("--no-gates", action="store_true"); ap.add_argument("--max-nan", type=float, default=0.3)
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "gmm", "hdbscan"]); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-cluster-size", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--components", type=int, default=0, help="fix the number of PCA components (default: Kaiser criterion, 2-20); HDBSCAN wants few")
    ap.add_argument("--min-samples", type=int, default=10, help="HDBSCAN min_samples (density requirement); the default is loose")
    ap.add_argument("--join", action="append", default=[], metavar="name=populations.csv", help="merge another run's populations.csv on id.index as <name>.cluster etc., so a gate can chain on it")
    ap.add_argument("--store", default=None, help="virtual-stain store, to draw a gallery per population"); ap.add_argument("--gallery-n", type=int, default=8)
    a = ap.parse_args(argv)
    import pandas as pd
    t0 = time.time(); df = pd.read_csv(a.table); n_all = len(df)
    global MIN_SAMPLES; MIN_SAMPLES = a.min_samples
    for spec in a.join:
        name, path = spec.split("=", 1); other = pd.read_csv(path)
        keep = [c for c in other.columns if c in ("cluster", "emb_x", "emb_y") or c.startswith(("gate.", "pc"))]
        other = other[["id.index"] + keep].rename(columns={c: "%s.%s" % (name, c) for c in keep})
        df = df.merge(other, on="id.index", how="left")
    # ---- gates ----
    gates = [] if a.no_gates else list(DEFAULT_GATES)
    if a.within: gates.insert(0, ("within", "`id.pop_pred` == '%s'" % a.within))
    for g in a.gate:
        name, q = g.split("=", 1); gates = [(n_, q_) for n_, q_ in gates if n_ != name] + [(name, q)]
    focus_p5 = float(np.nanpercentile(df["qc.focus_gradient_rms"], 5)) if "qc.focus_gradient_rms" in df else 0.0
    passed = np.ones(n_all, bool); gate_report = []
    for name, q in gates:
        try:
            res = df.eval(q, local_dict={"focus_p5": focus_p5})
        except Exception as e:
            raise SystemExit("gate %s (%s): %s" % (name, q, e))
        ok = np.asarray(res.fillna(False), dtype=bool) if hasattr(res, "fillna") else np.asarray(res, dtype=bool)   # a NaN never passes a gate
        lost = int((passed & ~ok).sum()); passed &= ok; df["gate." + name] = ok
        gate_report.append({"gate": name, "query": q, "lost": lost, "remaining": int(passed.sum())})
        print("gate %-10s %-55s lost %5d  remaining %5d" % (name, q, lost, passed.sum()))
    df["gate.passed"] = passed
    # ---- features ----
    cols, dropped = select_columns(df, SETS[a.set] + a.features, a.max_nan)
    if len(cols) < 2: raise SystemExit("fewer than two usable columns for set %s" % a.set)
    sub = df[passed]; X = sub[cols].to_numpy(dtype=float)
    Z, med, scale = standardise(X)
    scores, pca = reduce_pca(Z, fixed=a.components or None)
    print("set %s: %d columns kept, %d dropped; %d objects; PCA keeps %d components (%.0f%% of variance)" % (a.set, len(cols), len(dropped), len(sub), pca["n_components"], 100 * pca["cumulative"]))
    # ---- clustering ----
    labels, model = fit_cluster(scores, a.method, a.k, a.seed, a.min_cluster_size); model._fit_X = scores
    sizes = {str(c): int((labels == c).sum()) for c in sorted(set(labels))}
    print("clusters (%s): %s" % (a.method, sizes))
    order = sub["id.index"].to_numpy(dtype=float)
    stab = stability(scores, labels, order, a.method, a.k, a.seed, a.min_cluster_size)
    print("stability: halves ARI %s, bootstrap ARI mean %s" % (["%.2f" % v for v in stab["halves_ari"]], "%.2f" % stab["bootstrap_ari_mean"] if stab["bootstrap_ari_mean"] is not None else "-"))
    disc = discriminators(sub, cols, labels)
    for c, feats in disc.items():
        print("  cluster %s (n=%d): " % (c, sizes[c]) + "; ".join("%s %.3g vs %.3g" % (f["feature"], f["median_in"], f["median_rest"]) for f in feats[:4]))
    emb, emb_kind = embed_2d(scores, a.seed)
    # ---- output table ----
    os.makedirs(a.out, exist_ok=True)
    out = df[[c for c in df.columns if c.startswith(("id.", "gate."))]].copy()
    out["cluster"] = -2; out.loc[passed, "cluster"] = labels
    for j in range(scores.shape[1]): out["pc%d" % (j + 1)] = np.nan; out.loc[passed, "pc%d" % (j + 1)] = scores[:, j]
    out["emb_x"] = np.nan; out["emb_y"] = np.nan; out.loc[passed, "emb_x"] = emb[:, 0]; out.loc[passed, "emb_y"] = emb[:, 1]
    out.to_csv(os.path.join(a.out, "populations.csv"), index=False)
    summary = {"table": os.path.abspath(a.table), "n": n_all, "gates": gate_report, "set": a.set, "columns": cols, "dropped": dropped,
               "standardise": {"median": med.tolist(), "scale": scale.tolist()}, "pca": {k: v for k, v in pca.items() if k != "loadings"},
               "pca_loadings": pca["loadings"], "method": a.method, "k": a.k, "min_cluster_size": a.min_cluster_size, "seed": a.seed,
               "cluster_sizes": sizes, "stability": stab, "discriminators": disc, "embedding": emb_kind, "created": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    from .gallery import embedding_plot, galleries
    # the measured side, when the table has reference features: the same recipe on ref.* columns, for the plot and the summary
    measured_labels = None
    ref_cols, _ = select_columns(sub, [p.replace("pred.", "ref.") for p in SETS[a.set] + a.features if p.startswith("pred.")], a.max_nan)
    if len(ref_cols) >= 2 and a.method != "hdbscan":
        Zr, _, _ = standardise(sub[ref_cols].to_numpy(float)); sr, _ = reduce_pca(Zr, fixed=a.components or None)
        measured_labels, _ = fit_cluster(sr, a.method, a.k, a.seed, a.min_cluster_size)
        from sklearn.metrics import adjusted_rand_score
        summary["measured_side"] = {"columns": ref_cols, "ari_vs_label_free": float(adjusted_rand_score(labels, measured_labels)),
                                    "contingency": {str(c): {str(m): int(((labels == c) & (measured_labels == m)).sum()) for m in sorted(set(measured_labels))} for c in sorted(set(labels))}}
        print("measured side (same recipe on ref.*): ARI vs label-free %.3f" % summary["measured_side"]["ari_vs_label_free"])
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
        out["measured_cluster"] = -2; out.loc[passed, "measured_cluster"] = measured_labels; out.to_csv(os.path.join(a.out, "populations.csv"), index=False)
    embedding_plot(out[passed], labels, emb_kind, os.path.join(a.out, "embedding.png"), title="%s: %s, %s" % (os.path.basename(a.out), a.set, a.method),
                   measured_labels=measured_labels, measured=sub)
    if a.store:
        galleries(a.store, out[passed], scores, labels, a.set, a.out, a.gallery_n)
    print("wrote %s (%.0fs)" % (a.out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
