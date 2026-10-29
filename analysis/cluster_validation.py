"""Validation metrics for an unsupervised clustering, the set a paper reports, computed for a
latent embedding (or feature set) over a range of k and drawn as one figure:

  external   adjusted Rand index and normalised mutual information against a label the
             clustering never saw (wells on the A8); Hungarian-matched balanced accuracy;
             per-cluster purity and recall; the contingency table at the chosen k
  stability  bootstrap ARI (mean and range), acquisition-halves ARI, seed-to-seed ARI,
             per-cluster persistence (fraction of bootstraps in which the cluster reappears
             with Jaccard >= 0.75)
  internal   silhouette (overall and per cluster), Calinski-Harabasz, Davies-Bouldin, and
             for the Gaussian mixture the BIC curve over k

    python analysis/cluster_validation.py data/a8_vstain/features.csv --embedding data/a8_vstain/emb_dcae.npz \\
        --label id.well --label-map "S=sperm,SP=sperm,P=PBMC" --ks 2,3,4,5,6 --k 2 --out figures/a8_cluster_validation.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sperm_fluorescence.population.engine import CLIP, SETS, select_columns, standardise, reduce_pca, halves_order  # noqa: E402


def fit(scores, k, seed):
    from sklearn.mixture import GaussianMixture
    m = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed).fit(scores)
    return m.predict(scores), m


def matched_balanced_accuracy(labels, truth):
    """Clusters assigned to truth classes by the Hungarian method on the contingency table
    (several clusters may be left unassigned when k exceeds the number of classes; they
    count as errors), then the mean per-class recall."""
    from scipy.optimize import linear_sum_assignment
    classes = sorted(set(truth)); ks = sorted(set(labels))
    C = np.array([[np.sum((labels == c) & (truth == t)) for t in classes] for c in ks])
    r, c = linear_sum_assignment(-C)
    pred = np.full(len(labels), "", dtype=object)
    for ci, ti in zip(r, c): pred[labels == ks[ci]] = classes[ti]
    rec = [np.mean(pred[truth == t] == t) for t in classes]
    return float(np.mean(rec)), dict(zip([str(t) for t in classes], [float(v) for v in rec]))


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table"); ap.add_argument("--embedding", default=None); ap.add_argument("--set", default=None, choices=sorted(SETS))
    ap.add_argument("--components", type=int, default=16); ap.add_argument("--label", required=True); ap.add_argument("--label-map", default="")
    ap.add_argument("--truth-exclude", default="", help="comma list of raw label values that are not a per-cell truth (A8: SP, the mixed well); such objects are clustered but left out of the external metrics")
    ap.add_argument("--ks", default="2,3,4,5,6"); ap.add_argument("--k", type=int, default=2); ap.add_argument("--n-boot", type=int, default=30); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi, silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
    from sklearn.decomposition import PCA
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pd.read_csv(a.table); rng = np.random.default_rng(a.seed)
    lm = dict(s.split("=") for s in a.label_map.split(",") if s); raw = df[a.label].astype(str).to_numpy(); truth = np.array([lm.get(v, v) for v in raw], dtype=object)
    scored = ~np.isin(raw, [s for s in a.truth_exclude.split(",") if s])
    if a.embedding:
        e = np.load(a.embedding); E = e["embedding"].astype(np.float32); pos = {int(i): q for q, i in enumerate(e["index"])}
        X = E[[pos[int(i)] for i in df["id.index"]]]; X = np.clip((X - X.mean(0)) / (X.std(0) + 1e-6), -CLIP, CLIP)
        Z = PCA(n_components=min(a.components, X.shape[1]), random_state=a.seed).fit_transform(X); Z = Z / (Z.std(0) + 1e-6); what = "autoencoder latent (%d-d)" % E.shape[1]
    else:
        cols, _ = select_columns(df, SETS[a.set], 0.3); Zs, _, _ = standardise(df[cols].to_numpy(float)); Z, _ = reduce_pca(Zs); what = "%s features (%d columns)" % (a.set, len(cols))
    order = halves_order(df); half = order < np.median(order)      # first / second half of every well
    ks = [int(v) for v in a.ks.split(",")]; R = {}
    for k in ks:
        lab, model = fit(Z, k, a.seed)
        boots, pers = [], {c: 0 for c in range(k)}
        full_sets = {c: set(np.flatnonzero(lab == c)) for c in range(k)}
        for b in range(a.n_boot):
            idx = rng.choice(len(Z), len(Z), replace=True); lb, _ = fit(Z[idx], k, a.seed + 1 + b)
            boots.append(ari(lab[idx], lb))
            uniq = np.unique(idx); lab_b = np.full(len(Z), -1); lab_b[idx] = lb                                    # label of each object in this bootstrap (last draw wins)
            for c in range(k):
                best = max(jaccard(full_sets[c] & set(uniq), set(np.flatnonzero(lab_b == cb))) for cb in range(k))
                pers[c] += best >= 0.75
        halves = []
        for sel, oth in ((half, ~half), (~half, half)):
            lb, m = fit(Z[sel], k, a.seed); halves.append(ari(lab[oth], m.predict(Z[oth])))
        seeds = [ari(lab, fit(Z, k, a.seed + 100 + s)[0]) for s in range(a.seeds)]
        sil = silhouette_score(Z, lab); sil_c = silhouette_samples(Z, lab)
        bal, rec = matched_balanced_accuracy(lab[scored], truth[scored])
        ct = pd.crosstab(pd.Series(lab[scored], name="cluster"), pd.Series(truth[scored], name="truth"))
        purity = {int(c): float(ct.loc[c].max() / ct.loc[c].sum()) for c in ct.index}
        R[k] = dict(labels=lab, ari=ari(lab[scored], truth[scored]), nmi=nmi(lab[scored], truth[scored]), balanced_accuracy=bal, recall=rec, purity=purity, contingency=ct,
                    boot_mean=float(np.mean(boots)), boot_lo=float(np.percentile(boots, 5)), boot_hi=float(np.percentile(boots, 95)),
                    halves=halves, seeds_mean=float(np.mean(seeds)), persistence={c: pers[c] / a.n_boot for c in range(k)},
                    silhouette=float(sil), silhouette_per=[float(sil_c[lab == c].mean()) for c in range(k)], sizes=[int((lab == c).sum()) for c in range(k)],
                    ch=float(calinski_harabasz_score(Z, lab)), db=float(davies_bouldin_score(Z, lab)), bic=float(model.bic(Z)))
    # ---- print
    print("representation: %s; truth: %s on %d of %d objects (%s excluded from scoring); %d bootstraps, %d seeds" % (what, a.label, int(scored.sum()), len(Z), a.truth_exclude or "none", a.n_boot, a.seeds))
    print("%3s %6s %6s %8s | %8s %12s %8s %6s | %6s %8s %8s %12s" % ("k", "ARI", "NMI", "bal.acc", "boot", "(5-95%)", "halves", "seeds", "silh", "CH", "DB", "BIC"))
    for k in ks:
        r = R[k]; print("%3d %6.3f %6.3f %8.3f | %8.3f %5.2f-%-5.2f %4.2f/%-4.2f %6.2f | %6.3f %8.0f %8.2f %12.0f" % (
            k, r["ari"], r["nmi"], r["balanced_accuracy"], r["boot_mean"], r["boot_lo"], r["boot_hi"], r["halves"][0], r["halves"][1], r["seeds_mean"], r["silhouette"], r["ch"], r["db"], r["bic"]))
    r = R[a.k]; print("\nk = %d: contingency" % a.k); print(r["contingency"])
    print("purity per cluster", {c: round(v, 3) for c, v in r["purity"].items()}, " recall per class", {c: round(v, 3) for c, v in r["recall"].items()})
    print("persistence per cluster (Jaccard >= 0.75 over bootstraps)", {c: round(v, 2) for c, v in r["persistence"].items()})
    print("silhouette per cluster", [round(v, 3) for v in r["silhouette_per"]], "sizes", r["sizes"])
    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor="white")
    kk = np.array(ks)
    ax = axes[0, 0]; ax.plot(kk, [R[k]["ari"] for k in ks], "o-", color="#2a78d6", label="ARI vs %s" % a.label); ax.plot(kk, [R[k]["nmi"] for k in ks], "s--", color="#2a78d6", alpha=0.6, label="NMI")
    ax.plot(kk, [R[k]["balanced_accuracy"] for k in ks], "^-", color="#1baf7a", label="matched balanced accuracy"); ax.set_ylim(0, 1.05); ax.set_title("external: agreement with truth never seen", fontsize=10, loc="left"); ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]; ax.plot(kk, [R[k]["boot_mean"] for k in ks], "o-", color="#eb6834", label="bootstrap ARI (mean)")
    ax.fill_between(kk, [R[k]["boot_lo"] for k in ks], [R[k]["boot_hi"] for k in ks], color="#eb6834", alpha=0.15, label="5–95 %")
    ax.plot(kk, [min(R[k]["halves"]) for k in ks], "v-", color="#8e5bd6", label="halves ARI (worse half)"); ax.plot(kk, [R[k]["seeds_mean"] for k in ks], "d-", color="#eda100", label="seed-to-seed ARI")
    ax.set_ylim(0, 1.05); ax.set_title("stability", fontsize=10, loc="left"); ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 2]; ax.plot(kk, [R[k]["silhouette"] for k in ks], "o-", color="#2a78d6", label="silhouette"); ax.set_ylabel("silhouette", color="#2a78d6")
    ax2 = ax.twinx(); ax2.plot(kk, [R[k]["bic"] for k in ks], "s--", color="#52514e", label="BIC (lower is better)"); ax2.set_ylabel("BIC", color="#52514e")
    ax.set_title("internal: silhouette and BIC over k", fontsize=10, loc="left"); h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False)
    ax = axes[1, 0]; ct = r["contingency"]; im = ax.imshow(ct.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, fontsize=9); ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(["cluster %d" % c for c in ct.index], fontsize=9)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]): ax.text(j, i, int(ct.iloc[i, j]), ha="center", va="center", fontsize=10, color="white" if ct.iloc[i, j] > ct.to_numpy().max() / 2 else "black")
    ax.set_title("k = %d: contingency against %s (%d scored objects)" % (a.k, a.label, int(scored.sum())), fontsize=10, loc="left")
    ax = axes[1, 1]; x = np.arange(a.k)
    ax.bar(x - 0.2, [r["purity"][c] for c in range(a.k)], 0.4, color="#2a78d6", label="purity"); ax.bar(x + 0.2, [r["persistence"][c] for c in range(a.k)], 0.4, color="#eb6834", label="persistence (bootstrap)")
    ax.set_xticks(x); ax.set_xticklabels(["cluster %d\n(n = %d)" % (c, r["sizes"][c]) for c in range(a.k)], fontsize=8); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, frameon=False); ax.set_title("k = %d: per cluster" % a.k, fontsize=10, loc="left")
    ax = axes[1, 2]; sil_c = silhouette_samples(Z, r["labels"]); y0 = 0
    for c in range(a.k):
        v = np.sort(sil_c[r["labels"] == c]); ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6", "#c9357a"][c % 6], alpha=0.8); ax.text(-0.05, y0 + len(v) / 2, "cluster %d" % c, fontsize=8, ha="right", va="center"); y0 += len(v) + 10
    ax.axvline(r["silhouette"], color="#52514e", ls="--", lw=0.8); ax.set_yticks([]); ax.set_xlabel("silhouette per object", fontsize=9); ax.set_title("k = %d: silhouette profile (dashed = mean %.2f)" % (a.k, r["silhouette"]), fontsize=10, loc="left")
    for ax in axes.ravel():
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[0]: ax.set_xlabel("k", fontsize=9); ax.set_xticks(kk)
    fig.suptitle(a.title or "Clustering validation: %s, n = %d, Gaussian mixture; truth = %s" % (what, len(Z), a.label), fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=160, facecolor="white"); plt.close(fig)
    json.dump({k: {kk_: (v if not hasattr(v, "to_dict") and not isinstance(v, np.ndarray) else None) for kk_, v in R[k].items()} for k in ks}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
