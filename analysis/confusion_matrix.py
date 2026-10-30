"""Confusion matrix of a label-free clustering against the measured stains, taken as ground
truth: the measured-side clustering (same unsupervised recipe on the measured stain images,
`measured_cluster` in populations.csv) names each object, the label-free cluster is the
prediction. Clusters are matched to truth classes by the Hungarian assignment on the counts.

    python analysis/confusion_matrix.py runs/pop_srv_a8_feat_k2 --names sperm leukocyte --out figures/srv_a8_pop_feat_k2_confusion.png

--names are given in the order of the TRUTH classes after matching, largest first by default; pass
--positive <name> to choose which class the sensitivity / PPV refer to (default: the first name).
Writes the PNG and a JSON of the counts and rates beside it, overall and per well.
"""
import argparse
import json
import os

import numpy as np

WELL = {"3S": "sperm well", "3P": "PBMC well", "3SP": "mixed well", "S": "sperm well", "P": "PBMC well", "SP": "mixed well"}


def rates(cm, pos):
    """cm[truth, pred], pos = index of the positive class (binary only for sens/spec)."""
    n = cm.sum(); acc = np.trace(cm) / n
    out = {"n": int(n), "accuracy": float(acc), "balanced_accuracy": float(np.mean([cm[i, i] / cm[i].sum() for i in range(len(cm)) if cm[i].sum()]))}
    if len(cm) == 2:
        neg = 1 - pos; tp, fn, fp, tn = cm[pos, pos], cm[pos, neg], cm[neg, pos], cm[neg, neg]
        div = lambda x, y: float(x / y) if y else None      # a well with no objects of one class has no rate there
        out.update({"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                    "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
                    "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "f1": div(2 * tp, 2 * tp + fp + fn)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--out", required=True)
    ap.add_argument("--truth", default="measured_cluster", help="column holding the ground-truth label (default: the measured-side clustering)")
    ap.add_argument("--names", nargs="*", default=[], help="class names: either in order of the matched truth classes (largest truth class first) or as <label-free cluster id>=name, which is safer")
    ap.add_argument("--positive", default=None, help="which named class is 'positive' for sensitivity / PPV (default: the first)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = pd.read_csv(os.path.join(a.run, "populations.csv"), low_memory=False); s = json.load(open(os.path.join(a.run, "summary.json")))
    t = t[(t["cluster"] >= 0) & t[a.truth].notna()].copy(); t[a.truth] = t[a.truth].astype(int)
    truths = sorted(t[a.truth].unique(), key=lambda v: -(t[a.truth] == v).sum()); preds = sorted(t["cluster"].unique())
    C = np.array([[int(((t[a.truth] == u) & (t["cluster"] == p)).sum()) for p in preds] for u in truths])
    r, c = linear_sum_assignment(-C); pred_order = [preds[j] for _, j in sorted(zip(r, c))] + [p for p in preds if p not in [preds[j] for j in c]]
    cm = np.array([[int(((t[a.truth] == u) & (t["cluster"] == p)).sum()) for p in pred_order] for u in truths])
    if a.names and all("=" in n for n in a.names):       # "<label-free cluster>=name": name each truth row by the cluster it matched
        by_pred = {int(n.split("=", 1)[0]): n.split("=", 1)[1] for n in a.names}
        names = [by_pred.get(int(p), "cluster %d" % p) for p in pred_order]
    else:
        names = list(a.names) + ["class %d" % u for u in truths[len(a.names):]]
    print("matched: " + ", ".join("label-free cluster %d <-> measured class %d = %s" % (p, u, n) for p, u, n in zip(pred_order, truths, names)))
    pos = names.index(a.positive) if a.positive else 0
    res = {"run": os.path.abspath(a.run), "truth": a.truth, "classes": names, "truth_ids": [int(u) for u in truths], "matched_clusters": [int(p) for p in pred_order],
           "matrix_truth_rows_pred_cols": cm.tolist(), "overall": rates(cm, pos),
           "ari": float(adjusted_rand_score(t[a.truth], t["cluster"])), "kappa": float(cohen_kappa_score(t[a.truth].map({u: i for i, u in enumerate(truths)}), t["cluster"].map({p: i for i, p in enumerate(pred_order)})))}
    wells = [w for w in ("3S", "3P", "3SP", "S", "P", "SP") if w in set(t["id.class"])] + sorted(set(t["id.class"]) - set(WELL))
    res["per_well"] = {}
    for w in wells:
        tw = t[t["id.class"] == w]; cw = np.array([[int(((tw[a.truth] == u) & (tw["cluster"] == p)).sum()) for p in pred_order] for u in truths])
        res["per_well"][w] = {"matrix": cw.tolist(), **rates(cw, pos)}
    # ---- print
    print("%s: label-free clusters vs %s (truth), %d objects" % (os.path.basename(a.run.rstrip("/")), a.truth, len(t)))
    print("%-22s" % "truth \\ predicted" + "".join("%14s" % n for n in names) + "%10s" % "recall")
    for i, n in enumerate(names):
        print("%-22s" % n + "".join("%14d" % v for v in cm[i]) + "%9.1f%%" % (100 * cm[i, i] / cm[i].sum()))
    print("%-22s" % "precision" + "".join("%13.1f%%" % (100 * cm[j, j] / cm[:, j].sum()) for j in range(len(names))))
    o = res["overall"]; print("accuracy %.4f  balanced %.4f  kappa %.3f  ARI %.3f" % (o["accuracy"], o["balanced_accuracy"], res["kappa"], res["ari"]))
    if "sensitivity" in o: print("positive = %s: sensitivity %.4f  specificity %.4f  PPV %.4f  NPV %.4f  F1 %.4f" % (names[pos], o["sensitivity"], o["specificity"], o["ppv"], o["npv"], o["f1"]))
    for w in wells: print("  %-11s n=%6d  accuracy %.4f" % (WELL.get(w, w), res["per_well"][w]["n"], res["per_well"][w]["accuracy"]))
    # ---- figure: the matrix (counts + row %), then one small matrix per well
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"; blue = "#2a78d6"
    k = len(names); ncol = 1 + len(wells)
    fig = plt.figure(figsize=(3.9 + 2.5 * len(wells), 4.6), facecolor="#fcfcfb")
    gs = fig.add_gridspec(1, ncol, width_ratios=[1.6] + [1] * len(wells), wspace=0.45, left=0.09, right=0.98, top=0.78, bottom=0.2)
    def draw(ax, M, title, big):
        rowsum = M.sum(1, keepdims=True); frac = M / np.maximum(rowsum, 1)
        ax.imshow(frac, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", blue]), vmin=0, vmax=1, aspect="equal")
        for i in range(k):
            for j in range(k):
                col = "white" if frac[i, j] > 0.6 else ink
                ax.text(j, i - (0.1 if big else 0.12), "{:,}".format(M[i, j]), ha="center", va="center", fontsize=12 if big else 9.5, fontweight="bold", color=col)
                ax.text(j, i + (0.22 if big else 0.24), "%.1f%%" % (100 * frac[i, j]), ha="center", va="center", fontsize=8.5 if big else 7.5, color=col if frac[i, j] > 0.6 else muted)
        ax.set_xticks(range(k)); ax.set_yticks(range(k)); ax.set_xticklabels(names, fontsize=9 if big else 8, color=ink); ax.set_yticklabels(names, fontsize=9 if big else 8, color=ink, rotation=90, va="center")
        ax.set_xlabel("label-free cluster", fontsize=9 if big else 8, color=muted); ax.set_ylabel("measured stains (truth)", fontsize=9 if big else 8, color=muted)
        ax.set_title(title, fontsize=10.5 if big else 9, fontweight="bold", color=ink, pad=8)
        for sp in ax.spines.values(): sp.set_color(grid)
        ax.set_xticks(np.arange(-0.5, k, 1), minor=True); ax.set_yticks(np.arange(-0.5, k, 1), minor=True); ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
    draw(fig.add_subplot(gs[0, 0]), cm, "all wells, n = {:,}".format(len(t)), True)
    for q, w in enumerate(wells):
        draw(fig.add_subplot(gs[0, 1 + q]), np.array(res["per_well"][w]["matrix"]), "%s, n = %s" % (WELL.get(w, w), "{:,}".format(res["per_well"][w]["n"])), False)
    title = a.title or "%s: label-free clusters against the measured stains" % os.path.basename(a.run.rstrip("/"))
    fig.suptitle(title, fontsize=12, fontweight="bold", color=ink, y=0.97)
    sub = "accuracy %.1f%%  ·  balanced accuracy %.1f%%  ·  Cohen's κ %.3f  ·  ARI %.3f" % (100 * o["accuracy"], 100 * o["balanced_accuracy"], res["kappa"], res["ari"])
    if "sensitivity" in o: sub += "\n%s: sensitivity %.1f%%  ·  specificity %.1f%%  ·  PPV %.1f%%  ·  NPV %.1f%%" % (names[pos], 100 * o["sensitivity"], 100 * o["specificity"], 100 * o["ppv"], 100 * o["npv"])
    fig.text(0.5, 0.855, sub, ha="center", va="center", fontsize=9, color=muted, linespacing=1.5)
    fig.text(0.5, 0.04, "truth = the same unsupervised clustering (%s, k = %d) run on the MEASURED stain images; clusters matched to truth classes by Hungarian assignment; cells show count and row %%" % (s["method"], s["k"]), ha="center", fontsize=7.5, color=muted)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=200, facecolor="#fcfcfb"); plt.close(fig)
    json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1); print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
