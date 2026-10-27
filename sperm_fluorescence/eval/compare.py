"""Within-population event r of several runs side by side: one panel per marker, one bar
group per population, one bar per run.

    python -m sperm_fluorescence compare "laptop sample=runs/abl_dropout" "full run=runs/srv_a8_stems" \\
        --title "A8: laptop sample vs the full run" --out figures/srv_vs_laptop_a8.png

Each run's numbers come from its ablate.json entry for all of its roles (so the ISX
populations sperm / round / dim are available) and otherwise from results.json's
within-well event r. Writes the PNG and a JSON of the numbers beside it.
"""
import argparse
import json
import os

import numpy as np

POP_LABEL = {"S": "sperm well", "P": "PBMC well", "SP": "mixed well", "I": "whole tube",
             "sperm": "sperm", "round": "round cells", "dim": "DAPI-dim"}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # fixed order, never cycled


def within(run):
    res = json.load(open(os.path.join(run, "results.json")))
    out = {}
    abl_path = os.path.join(run, "ablate.json")
    abl = json.load(open(abl_path)) if os.path.exists(abl_path) else {}
    key = ",".join(res.get("roles") or [])
    for name, r in res["results"].items():
        if r.get("event_r") is None: continue
        if key in abl and name in abl[key] and abl[key][name].get("within_population_event_r"):
            out[name] = dict(abl[key][name]["within_population_event_r"])
        else:
            out[name] = {k: v for k, v in r["within_well_event_r"].items() if v is not None}
    return out, res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence compare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path, in the order the bars should appear")
    ap.add_argument("--title", default="Within-population event r"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in a.runs]
    table = {label: within(path)[0] for label, path in runs}
    markers = [m for m in table[runs[0][0]] if all(m in t for t in table.values())]
    seen = {p for m in markers for t in table.values() for p in t[m]}
    pops = [p for p in ("S", "SP", "P", "I", "sperm", "round", "dim") if p in seen] + sorted(p for p in seen if p not in POP_LABEL)
    print("%-12s %-12s " % ("marker", "population") + " ".join("%14s" % l for l, _ in runs))
    for m in markers:
        for p in pops:
            print("%-12s %-12s " % (m, POP_LABEL.get(p, p)) + " ".join("%14s" % ("%.3f" % t[m][p] if p in t[m] else "") for t in table.values()))
    n = len(runs); bw = 0.8 / n
    fig, axes = plt.subplots(1, len(markers), figsize=(1.1 + 2.6 * len(markers), 3.8), facecolor="#fcfcfb", squeeze=False, sharey=True)
    for ax, m in zip(axes[0], markers):
        x = np.arange(len(pops))
        for j, (label, _) in enumerate(runs):
            vals = [table[label][m].get(p, np.nan) for p in pops]
            pos = x - 0.4 + bw * (j + 0.5)
            ax.bar(pos, vals, bw * 0.92, color=COLOURS[j % len(COLOURS)], label=label if m == markers[0] else None, zorder=3)
            for xx, v in zip(pos, vals):
                if np.isfinite(v): ax.text(xx, v + 0.015 if v >= 0 else v - 0.05, "%.2f" % v, ha="center", va="bottom", fontsize=7.5, color="#0b0b0b")
        ax.set_xticks(x); ax.set_xticklabels([POP_LABEL.get(p, p) for p in pops], fontsize=8.5)
        ax.set_title(m, fontsize=11, fontweight="bold", color="#0b0b0b"); ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_ylim(min(-0.1, min(min(t[m].values()) for t in table.values()) - 0.05), 1.08); ax.grid(axis="y", color="#e6e5e1", zorder=0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    axes[0][0].set_ylabel("event r within one population", fontsize=9.5, color="#52514e")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, frameon=False, loc="upper center", ncol=len(runs), bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(a.title, fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=200, facecolor="#fcfcfb"); plt.close(fig)
    json.dump({"runs": dict(runs), "within_population_event_r": table}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
