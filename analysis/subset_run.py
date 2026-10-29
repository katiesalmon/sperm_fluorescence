"""Carve named groups out of a population run so the gallery scripts can draw them: a derived run
directory whose "clusters" are the groups. Each group is a pandas query over populations.csv.

    python analysis/subset_run.py runs/pop_srv_a8_feat_k2 --out runs/pop_srv_a8_feat_k2_misplaced \\
        --group "PBMC well, sperm cluster, sperm by measured stains=`id.class` == '3P' and cluster == 1 and measured_cluster == 1" \\
        --group "PBMC well, sperm cluster, leukocyte by measured stains=`id.class` == '3P' and cluster == 1 and measured_cluster == 0" \\
        --group "sperm well, leukocyte cluster, leukocyte by measured stains=`id.class` == '3S' and cluster == 0 and measured_cluster == 0" \\
        --group "sperm well, leukocyte cluster, sperm by measured stains=`id.class` == '3S' and cluster == 0 and measured_cluster == 1"
    python analysis/stain_gallery.py runs/pop_srv_a8_feat_k2_misplaced --store <store> --rows 4 --out figures/...png

Writes populations.csv (cluster = group number, the PC columns kept so "centre outward" ordering
still works), summary.json (the source summary with k and n replaced and the group names under
"group_names"), and prints the group sizes. Objects matching several queries go to the first.
"""
import argparse
import json
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--out", required=True)
    ap.add_argument("--group", action="append", required=True, help="name=pandas query; repeat")
    a = ap.parse_args(argv)
    import pandas as pd
    t = pd.read_csv(os.path.join(a.run, "populations.csv"), low_memory=False)
    s = json.load(open(os.path.join(a.run, "summary.json")))
    names, label = [], np.full(len(t), -1)
    for g, spec in enumerate(a.group):
        name, q = spec.split("=", 1); names.append(name.strip())
        m = t.eval(q).to_numpy(bool) & (label < 0); label[m] = g
        print("group %d  %-70s %6d" % (g, name.strip(), int(m.sum())))
    sub = t[label >= 0].copy(); sub["source_cluster"] = sub["cluster"]; sub["cluster"] = label[label >= 0]
    os.makedirs(a.out, exist_ok=True); sub.to_csv(os.path.join(a.out, "populations.csv"), index=False)
    s = dict(s); s["k"] = len(names); s["n"] = int(len(sub)); s["derived_from"] = os.path.abspath(a.run); s["group_names"] = names
    s["groups"] = [{"name": n, "query": g.split("=", 1)[1]} for n, g in zip(names, a.group)]
    s.pop("stability", None); s.pop("measured_side", None); s.pop("discriminators", None)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("wrote %s: %d objects in %d groups" % (a.out, len(sub), len(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
