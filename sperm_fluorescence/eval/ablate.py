"""Score one trained model under several input-role subsets (the stems architecture).

    python -m sperm_fluorescence ablate runs/a8_stems_dropout --cache data/a8_roles --test 3 \\
        --subsets "lightloss; lightloss,fsc; lightloss,fsc,ssc"

For each subset the roles outside it are marked absent for every object, and the held-out
set is scored exactly as at training time: pixel r, event r, within-well event r per marker.
Writes ablate.json beside the run.
"""
import argparse
import json
import os

import numpy as np

from .accuracy import groups_for
from .load import load_run
from .metrics import evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence ablate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--include", default=None); ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--subsets", required=True, help="semicolon-separated role lists, e.g. 'lightloss; lightloss,fsc; lightloss,fsc,ssc'")
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args(argv)
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint)
    roles = list(R.te.roles)
    out = {}
    print("%s  roles trained: %s\n" % (a.run, roles))
    for sub in [s.strip() for s in a.subsets.split(";") if s.strip()]:
        want = [r.strip() for r in sub.split(",")]
        bad = [r for r in want if r not in roles]
        if bad: raise SystemExit("roles %s not in the model (%s)" % (bad, roles))
        mask = np.array([r in want for r in roles], bool)
        res, P = evaluate(R.model, R.te, R.wte, R.scale, R.dev, R.names, tta=a.tta, roles_mask=mask)
        # within-population r (ISX: sperm / round / dim by the DAPI scalar; A8: the well types are already the populations)
        idx = np.arange(len(R.te)); g = groups_for(R.te, R.wte, idx); pops = {}
        for k, n in enumerate(R.names):
            if res[n]["event_r"] is None: continue
            av = R.te.avail[:, k] if R.te.avail is not None else np.ones(len(R.te), bool)
            et = np.array([(R.te.item(i)[1][k] * R.te.item(i)[2]).sum() if av[i] else np.nan for i in idx])
            ep = np.array([(P[q][k] * R.te.item(i)[2]).sum() if av[i] else np.nan for q, i in enumerate(idx)])
            pops[n] = {}
            for grp in sorted(set(g)):
                sel = (g == grp) & np.isfinite(et)
                if sel.sum() > 10: pops[n][str(grp)] = float(np.corrcoef(et[sel], ep[sel])[0, 1])
            res[n]["within_population_event_r"] = pops[n]
        out[sub] = res
        print("input %-28s" % sub + "  ".join("%s ev %s (%s)" % (n, "%.3f" % res[n]["event_r"] if res[n]["event_r"] is not None else "n/a",
              " ".join("%s %.2f" % (k, v) for k, v in res[n].get("within_population_event_r", {}).items())) for n in R.names))
    json.dump(out, open(os.path.join(a.run, "ablate.json"), "w"), indent=1)
    print("wrote", os.path.join(a.run, "ablate.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
