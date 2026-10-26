"""Score one trained model under several input-role subsets (the stems architecture).

    python -m stainless ablate runs/a8_stems_dropout --cache data/a8_roles --test 3 \\
        --subsets "lightloss; lightloss,fsc; lightloss,fsc,ssc"

For each subset the roles outside it are marked absent for every object, and the held-out
set is scored exactly as at training time: pixel r, event r, within-well event r per marker.
Writes ablate.json beside the run.
"""
import argparse
import json
import os

import numpy as np

from .load import load_run
from .metrics import evaluate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stainless ablate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        res, _ = evaluate(R.model, R.te, R.wte, R.scale, R.dev, R.names, tta=a.tta, roles_mask=mask)
        out[sub] = res
        print("input %-28s" % sub + "  ".join("%s ev %s (%s)" % (n, "%.3f" % res[n]["event_r"] if res[n]["event_r"] is not None else "n/a",
              " ".join("%s %.2f" % (k, v) for k, v in res[n]["within_well_event_r"].items() if v is not None)) for n in R.names))
    json.dump(out, open(os.path.join(a.run, "ablate.json"), "w"), indent=1)
    print("wrote", os.path.join(a.run, "ablate.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
