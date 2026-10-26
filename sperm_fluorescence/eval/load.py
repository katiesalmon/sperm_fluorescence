"""Load a finished run from its checkpoint, and redraw its IDEAS-style sheet (`sperm_fluorescence sheet`).

    python -m sperm_fluorescence sheet runs/gen_a8_scratch --cache data/gen_cache --test 3 [--per-well 3] [--seed 0]

Rows are held-out objects, a few per well type; columns are BF, each marker true and
predicted in its colour, then the true and predicted all-channel overlay. Channels the
object's instrument did not stain are not drawn.
"""
import argparse, glob, json, os, types

import numpy as np

from ..data.batching import avail_of
from ..data.cache import open_cache
from ..models import stems
from ..train.device import pick_device
from ..train.losses import target_scale
from .metrics import evaluate
from .sheets import sheet


def load_run(run, cache, test, include=None, checkpoint="best.pt"):
    """A finished run, loaded: model, dev, scale, te, wte, te_cls, res, names (as a namespace)."""
    import torch
    meta = json.load(open(os.path.join(cache, "meta.json")))
    names = list(meta["target_names"]) if "target_names" in meta else ["LDHC/AKAP4", "CD45", "ACRV1"]
    res = json.load(open(os.path.join(run, "results.json")))
    classes = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(cache, "*_x.npy"))})
    if include: classes = [c for c in classes if c in include.split(",")]
    te_cls = [c for c in classes if c[0] in test]
    te, wte, scal = open_cache(cache, te_cls, roles=res.get("roles") if res.get("roles") and res.get("arch") else None); te.scalars = scal   # ISX: DAPI intensity; NaN on the A8
    dev = pick_device()
    model = stems.build(res.get("arch", "stack"), te.roles, res.get("width", 32), te.cin, te.cout).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, checkpoint), map_location=dev)); model.eval()
    if "scale" in res:
        scale_vec = np.array(res["scale"], np.float32)
    else:   # older run: rebuild the divisor the way train() did, from the same training classes
        tr, _, _ = open_cache(cache, [c for c in classes if c in res["train"]])
        scale_vec = target_scale(tr, res.get("channel_scale", False))
    scale = torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1)
    if res.get("target"): names = [res["target"]]
    return types.SimpleNamespace(model=model, dev=dev, scale=scale, te=te, wte=wte, te_cls=te_cls, res=res, names=names)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence sheet")
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--per-well", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include", default=None, help="Restrict to these cache classes (comma list)")
    ap.add_argument("--checkpoint", default="best.pt"); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    R = load_run(a.run, a.cache, a.test, a.include, a.checkpoint); model, dev, scale, te, wte, names = R.model, R.dev, R.scale, R.te, R.wte, R.names
    rng = np.random.default_rng(a.seed)
    pick = []
    for w in sorted(set(wte)):
        idx = np.flatnonzero(wte == w); pick += [int(i) for i in rng.choice(idx, min(a.per_well, len(idx)), replace=False)]
    _, P = evaluate(model, te, wte, scale, dev, names, idx=np.array(pick))
    xb, yb, mb = te.batch(pick)
    H, W = yb.shape[2], yb.shape[3]
    pb = np.stack([np.pad(P[j], [(0, 0), ((H - P[j].shape[1]) // 2, H - P[j].shape[1] - (H - P[j].shape[1]) // 2),
                                 ((W - P[j].shape[2]) // 2, W - P[j].shape[2] - (W - P[j].shape[2]) // 2)]) for j in range(len(pick))])
    out = a.out or os.path.join(a.run, "sheet.png")
    sheet(out, "%s: held-out predictions" % os.path.basename(a.run.rstrip("/")), xb.numpy(), yb.numpy(), pb, mb.numpy()[:, 0], wte[pick], names, avail_of(te, pick).numpy() > 0)
    print("wrote", out)


if __name__ == "__main__":
    main()
