"""Redraw the IDEAS-style prediction sheet for a finished run, from its checkpoint.

    python analysis/make_sheet.py runs/gen_a8_scratch --cache data/gen_cache --test 3 [--per-well 3] [--seed 0]

Rows are held-out objects, a few per well type; columns are BF, each marker true and
predicted in its colour, then the true and predicted all-channel overlay. Channels the
object's instrument did not stain are not drawn.
"""
import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import train_a8 as T


def load_run(run, cache, test, include=None, checkpoint="best.pt"):
    """(model, device, scale tensor, test dataset, wells, class names, results dict) for a finished run."""
    import torch
    meta = json.load(open(os.path.join(cache, "meta.json")))
    if "target_names" in meta: T.NAMES = list(meta["target_names"])
    res = json.load(open(os.path.join(run, "results.json")))
    classes = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(cache, "*_x.npy"))})
    if include: classes = [c for c in classes if c in include.split(",")]
    te_cls = [c for c in classes if c[0] in test]
    te, wte, scal = T.open_cache(cache, te_cls); te.scalars = scal   # ISX: DAPI intensity; NaN on the A8
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = T.build_model(res.get("width", 32), te.cin, te.cout).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, checkpoint), map_location=dev)); model.eval()
    if "scale" in res:
        scale_vec = np.array(res["scale"], np.float32)
    else:   # older run: rebuild the divisor the way train() did, from the same training classes
        tr, _, _ = T.open_cache(cache, [c for c in classes if c in res["train"]])
        scale_vec = T.target_scale(tr, res.get("channel_scale", False))
    scale = torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1)
    return model, dev, scale, te, wte, te_cls, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--per-well", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include", default=None, help="Restrict to these cache classes (comma list)")
    ap.add_argument("--checkpoint", default="best.pt"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, dev, scale, te, wte, _, _ = load_run(a.run, a.cache, a.test, a.include, a.checkpoint)
    rng = np.random.default_rng(a.seed)
    pick = []
    for w in sorted(set(wte)):
        idx = np.flatnonzero(wte == w); pick += [int(i) for i in rng.choice(idx, min(a.per_well, len(idx)), replace=False)]
    _, P = T.evaluate(model, te, wte, scale, dev, idx=np.array(pick))
    xb, yb, mb = te.batch(pick)
    H, W = yb.shape[2], yb.shape[3]
    pb = np.stack([np.pad(P[j], [(0, 0), ((H - P[j].shape[1]) // 2, H - P[j].shape[1] - (H - P[j].shape[1]) // 2),
                                 ((W - P[j].shape[2]) // 2, W - P[j].shape[2] - (W - P[j].shape[2]) // 2)]) for j in range(len(pick))])
    out = a.out or os.path.join(a.run, "sheet.png")
    T._sheet(out, "%s: held-out predictions" % os.path.basename(a.run.rstrip("/")), xb.numpy(), yb.numpy(), pb, mb.numpy()[:, 0], wte[pick], T._avail_of(te, pick).numpy() > 0)
    print("wrote", out)


if __name__ == "__main__":
    main()
