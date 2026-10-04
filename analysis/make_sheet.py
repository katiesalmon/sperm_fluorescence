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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--per-well", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include", default=None, help="Restrict to these cache classes (comma list)")
    ap.add_argument("--checkpoint", default="best.pt"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    if "target_names" in meta: T.NAMES = list(meta["target_names"])
    res = json.load(open(os.path.join(a.run, "results.json")))
    classes = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(a.cache, "*_x.npy"))})
    if a.include: classes = [c for c in classes if c in a.include.split(",")]
    te_cls = [c for c in classes if c[0] in a.test]
    te, wte, _ = T.open_cache(a.cache, te_cls)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = T.build_model(res.get("width", 32), te.cin, te.cout).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, a.checkpoint), map_location=dev)); model.eval()
    if "scale" in res:
        scale_vec = np.array(res["scale"], np.float32)
    else:   # older run: rebuild the divisor the way train() did, from the same training classes
        tr, _, _ = T.open_cache(a.cache, [c for c in classes if c[0] in "".join(str(d) for d in range(10) if str(d) not in a.test) and c in res["train"]])
        scale_vec = T.target_scale(tr, res.get("channel_scale", False))
    scale = torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1)
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
