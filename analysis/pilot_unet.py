#!/usr/bin/env python
"""Pilot image-to-image model: three label-free pages in, three unmixed stain pages out.

A small U-Net, L1 loss, trained on replicate 2 and tested on replicate 3 of the local
bundle. At 600 training events this is a feasibility check, not a result: it answers
whether the pipeline, the unmixing and the loss behave, and whether a convolutional model
clears the per-pixel linear floor (analysis/pilot_floor.py) on the cell-type controls that
floor fails. The full run is 120,000 events on the server GPU.

Reports per marker: pixel r, event r (sum of predicted vs true page), and the predicted
P:S ratio -- median page sum on PBMC-well events over sperm-well events -- against the true
ratio. ACRV1 and LDHC should be << 1, CD45 >> 1. Writes a prediction sheet.

Example
-------
  .venv/bin/python analysis/pilot_unet.py data/a8_bundle_seed0.zip --epochs 30
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a8_io  # noqa: E402

H, W = 80, 104  # pad every event to this; the flow axis varies 57-85, the scan axis is always 104


def prep(st, M):
    """(3, H, W) normalised label-free input and (3, H, W) unmixed fluorescence target, plus a mask."""
    um = a8_io.unmix(st, M)
    x = um[a8_io.LABEL_FREE]
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)
    y = um[a8_io.FLUOR]
    h = min(st.shape[1], H)
    X = np.zeros((3, H, W), np.float32); Y = np.zeros((3, H, W), np.float32); m = np.zeros((1, H, W), np.float32)
    top = (H - h) // 2
    X[:, top:top + h] = x[:, :h]; Y[:, top:top + h] = y[:, :h]; m[:, top:top + h] = 1
    return X, Y, m


def load(bundle, classes, M):
    X, Y, Mk, well = [], [], [], []
    for c, r in bundle.events(classes):
        x, y, m = prep(bundle.image(c, r), M)
        X.append(x); Y.append(y); Mk.append(m); well.append(c.lstrip("123"))
    return (torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)), torch.tensor(np.stack(Mk)), np.array(well))


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    """Three levels, ~0.5M parameters. No global pooling anywhere: output keeps full resolution."""
    def __init__(self, cin=3, cout=3, w=24):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w)
        self.d2, self.d1 = Block(4 * w + 2 * w, 2 * w), Block(2 * w + w, w)
        self.out = nn.Conv2d(w, cout, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.out(d1)


def evaluate(model, X, Y, Mk, well, scale, dev):
    model.eval()
    with torch.no_grad():
        P = torch.cat([model(X[i:i + 64].to(dev)).cpu() for i in range(0, len(X), 64)]) * scale
    P, Yt, m = P.numpy(), Y.numpy(), Mk.numpy()
    out = []
    for k in range(3):
        pp, tt = P[:, k][m[:, 0] > 0], Yt[:, k][m[:, 0] > 0]
        ev_p, ev_t = (P[:, k] * m[:, 0]).sum(axis=(1, 2)), (Yt[:, k] * m[:, 0]).sum(axis=(1, 2))
        ps = lambda v: np.median(v[well == "P"]) / max(abs(np.median(v[well == "S"])), 1e-9)
        out.append((np.corrcoef(pp, tt)[0, 1], np.corrcoef(ev_p, ev_t)[0, 1], ps(ev_p), ps(ev_t)))
    return out, P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?", default="data/a8_bundle_seed0.zip")
    ap.add_argument("--train", default="2"); ap.add_argument("--test", default="3")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args(argv)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    b = a8_io.Bundle(args.bundle); M, _ = a8_io.spillover(b)
    tr = [c for c in b.classes if c[0] in args.train]; te = [c for c in b.classes if c[0] in args.test]
    Xtr, Ytr, Mtr, wtr = load(b, tr, M); Xte, Yte, Mte, wte = load(b, te, M)
    # targets are in tiny float units; scale so L1 is O(1), undo at evaluation
    scale = float(Ytr.abs().mean()) * 10
    print("device %s   train %s %d events   test %s %d events   target scale %.4f" % (dev, tr, len(Xtr), te, len(Xte), scale))

    model = UNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    names = ["LDHC/AKAP4", "CD45", "ACRV1"]
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            x, y, m = Xtr[idx].to(dev), (Ytr[idx] / scale).to(dev), Mtr[idx].to(dev)
            if np.random.rand() < 0.5: x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)   # scan axis is symmetric
            loss = (F.l1_loss(model(x), y, reduction="none") * m).sum() / m.sum() / 3
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            res, _ = evaluate(model, Xte, Yte, Mte, wte, scale, dev)
            print("epoch %3d  train L1 %.4f   test pixel r: %s   (%.0fs)" % (
                ep, tot / len(Xtr), "  ".join("%s %.3f" % (n, r[0]) for n, r in zip(names, res)), time.time() - t0))

    res, P = evaluate(model, Xte, Yte, Mte, wte, scale, dev)
    print("\n%-12s %9s %9s %12s %12s   %s" % ("marker", "pixel r", "event r", "pred P:S", "true P:S", "want"))
    for n, (pr, er, pps, tps), want in zip(names, res, ["<< 1", ">> 1", "<< 1"]):
        print("%-12s %9.3f %9.3f %12.2f %12.2f   %s" % (n, pr, er, pps, tps, want))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pick = [np.where(wte == w)[0][j] for w in ("S", "P", "SP") for j in (3, 40)]
    fig, axes = plt.subplots(len(pick), 7, figsize=(14, 1.9 * len(pick)))
    cols = ["LightLoss in", "true LDHC", "pred LDHC", "true CD45", "pred CD45", "true ACRV1", "pred ACRV1"]
    for i, j in enumerate(pick):
        panels = [Xte[j, 0].numpy()] + [v for k in range(3) for v in (Yte[j, k].numpy(), P[j, k])]
        for c, (ax, a) in enumerate(zip(axes[i], panels)):
            if c == 0: ax.imshow(a, cmap="gray")
            else:
                ref = panels[c if c % 2 == 1 else c - 1]; hi = max(np.percentile(ref, 99.5), 1e-4)
                ax.imshow(a, cmap="magma", vmin=0, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0: ax.set_ylabel("%s" % wte[j], fontsize=8)
            if i == 0: ax.set_title(cols[c], fontsize=8)
    fig.suptitle("pilot U-Net, replicate 2 -> 3: each pred shares its true panel's colour scale", fontsize=9)
    fig.tight_layout(); os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "pilot_unet_preds.png"); fig.savefig(path, dpi=120); print("\nwrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
