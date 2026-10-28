"""A convolutional autoencoder on the aligned virtual-stain stacks: trained with a
reconstruction loss on the label-free side only (the virtual stains plus the inputs), decoder
discarded, latent vectors clustered with a Gaussian mixture, stains and wells used only
afterwards to say what the clusters are.

    python analysis/autoencoder_embedding.py data/isx_vstain/features.pixels.npz --out data/isx_vstain/emb_dcae.npz --latent 32 --epochs 60

Inputs are 7 x 48 x 48 stacks (4 predicted stains + 3 inputs on the ImageStream, 6 channels on
the A8) aligned head-left. Light augmentation (small rotations, shifts and vertical flips, which
keep the head on the left) makes the latent space tolerant of pose.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_stack(path):
    d = np.load(path); P = d["pixels"].astype(np.float32); ch = list(d["channels"]); idx = d["index"]
    stain = [k for k, c in enumerate(ch) if not c.startswith(("brightfield", "lightloss", "fsc", "ssc", "darkfield"))]
    X = P.copy()
    for k in range(len(ch)):
        if k in stain:
            v = np.clip(X[:, k], 0, None); hi = max(float(np.percentile(v, 99.5)), 1e-6); X[:, k] = np.clip(v / hi, 0, 1.5)    # one scale per stain channel across the dataset
        else:
            X[:, k] = np.clip(X[:, k] / 3.0, -2, 2)                                                                     # inputs are standardised already
    return X, idx, ch


def build(cin, latent, side):
    import torch.nn as nn
    class AE(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Conv2d(cin, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),      # 24
                                  nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),       # 12
                                  nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),     # 6
                                  nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True))    # 3
            s.fc = nn.Linear(256 * (side // 16) ** 2, latent)
            s.dfc = nn.Linear(latent, 256 * (side // 16) ** 2)
            s.dec = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
                                  nn.ConvTranspose2d(32, cin, 4, 2, 1))
            s.side = side
        def encode(s, x): return s.fc(s.enc(x).flatten(1))
        def forward(s, x):
            z = s.encode(x); h = s.dfc(z).view(-1, 256, s.side // 16, s.side // 16)
            return s.dec(h), z
    return AE()


def augment(x, rng):
    """Small rotation, shift, vertical flip (head stays left), intensity jitter. x: (B, C, H, W) numpy."""
    from scipy import ndimage as ndi
    out = np.empty_like(x)
    for i in range(len(x)):
        a = x[i]
        if rng.random() < 0.5: a = a[:, ::-1, :]
        ang = rng.uniform(-15, 15); dy, dx = rng.integers(-3, 4, 2)
        a = np.stack([ndi.shift(ndi.rotate(c, ang, reshape=False, order=1, mode="nearest"), (dy, dx), order=1, mode="nearest") for c in a])
        out[i] = a * rng.uniform(0.85, 1.15)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pixels"); ap.add_argument("--out", required=True); ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-aug", action="store_true"); ap.add_argument("--stains-only", action="store_true")
    ap.add_argument("--inputs-only", action="store_true", help="train on the label-free input channels alone (no virtual stains)")
    a = ap.parse_args(argv)
    import torch
    from sperm_fluorescence.train.device import pick_device
    X, idx, ch = load_stack(a.pixels)
    is_input = [c.startswith(("brightfield", "lightloss", "fsc", "ssc", "darkfield")) for c in ch]
    if a.stains_only or a.inputs_only:
        keep = [k for k, inp in enumerate(is_input) if inp == a.inputs_only]; X = X[:, keep]; ch = [ch[k] for k in keep]
    dev = pick_device(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    model = build(X.shape[1], a.latent, X.shape[-1]).to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    n = len(X); print("training on %d stacks of %s (%s), latent %d, %d epochs on %s" % (n, "x".join(str(v) for v in X.shape[1:]), ",".join(ch), a.latent, a.epochs, dev))
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, a.batch):
            xb = X[perm[i:i + a.batch]]
            if not a.no_aug: xb = augment(xb, rng)
            xb = torch.tensor(np.ascontiguousarray(xb), device=dev)
            rec, _ = model(xb); loss = torch.nn.functional.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
        if ep % 10 == 9 or ep == a.epochs - 1: print("  epoch %3d  reconstruction MSE %.5f" % (ep + 1, tot / nb))
    model.eval(); Z = []
    with torch.no_grad():
        for i in range(0, n, 256): Z.append(model.encode(torch.tensor(X[i:i + 256], device=dev)).cpu().numpy())
    Z = np.concatenate(Z)
    np.savez(a.out, embedding=Z.astype(np.float32), index=idx, channels=np.array(ch), latent=a.latent, epochs=a.epochs)
    torch.save(model.state_dict(), os.path.splitext(a.out)[0] + ".pt")
    print("wrote %s: %d x %d latent" % (a.out, *Z.shape))


if __name__ == "__main__":
    main()
