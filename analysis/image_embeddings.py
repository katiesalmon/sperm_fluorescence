"""Representations of the virtual stain images that need no hand-made features, written as
embeddings `populate --embedding` can cluster:

    python analysis/image_embeddings.py data/isx_vstain runs/srv_isx_stems --classes 3I --out data/isx_vstain/emb

  blur12 / blur6   the aligned stain channels (features.pixels.npz, head left), each normalised to
                   unit sum, Gaussian-blurred (sigma 2 px = 1.3 um) and block-averaged to 12x12 or 6x6
  stainpix24       the same without blur at 24x24 (a cheaper raw-pixel baseline)
  bottleneck       the model's own deepest activations (U-Net e4, 8 x width channels at 1/8
                   resolution), mean- and max-pooled over the object: what the network itself
                   computed in order to paint the stains
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def blocks(a, n):
    """Block-average an (h, w) image to (n, n)."""
    h, w = a.shape; ys = np.linspace(0, h, n + 1).astype(int); xs = np.linspace(0, w, n + 1).astype(int)
    return np.array([[a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(n)] for i in range(n)])


def pixel_embeddings(pixels_npz, out, stains_only=True):
    d = np.load(pixels_npz); P = d["pixels"].astype(np.float32); ch = list(d["channels"]); idx = d["index"]
    keep = [k for k, c in enumerate(ch) if not c.startswith(("brightfield", "lightloss", "fsc", "ssc", "darkfield"))] if stains_only else list(range(len(ch)))
    S = np.clip(P[:, keep], 0, None); S = S / np.maximum(S.reshape(len(S), len(keep), -1).sum(-1), 1e-6)[:, :, None, None]
    for name, n, sigma in (("blur12", 12, 2.0), ("blur6", 6, 2.0), ("stainpix24", 24, 0.0)):
        E = np.stack([np.concatenate([blocks(ndi.gaussian_filter(S[i, k], sigma) if sigma else S[i, k], n).ravel() for k in range(len(keep))]) for i in range(len(S))])
        np.savez(out + "_%s.npz" % name, embedding=E.astype(np.float32), index=idx, channels=np.array([ch[k] for k in keep]))
        print("%-12s %d objects x %d dims" % (name, *E.shape))


def bottleneck_embedding(store, run, classes, out, checkpoint="best.pt", bs=32):
    import torch
    from sperm_fluorescence.data.batching import make_batches, present_of
    from sperm_fluorescence.data.store import open_store
    from sperm_fluorescence.eval.load import load_run
    from sperm_fluorescence.models.stems import call
    st = open_store(store, classes)
    digits = "".join(sorted({c[0] for c in st.classes}))
    R = load_run(run, st.meta["source_cache"], digits, ",".join(st.classes), checkpoint); ds, dev = R.te, R.dev
    feats = {}
    hook = R.model.trunk.e4.register_forward_hook(lambda m, i, o: feats.__setitem__("e4", o))
    E = np.zeros((len(ds), 2 * R.model.trunk.e4.net[-2].num_features), np.float32)   # net[-2] is the last BatchNorm
    with torch.no_grad():
        for b in make_batches(ds, np.arange(len(ds)), bs):
            x, _, m = ds.batch(b); call(R.model, x.to(dev), present_of(ds, b).to(dev))
            f = feats["e4"].float()                                    # (B, 8w, H/8, W/8)
            mm = torch.nn.functional.adaptive_max_pool2d(m.to(dev), f.shape[-2:])   # valid-frame mask at e4 resolution
            w = mm / mm.sum((2, 3), keepdim=True).clamp(min=1.0)
            mean = (f * w).sum((2, 3)); mx = (f * mm + (mm - 1) * 1e4).amax((2, 3))
            E[np.asarray(b)] = torch.cat([mean, mx], 1).cpu().numpy()
    hook.remove()
    np.savez(out + "_bottleneck.npz", embedding=E, index=np.arange(len(ds)))
    print("%-12s %d objects x %d dims (U-Net e4 mean+max pooled)" % ("bottleneck", *E.shape))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store"); ap.add_argument("run"); ap.add_argument("--classes", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--pixels", default=None, help="default: <store>/features.pixels.npz")
    a = ap.parse_args(argv)
    pixel_embeddings(a.pixels or os.path.join(a.store, "features.pixels.npz"), a.out)
    bottleneck_embedding(a.store, a.run, [c.strip() for c in a.classes.split(",")] if a.classes else None, a.out)


if __name__ == "__main__":
    main()
