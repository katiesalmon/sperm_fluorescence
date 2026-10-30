"""The on-disk cache: one set of arrays per class (sample), ragged native-size objects.

    <cache>/<cls>_x.npy        (C_in, total_px) float16   label-free roles, standardised per object
    <cache>/<cls>_y.npy        (C_out, total_px) float16  stain targets, background-subtracted, scaled
    <cache>/<cls>_m.npy        (total_px,) bool            valid (non-padding) pixels
    <cache>/<cls>_shapes.npy, _offsets.npy                 per-object (h, w) and flat offsets
    <cache>/<cls>_avail.npy    (N, C_out) bool             which targets this object's instrument stained
    <cache>/<cls>_scalars.npy, _rows.npy                   per-event scalars (ISX DAPI) and source rows
    <cache>/meta.json          roles, target_names, classes {instrument, avail, repeat}, scales

A fixed-frame cache ((N, C, H, W) arrays, no _shapes) is still read, for the older A8 runs.
"""
import json
import os

import numpy as np

from .batching import Cached, RaggedCached


def standardise(x):
    """Per-object, per-channel zero mean and unit variance for the label-free input."""
    return (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)


def to_multiple(a, k=8):
    """Pad (C, h, w) up to the next multiple of `k` on each axis with the per-channel border
    median; return the frame and its validity mask. Nothing is ever cropped."""
    C, h, w = a.shape
    H = -(-h // k) * k; W = -(-w // k) * k
    border = np.concatenate([a[:, 0, :], a[:, -1, :], a[:, :, 0], a[:, :, -1]], axis=1)
    fill = np.median(border, axis=1)
    out = np.empty((C, H, W), np.float32); out[:] = fill[:, None, None]
    bh, bw = (H - h) // 2, (W - w) // 2
    out[:, bh:bh + h, bw:bw + w] = a
    m = np.zeros((H, W), bool); m[bh:bh + h, bw:bw + w] = True
    return out, m


def write_ragged(cls, items, cache):
    """items: list of (x, y, m) at native size. Stored flat with an index, so every object
    keeps its own height and width and a batch is assembled from objects of like shape."""
    shapes = np.array([x.shape[1:] for x, _, _ in items], np.int32)
    px = (shapes[:, 0] * shapes[:, 1]).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(px)))
    cx, cy = items[0][0].shape[0], items[0][1].shape[0]
    X = np.lib.format.open_memmap(os.path.join(cache, cls + "_x.npy"), "w+", np.float16, (cx, int(offsets[-1])))
    Y = np.lib.format.open_memmap(os.path.join(cache, cls + "_y.npy"), "w+", np.float16, (cy, int(offsets[-1])))
    Mk = np.lib.format.open_memmap(os.path.join(cache, cls + "_m.npy"), "w+", bool, (int(offsets[-1]),))
    for i, (x, y, m) in enumerate(items):
        a, b = offsets[i], offsets[i + 1]
        X[:, a:b] = x.reshape(cx, -1); Y[:, a:b] = y.reshape(cy, -1); Mk[a:b] = m.ravel()
    X.flush(); Y.flush(); Mk.flush()
    np.save(os.path.join(cache, cls + "_shapes.npy"), shapes)
    np.save(os.path.join(cache, cls + "_offsets.npy"), offsets)


def load_cache(cache, classes):
    """Open the per-class arrays. Returns (xs, ys, ms, wells, scalars, extra): extra is None
    for a fixed-frame cache and [(shapes, offsets), ...] for a ragged one."""
    xs, ys, ms, wells, scal, extra, avail, clsarr, present = [], [], [], [], [], [], [], [], []
    for cls in classes:
        p = os.path.join(cache, cls + "_x.npy")
        if not os.path.exists(p): continue
        xs.append(np.load(p, mmap_mode="r")); ys.append(np.load(os.path.join(cache, cls + "_y.npy"), mmap_mode="r"))
        ms.append(np.load(os.path.join(cache, cls + "_m.npy"), mmap_mode="r"))
        sp = os.path.join(cache, cls + "_shapes.npy")
        if os.path.exists(sp):
            shapes = np.load(sp); extra.append((shapes, np.load(os.path.join(cache, cls + "_offsets.npy")))); n = len(shapes)
        else:
            n = len(xs[-1])
        wells += [cls.lstrip("123")] * n; clsarr += [cls] * n
        scal.append(np.load(os.path.join(cache, cls + "_scalars.npy")))
        ap_ = os.path.join(cache, cls + "_avail.npy")
        avail.append(np.load(ap_) if os.path.exists(ap_) else None)
        pp_ = os.path.join(cache, cls + "_present.npy")
        present.append(np.load(pp_) if os.path.exists(pp_) else None)
    avail = np.concatenate(avail) if avail and all(a is not None for a in avail) else None
    present = np.concatenate(present) if present and all(p is not None for p in present) else None
    return xs, ys, ms, np.array(wells), np.concatenate(scal), (extra if extra else None), avail, np.array(clsarr), present


def cache_roles(cache):
    """The x channel names of a cache: meta 'roles' (new), 'inputs' (generalist), or None."""
    p = os.path.join(cache, "meta.json")
    if not os.path.exists(p): return None
    meta = json.load(open(p))
    return list(meta.get("roles") or meta.get("inputs") or []) or None


def open_cache(cache, classes, target_k=None, roles=None, denoise=None):
    """A dataset plus its per-object wells, scalars, and class names. `ds.avail` is an
    (N, C) bool of which target channels each object actually has, or None for all;
    `ds.present` the same for input roles. `roles`: names to keep, in that order."""
    xs, ys, ms, wells, scal, extra, avail, clsarr, present = load_cache(cache, classes)
    if extra is None:
        if target_k is not None:
            ys = [y[:, target_k:target_k + 1] for y in ys]
        ds = Cached(xs, ys, ms)
    else:
        ds = RaggedCached(xs, ys, ms, extra, target_k)
    if avail is not None and target_k is not None:
        avail = avail[:, target_k:target_k + 1]
    ds.avail = avail; ds.classes = clsarr; ds.present = present
    if denoise and denoise != "none":
        if not getattr(ds, "ragged", False): raise SystemExit("denoise= is only implemented for ragged caches")
        ds.denoise = denoise
    all_roles = cache_roles(cache)
    ds.roles = list(all_roles) if all_roles else ["ch%d" % i for i in range(ds.cin)]
    if roles:
        missing = [r for r in roles if r not in ds.roles]
        if missing: raise SystemExit("roles %s not in this cache (has %s)" % (missing, ds.roles))
        ds.select_roles([ds.roles.index(r) for r in roles]); ds.roles = list(roles)
    return ds, wells, scal

