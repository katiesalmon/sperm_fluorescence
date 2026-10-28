"""The virtual-stain store: what a model predicted for every object of a cache, on disk, in
the cache's own ragged layout so the two index identically.

    <store>/meta.json           source cache and classes, run and checkpoint, roles used, marker
                                names, mask thresholds, per-class summaries
    <store>/<cls>_p.npy         (C_out, total_px) float16   predicted stains, in cache units
    <store>/<cls>_pm.npy        (C_out, total_px) bool      predicted masks (threshold rule, inside the valid frame)
    <store>/<cls>_im.npy        (C_out, total_px) bool      IDEAS masks of the measured channels (ISX, when the .cif was given)
    <store>/<cls>_shapes.npy, _offsets.npy, _m.npy, _rows.npy    copied from the cache, so the store indexes itself

Object i of class cls in the store is object i of class cls in the cache: `open_store` and
`open_cache` over the same class list give the same global index. Predictions are stored at
the cache's padded frame size (multiples of 8) with `m` marking the object's own pixels.
"""
import json
import os
import shutil

import numpy as np

COPIED = ("shapes", "offsets", "m", "rows")


def write_class_arrays(store, cls, cout, total_px, with_ideas):
    """Open the memmaps a class is written into. Returns (p, pm, im_or_None)."""
    p = np.lib.format.open_memmap(os.path.join(store, cls + "_p.npy"), "w+", np.float16, (cout, int(total_px)))
    pm = np.lib.format.open_memmap(os.path.join(store, cls + "_pm.npy"), "w+", bool, (cout, int(total_px)))
    im = np.lib.format.open_memmap(os.path.join(store, cls + "_im.npy"), "w+", bool, (cout, int(total_px))) if with_ideas else None
    return p, pm, im


def copy_index(cache, store, cls):
    """The cache's per-class index arrays, copied beside the predictions."""
    for suffix in COPIED:
        src = os.path.join(cache, "%s_%s.npy" % (cls, suffix))
        if os.path.exists(src): shutil.copyfile(src, os.path.join(store, "%s_%s.npy" % (cls, suffix)))


def source_rows(cache, cls, n_cif=None):
    """Object index in the source .cif for each object of an ISX class. New caches store it
    (<cls>_rows.npy, meta source_rows); legacy caches wrote 2I = first half, 3I = the rest."""
    meta = json.load(open(os.path.join(cache, "meta.json")))
    p = os.path.join(cache, cls + "_rows.npy")
    if meta.get("source_rows") and os.path.exists(p):
        return np.load(p).astype(np.int64)
    n = len(np.load(os.path.join(cache, cls + "_shapes.npy")))
    p2 = os.path.join(cache, "2I_shapes.npy")
    n2i = len(np.load(p2)) if os.path.exists(p2) else (n_cif or 2 * n) // 2
    return np.arange(n) if cls == "2I" else n2i + np.arange(n)


class Store:
    """Read side. item(i) -> (p (C, h, w) float32, pm (C, h, w) bool, m (h, w) bool, im (C, h, w) bool or None)."""

    def __init__(self, store, classes=None):
        self.path = store
        self.meta = json.load(open(os.path.join(store, "meta.json")))
        self.names = list(self.meta["target_names"]); self.cout = len(self.names)
        self.classes = list(classes) if classes else list(self.meta["classes"])
        self.ps, self.pms, self.ims, self.ms, self.per_class, self.rows = [], [], [], [], [], []
        for cls in self.classes:
            if cls not in self.meta["classes"]: raise SystemExit("class %s not in store %s (has %s)" % (cls, store, list(self.meta["classes"])))
            self.ps.append(np.load(os.path.join(store, cls + "_p.npy"), mmap_mode="r"))
            self.pms.append(np.load(os.path.join(store, cls + "_pm.npy"), mmap_mode="r"))
            ip = os.path.join(store, cls + "_im.npy")
            self.ims.append(np.load(ip, mmap_mode="r") if os.path.exists(ip) else None)
            self.ms.append(np.load(os.path.join(store, cls + "_m.npy"), mmap_mode="r"))
            self.per_class.append((np.load(os.path.join(store, cls + "_shapes.npy")), np.load(os.path.join(store, cls + "_offsets.npy"))))
            rp = os.path.join(store, cls + "_rows.npy")
            self.rows.append(np.load(rp) if os.path.exists(rp) else None)
        self.counts = np.cumsum([0] + [len(s) for s, _ in self.per_class])
        self.shapes = np.concatenate([s for s, _ in self.per_class]) if self.per_class else np.zeros((0, 2), int)
        self.class_of = np.array([c for c, (s, _) in zip(self.classes, self.per_class) for _ in range(len(s))])

    def __len__(self): return int(self.counts[-1])

    def locate(self, i):
        k = int(np.searchsorted(self.counts, i, side="right") - 1); j = int(i - self.counts[k])
        shapes, offsets = self.per_class[k]
        return k, j, tuple(int(v) for v in shapes[j]), int(offsets[j]), int(offsets[j + 1])

    def item(self, i):
        k, j, (h, w), a, b = self.locate(i)
        p = np.asarray(self.ps[k][:, a:b]).reshape(self.cout, h, w).astype(np.float32)
        pm = np.asarray(self.pms[k][:, a:b]).reshape(self.cout, h, w)
        m = np.asarray(self.ms[k][a:b]).reshape(h, w)
        im = np.asarray(self.ims[k][:, a:b]).reshape(self.cout, h, w) if self.ims[k] is not None else None
        return p, pm, m, im

    def source_row(self, i):
        k, j, _, _, _ = self.locate(i)
        return int(self.rows[k][j]) if self.rows[k] is not None else None


def open_store(store, classes=None):
    return Store(store, classes)
