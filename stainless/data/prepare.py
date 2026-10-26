"""One `prepare` for every instrument: read events, name their label-free roles and stain
targets from the instrument config, and write a ragged cache the trainer can select roles
and markers from.

    python -m stainless prepare --instrument a8  --source data/a8_bundle_seed0.zip --cache data/a8_roles
    python -m stainless prepare --instrument isx --source data/isx/ALL_1_400_1.cif --daf data/isx/ALL_1_400_1.daf --cache data/isx_roles
    python -m stainless prepare --instrument a8 --source ... --instrument isx --source ... --daf ... --cache data/both_roles

The cache layout is data/cache.py's. meta.json carries:
    roles          the x channel names, in order (union over instruments; a role an instrument
                   lacks is a zero channel, and <cls>_present.npy says which roles each object has)
    target_names   the marker names, in order (union over instruments; <cls>_avail.npy per object)
    classes        {cls: {instrument, family, avail, present, repeat}}
    scales         per instrument and marker: the divisor that brought targets to unit mean |y|

Targets are background-subtracted stain images scaled per instrument and marker to unit
mean magnitude, so instruments with different units meet on equal terms. Inputs are
standardised per object and channel. The A8 is resampled to the ImageStream pitch
(instrument config `zoom_to_reference`) so one model sees one scale.
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi

from ..io import a8 as a8_io
from ..io import isx as isx_io
from ..io import isx_daf
from ..models import registry
from .cache import standardise, to_multiple, write_ragged


def a8_records(cfg, source, limit=0):
    """Yield (cls, x (R, h, w), y (M, h, w), scalars) for an a8_bundle zip."""
    b = a8_io.Bundle(source); M, _ = a8_io.spillover(b)
    skip = set(cfg["classes"].get("skip", [])); zoom = float(cfg.get("zoom_to_reference", 1.0))
    roles = cfg["roles"]; stains = cfg["stains"]
    for cls in b.classes:
        if cls in skip: continue
        rows = b.rows(cls)[:limit] if limit else b.rows(cls)
        items = []
        for r in rows:
            st = a8_io.unmix(b.image(cls, r), M)
            if zoom != 1.0: st = ndi.zoom(st, (1, zoom, zoom), order=1).astype(np.float32)
            x = standardise(np.stack([st[p] for p in roles.values()]))
            y = np.stack([st[p] for p in stains.values()]).astype(np.float32)
            items.append((x, y, np.nan))
        yield cls, items


def isx_records(cfg, source, daf, limit=0):
    c = isx_io.CIF(source); roles = cfg["roles"]; stains = cfg["stains"]
    scal_name = cfg["classes"].get("scalar"); F = col = None
    if daf and scal_name:
        names, F, _ = isx_daf.read_features(daf); col = {n: i for i, n in enumerate(names)}
    n = len(c) if not limit else min(limit, len(c)); half = n // 2
    for cls, rows in (("2I", range(0, half)), ("3I", range(half, n))):
        items = []; t0 = time.time()
        for i in rows:
            img = c.image(i).astype(np.float32); mk = c.mask(i); mc = mk.any(axis=0)
            bg = np.array([np.median(img[k][~mc]) if (~mc).any() else np.median(img[k]) for k in range(img.shape[0])])
            x = standardise(np.stack([img[ch] for ch in roles.values()]))
            y = np.stack([img[ch] - bg[ch] for ch in stains.values()]).astype(np.float32)
            s = float(F[i, col[scal_name]]) if F is not None else np.nan
            items.append((x, y, s))
        print("isx %s: %d objects decoded (%.0fs)" % (cls, len(items), time.time() - t0))
        yield cls, items


READERS = {"a8_bundle": a8_records, "isx_cif": isx_records}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stainless prepare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", action="append", required=True, help="config name (a8, isx, ...) or path; repeat for several")
    ap.add_argument("--source", action="append", required=True, help="one per --instrument: bundle zip, .cif, ...")
    ap.add_argument("--daf", action="append", default=[], help="ImageStream .daf (features), one per isx instrument")
    ap.add_argument("--cache", required=True); ap.add_argument("--limit", type=int, default=0, help="events per class (0 = all)")
    a = ap.parse_args(argv)
    if len(a.source) != len(a.instrument): ap.error("one --source per --instrument")
    cfgs = [registry.instrument(n) for n in a.instrument]
    roles = []; markers = []
    for cfg in cfgs:
        roles += [r for r in cfg["roles"] if r not in roles]; markers += [m for m in cfg["stains"] if m not in markers]
    os.makedirs(a.cache, exist_ok=True)
    meta = {"layout": "ragged", "roles": roles, "target_names": markers, "classes": {}, "scales": {}, "instruments": {}}
    dafs = list(a.daf)
    for cfg, src in zip(cfgs, a.source):
        inst = cfg["name"]; meta["instruments"][inst] = {"family": cfg.get("family"), "pixel_um": cfg.get("pixel_um"), "zoom_to_reference": cfg.get("zoom_to_reference", 1.0)}
        present = [r in cfg["roles"] for r in roles]; avail = [m in cfg["stains"] for m in markers]
        rk = [list(cfg["roles"]).index(r) if r in cfg["roles"] else None for r in roles]
        mk = [list(cfg["stains"]).index(m) if m in cfg["stains"] else None for m in markers]
        reader = READERS[cfg["reader"]]
        gen = reader(cfg, src, dafs.pop(0) if cfg["reader"] == "isx_cif" and dafs else None, a.limit) if cfg["reader"] == "isx_cif" else reader(cfg, src, a.limit)
        raw = {}
        for cls, items in gen:
            full = []
            for x, y, s in items:
                X = np.zeros((len(roles),) + x.shape[1:], np.float32); Y = np.zeros((len(markers),) + y.shape[1:], np.float32)
                for j, k in enumerate(rk):
                    if k is not None: X[j] = x[k]
                for j, k in enumerate(mk):
                    if k is not None: Y[j] = y[k]
                full.append((X, Y, s))
            raw[cls] = full
            print("%s %s: %d objects" % (inst, cls, len(full)))
        # per-instrument, per-marker scale to unit mean |y| from the training-digit classes (fallback: all)
        tr = [c for c in raw if c[0] == "2"] or list(raw)
        scale = np.ones(len(markers), np.float32)
        for j in range(len(markers)):
            if avail[j]:
                vals = np.concatenate([np.abs(y[j]).ravel()[::7] for c in tr for _, y, _ in raw[c][:300]])
                scale[j] = max(float(vals.mean()), 1e-6)
        meta["scales"][inst] = scale.tolist()
        for cls, full in raw.items():
            out = []
            for X, Y, _ in full:
                fx, m = to_multiple(X); fy, _ = to_multiple(Y / scale[:, None, None]); out.append((fx, fy, m))
            write_ragged(cls, out, a.cache)
            np.save(os.path.join(a.cache, cls + "_avail.npy"), np.tile(np.array(avail, bool), (len(out), 1)))
            np.save(os.path.join(a.cache, cls + "_present.npy"), np.tile(np.array(present, bool), (len(out), 1)))
            np.save(os.path.join(a.cache, cls + "_rows.npy"), np.arange(len(out)))
            np.save(os.path.join(a.cache, cls + "_scalars.npy"), np.array([s for _, _, s in full], np.float32)[:, None])
            meta["classes"][cls] = {"instrument": inst, "family": cfg.get("family"), "avail": avail, "present": present, "repeat": int(cfg["classes"].get("repeat", 1))}
            print("wrote %s: %d objects, roles %s, markers %s" % (cls, len(out), [r for r, p in zip(roles, present) if p], [m for m, v in zip(markers, avail) if v]))
    json.dump(meta, open(os.path.join(a.cache, "meta.json"), "w"), indent=1)
    print("cache %s: roles %s, markers %s" % (a.cache, roles, markers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
