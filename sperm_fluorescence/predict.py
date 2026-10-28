"""Virtual stains for every object of a cache: run a trained model and write the predicted
stain images and their masks to a store (data/store.py) that indexes like the cache.

    python -m sperm_fluorescence predict runs/srv_isx_stems --cache data/isx_roles --classes 3I \\
        --isx data/isx/ALL_1_400_1.cif --tta --out data/isx_vstain
    python -m sperm_fluorescence predict runs/srv_isx_stems --cache data/isx_unstained --out data/isx_unstained_vstain

The cache may be a stained one (the measured channels then sit beside the predictions, in the
cache) or an unstained one (`prepare --unstained`: zero targets, nothing marked available).
Mask thresholds come from the run's masks.json when it has them (the rule that best matched the
instrument's masks on training objects), else ("rel", 0.15); --threshold overrides one marker.
With --isx, the IDEAS masks of the measured channels are stored too, for feature validation.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from .data.batching import make_batches, present_of
from .data.store import copy_index, source_rows, write_class_arrays
from .eval.load import load_run
from .eval.masks import ideas_mask, pred_mask
from .eval.metrics import predict
from .models import registry
from .train.device import release_device_memory

DEFAULT_RULE = ("rel", 0.15)


def thresholds_for(run, names, instruments, overrides):
    """{(instrument, marker): (kind, value)} from the run's masks.json, the default, and overrides."""
    out = {}
    mp = os.path.join(run, "masks.json")
    saved = json.load(open(mp)).get("thresholds", {}) if os.path.exists(mp) else {}
    for inst in instruments:
        for n in names:
            kind, v = saved.get("%s:%s" % (inst, n), DEFAULT_RULE)
            out[(inst, n)] = (kind, float(v))
    for spec in overrides:
        name, rule = spec.split("=", 1); kind, v = rule.split(":")
        for inst in instruments: out[(inst, name)] = (kind, float(v))
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence predict", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default=None, help="comma list of cache classes (default: every class in the cache)")
    ap.add_argument("--checkpoint", default="best.pt"); ap.add_argument("--tta", action="store_true"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--isx", default=None, help=".cif the ISX classes were cut from: stores the IDEAS masks of the measured channels")
    ap.add_argument("--threshold", action="append", default=[], metavar="MARKER=rel:0.15", help="override a marker's mask rule (rel:f or abs:v)")
    ap.add_argument("--limit", type=int, default=0, help="objects per class (0 = all), for a quick look")
    a = ap.parse_args(argv)
    import torch
    cache_meta = json.load(open(os.path.join(a.cache, "meta.json")))
    classes = [c.strip() for c in a.classes.split(",")] if a.classes else list(cache_meta["classes"])
    missing = [c for c in classes if c not in cache_meta["classes"]]
    if missing: raise SystemExit("classes %s not in %s (has %s)" % (missing, a.cache, list(cache_meta["classes"])))
    digits = "".join(sorted({c[0] for c in classes}))
    R = load_run(a.run, a.cache, digits, ",".join(classes), a.checkpoint)
    ds, names, dev, scale = R.te, R.names, R.dev, R.scale
    if list(ds.classes[ds.counts[:-1]]) != classes:                     # keep the caller's order for the store
        classes = [str(c) for c in ds.classes[ds.counts[:-1]]]
    inst_of = {c: cache_meta["classes"][c]["instrument"] for c in classes}
    thr = thresholds_for(a.run, names, sorted(set(inst_of.values())), a.threshold)
    cif = None
    if a.isx:
        from .io import isx as isx_io
        cif = isx_io.CIF(a.isx); isx_stains = registry.instrument("isx")["stains"]
    os.makedirs(a.out, exist_ok=True)
    meta = {"layout": "virtual-stain", "source_cache": os.path.abspath(a.cache), "run": os.path.abspath(a.run), "checkpoint": a.checkpoint,
            "tta": bool(a.tta), "roles": list(ds.roles), "target_names": list(names), "scale": R.scale.view(-1).tolist(),
            "thresholds": {"%s:%s" % (i, n): list(v) for (i, n), v in thr.items()}, "classes": {},
            "source_rows": True, "created": time.strftime("%Y-%m-%d %H:%M"), "commit": git_commit(), "device": str(dev)}
    sc = scale.view(-1).numpy()
    for k, cls in enumerate(classes):
        lo, hi = int(ds.counts[k]), int(ds.counts[k + 1])
        idx = np.arange(lo, hi if not a.limit else min(hi, lo + a.limit))
        shapes, offsets = ds.per_class[k]
        total = int(offsets[len(idx)]) if a.limit else int(offsets[-1])
        avail = ds.avail[lo] if ds.avail is not None else np.zeros(ds.cout, bool)
        with_ideas = cif is not None and inst_of[cls] == "isx" and bool(avail.any())
        rows = source_rows(a.cache, cls, len(cif) if cif is not None else None) if with_ideas else None
        P, PM, IM = write_class_arrays(a.out, cls, ds.cout, total, with_ideas)
        rules = [thr[(inst_of[cls], n)] for n in names]
        ev_sum = np.zeros((len(idx), ds.cout), np.float32); has_mask = np.zeros((len(idx), ds.cout), bool)
        t0 = time.time(); done = 0
        with torch.no_grad():
            for nb, b in enumerate(make_batches(ds, idx, a.batch)):
                x, _, m = ds.batch(b); present = present_of(ds, b)
                p = (predict(R.model, x.to(dev), a.tta, present.to(dev)).float().cpu() * scale).numpy(); mb = m.numpy()[:, 0]
                H, W = p.shape[-2:]
                for q, j in enumerate(b):
                    jj = int(j) - lo; h, w = ds.shape(int(j)); t, l = (H - h) // 2, (W - w) // 2
                    pj = p[q][:, t:t + h, l:l + w]; mj = mb[q][t:t + h, l:l + w] > 0
                    aa, bb = int(offsets[jj]), int(offsets[jj + 1])
                    P[:, aa:bb] = pj.reshape(ds.cout, -1).astype(np.float16)
                    for c in range(ds.cout):
                        pm = pred_mask(pj[c], rules[c], mj.astype(np.float32))
                        PM[c, aa:bb] = pm.ravel(); has_mask[jj, c] = pm.any(); ev_sum[jj, c] = float((pj[c] * mj).sum())
                        if with_ideas and avail[c]:
                            IM[c, aa:bb] = ideas_mask(cif, int(rows[jj]), isx_stains[names[c]], (h, w)).ravel()
                done += len(b)
                if nb % 20 == 19: release_device_memory(dev)
                if done % 2000 < len(b): print("  %s %d / %d  (%.0fs)" % (cls, done, len(idx), time.time() - t0))
        P.flush(); PM.flush()
        if IM is not None: IM.flush()
        copy_index(a.cache, a.out, cls)
        if inst_of[cls] == "isx":                       # resolve the .cif row for every object, even from a legacy cache that wrote arange
            np.save(os.path.join(a.out, cls + "_rows.npy"), source_rows(a.cache, cls, len(cif) if cif is not None else None))
        if a.limit:                                     # the copied index covers the whole class; trim it to what was written
            np.save(os.path.join(a.out, cls + "_shapes.npy"), shapes[:len(idx)]); np.save(os.path.join(a.out, cls + "_offsets.npy"), offsets[:len(idx) + 1])
            np.save(os.path.join(a.out, cls + "_m.npy"), np.asarray(ds.ms[k][:total]))
            rp = os.path.join(a.out, cls + "_rows.npy")
            if os.path.exists(rp): np.save(rp, np.load(rp)[:len(idx)])
        meta["classes"][cls] = {"n": int(len(idx)), "instrument": inst_of[cls], "avail": [bool(v) for v in avail], "ideas_masks": with_ideas,
                                "summary": {n: {"mean_event_sum": float(ev_sum[:, c].mean()), "mask_fraction": float(has_mask[:, c].mean())} for c, n in enumerate(names)}}
        print("%s: %d objects, %.0fs; mask fraction %s" % (cls, len(idx), time.time() - t0, " ".join("%s %.2f" % (n, has_mask[:, c].mean()) for c, n in enumerate(names))))
        release_device_memory(dev)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("wrote store", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
