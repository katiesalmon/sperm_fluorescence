"""One `prepare` for every instrument: read events, name their label-free roles and stain
targets from the instrument config, and write a ragged cache the trainer can select roles
and markers from.

    python -m sperm_fluorescence prepare --instrument a8  --source data/a8_bundle_seed0.zip --cache data/a8_roles
    python -m sperm_fluorescence prepare --instrument isx --source data/isx/ALL_1_400_1.cif --daf data/isx/ALL_1_400_1.daf --cache data/isx_roles
    python -m sperm_fluorescence prepare --instrument a8 --source ... --instrument isx --source ... --daf ... --cache data/both_roles
    python -m sperm_fluorescence prepare --instrument isx --source unstained.cif --unstained --cache data/isx_unstained

The cache layout is data/cache.py's. meta.json carries:
    roles          the x channel names, in order (union over instruments; a role an instrument
                   lacks is a zero channel, and <cls>_present.npy says which roles each object has)
    target_names   the marker names, in order (union over instruments; <cls>_avail.npy per object)
    classes        {cls: {instrument, family, avail, present, repeat}}
    scales         per instrument and marker: the divisor that brought targets to unit mean |y|

An unstained source (--unstained) gets zero targets and nothing marked available; it exists
to be run through `predict`. Targets are background-subtracted stain images scaled per
instrument and marker to unit mean magnitude, so instruments with different units meet on equal terms. Inputs are
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
            items.append((x, y, np.nan, int(r)))
        yield cls, items


def isx_records(cfg, source, daf, limit=0, unstained=False):
    """Stained: the tube is split into 2I (first half) and 3I (second half) by acquisition
    order. Unstained: one class, 1I, every object (the digit 1 is the brightfield-only
    replicate, as on the A8)."""
    c = isx_io.CIF(source); roles = cfg["roles"]; stains = cfg["stains"]
    scal_name = cfg["classes"].get("scalar"); F = col = None
    if daf and scal_name:
        names, F, _ = isx_daf.read_features(daf); col = {n: i for i, n in enumerate(names)}
    n = len(c) if not limit else min(limit, len(c)); half = n // 2
    splits = (("1I", range(0, n)),) if unstained else (("2I", range(0, half)), ("3I", range(half, n)))
    for cls, rows in splits:
        items = []; t0 = time.time()
        for i in rows:
            img = c.image(i).astype(np.float32); mk = c.mask(i); mc = mk.any(axis=0)
            bg = np.array([np.median(img[k][~mc]) if (~mc).any() else np.median(img[k]) for k in range(img.shape[0])])
            x = standardise(np.stack([img[ch] for ch in roles.values()]))
            y = np.stack([img[ch] - bg[ch] for ch in stains.values()]).astype(np.float32)
            s = float(F[i, col[scal_name]]) if F is not None else np.nan
            items.append((x, y, s, int(i)))
        print("isx %s: %d objects decoded (%.0fs)" % (cls, len(items), time.time() - t0))
        yield cls, items


def a8_run_records(cfg, source, limit=0):
    """Yield (cls, items) for a full A8 run folder on the server. Per sample: <stem>.fcs beside
    either <stem>_images/**/*_<row>.tiff or the instrument's <stem>.zip (read in place, no
    extraction). Samples are found by the -<replicate><S|P|SP> suffix (prepare_a8.CLASS_RE) or
    named in the instrument config:

        [classes.samples]                 # file stem -> class letter
        "B5-All Primary 1_002" = "M"
        [classes]
        split_halves = true               # one acquisition -> 2M (first half of events) and 3M (second half)

    Spillover ([spillover] source): "estimate" from the S and P wells (io.a8.spillover; the July
    design), "identity" (no unmixing), or the path of a spillover.json / cache holding "M" from an
    earlier run, or "matrix" with M = [[...]] inline. The matrix used is saved as <cache>/spillover.json.
    The scan-line streak is removed (prepare_a8.destreak) as before."""
    import glob
    import zipfile
    from ..io.fcs import fcs_data, fcs_probe
    from .prepare_a8 import CLASS_RE, INDEX_RE, destreak
    skip = set(cfg["classes"].get("skip", [])); zoom = float(cfg.get("zoom_to_reference", 1.0))
    roles = cfg["roles"]; stains = cfg["stains"]
    named = dict(cfg["classes"].get("samples", {})); halves = bool(cfg["classes"].get("split_halves", False))
    samples = {}                                   # class -> (stem, rows filter)
    stems = set()
    for d in sorted(glob.glob(os.path.join(source, "*_images"))): stems.add(os.path.basename(d)[:-len("_images")])
    for z in sorted(glob.glob(os.path.join(source, "*.zip"))): stems.add(os.path.splitext(os.path.basename(z))[0])
    for stem in sorted(stems):
        m = CLASS_RE.search(stem)
        if stem in named:
            letter = named[stem]
            if halves: samples["2" + letter] = (stem, "first"); samples["3" + letter] = (stem, "second")
            else: samples[letter if letter[0].isdigit() else "2" + letter] = (stem, None)
        elif m and m.group("cls").upper() not in skip: samples[m.group("cls").upper()] = (stem, None)
    if not samples: raise SystemExit("no samples under %s: neither <stem>_images / <stem>.zip with a -2S style suffix nor a [classes.samples] entry" % source)
    print("samples:", {c: st for c, (st, _) in samples.items()})

    class _Source:
        """row -> TIFF bytes for one sample, from the folder or straight from the zip."""
        def __init__(self, stem):
            self.paths, self.zf = {}, None
            d = os.path.join(source, stem + "_images")
            if os.path.isdir(d):
                for p in glob.iglob(os.path.join(d, "**", "*.tif*"), recursive=True):
                    m = INDEX_RE.search(os.path.basename(p))
                    if m: self.paths[int(m.group("idx"))] = p
            else:
                self.zf = zipfile.ZipFile(os.path.join(source, stem + ".zip"))
                for n in self.zf.namelist():
                    m = INDEX_RE.search(os.path.basename(n))
                    if m and not n.endswith("/"): self.paths[int(m.group("idx"))] = n
        def read(self, r): return self.zf.read(self.paths[r]) if self.zf else open(self.paths[r], "rb").read()
    sources = {stem: _Source(stem) for stem in {st for st, _ in samples.values()}}

    # spillover
    sp = cfg.get("spillover", {}); how = sp.get("source", "estimate")
    if how == "estimate":
        class _Mini:
            def __init__(self): self.imgs = {}
            def image(self, c, r): return a8_io.read_stack(sources[self.imgs[c][r][0]].read(r))[0]
            def events(self, classes):
                good = []
                for c in classes:
                    for r, (stem, _) in list(self.imgs.get(c, {}).items()):
                        try: a8_io.read_stack(sources[stem].read(r))
                        except Exception: del self.imgs[c][r]; continue
                        good.append((c, r))
                return good
        mini = _Mini()
        for cls in ("2S", "3S", "2P", "3P"):
            if cls not in samples: continue
            stem = samples[cls][0]; rows = sorted(sources[stem].paths)[::max(1, len(sources[stem].paths) // 200)]
            mini.imgs[cls] = {r: (stem, None) for r in rows}
        if not ({"2S", "3S"} & set(mini.imgs)) or not ({"2P", "3P"} & set(mini.imgs)):
            raise SystemExit("spillover 'estimate' needs an S well and a P well (have %s); set [spillover] source = \"identity\" or the path of a spillover.json from the July run" % sorted(samples))
        M, fits = a8_io.spillover(mini); print("spillover:", {k: round(v[0], 3) for k, v in fits.items()})
    elif how == "identity":
        M = np.eye(3, dtype=np.float32); print("spillover: identity (no unmixing)")
    elif how == "matrix":
        M = np.array(sp["M"], np.float32); print("spillover: matrix from the config (%s)" % sp.get("note", "no note"))
    else:
        pth = how if how.endswith(".json") else os.path.join(how, "spillover.json")
        M = np.array(json.load(open(pth))["M"], np.float32); print("spillover: from %s" % pth)
    if len(stains) != 3 or M.shape != (3, 3): raise SystemExit("the A8 unmixing assumes three fluorescence pages; config has %d stains" % len(stains))
    cache_dir = os.environ.get("SF_CACHE_DIR")
    if cache_dir: json.dump({"M": M.tolist(), "source": how, "fits": {k: list(v) for k, v in fits.items()} if how == "estimate" else None}, open(os.path.join(cache_dir, "spillover.json"), "w"), indent=1)

    for cls, (stem, part) in samples.items():
        src = sources[stem]
        fcs_path = os.path.join(source, stem + ".fcs"); n_fcs = None
        if os.path.exists(fcs_path):
            raw = open(fcs_path, "rb").read(); kw = fcs_probe.read_text_segment(raw[: 1024 * 1024]); _, _, n_fcs = fcs_data.read_matrix(raw, kw)
        rows = sorted(r for r in src.paths if n_fcs is None or r < n_fcs)
        if part == "first": rows = rows[: len(rows) // 2]
        elif part == "second": rows = rows[len(rows) // 2:]
        if limit: rows = rows[:limit]
        items = []; bad = 0; t0 = time.time()
        for r in rows:
            try:
                st, _ = a8_io.read_stack(src.read(r))
                st = destreak(a8_io.unmix(st, M))
            except Exception:
                bad += 1; continue
            if zoom != 1.0: st = ndi.zoom(st, (1, zoom, zoom), order=1).astype(np.float32)
            x = standardise(np.stack([st[p] for p in roles.values()]))
            y = np.stack([st[p] for p in stains.values()]).astype(np.float32)
            items.append((x, y, np.nan, int(r)))
            if len(items) % 5000 == 0: print("  %s %d / %d  (%.0fs)" % (cls, len(items), len(rows), time.time() - t0))
        if bad: print("  %s: skipped %d unreadable file(s)" % (cls, bad))
        yield cls, items


READERS = {"a8_bundle": a8_records, "a8_run": a8_run_records, "isx_cif": isx_records}


from .denoise import denoise_target


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence prepare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", action="append", required=True, help="config name (a8, isx, ...) or path; repeat for several")
    ap.add_argument("--source", action="append", required=True, help="one per --instrument: bundle zip, .cif, ...")
    ap.add_argument("--daf", action="append", default=[], help="ImageStream .daf (features), one per isx instrument")
    ap.add_argument("--cache", required=True); ap.add_argument("--limit", type=int, default=0, help="events per class (0 = all)")
    ap.add_argument("--reader", default=None, help="override the config's reader, e.g. a8_run for a full run folder on the server")
    ap.add_argument("--denoise-targets", default="none", choices=["none", "floor", "smooth"], help="denoise every measured stain image before it becomes a target: floor = subtract the frame's noise floor (median + 3 MAD) and clip at 0; smooth = a 1-px Gaussian first, then the floor. The model then learns, and is scored against, the stain without its read noise (recorded in meta.json)")
    ap.add_argument("--unstained", action="store_true", help="the source carries no stains: targets are written as zeros and no marker is marked available (ISX: one class, 1I)")
    a = ap.parse_args(argv)
    if len(a.source) != len(a.instrument): ap.error("one --source per --instrument")
    cfgs = [registry.instrument(n) for n in a.instrument]
    roles = []; markers = []
    for cfg in cfgs:
        roles += [r for r in cfg["roles"] if r not in roles]; markers += [m for m in cfg["stains"] if m not in markers]
    os.makedirs(a.cache, exist_ok=True); os.environ["SF_CACHE_DIR"] = os.path.abspath(a.cache)      # readers may drop side files (spillover.json) there
    meta = {"layout": "ragged", "roles": roles, "target_names": markers, "classes": {}, "scales": {}, "instruments": {}, "source_rows": True, "denoise_targets": a.denoise_targets}
    dafs = list(a.daf)
    for cfg, src in zip(cfgs, a.source):
        inst = cfg["name"]; meta["instruments"][inst] = {"family": cfg.get("family"), "pixel_um": cfg.get("pixel_um"), "zoom_to_reference": cfg.get("zoom_to_reference", 1.0)}
        present = [r in cfg["roles"] for r in roles]; avail = [m in cfg["stains"] and not a.unstained for m in markers]
        rk = [list(cfg["roles"]).index(r) if r in cfg["roles"] else None for r in roles]
        mk = [list(cfg["stains"]).index(m) if m in cfg["stains"] else None for m in markers]
        rname = a.reader or cfg["reader"]; reader = READERS[rname]
        def run_reader(limit):
            return reader(cfg, src, dafs[0] if rname == "isx_cif" and dafs else None, limit, a.unstained) if rname == "isx_cif" else reader(cfg, src, limit)
        def widen(items):
            full = []
            for x, y, s, row in items:
                X = np.zeros((len(roles),) + x.shape[1:], np.float32); Y = np.zeros((len(markers),) + y.shape[1:], np.float32)
                for j, k in enumerate(rk):
                    if k is not None: X[j] = x[k]
                for j, k in enumerate(mk):
                    if k is not None and not a.unstained: Y[j] = denoise_target(y[k], a.denoise_targets)
                full.append((X, Y, s, row))
            return full
        # pass 1: the per-marker scale (unit mean |y|) from a short draw of the training-digit classes
        draw = {cls: widen(items) for cls, items in run_reader(300)}
        tr = [c for c in draw if c[0] == "2"] or list(draw)
        scale = np.ones(len(markers), np.float32)
        for j in range(len(markers)):
            if avail[j]:
                vals = np.concatenate([np.abs(y[j]).ravel()[::7] for c in tr for _, y, _, _ in draw[c]])
                scale[j] = max(float(vals.mean()), 1e-6)
        meta["scales"][inst] = scale.tolist(); del draw
        # pass 2: stream, one class in memory at a time
        for cls, items in run_reader(a.limit):
            full = widen(items); del items
            print("%s %s: %d objects" % (inst, cls, len(full)))
            out = []
            for X, Y, _, _ in full:
                fx, m = to_multiple(X); fy, _ = to_multiple(Y / scale[:, None, None]); out.append((fx, fy, m))
            write_ragged(cls, out, a.cache)
            np.save(os.path.join(a.cache, cls + "_avail.npy"), np.tile(np.array(avail, bool), (len(out), 1)))
            np.save(os.path.join(a.cache, cls + "_present.npy"), np.tile(np.array(present, bool), (len(out), 1)))
            np.save(os.path.join(a.cache, cls + "_rows.npy"), np.array([row for _, _, _, row in full], np.int64))   # source object index
            np.save(os.path.join(a.cache, cls + "_scalars.npy"), np.array([s for _, _, s, _ in full], np.float32)[:, None])
            meta["classes"][cls] = {"instrument": inst, "family": cfg.get("family"), "avail": avail, "present": present, "repeat": int(cfg["classes"].get("repeat", 1))}
            print("wrote %s: %d objects, roles %s, markers %s" % (cls, len(out), [r for r, p in zip(roles, present) if p], [m for m, v in zip(markers, avail) if v]))
            del full, out
        if rname == "isx_cif" and dafs: dafs.pop(0)
    json.dump(meta, open(os.path.join(a.cache, "meta.json"), "w"), indent=1)
    print("cache %s: roles %s, markers %s" % (a.cache, roles, markers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
