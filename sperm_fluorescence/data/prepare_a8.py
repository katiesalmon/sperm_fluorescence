"""Cache an A8 run (fixed 80 x 104 frames, destreaked, unmixed) -- the layout the
full-scale server runs used. One cache per sample class:

    <cache>/<cls>_x.npy   (N, 3, 80, 104) float16   LightLoss, FSC, SSC, standardised
    <cache>/<cls>_y.npy   (N, 3, 80, 104) float16   AF488, PE, PerCP pages, unmixed
    <cache>/<cls>_m.npy   (N, 80, 104)    bool      valid region
    <cache>/<cls>_rows.npy, <cls>_scalars.npy        FCS row and the unmixed scalars

    python -m sperm_fluorescence prepare-a8 "Z:\\...\\260709_Blair_Sperm_A8\\260709" --cache D:\\a8_cache
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np

from ..io import a8 as a8_io
from ..io.fcs import fcs_data, fcs_probe

H, W = 80, 104
CLASS_RE = re.compile(r"-(?P<cls>[123](?:S|P|SP))$", re.IGNORECASE)
INDEX_RE = re.compile(r"_(?P<idx>\d{6,10})\.tiff?$", re.IGNORECASE)
SCALARS = ["LDHC_AKAP4-A", "CD45-A", "ACRV-1-A", "DAPI-A"]


def destreak(pages):
    """Subtract each row's median from every page but LightLoss. A cell spans ~30 of the
    104 columns, so the row median is background plus streak, not cell."""
    out = pages.copy()
    out[1:] -= np.median(out[1:], axis=2, keepdims=True)
    return out


def to_fixed(st, M, do_destreak):
    """One event -> (x (3,H,W), y (3,H,W), mask (H,W)). Reflect-pad the flow axis."""
    um = a8_io.unmix(st, M)
    if do_destreak:
        um = destreak(um)
    x = um[a8_io.LABEL_FREE]
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)
    y = um[a8_io.FLUOR]
    h = st.shape[1]
    if h >= H:
        top = (h - H) // 2
        return x[:, top:top + H], y[:, top:top + H], np.ones((H, W), bool)
    before = (H - h) // 2; after = H - h - before
    pad = lambda a: np.pad(a, ((0, 0), (before, after), (0, 0)), mode="reflect")
    m = np.zeros((H, W), bool); m[before:before + h] = True
    return pad(x), pad(y), m


def prepare(args):
    run = args.run_folder
    samples = []
    for p in sorted(glob.glob(os.path.join(run, "*.fcs"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = CLASS_RE.search(stem)
        if m and (not args.classes or m.group("cls").upper() in {c.upper() for c in args.classes}):
            samples.append((m.group("cls").upper(), stem, p))
    os.makedirs(args.cache, exist_ok=True)

    # Spillover from a modest draw of S and P events, same method as the pilot.
    class _Mini:
        def __init__(self): self.imgs = {}
        def events(self, classes): return [(c, r) for c in classes for r in self.imgs.get(c, {})]
        def image(self, c, r): return a8_io.read_stack(open(self.imgs[c][r], "rb").read())[0]
        def events(self, classes):
            # drop anything unreadable up front so the spillover fit never sees it
            good = []
            for c in classes:
                for r, path in list(self.imgs.get(c, {}).items()):
                    try: a8_io.read_stack(open(path, "rb").read())
                    except Exception: del self.imgs[c][r]; continue
                    good.append((c, r))
            return good
    mini = _Mini()
    # a deterministic subsample -- every 100th file -- of each fluorophore-dominant well
    for cls, stem, _ in samples:
        if cls in ("2S", "3S", "2P", "3P"):
            paths = sorted(glob.glob(os.path.join(run, stem + "_images", "**", "*.tif*"), recursive=True))
            paths = paths[::max(1, len(paths) // 200)]
            mini.imgs[cls] = {int(INDEX_RE.search(os.path.basename(p)).group("idx")): p for p in paths}
    M, fits = a8_io.spillover(mini)
    print("spillover:", {k: round(v[0], 3) for k, v in fits.items()})
    json.dump({"M": M.tolist(), "fits": {k: list(v) for k, v in fits.items()}, "destreak": args.destreak},
              open(os.path.join(args.cache, "spillover.json"), "w"), indent=1)

    for cls, stem, fcs_path in samples:
        raw = open(fcs_path, "rb").read()
        kw = fcs_probe.read_text_segment(raw[: 1024 * 1024]); params = fcs_probe.parameters(kw)
        values, par, n = fcs_data.read_matrix(raw, kw)
        idx = {nm: fcs_data.column_index(params, nm) for nm in SCALARS}
        paths = {}
        for p in glob.iglob(os.path.join(run, stem + "_images", "**", "*.tif*"), recursive=True):
            paths[int(INDEX_RE.search(os.path.basename(p)).group("idx"))] = p
        rows = sorted(r for r in paths if r < n)
        if not rows:
            print("%-4s no images -- skipped" % cls); continue
        if args.limit: rows = rows[:args.limit]
        N = len(rows)
        X = np.lib.format.open_memmap(os.path.join(args.cache, cls + "_x.npy"), "w+", np.float16, (N, 3, H, W))
        Y = np.lib.format.open_memmap(os.path.join(args.cache, cls + "_y.npy"), "w+", np.float16, (N, 3, H, W))
        Mk = np.lib.format.open_memmap(os.path.join(args.cache, cls + "_m.npy"), "w+", bool, (N, H, W))
        S = np.zeros((N, len(SCALARS)), np.float32)
        t0 = time.time(); bad = []; i = 0
        for r in rows:
            try:
                st, _ = a8_io.read_stack(open(paths[r], "rb").read())
                x, y, m = to_fixed(st, M, args.destreak)
            except Exception as exc:
                bad.append((r, str(exc)[:60])); continue
            X[i], Y[i], Mk[i] = x, y, m
            S[i] = [values[r * par + idx[nm]] if idx[nm] is not None else np.nan for nm in SCALARS]
            i += 1
            if i % 2000 == 0:
                print("%-4s %6d / %d   %.0fs" % (cls, i, N, time.time() - t0))
        X.flush(); Y.flush(); Mk.flush()
        kept = [r for r in rows if r not in {b[0] for b in bad}][:i]
        if bad:
            # shrink the memmaps to what was actually written, so a bad file never leaves a zero frame
            for name, arr in (("_x.npy", X), ("_y.npy", Y), ("_m.npy", Mk)):
                np.save(os.path.join(args.cache, cls + name), np.asarray(arr[:i]))
            print("%-4s skipped %d unreadable file(s), e.g. row %s: %s" % (cls, len(bad), bad[0][0], bad[0][1]))
        np.save(os.path.join(args.cache, cls + "_rows.npy"), np.array(kept))
        np.save(os.path.join(args.cache, cls + "_scalars.npy"), S[:i])
        print("%-4s %d events cached   (%.0fs)" % (cls, i, time.time() - t0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence prepare-a8", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p = ap; p.add_argument("run_folder"); p.add_argument("--cache", required=True)
    p.add_argument("--classes", nargs="+"); p.add_argument("--limit", type=int, default=0, help="events per sample (0 = all)")
    p.add_argument("--no-destreak", dest="destreak", action="store_false")
    return prepare(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
