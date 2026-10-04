#!/usr/bin/env python
"""Full-scale virtual staining on the A8 run: three label-free pages -> three stain pages.

Two steps, because the images live on a network share and Windows multiprocessing is not
worth fighting:

  prepare   read every imaged event once and write one cache per sample:
              <cache>/<cls>_x.npy   (N, 3, 80, 104) float16   label-free pages, normalised
              <cache>/<cls>_y.npy   (N, 3, 80, 104) float16   fluorescence pages, unmixed
              <cache>/<cls>_m.npy   (N, 80, 104)    bool      valid region
              <cache>/<cls>_rows.npy, <cls>_scalars.npy        FCS row and the unmixed scalars
            ~2 GB per sample, memory-mapped at training time.

  train     a U-Net on one replicate, tested on the other, with the cell-type controls.

The join is the FCS row index (verified from the data; docs/task_brief.md). The imaging
spillover is estimated at prepare time by the physically-constrained method in
a8_io.spillover and applied per pixel. Two pipeline defects from the pilot are fixed:
inputs are reflect-padded (no learnable zero border) and predictions are evaluated only
inside the valid region; and a row-median subtraction removes the CellView scan-line
streak from the scatter and fluorescence pages (--no-destreak to ablate).

Examples
--------
  python analysis/train_a8.py prepare "Z:\\...\\260709_Blair_Sperm_A8\\260709" --cache D:\\a8_cache
  python analysis/train_a8.py train --cache D:\\a8_cache --train 2 --test 3 --epochs 40 --out runs/fold_2to3
  python analysis/train_a8.py train --cache D:\\a8_cache --train 3 --test 2 --epochs 40 --out runs/fold_3to2
"""

import argparse
import contextlib
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import a8_io  # noqa: E402
import fcs_data  # noqa: E402
import fcs_probe  # noqa: E402

H, W = 80, 104
CLASS_RE = re.compile(r"-(?P<cls>[123](?:S|P|SP))$", re.IGNORECASE)
INDEX_RE = re.compile(r"_(?P<idx>\d{6,10})\.tiff?$", re.IGNORECASE)
SCALARS = ["LDHC_AKAP4-A", "CD45-A", "ACRV-1-A", "DAPI-A"]
NAMES = ["LDHC/AKAP4", "CD45", "ACRV1"]   # A8 default; overridden by <cache>/meta.json when present


# ------------------------------------------------------------------------- prepare ----

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


# --------------------------------------------------------------------------- train ----

def load_cache(cache, classes):
    import torch
    xs, ys, ms, wells, scal = [], [], [], [], []
    for cls in classes:
        p = os.path.join(cache, cls + "_x.npy")
        if not os.path.exists(p): continue
        xs.append(np.load(p, mmap_mode="r")); ys.append(np.load(os.path.join(cache, cls + "_y.npy"), mmap_mode="r"))
        ms.append(np.load(os.path.join(cache, cls + "_m.npy"), mmap_mode="r"))
        n = len(xs[-1]); wells += [cls.lstrip("123")] * n
        scal.append(np.load(os.path.join(cache, cls + "_scalars.npy")))
    return xs, ys, ms, np.array(wells), np.concatenate(scal)


class Cached:
    """Index into several memmaps as one dataset."""
    def __init__(self, xs, ys, ms):
        self.xs, self.ys, self.ms = xs, ys, ms
        self.offsets = np.cumsum([0] + [len(x) for x in xs])
    def __len__(self): return int(self.offsets[-1])
    def batch(self, idx):
        import torch
        out = [[], [], []]
        for i in idx:
            k = np.searchsorted(self.offsets, i, side="right") - 1; j = i - self.offsets[k]
            out[0].append(self.xs[k][j]); out[1].append(self.ys[k][j]); out[2].append(self.ms[k][j])
        return (torch.tensor(np.stack(out[0]).astype(np.float32)), torch.tensor(np.stack(out[1]).astype(np.float32)),
                torch.tensor(np.stack(out[2])[:, None].astype(np.float32)))


def build_model(width, cin=3, cout=3):
    import torch, torch.nn as nn, torch.nn.functional as F

    class Block(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(o), nn.ReLU(True),
                                  nn.Conv2d(o, o, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(o), nn.ReLU(True))
        def forward(s, x): return s.net(x)

    class UNet(nn.Module):
        def __init__(s, w, cin=3, cout=3):
            super().__init__()
            s.e1, s.e2, s.e3, s.e4 = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w), Block(4 * w, 8 * w)
            s.d3, s.d2, s.d1 = Block(8 * w + 4 * w, 4 * w), Block(4 * w + 2 * w, 2 * w), Block(2 * w + w, w)
            s.out = nn.Conv2d(w, cout, 1)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); e4 = s.e4(F.max_pool2d(e3, 2))
            up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            d3 = s.d3(torch.cat([up(e4, e3), e3], 1)); d2 = s.d2(torch.cat([up(d3, e2), e2], 1)); d1 = s.d1(torch.cat([up(d2, e1), e1], 1))
            return s.out(d1)
    return UNet(width, cin, cout)


def evaluate(model, ds, wells, scale, dev, bs=256, idx=None):
    """`scale` broadcasts over (N, 3, H, W): a scalar, or a (1, 3, 1, 1) tensor per channel.
    `idx` restricts to a subset of ds; `wells` must already be that subset's wells."""
    import torch
    idx = np.arange(len(ds)) if idx is None else np.asarray(idx)
    model.eval(); P, Yt, Mk = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, y, m = ds.batch(idx[i:i + bs])
            P.append((model(x.to(dev)).float().cpu() * scale).numpy()); Yt.append(y.numpy()); Mk.append(m.numpy())
    P, Yt, Mk = np.concatenate(P), np.concatenate(Yt), np.concatenate(Mk)
    res = {}
    has_ps = (wells == "P").any() and (wells == "S").any()
    for k, n in enumerate(NAMES):
        valid = Mk[:, 0] > 0
        pix = np.corrcoef(P[:, k][valid], Yt[:, k][valid])[0, 1]
        ep, et = (P[:, k] * Mk[:, 0]).sum(axis=(1, 2)), (Yt[:, k] * Mk[:, 0]).sum(axis=(1, 2))
        ps = (lambda v: float(np.median(v[wells == "P"]) / max(abs(np.median(v[wells == "S"])), 1e-9))) if has_ps else (lambda v: None)
        # event r WITHIN one well type is the metric that cannot be earned by recognising the
        # cell. For ACRV1 the within-sperm value is the whole question: across types the model
        # scores ~0.5 just by knowing sperm have acrosomes; within sperm the pilot scored 0.0.
        within = {}
        for w in sorted(set(wells)):
            sel = wells == w
            within[w] = float(np.corrcoef(ep[sel], et[sel])[0, 1]) if sel.sum() > 2 else None
        res[n] = {"pixel_r": float(pix), "event_r": float(np.corrcoef(ep, et)[0, 1]),
                  "pred_PS": ps(ep), "true_PS": ps(et), "within_well_event_r": within}
    return res, P


def train(args):
    import torch, torch.nn.functional as F
    global NAMES
    meta_path = os.path.join(args.cache, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if meta.get("target_names"):
        NAMES = list(meta["target_names"])
    target_k = None
    if args.target:
        if args.target not in NAMES:
            raise SystemExit("--target must be one of %s" % NAMES)
        target_k = NAMES.index(args.target)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    all_cls = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(args.cache, "*_x.npy"))})
    tr_cls = [c for c in all_cls if c[0] in args.train]; te_cls = [c for c in all_cls if c[0] in args.test]
    xs, ys, ms, wtr, _ = load_cache(args.cache, tr_cls)
    if target_k is not None:
        ys = [y[:, target_k:target_k + 1] for y in ys]   # memmap views: one target channel
    tr = Cached(xs, ys, ms)
    xs, ys, ms, wte, _ = load_cache(args.cache, te_cls)
    if target_k is not None:
        ys = [y[:, target_k:target_k + 1] for y in ys]
        NAMES = [args.target]
    te = Cached(xs, ys, ms)
    # hold a slice of the training replicate out for model selection; the test replicate is never used for it
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(tr))
    nval = min(max(500, len(tr) // 20), max(1, len(tr) // 5))   # 5% of a big set, never more than 20% of a small one
    val_idx, fit_idx = perm[:nval], perm[nval:]
    ysample = np.concatenate([np.asarray(y[:200]) for y in tr.ys]).astype(np.float32)
    if args.channel_scale:
        # One scale per channel. With a single shared scale the L1 is dominated by the
        # brightest channel: AF488 is ~3x PerCP, so ACRV1 -- the dimmest and the one with the
        # least headroom to waste -- got roughly a fifth of the gradient in the first runs.
        scale_vec = np.abs(ysample).mean(axis=(0, 2, 3)) * 10
    else:
        scale_vec = np.full(ysample.shape[1], np.abs(ysample).mean() * 10, np.float32)
    scale = torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1)
    scale_dev = scale.to(dev)
    os.makedirs(args.out, exist_ok=True)
    print("device %s   train %s: %d fit + %d val   test %s: %d   width %d   target scale %s   select on %s"
          % (dev, tr_cls, len(fit_idx), nval, te_cls, len(te), args.width,
             "/".join("%.4f" % v for v in scale_vec), args.select_on))

    cin, cout = tr.xs[0].shape[1], tr.ys[0].shape[1]
    model = build_model(args.width, cin, cout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (len(fit_idx) // args.batch + 1))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    log = open(os.path.join(args.out, "log.csv"), "w", newline=""); lw = csv.writer(log)
    lw.writerow(["epoch", "train_l1", "val_l1"] + ["%s_%s" % (n, k) for n in NAMES for k in ("pixel_r", "event_r", "pred_PS")])
    best = float("inf"); t0 = time.time()
    best_ch = np.full(len(NAMES), np.inf); best_ep = [0] * len(NAMES)
    val_ds = Cached(tr.xs, tr.ys, tr.ms)

    # Fixed test objects for per-epoch snapshots: a spread over well types and over the
    # brightest target channel, so both strong and faint stains are watched.
    snap_idx = []
    if args.snapshot:
        kinds_te = sorted(set(wte))
        for w in kinds_te:
            cand = np.flatnonzero(wte == w)
            if len(cand) == 0: continue
            sums = np.array([np.asarray(te.batch([j])[1][0, -1]).sum() for j in cand[:3000]])
            order = cand[:3000][np.argsort(sums)]
            per = max(1, 6 // len(kinds_te))
            snap_idx += [int(order[int(q * (len(order) - 1))]) for q in np.linspace(0.15, 0.95, per)]
        snap_x = np.stack([np.asarray(te.batch([j])[0][0]) for j in snap_idx])
        snap_y = np.stack([np.asarray(te.batch([j])[1][0]) for j in snap_idx])
        snap_m = np.stack([np.asarray(te.batch([j])[2][0, 0]) for j in snap_idx])
        snap_preds = {}

    for ep in range(1, args.epochs + 1):
        model.train(); rng.shuffle(fit_idx); tot = 0.0; nb = 0
        for i in range(0, len(fit_idx), args.batch):
            x, y, m = tr.batch(fit_idx[i:i + args.batch]); x, y, m = x.to(dev), y.to(dev) / scale_dev, m.to(dev)
            if args.full_frame:
                # Train on every pixel. The padded zone is background with a zero target,
                # and a network never penalised there paints stain onto it (seen in the
                # ISX progression sheets). The validity mask still scopes the metrics.
                m = torch.ones_like(m)
            if np.random.rand() < 0.5: x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
            # mixed precision on CUDA only; torch 2.2 rejects autocast on mps even when disabled
            amp = torch.autocast("cuda") if dev.type == "cuda" else contextlib.nullcontext()
            with amp:
                loss = (F.l1_loss(model(x), y, reduction="none") * m).sum() / m.sum() / 3
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        # validation L1 on the held-out slice of the TRAINING replicate
        model.eval(); vl = 0.0; vn = 0; vch = np.zeros(len(NAMES)); vm = 0.0
        with torch.no_grad():
            for i in range(0, len(val_idx), 256):
                x, y, m = val_ds.batch(val_idx[i:i + 256]); x, y, m = x.to(dev), y.to(dev) / scale_dev, m.to(dev)
                if args.full_frame: m = torch.ones_like(m)
                per = (F.l1_loss(model(x), y, reduction="none") * m).sum(dim=(0, 2, 3)).cpu().numpy()
                vch += per; vm += m.sum().item()
                vl += (per.sum() / m.sum().item() / len(NAMES)); vn += 1
        vl /= max(vn, 1); vch /= max(vm, 1.0)
        # Each marker keeps its own best epoch: a shared trunk's best moment differs per
        # marker, and one checkpoint cannot serve all of them.
        for k, nm in enumerate(NAMES):
            if vch[k] < best_ch[k]:
                best_ch[k] = vch[k]; best_ep[k] = ep
                torch.save(model.state_dict(), os.path.join(args.out, "best_%s.pt" % nm.replace("/", "_")))
        res, _ = evaluate(model, te, wte, scale, dev) if (ep % args.eval_every == 0 or ep == args.epochs) else ({}, None)
        lw.writerow([ep, tot / nb, vl] + [res[n][k] if res else "" for n in NAMES for k in ("pixel_r", "event_r", "pred_PS")]); log.flush()
        line = "epoch %3d  train %.4f  val %.4f" % (ep, tot / nb, vl)
        if res: line += "   test event r: " + "  ".join("%s %.3f" % (n, res[n]["event_r"]) for n in NAMES)
        if res and "ACRV1" in res and res["ACRV1"]["within_well_event_r"].get("S") is not None:
            line += "   ACRV1 within-sperm %.3f" % res["ACRV1"]["within_well_event_r"]["S"]
        print(line + "   (%.0fs)" % (time.time() - t0))
        # Checkpoint selection, on the training replicate's held-out slice only.
        if args.select_on == "acrv1_sperm":
            rv, _ = evaluate(model, tr, wtr[val_idx], scale, dev, idx=val_idx)
            w_acrv = rv["ACRV1"]["within_well_event_r"]; r_s = w_acrv.get("S", next(iter(w_acrv.values())))
            score = -(r_s if r_s is not None else -1.0)
            line_sel = "   val ACRV1 within-sperm %.3f" % (r_s if r_s is not None else float("nan"))
        else:
            score, line_sel = vl, ""
        if line_sel: print("          " + line_sel.strip())
        if score < best:
            best = score; torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        if args.snapshot and (ep % args.snapshot == 0 or ep == 1 or ep == args.epochs):
            model.eval()
            with torch.no_grad():
                sp = (model(torch.tensor(snap_x).to(dev)).float().cpu() * scale).numpy()
            snap_preds[ep] = sp
            np.savez_compressed(os.path.join(args.out, "epoch_%02d.npz" % ep), pred=sp, true=snap_y, x=snap_x, mask=snap_m, idx=np.array(snap_idx))
            _snapshot_sheet(args.out, ep, snap_x, snap_y, sp, snap_m, wte[snap_idx])

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    res, P = evaluate(model, te, wte, scale, dev)
    print("\nbest-by-validation model on the held-out replicate:")
    print("%-12s %9s %9s %12s %12s   %s" % ("marker", "pixel r", "event r", "pred P:S", "true P:S", "want"))
    wants = dict(zip(["LDHC/AKAP4", "CD45", "ACRV1"], ["<< 1", ">> 1", "<< 1"]))
    for n in NAMES:
        r = res[n]
        ps = ("%12.2f %12.2f   %s" % (r["pred_PS"], r["true_PS"], wants.get(n, ""))) if r["pred_PS"] is not None else "%12s %12s" % ("n/a", "n/a")
        print("%-12s %9.3f %9.3f %s" % (n, r["pixel_r"], r["event_r"], ps))
    kinds = sorted({k for n in NAMES for k in res[n]["within_well_event_r"]})
    print("\nevent r WITHIN one well type -- the number that cannot be earned by recognising the cell:")
    print("%-12s " % "marker" + " ".join("%14s" % k for k in kinds))
    for n in NAMES:
        w = res[n]["within_well_event_r"]
        print("%-12s " % n + " ".join("%14s" % ("%.3f" % w[k] if w.get(k) is not None else "n/a") for k in kinds))
    per_marker = {}
    if len(NAMES) > 1:
        print("\nsame joint model, but each marker evaluated at ITS OWN best validation epoch:")
        print("%-12s %6s %9s %9s   %s" % ("marker", "epoch", "pixel r", "event r", "within-well event r"))
        for k, nm in enumerate(NAMES):
            ck = os.path.join(args.out, "best_%s.pt" % nm.replace("/", "_"))
            if not os.path.exists(ck): continue
            model.load_state_dict(torch.load(ck, map_location=dev))
            rk, _ = evaluate(model, te, wte, scale, dev)
            per_marker[nm] = dict(rk[nm], epoch=best_ep[k])
            w = rk[nm]["within_well_event_r"]
            print("%-12s %6d %9.3f %9.3f   %s" % (nm, best_ep[k], rk[nm]["pixel_r"], rk[nm]["event_r"],
                  "  ".join("%s %.3f" % (kk, v) for kk, v in w.items() if v is not None)))
        model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    json.dump({"train": tr_cls, "test": te_cls, "epochs": args.epochs, "width": args.width, "target": args.target, "full_frame": args.full_frame,
               "channel_scale": args.channel_scale, "select_on": args.select_on, "results": res,
               "per_marker_checkpoint": per_marker},
              open(os.path.join(args.out, "results.json"), "w"), indent=1)

    if args.snapshot and snap_preds:
        _progression_sheet(args.out, snap_x, snap_y, snap_preds, snap_m)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    kinds_te = sorted(set(wte)); per = max(1, 9 // len(kinds_te))
    pick = [np.where(wte == w)[0][j] for w in kinds_te for j in [3, 40, 400, 900, 2000, 5000][:per] if (wte == w).sum() > j]
    ncol = 1 + 2 * cout
    fig, axes = plt.subplots(len(pick), ncol, figsize=(2 * ncol, 1.9 * len(pick)))
    for i, j in enumerate(pick):
        x, y, m = te.batch([j]); x, y = x[0].numpy(), y[0].numpy()
        panels = [x[0]] + [v for k in range(cout) for v in (y[k], P[j, k])]
        for c, (ax, a) in enumerate(zip(axes[i], panels)):
            if c == 0: ax.imshow(a, cmap="gray")
            else:
                ref = panels[c if c % 2 == 1 else c - 1]; ax.imshow(a, cmap="magma", vmin=0, vmax=max(np.percentile(ref, 99.5), 1e-4))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0: ax.set_ylabel(wte[j], fontsize=8)
            if i == 0: ax.set_title((["input ch0"] + [t % n for n in NAMES for t in ("true %s", "pred %s")])[c], fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "preds.png"), dpi=120)
    print("wrote", os.path.join(args.out, "results.json"), "and preds.png")
    return 0


def _snapshot_sheet(out, ep, x, y, p, m, wells):
    """One epoch: rows = fixed objects, columns = input, then true/pred per marker."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n, cout = len(x), y.shape[1]
    fig, axes = plt.subplots(n, 1 + 2 * cout, figsize=(1.6 * (1 + 2 * cout), 1.9 * n))
    for i in range(n):
        panels = [x[i, 0]] + [v for k in range(cout) for v in (y[i, k], p[i, k])]
        for c, (ax, a) in enumerate(zip(axes[i], panels)):
            if c == 0: ax.imshow(a, cmap="gray")
            else:
                ref = panels[c if c % 2 == 1 else c - 1]; ax.imshow(a, cmap="magma", vmin=0, vmax=max(np.percentile(ref[m[i] > 0], 99.5), 1e-4))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0: ax.set_ylabel(str(wells[i]), fontsize=7)
            if i == 0: ax.set_title((["input"] + [t % nm for nm in NAMES for t in ("true %s", "pred %s")])[c], fontsize=7)
    fig.suptitle("epoch %d" % ep, fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out, "epoch_%02d.png" % ep), dpi=100); plt.close(fig)


def _progression_sheet(out, x, y, preds, m):
    """Per marker: rows = fixed objects, columns = input, truth, then prediction at each
    snapshot epoch. The picture of what training actually changes."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    eps = sorted(preds); n, cout = len(x), y.shape[1]
    for k, nm in enumerate(NAMES):
        fig, axes = plt.subplots(n, 2 + len(eps), figsize=(1.5 * (2 + len(eps)), 1.8 * n))
        for i in range(n):
            hi = max(np.percentile(y[i, k][m[i] > 0], 99.5), 1e-4)
            axes[i, 0].imshow(x[i, 0], cmap="gray"); axes[i, 1].imshow(y[i, k], cmap="magma", vmin=0, vmax=hi)
            for j, ep in enumerate(eps):
                axes[i, 2 + j].imshow(preds[ep][i, k], cmap="magma", vmin=0, vmax=hi)
                if i == 0: axes[i, 2 + j].set_title("ep %d" % ep, fontsize=8)
            if i == 0: axes[i, 0].set_title("input", fontsize=8); axes[i, 1].set_title("true", fontsize=8)
            for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("%s -- prediction by epoch, fixed test objects, truth's colour scale" % nm, fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, "progression_%s.png" % nm.replace("/", "_")), dpi=100); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("run_folder"); p.add_argument("--cache", required=True)
    p.add_argument("--classes", nargs="+"); p.add_argument("--limit", type=int, default=0, help="events per sample (0 = all)")
    p.add_argument("--no-destreak", dest="destreak", action="store_false")
    t = sub.add_parser("train"); t.add_argument("--cache", required=True); t.add_argument("--train", default="2"); t.add_argument("--test", default="3")
    t.add_argument("--epochs", type=int, default=40); t.add_argument("--batch", type=int, default=64); t.add_argument("--width", type=int, default=48)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--eval-every", type=int, default=5); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs/fold")
    t.add_argument("--channel-scale", action="store_true", help="Normalise the L1 per target channel instead of one shared scale")
    t.add_argument("--full-frame", action="store_true",
                   help="Compute the training loss over the whole frame, padding included (requires a "
                        "background-padded cache, i.e. prepare_isx after 33089b3). Metrics stay masked.")
    t.add_argument("--target", default=None, metavar="NAME",
                   help="Train a single-output model for one marker (name as in the cache's meta.json)")
    t.add_argument("--snapshot", type=int, default=0, metavar="N",
                   help="Every N epochs, save predictions for a fixed set of test objects (epoch_XX.png + .npz) and, at the end, a progression sheet")
    t.add_argument("--select-on", choices=["val_l1", "acrv1_sperm"], default="val_l1",
                   help="Checkpoint selection: validation L1 (default) or within-sperm ACRV1 event r on the validation slice")
    args = ap.parse_args(argv)
    return prepare(args) if args.cmd == "prepare" else train(args)


if __name__ == "__main__":
    sys.exit(main())
