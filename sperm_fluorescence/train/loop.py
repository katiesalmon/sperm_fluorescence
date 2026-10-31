"""The training loop: one replicate (or class set) to fit, another held out, per-marker
checkpoints, IDEAS-style sheets. Behaviour is that of analysis/train_a8.py at a792902."""
import contextlib
import csv
import glob
import json
import os
import time

import numpy as np

from ..data.augment import augment
from ..data.batching import avail_of, make_batches, present_of
from ..data.cache import open_cache
from ..eval.metrics import evaluate
from ..eval.sheets import progression_sheet, sheet, snapshot_sheet
from ..models import stems
from .device import pick_device, release_device_memory
from .losses import grad_loss, target_scale

DEFAULT_NAMES = ["LDHC/AKAP4", "CD45", "ACRV1"]   # A8 fixed-frame caches have no meta.json


def train(args):
    import torch, torch.nn.functional as F
    names = list(DEFAULT_NAMES)
    meta_path = os.path.join(args.cache, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if meta.get("target_names"):
        names = list(meta["target_names"])
    target_k = None
    if args.target:
        if args.target not in names:
            raise SystemExit("--target must be one of %s" % names)
        target_k = names.index(args.target)
    dev = pick_device()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    all_cls = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(args.cache, "*_x.npy"))})
    if args.include:
        keep = {c.strip() for c in args.include.split(",")}
        all_cls = [c for c in all_cls if c in keep]
    tr_cls = [c for c in all_cls if c[0] in args.train]; te_cls = [c for c in all_cls if c[0] in args.test]
    roles = [r.strip() for r in args.roles.split(",")] if args.roles else None
    tr, wtr, _ = open_cache(args.cache, tr_cls, target_k, roles)
    te, wte, _ = open_cache(args.cache, te_cls, target_k, roles)
    if target_k is not None:
        names = [args.target]
    # hold a slice of the training replicate out for model selection; the test replicate is never used for it
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(tr))
    nval = min(max(500, len(tr) // 20), max(1, len(tr) // 5))   # 5% of a big set, never more than 20% of a small one
    val_idx, fit_idx = perm[:nval], perm[nval:]
    # a class may ask to be seen several times per epoch (meta.json "repeat"), so a small
    # instrument is not drowned by a large one
    repeats = {c: int(v.get("repeat", 1)) for c, v in meta.get("classes", {}).items()} if isinstance(meta.get("classes"), dict) else {}
    # the repeat exists to balance instruments; with one instrument in the training classes it only multiplies the epoch
    instruments = {meta["classes"][c].get("instrument") for c in tr_cls if c in meta.get("classes", {})}
    if len(instruments) < 2: repeats = {}
    if any(r > 1 for r in repeats.values()):
        fit_idx = np.concatenate([np.repeat(fit_idx[tr.classes[fit_idx] == c], repeats.get(c, 1)) for c in np.unique(tr.classes[fit_idx])])
    scale_vec = target_scale(tr, args.channel_scale)
    scale = torch.tensor(scale_vec, dtype=torch.float32).view(1, -1, 1, 1)
    scale_dev = scale.to(dev)
    os.makedirs(args.out, exist_ok=True)
    print("device %s   train %s: %d fit + %d val   test %s: %d   width %d   layout %s   target scale %s   select on %s"
          % (dev, tr_cls, len(fit_idx), nval, te_cls, len(te), args.width, "ragged, native size" if tr.ragged else "fixed frame",
             "/".join("%.4f" % v for v in scale_vec), args.select_on))

    cin, cout = tr.cin, tr.cout
    model = stems.build(args.arch, tr.roles, args.width, cin, cout).to(dev)
    print("arch %s   roles %s%s" % (args.arch, tr.roles, "   role dropout %.2f" % args.role_dropout if args.role_dropout else ""))
    if args.init:
        state = torch.load(args.init, map_location=dev)
        missing = model.load_state_dict(state, strict=False)
        print("initialised from %s (%d missing, %d unexpected keys)" % (args.init, len(missing.missing_keys), len(missing.unexpected_keys)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (len(fit_idx) // args.batch + 1))
    # Mixed precision on CUDA only. bfloat16 where the GPU has it (Ampere and later): it keeps
    # fp32's exponent range, so activations cannot overflow the way fp16 did in the first full
    # A8 run (NaN from epoch 22 of 40; the epoch-20 checkpoint survived). fp16 with a GradScaler
    # remains the fallback for older GPUs, and every non-finite batch is skipped, never stepped.
    amp_dtype = None
    if dev.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("mixed precision: %s" % str(amp_dtype).replace("torch.", ""))
    use_scaler = amp_dtype is torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    best_epoch = 0
    log = open(os.path.join(args.out, "log.csv"), "w", newline=""); lw = csv.writer(log)
    lw.writerow(["epoch", "train_l1", "val_l1"] + ["%s_%s" % (n, k) for n in names for k in ("pixel_r", "event_r", "pred_PS")])
    best = float("inf"); t0 = time.time()
    best_ch = np.full(len(names), np.inf); best_ep = [0] * len(names); vm = 0.0
    val_ds = tr

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
        xb, yb, mb = te.batch(snap_idx)           # padded to the largest of the six, display only
        snap_x, snap_y, snap_m = xb.numpy(), yb.numpy(), mb.numpy()[:, 0]
        snap_preds = {}

    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0; skipped = 0
        for b in make_batches(tr, fit_idx, args.batch, rng):
            x, y, m = tr.batch(b); x, y, m = x.to(dev), y.to(dev) / scale_dev, m.to(dev)
            m = m * avail_of(tr, b).to(dev)[:, :, None, None]     # (B, C, H, W): only channels this object has
            if args.full_frame:
                # Train on every pixel. The padded zone is background with a zero target,
                # and a network never penalised there paints stain onto it (seen in the
                # ISX progression sheets). The validity mask still scopes the metrics.
                m = torch.ones_like(m) * avail_of(tr, b).to(dev)[:, :, None, None]
            x, y, m = augment(x, y, m, args.aug)
            present = present_of(tr, b)
            if args.role_dropout > 0:
                # drop each present role with probability p, never all of an object's roles
                keep = torch.rand(present.shape) >= args.role_dropout
                keep[torch.arange(len(b)), torch.multinomial(present.float(), 1)[:, 0]] = True
                present = present & keep
            # mixed precision on CUDA only; torch 2.2 rejects autocast on mps even when disabled
            amp = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
            with amp:
                out = stems.call(model, x, present.to(dev))
                loss = (F.l1_loss(out, y, reduction="none") * m).sum() / m.sum().clamp(min=1.0) / out.shape[1]
                if args.grad_loss > 0:
                    loss = loss + args.grad_loss * grad_loss(out, y, m)
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += loss.item(); nb += 1
            else:
                skipped += 1        # the batch is dropped; the weights never see it
            try: sched.step()
            except ValueError: pass   # OneCycle's step budget is an estimate when batches vary
        if skipped:
            print("epoch %3d: %d of %d batches had a non-finite loss and were skipped" % (ep, skipped, skipped + nb))
        if nb == 0:
            raise SystemExit("epoch %d: every batch was non-finite; stopping. best.pt is from epoch %d" % (ep, best_epoch))
        tot /= nb
        # validation L1 on the held-out slice of the TRAINING replicate
        model.eval(); vl = 0.0; vn = 0; vch = np.zeros(len(names)); vm = 0.0
        with torch.no_grad():
            for b in make_batches(val_ds, val_idx, 256):
                x, y, m = val_ds.batch(b); x, y, m = x.to(dev), y.to(dev) / scale_dev, m.to(dev)
                if args.full_frame: m = torch.ones_like(m)
                m = m * avail_of(val_ds, b).to(dev)[:, :, None, None]
                per = (F.l1_loss(stems.call(model, x, present_of(val_ds, b).to(dev)), y, reduction="none") * m).sum(dim=(0, 2, 3)).cpu().numpy()
                cnt = m.sum(dim=(0, 2, 3)).cpu().numpy()
                vch += per; vm = vm + cnt
                vl += (per.sum() / max(cnt.sum(), 1.0)); vn += 1
        vl /= max(vn, 1); vch = vch / np.maximum(np.asarray(vm, dtype=np.float64), 1.0)
        release_device_memory(dev)
        # Each marker keeps its own best epoch: a shared trunk's best moment differs per
        # marker, and one checkpoint cannot serve all of them.
        for k, nm in enumerate(names):
            if vch[k] < best_ch[k]:
                best_ch[k] = vch[k]; best_ep[k] = ep
                torch.save(model.state_dict(), os.path.join(args.out, "best_%s.pt" % nm.replace("/", "_")))
        res, _ = evaluate(model, te, wte, scale, dev, names) if (ep % args.eval_every == 0 or ep == args.epochs) else ({}, None)
        release_device_memory(dev)
        lw.writerow([ep, tot, vl] + [res[n][k] if res else "" for n in names for k in ("pixel_r", "event_r", "pred_PS")]); log.flush()
        line = "epoch %3d  train %.4f  val %.4f" % (ep, tot, vl)
        if res: line += "   test event r: " + "  ".join("%s %s" % (n, "%.3f" % res[n]["event_r"] if res[n]["event_r"] is not None else "n/a") for n in names)
        if res and "ACRV1" in res and res["ACRV1"]["within_well_event_r"].get("S") is not None:
            line += "   ACRV1 within-sperm %.3f" % res["ACRV1"]["within_well_event_r"]["S"]
        print(line + "   (%.0fs)" % (time.time() - t0))
        # Checkpoint selection, on the training replicate's held-out slice only.
        if args.select_on == "val_event_r":
            rv, _ = evaluate(model, tr, wtr[val_idx], scale, dev, names, idx=val_idx)
            mean_r = float(np.mean([rv[n]["event_r"] for n in names if rv[n]["event_r"] is not None]))
            score, line_sel = -mean_r, "   val event r %.3f" % mean_r
        elif args.select_on == "acrv1_sperm":
            rv, _ = evaluate(model, tr, wtr[val_idx], scale, dev, names, idx=val_idx)
            w_acrv = rv["ACRV1"]["within_well_event_r"]; r_s = w_acrv.get("S", next(iter(w_acrv.values()), None)) if w_acrv else None
            score = -(r_s if r_s is not None else -1.0)
            line_sel = "   val ACRV1 within-sperm %.3f" % (r_s if r_s is not None else float("nan"))
        else:
            score, line_sel = vl, ""
        if line_sel: print("          " + line_sel.strip())
        if score < best:
            best = score; best_epoch = ep; torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        if args.snapshot and (ep % args.snapshot == 0 or ep == 1 or ep == args.epochs):
            model.eval()
            with torch.no_grad():
                sp = (stems.call(model, torch.tensor(snap_x).to(dev), present_of(te, snap_idx).to(dev)).float().cpu() * scale).numpy()
            snap_preds[ep] = sp
            np.savez_compressed(os.path.join(args.out, "epoch_%02d.npz" % ep), pred=sp, true=snap_y, x=snap_x, mask=snap_m, idx=np.array(snap_idx))
            snapshot_sheet(args.out, ep, snap_x, snap_y, sp, snap_m, wte[snap_idx], names, avail_of(te, snap_idx).numpy() > 0)

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    res, P = evaluate(model, te, wte, scale, dev, names, tta=args.tta)
    print("\nbest-by-validation model on the held-out replicate:")
    print("%-12s %9s %9s %12s %12s   %s" % ("marker", "pixel r", "event r", "pred P:S", "true P:S", "want"))
    wants = dict(zip(["LDHC/AKAP4", "LDHC", "CD45", "ACRV1"], ["<< 1", "<< 1", ">> 1", "<< 1"]))
    for n in names:
        r = res[n]
        if r["event_r"] is None:
            print("%-12s %9s %9s   (not stained on the test instrument)" % (n, "n/a", "n/a")); continue
        ps = ("%12.2f %12.2f   %s" % (r["pred_PS"], r["true_PS"], wants.get(n, ""))) if r["pred_PS"] is not None else "%12s %12s" % ("n/a", "n/a")
        print("%-12s %9.3f %9.3f %s" % (n, r["pixel_r"], r["event_r"], ps))
    kinds = sorted({k for n in names for k in res[n]["within_well_event_r"]})
    print("\nevent r WITHIN one well type -- the number that cannot be earned by recognising the cell:")
    print("%-12s " % "marker" + " ".join("%14s" % k for k in kinds))
    for n in names:
        w = res[n]["within_well_event_r"]
        print("%-12s " % n + " ".join("%14s" % ("%.3f" % w[k] if w.get(k) is not None else "n/a") for k in kinds))
    per_marker = {}
    if len(names) > 1:
        print("\nsame joint model, but each marker evaluated at ITS OWN best validation epoch:")
        print("%-12s %6s %9s %9s   %s" % ("marker", "epoch", "pixel r", "event r", "within-well event r"))
        for k, nm in enumerate(names):
            ck = os.path.join(args.out, "best_%s.pt" % nm.replace("/", "_"))
            if not os.path.exists(ck): continue
            model.load_state_dict(torch.load(ck, map_location=dev))
            rk, _ = evaluate(model, te, wte, scale, dev, names, tta=args.tta)
            per_marker[nm] = dict(rk[nm], epoch=best_ep[k])
            if rk[nm]["event_r"] is None: continue
            w = rk[nm]["within_well_event_r"]
            print("%-12s %6d %9.3f %9.3f   %s" % (nm, best_ep[k], rk[nm]["pixel_r"], rk[nm]["event_r"],
                  "  ".join("%s %.3f" % (kk, v) for kk, v in w.items() if v is not None)))
        model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    json.dump({"train": tr_cls, "test": te_cls, "epochs": args.epochs, "width": args.width, "target": args.target, "full_frame": args.full_frame,
               "aug": args.aug, "grad_loss": args.grad_loss, "tta": args.tta, "seed": args.seed,
               "channel_scale": args.channel_scale, "select_on": args.select_on, "results": res,
               "scale": [float(v) for v in scale_vec], "include": args.include, "init": args.init,
               "arch": args.arch, "roles": list(tr.roles), "role_dropout": args.role_dropout,
               "n_fit": int(len(fit_idx)), "n_val": int(nval), "n_test": int(len(te)), "best_epoch": best_epoch,
               "device": str(dev), "mixed_precision": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
               "per_marker_checkpoint": per_marker},
              open(os.path.join(args.out, "results.json"), "w"), indent=1)

    if args.snapshot and snap_preds:
        progression_sheet(args.out, snap_x, snap_y, snap_preds, snap_m, names)

    kinds_te = sorted(set(wte)); per = max(1, 9 // len(kinds_te))
    pick = [np.where(wte == w)[0][j] for w in kinds_te for j in [3, 40, 400, 900, 2000, 5000][:per] if (wte == w).sum() > j]
    xb, yb, mb = te.batch(pick)
    pb = np.stack([np.pad(P[j], [(0, 0), ((yb.shape[2] - P[j].shape[1]) // 2, yb.shape[2] - P[j].shape[1] - (yb.shape[2] - P[j].shape[1]) // 2),
                                 ((yb.shape[3] - P[j].shape[2]) // 2, yb.shape[3] - P[j].shape[2] - (yb.shape[3] - P[j].shape[2]) // 2)]) for j in pick])
    sheet(os.path.join(args.out, "preds.png"), "held-out predictions (best checkpoint)", xb.numpy(), yb.numpy(), pb, mb.numpy()[:, 0], wte[pick], names, avail_of(te, pick).numpy() > 0)
    print("wrote", os.path.join(args.out, "results.json"), "and preds.png")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="sperm_fluorescence train", description="Train on a cache; one replicate (or class set) fits, another is held out.")
    t = ap; t.add_argument("--cache", required=True); t.add_argument("--train", default="2"); t.add_argument("--test", default="3")
    t.add_argument("--epochs", type=int, default=40); t.add_argument("--batch", type=int, default=64); t.add_argument("--width", type=int, default=48)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--eval-every", type=int, default=5); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs/fold")
    t.add_argument("--channel-scale", action="store_true", help="Normalise the L1 per target channel instead of one shared scale")
    t.add_argument("--full-frame", action="store_true",
                   help="Compute the training loss over the whole frame, padding included (requires a "
                        "background-padded cache, i.e. prepare_isx after 33089b3). Metrics stay masked.")
    t.add_argument("--include", default=None, metavar="CLS,CLS", help="Restrict to these cache classes before the train/test split")
    t.add_argument("--init", default=None, metavar="CKPT", help="Initialise weights from a checkpoint (warm start / fine-tune)")
    t.add_argument("--target", default=None, metavar="NAME",
                   help="Train a single-output model for one marker (name as in the cache's meta.json)")
    t.add_argument("--snapshot", type=int, default=0, metavar="N",
                   help="Every N epochs, save predictions for a fixed set of test objects (epoch_XX.png + .npz) and, at the end, a progression sheet")
    t.add_argument("--select-on", choices=["val_l1", "val_event_r", "acrv1_sperm"], default="val_l1",
                   help="Checkpoint selection on the validation slice: L1 (default), mean event r across markers, or within-sperm ACRV1 event r")
    t.add_argument("--aug", type=int, default=0, choices=[0, 1],
                   help="0: horizontal flip only. 1: + vertical flip, +-6 px shift, input gain/offset jitter")
    t.add_argument("--grad-loss", type=float, default=0.0, metavar="W",
                   help="Weight of a gradient-difference term beside the L1, for sharper edges (try 0.5)")
    t.add_argument("--tta", action="store_true", help="Average predictions over the four flips at evaluation")
    t.add_argument("--arch", choices=["stack", "stems"], default="stack",
                   help="stack: fixed input channels (default). stems: one stem per input role, any subset at inference")
    t.add_argument("--roles", default=None, metavar="R,R", help="Input roles to use, by cache name (default: all)")
    t.add_argument("--role-dropout", type=float, default=0.0, metavar="P", help="stems: drop each role with probability P in training")
    return train(ap.parse_args(argv))
