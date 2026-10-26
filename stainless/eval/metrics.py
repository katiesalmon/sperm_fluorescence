"""Native-size evaluation: pixel r, per-event r, within-population r, P:S ratios."""
import numpy as np

from ..data.batching import avail_of, make_batches
from ..train.device import release_device_memory


def predict(model, x, tta=False):
    """Model output, optionally averaged over the four flips (test-time augmentation)."""
    import torch
    if not tta:
        return model(x)
    outs = [model(x), model(x.flip(-1)).flip(-1), model(x.flip(-2)).flip(-2), model(x.flip(-1).flip(-2)).flip(-2).flip(-1)]
    return torch.stack(outs).mean(0)


def evaluate(model, ds, wells, scale, dev, names, bs=64, idx=None, tta=False):
    """Score on `idx` (default: everything); `wells` is already that subset's wells. Native
    size on a ragged cache: predictions are made per object, pixel r pooled over valid
    pixels, event r over per-object sums. Returns (res, preds) with preds[i] = (C, h, w)."""
    import torch
    idx = np.arange(len(ds)) if idx is None else np.asarray(idx)
    model.eval()
    preds = [None] * len(idx); trues = [None] * len(idx); masks = [None] * len(idx)
    pos = {int(j): q for q, j in enumerate(idx)}
    with torch.no_grad():
        for nb, b in enumerate(make_batches(ds, idx, bs)):
            x, y, m = ds.batch(b)
            p = (predict(model, x.to(dev), tta).float().cpu() * scale).numpy(); y = y.numpy(); m = m.numpy()[:, 0]
            # Every shape bucket is a new allocation size; the allocator cached them all
            # (22 GB during one evaluation of 10,600 ragged objects on MPS). Hand them back.
            if nb % 20 == 19: release_device_memory(dev)
            H, W = p.shape[-2:]
            for q, j in enumerate(b):
                # crop back to the object's own size: a batch is padded to its largest member
                h, w = ds.shape(int(j)); t, l = (H - h) // 2, (W - w) // 2
                preds[pos[int(j)]] = p[q][:, t:t + h, l:l + w]; trues[pos[int(j)]] = y[q][:, t:t + h, l:l + w]; masks[pos[int(j)]] = m[q][t:t + h, l:l + w]
    release_device_memory(dev)
    res = {}
    av = avail_of(ds, idx).numpy() > 0
    for k, n in enumerate(names):
        ok = np.flatnonzero(av[:, k])            # objects whose instrument stained this channel
        if len(ok) < 3:
            res[n] = {"pixel_r": None, "event_r": None, "pred_PS": None, "true_PS": None, "within_well_event_r": {}}
            continue
        w_ok = wells[ok]
        has_ps = (w_ok == "P").any() and (w_ok == "S").any()
        pp = np.concatenate([preds[i][k][masks[i] > 0] for i in ok]); tt = np.concatenate([trues[i][k][masks[i] > 0] for i in ok])
        ep = np.array([(preds[i][k] * masks[i]).sum() for i in ok]); et = np.array([(trues[i][k] * masks[i]).sum() for i in ok])
        ps = (lambda v: float(np.median(v[w_ok == "P"]) / max(abs(np.median(v[w_ok == "S"])), 1e-9))) if has_ps else (lambda v: None)
        within = {}
        for w in sorted(set(w_ok)):
            sel = w_ok == w
            within[w] = float(np.corrcoef(ep[sel], et[sel])[0, 1]) if sel.sum() > 2 else None
        res[n] = {"pixel_r": float(np.corrcoef(pp, tt)[0, 1]), "event_r": float(np.corrcoef(ep, et)[0, 1]),
                  "pred_PS": ps(ep), "true_PS": ps(et), "within_well_event_r": within}
    return res, preds

