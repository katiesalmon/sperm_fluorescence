"""Losses and target scaling."""
import numpy as np


def grad_loss(p, y, m):
    """L1 between spatial gradients of prediction and target, masked. L1 on pixels alone
    rewards blur; matching gradients rewards edges in the right places."""
    import torch.nn.functional as F
    dpx, dpy = p[..., :, 1:] - p[..., :, :-1], p[..., 1:, :] - p[..., :-1, :]
    dyx, dyy = y[..., :, 1:] - y[..., :, :-1], y[..., 1:, :] - y[..., :-1, :]
    mx, my = m[..., :, 1:] * m[..., :, :-1], m[..., 1:, :] * m[..., :-1, :]
    return ((F.l1_loss(dpx, dyx, reduction="none") * mx).sum() + (F.l1_loss(dpy, dyy, reduction="none") * my).sum()) / (mx.sum() + my.sum() + 1e-6)


def target_scale(tr, channel_scale):
    """Target divisor(s) from a deterministic sample of the training set."""
    if getattr(tr, "avail", None) is not None:
        # Mixed-instrument cache: a channel's scale comes only from objects that carry it,
        # otherwise the unstained objects' zeros drag it towards 0 (and the loss to NaN).
        sidx = tr.sample_idx(200); per_ch = []
        for k in range(tr.cout):
            ok = [i for i in sidx if tr.avail[i, k]]
            per_ch.append(float(np.mean([np.abs(tr.item(i)[1][k]).mean() for i in ok])) if ok else 1.0)
        ysample = np.array(per_ch, np.float32)[None, :, None, None]
    else:
        ysample = tr.sample_targets(200)
    if channel_scale:
        # One scale per channel. With a single shared scale the L1 is dominated by the
        # brightest channel: AF488 is ~3x PerCP, so ACRV1 -- the dimmest and the one with the
        # least headroom to waste -- got roughly a fifth of the gradient in the first runs.
        scale_vec = np.abs(ysample).mean(axis=(0, 2, 3)) * 10
    else:
        scale_vec = np.full(ysample.shape[1], np.abs(ysample).mean() * 10, np.float32)
    return np.where(scale_vec > 0, scale_vec, 1.0).astype(np.float32)

