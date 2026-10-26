"""Training-time augmentation. Nothing here changes what a target means."""
import numpy as np


def augment(x, y, m, level):
    """level 0: horizontal flip only (the scan axis is symmetric). level 1 adds a vertical
    flip, a small joint shift of input, target and mask, and a gain/offset jitter on the
    label-free input -- the usual recipe; none of it changes what the target means."""
    import torch
    if np.random.rand() < 0.5: x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
    if level >= 1:
        if np.random.rand() < 0.5: x, y, m = x.flip(-2), y.flip(-2), m.flip(-2)
        dy, dx = np.random.randint(-6, 7), np.random.randint(-6, 7)
        if dy or dx:
            x, y, m = (torch.roll(t, shifts=(dy, dx), dims=(-2, -1)) for t in (x, y, m))
        gain = 1.0 + 0.1 * torch.randn(x.shape[0], x.shape[1], 1, 1, device=x.device)
        offset = 0.1 * torch.randn(x.shape[0], x.shape[1], 1, 1, device=x.device)
        x = x * gain + offset
    return x, y, m

