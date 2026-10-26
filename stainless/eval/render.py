"""IDEAS-style rendering of events: one colour per channel on black, and additive overlays.

IDEAS draws each channel in its own colour on a black field and composites them by
adding light, which is how the gallery in the .daf is set up (Ch02 Lime, Ch03 Yellow,
Ch07 DarkOrchid, Ch11 Red). The same convention is used here for predictions, so a true
and a predicted channel are drawn with the same colour and the same display range, and
an overlay of predictions can be put beside an overlay of truth.
"""

import numpy as np

# Marker -> RGB in [0, 1]. Brightfield-like inputs are drawn in grey.
PALETTE = {
    "LDHC/AKAP4": (0.35, 1.00, 0.35),   # lime
    "ACRV1":      (1.00, 1.00, 0.30),   # yellow
    "DAPI":       (0.65, 0.35, 0.95),   # dark orchid
    "TOMM20":     (1.00, 0.30, 0.30),   # red
    "CD45":       (1.00, 0.40, 0.85),   # magenta
}
FALLBACK = [(0.4, 0.8, 1.0), (1.0, 0.6, 0.2), (0.6, 1.0, 0.8), (1.0, 0.8, 0.9)]


def color_for(name, k=0):
    return PALETTE.get(name, FALLBACK[k % len(FALLBACK)])


def colorize(a, color, vmin, vmax):
    """(h, w) intensities -> (h, w, 3) RGB: black at vmin, `color` at vmax."""
    t = np.clip((a.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return t[..., None] * np.asarray(color, np.float32)[None, None, :]


def grey(a, lo=0.5, hi=99.5):
    p0, p1 = np.percentile(a, [lo, hi])
    t = np.clip((a.astype(np.float32) - p0) / max(p1 - p0, 1e-6), 0, 1)
    return np.repeat(t[..., None], 3, axis=-1)


def composite(layers):
    """Additive blend of (h, w, 3) layers, clipped -- IDEAS's 'combine' view."""
    return np.clip(np.sum(layers, axis=0), 0, 1)


def ranges(trues, mask=None, hi=99.5):
    """Per-channel display range from the TRUE images of a set of objects, so a prediction
    is drawn on exactly the scale its truth is. trues: list of (C, h, w); returns [(0, vmax)]."""
    C = trues[0].shape[0]
    out = []
    for k in range(C):
        vals = np.concatenate([(t[k][m > 0] if m is not None else t[k].ravel()) for t, m in zip(trues, mask if mask is not None else [None] * len(trues))])
        out.append((0.0, max(float(np.percentile(vals, hi)), 1e-4)))
    return out
