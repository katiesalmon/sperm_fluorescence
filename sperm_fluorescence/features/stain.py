"""Features from the stain images themselves, with no threshold anywhere: intensity moments
per channel, radial and axial intensity profiles, cross-channel centroid geometry, and a
canonical aligned pixel stack (every channel rotated onto the cell's principal axis, head
to the left, resampled to a fixed grid) for pixel-level PCA."""
import numpy as np
from scipy import ndimage as ndi

CANON_PX = 48          # canonical grid side
CANON_UM = 0.667       # canonical pixel size, um (48 px = 32 um)
N_AXIAL = 16           # axial profile bins over +-8 um
RADII_UM = (1.0, 2.0, 4.0, 8.0)
HEAD_PREFERENCE = ("DAPI", "ACRV1", "TOMM20", "LDHC/AKAP4", "CD45")

DOC = {
    "total": "sum of the channel (clipped at 0) over the valid frame", "peak": "maximum pixel",
    "cy": "intensity-weighted centroid row, px", "cx": "intensity-weighted centroid column, px",
    "sigma_major_um": "intensity-weighted standard deviation along the channel's own major axis, um",
    "sigma_minor_um": "the same along its minor axis, um", "anisotropy": "1 - sigma_minor / sigma_major",
    "orientation_deg": "orientation of the channel's intensity ellipse, degrees from the row axis",
    "entropy": "entropy of the normalised intensity distribution over the valid frame, nats (spread)",
    "peak_frac": "peak / total (how concentrated)",
}
for r in RADII_UM: DOC["within_%gum" % r] = "fraction of the channel's intensity within %g um of its centroid" % r
for i in range(N_AXIAL): DOC["axial_%02d" % i] = "fraction of the channel's intensity in axial bin %d of %d (-8..+8 um along the cell axis, head at the negative end)" % (i, N_AXIAL)
X_DOC = {"axis_angle_deg": "orientation of the whole-cell intensity ellipse", "cell_sigma_major_um": "whole-cell intensity sigma along the axis, um",
         "cell_sigma_minor_um": "whole-cell intensity sigma across the axis, um"}


def moments(img):
    """total, centroid (y, x), covariance of an intensity image (clipped at 0)."""
    w = np.clip(img, 0, None).astype(np.float64); tot = w.sum()
    if tot <= 0: return 0.0, None, None
    yy, xx = np.indices(w.shape); cy, cx = (w * yy).sum() / tot, (w * xx).sum() / tot
    dy, dx = yy - cy, xx - cx
    cov = np.array([[(w * dy * dy).sum(), (w * dy * dx).sum()], [(w * dy * dx).sum(), (w * dx * dx).sum()]]) / tot
    return float(tot), (cy, cx), cov


def ellipse(cov):
    ev, vec = np.linalg.eigh(cov); ev = np.clip(ev, 0, None)
    major = vec[:, 1]; ang = np.degrees(np.arctan2(major[1], major[0]))
    return float(np.sqrt(ev[1])), float(np.sqrt(ev[0])), float(ang)


def channel_features(img, valid, px_um, axis, centre):
    """DOC's keys for one channel. axis: unit vector (dy, dx) of the cell's head->tail axis;
    centre: the cell's intensity centroid (row, col)."""
    out = {k: np.nan for k in DOC}
    w = np.clip(img, 0, None) * valid; tot, c, cov = moments(w)
    out["total"] = float(tot); out["peak"] = float(w.max())
    if tot <= 0 or c is None: return out
    out["cy"], out["cx"] = c; sM, sm, ang = ellipse(cov)
    out["sigma_major_um"], out["sigma_minor_um"], out["orientation_deg"] = sM * px_um, sm * px_um, ang
    out["anisotropy"] = 1 - sm / sM if sM > 0 else np.nan
    p = (w / tot)[valid]; p = p[p > 0]; out["entropy"] = float(-(p * np.log(p)).sum()); out["peak_frac"] = out["peak"] / tot
    yy, xx = np.indices(w.shape); r = np.hypot(yy - c[0], xx - c[1]) * px_um
    for rad in RADII_UM: out["within_%gum" % rad] = float(w[r <= rad].sum() / tot)
    s = ((yy - centre[0]) * axis[0] + (xx - centre[1]) * axis[1]) * px_um
    edges = np.linspace(-8, 8, N_AXIAL + 1); b = np.clip(np.digitize(s, edges) - 1, 0, N_AXIAL - 1)
    inside = (s >= -8) & (s < 8)
    for i in range(N_AXIAL): out["axial_%02d" % i] = float(w[inside & (b == i)].sum() / tot)
    return out


def cell_axis(chans, names, valid, px_um):
    """The whole-cell intensity ellipse from the summed (normalised) channels, oriented so the
    head channel (DAPI, else ACRV1, ...) lies at the negative end. Returns (axis, centre, info)."""
    acc = np.zeros(valid.shape, np.float64)
    for k, nm in enumerate(names):
        w = np.clip(chans[k], 0, None) * valid
        if w.sum() > 0: acc += w / w.sum()
    tot, c, cov = moments(acc)
    if tot <= 0 or c is None: return np.array([1.0, 0.0]), (valid.shape[0] / 2, valid.shape[1] / 2), {k: np.nan for k in X_DOC}
    sM, sm, ang = ellipse(cov); a = np.deg2rad(ang); axis = np.array([np.cos(a), np.sin(a)])
    head = next((names.index(h) for h in HEAD_PREFERENCE if h in names), 0)
    _, ch, _ = moments(np.clip(chans[head], 0, None) * valid)
    if ch is not None and ((ch[0] - c[0]) * axis[0] + (ch[1] - c[1]) * axis[1]) > 0: axis = -axis; ang = (ang + 180) % 360 - 180
    return axis, c, {"axis_angle_deg": float(ang), "cell_sigma_major_um": sM * px_um, "cell_sigma_minor_um": sm * px_um}


def canonical_stack(images, valid, px_um, axis, centre, fill):
    """(C, CANON_PX, CANON_PX): every image rotated so the cell axis runs left to right with the
    head at the left, centred on the cell centroid, resampled to CANON_UM per pixel. `fill`:
    per-channel value outside the frame (0 for stains, the border median for inputs)."""
    C = images.shape[0]; out = np.empty((C, CANON_PX, CANON_PX), np.float32)
    ang = np.degrees(np.arctan2(axis[1], axis[0]))          # angle of the axis from the row direction
    z = px_um / CANON_UM; half = CANON_PX / 2.0
    # rotation that maps the axis onto +x (columns): rotate the image by -(90 - ang)... do it with an affine map instead
    th = np.arctan2(axis[1], axis[0])
    # canonical (r, c) -> source (y, x): y = cy + (c - half)*CANON/px * axis_y + (r - half)*CANON/px * perp_y, same for x
    perp = np.array([-axis[1], axis[0]])
    M = np.array([[perp[0] / z, axis[0] / z], [perp[1] / z, axis[1] / z]])
    offset = np.array(centre) - M @ np.array([half, half])
    for c in range(C):
        out[c] = ndi.affine_transform(images[c].astype(np.float32), M, offset=offset, output_shape=(CANON_PX, CANON_PX), order=1, mode="constant", cval=float(fill[c]))
    return out
