"""Intensity and texture of one channel inside one mask, and similarity between two channels."""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops

DOC = {
    "sum": "integrated intensity inside the mask (background not subtracted here; cache targets already are)",
    "max": "maximum pixel inside the mask", "mean": "mean pixel inside the mask", "median": "median pixel inside the mask",
    "std": "standard deviation inside the mask", "bkgd": "median of the valid pixels outside the mask",
    "bright_detail": "sum inside the mask of (image - grey opening with a 7 px disc), clipped at 0 (IDEAS Bright Detail Intensity R3-like)",
    "gradient_rms": "root mean square of the Sobel gradient magnitude inside the mask (IDEAS Gradient RMS-like; focus)",
    "contrast": "std of the Sobel gradient magnitude inside the mask / mean intensity (IDEAS Contrast-like)",
    "glcm_entropy": "entropy of the 8-level co-occurrence matrix at distance 1 over the mask's bounding box",
    "glcm_homogeneity": "homogeneity of the same co-occurrence matrix",
    "frac_in_mask": "sum inside the mask / sum over the valid frame (how much of the channel's signal the mask captures)",
}
NAN = {k: np.nan for k in DOC}
DISC = np.array([[int(np.hypot(y - 3, x - 3) <= 3.0) for x in range(7)] for y in range(7)], bool)


def intensity_features(img, mask, valid):
    """dict of DOC's keys for one (h, w) image and mask within the valid frame."""
    m = mask & valid
    if m.sum() == 0: return dict(NAN)
    v = img[m]; out = {"sum": float(v.sum()), "max": float(v.max()), "mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std())}
    outside = valid & ~mask
    out["bkgd"] = float(np.median(img[outside])) if outside.any() else np.nan
    opened = ndi.grey_opening(img, footprint=DISC)
    out["bright_detail"] = float(np.clip(img - opened, 0, None)[m].sum())
    gy, gx = ndi.sobel(img, axis=0), ndi.sobel(img, axis=1); g = np.hypot(gy, gx)[m]
    out["gradient_rms"] = float(np.sqrt((g ** 2).mean()))
    out["contrast"] = float(g.std() / abs(v.mean())) if abs(v.mean()) > 1e-6 else np.nan
    ys, xs = np.nonzero(m); crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    lo, hi = np.percentile(crop, [1, 99])
    q = np.clip(((crop - lo) / max(hi - lo, 1e-6) * 7.999).astype(np.uint8), 0, 7)
    if q.shape[0] > 1 and q.shape[1] > 1:
        G = graycomatrix(q, [1], [0, np.pi / 2], levels=8, symmetric=True, normed=True)
        p = G[..., 0, :].mean(-1); p = p[p > 0]
        out["glcm_entropy"] = float(-(p * np.log2(p)).sum()); out["glcm_homogeneity"] = float(graycoprops(G, "homogeneity").mean())
    else:
        out["glcm_entropy"] = out["glcm_homogeneity"] = np.nan
    tot = float(img[valid].sum())
    out["frac_in_mask"] = out["sum"] / tot if abs(tot) > 1e-6 else np.nan
    return out


def similarity(a, b, mask):
    """Pearson r between two channels inside a mask, and its Fisher z (IDEAS Similarity is a
    log-transformed Pearson coefficient; z serves the same purpose)."""
    m = mask
    if m.sum() < 3: return {"r": np.nan, "z": np.nan}
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    if x.std() == 0 or y.std() == 0: return {"r": np.nan, "z": np.nan}
    r = float(np.corrcoef(x, y)[0, 1]); r = max(min(r, 0.999999), -0.999999)
    return {"r": r, "z": float(np.arctanh(r))}
