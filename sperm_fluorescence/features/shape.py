"""Shape features of one binary mask. Every value is defined here, in one place, because the
literature's definitions of "elongation" and "ellipticity" disagree and IDEAS does not
document which chord its Length and Width are. Several candidates are computed (bounding
box, Feret calipers, fitted ellipse) and the validation against the instrument's own table
says which matches."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

DOC = {
    "n_px": "mask pixels", "area_um2": "mask pixels x pixel area", "n_components": "8-connected components",
    "perimeter_um": "skimage perimeter of the largest component, in um",
    "bbox_h_um": "bounding-box height (rows, the flow direction), um", "bbox_w_um": "bounding-box width (columns), um",
    "feret_max_um": "largest caliper diameter of the convex hull (at 2 degree steps), um",
    "feret_min_um": "smallest caliper width of the convex hull (at 2 degree steps), um",
    "major_um": "major axis of the moments ellipse (skimage major_axis_length), um", "minor_um": "minor axis of the moments ellipse, um",
    "orientation_deg": "orientation of the moments ellipse, degrees from the row axis, (-90, 90]",
    "aspect_ratio": "minor / major of the moments ellipse (IDEAS Aspect Ratio convention)",
    "ellipticity": "feret_max / feret_min (CASA-Morph L/W)", "elongation": "(feret_max - feret_min) / (feret_max + feret_min)",
    "rugosity": "4 pi area / perimeter^2 (1 = circle)", "regularity": "pi L W / (4 area), L and W the Feret calipers",
    "circularity": "mean boundary radius / std of boundary radius, from the largest component's centroid (IDEAS-like)",
    "solidity": "area / convex-hull area", "extent": "area / bounding-box area",
    "thickness_max_um": "2 x the largest distance-transform value, um (IDEAS Thickness Max)",
    "thickness_mean_um": "2 x the mean distance-transform value over the mask, um",
    "n_holes": "holes in the largest component (Euler number)",
    "centroid_y": "row of the centroid, px (frame coordinates)", "centroid_x": "column of the centroid, px",
}
NAN = {k: np.nan for k in DOC}


def _feret(coords):
    """(max caliper, min caliper) over the convex hull of pixel-corner points, px."""
    from scipy.spatial import ConvexHull
    pts = np.concatenate([coords + d for d in ((0, 0), (0, 1), (1, 0), (1, 1))]).astype(np.float64)
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    ang = np.deg2rad(np.arange(0, 180, 2.0)); dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    proj = hull @ dirs.T
    widths = proj.max(0) - proj.min(0)
    return float(widths.max()), float(widths.min())


def shape_features(mask, px_um):
    """dict of DOC's keys for one (h, w) bool mask; NaNs for an empty mask."""
    n = int(mask.sum())
    if n == 0: return dict(NAN, n_px=0, area_um2=0.0, n_components=0)
    lab, nc = ndi.label(mask, structure=np.ones((3, 3)))
    if nc > 1:
        sizes = ndi.sum(mask, lab, range(1, nc + 1)); big = lab == (1 + int(np.argmax(sizes)))
    else:
        big = mask
    out = {"n_px": n, "area_um2": n * px_um ** 2, "n_components": int(nc)}
    rp = measure.regionprops(big.astype(np.uint8))[0]
    out["perimeter_um"] = float(rp.perimeter) * px_um
    out["bbox_h_um"] = float(rp.bbox[2] - rp.bbox[0]) * px_um; out["bbox_w_um"] = float(rp.bbox[3] - rp.bbox[1]) * px_um
    fmax, fmin = _feret(rp.coords)
    out["feret_max_um"] = fmax * px_um; out["feret_min_um"] = fmin * px_um
    out["major_um"] = float(rp.major_axis_length) * px_um; out["minor_um"] = float(rp.minor_axis_length) * px_um
    out["orientation_deg"] = float(np.rad2deg(rp.orientation))
    out["aspect_ratio"] = float(rp.minor_axis_length / rp.major_axis_length) if rp.major_axis_length > 0 else np.nan
    out["ellipticity"] = fmax / fmin if fmin > 0 else np.nan
    out["elongation"] = (fmax - fmin) / (fmax + fmin) if (fmax + fmin) > 0 else np.nan
    a = float(big.sum()); P = float(rp.perimeter)
    out["rugosity"] = 4 * np.pi * a / P ** 2 if P > 0 else np.nan
    out["regularity"] = np.pi * fmax * fmin / (4 * a) if a > 0 else np.nan
    cy, cx = rp.centroid; by, bx = np.nonzero(big & ~ndi.binary_erosion(big))
    r = np.hypot(by - cy, bx - cx)
    out["circularity"] = float(r.mean() / r.std()) if len(r) > 2 and r.std() > 0 else np.nan
    out["solidity"] = float(rp.solidity); out["extent"] = float(rp.extent)
    dt = ndi.distance_transform_edt(big)
    out["thickness_max_um"] = 2 * float(dt.max()) * px_um; out["thickness_mean_um"] = 2 * float(dt[big].mean()) * px_um
    out["n_holes"] = int(1 - rp.euler_number)
    out["centroid_y"] = float(cy); out["centroid_x"] = float(cx)
    return out
