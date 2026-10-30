"""`features`: one row per object of a virtual-stain store, the catalogue of generic features
on the predicted compartments (and, as reference, on the instrument's masks of the measured
stains), written as CSV with a JSON of column definitions beside it.

    python -m sperm_fluorescence features data/isx_vstain --out data/isx_vstain/features.csv \\
        --daf data/isx/ALL_1_400_1.daf

Columns:  id.*   class, index, .cif row, well, DAPI scalar, population labels
          qc.*   focus, object and nucleus counts, tail at the frame edge
          pred.<mask>.<shape>            shape of each predicted compartment mask (features/shape.py)
          pred.<mask>.skel.<f>           skeleton geometry of midpiece, principal piece, tail, cell (features/skeleton.py)
          pred.<marker>.<mask>.<f>       predicted channel intensity and texture inside a mask (features/intensity.py)
          in.<role>.<mask>.<f>           the label-free inputs inside the cell and head masks
          pred.sim.<a>_<b>.r|z           similarity of two predicted channels inside the cell
          pred.x.<f>                     cross-mask geometry: delta centroids, head-midpiece angle, insertion offset, area ratios
          preds.<channel>.<f>            the stain IMAGES, no mask: intensity moments, radial and axial profiles (features/stain.py), above the
                                         channel's noise floor and within the brightfield object dilated by 3 um (the same region for pred and ref)
          preds.d|along.<a>_<b>          distances between channel intensity centroids, and along the cell axis
          ref.*, refs.*                  the same on the IDEAS masks / measured channels (stained cache + .cif), for validation
Also <out>.pixels.npz: every channel (predicted stains + inputs) rotated onto the cell axis, head left,
on a 48 x 48 grid at 0.667 um/px, for pixel-level PCA (`populate --pixels`); and .pixels_ref.npz for the measured stains.
Masks: nucleus (DAPI), acrosome (ACRV1), head = filled nucleus|acrosome, midpiece (TOMM20),
flagellum (LDHC/AKAP4), principal = flagellum minus the dilated midpiece, tail = flagellum|midpiece,
cell = brightfield object (a proxy: |standardised brightfield| > 1.5, closed, filled, largest
component, united with every stain mask; light loss plays brightfield on the A8). Marker names in column names have '/' as '_'.
With --daf the reference features are checked against the instrument's own feature table and
the predicted ones against the reference (<out>.validation.json).
"""
import argparse
import csv
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi

from ..data.cache import open_cache
from ..data.store import open_store
from ..eval.masks import RULE, mask_from_image
from . import intensity as inten
from . import shape as shp
from . import skeleton as skel
from . import stain as stn

COMPARTMENT = {"DAPI": "nucleus", "ACRV1": "acrosome", "TOMM20": "midpiece", "LDHC/AKAP4": "flagellum", "CD45": "cd45"}
SKELETON_MASKS = ("midpiece", "principal", "flagellum", "tail", "cell")
INPUT_MASKS = ("cell", "head")
X_DOC = {
    "d_nucleus_acrosome_um": "distance between nucleus and acrosome centroids, um",
    "d_nucleus_midpiece_um": "distance between nucleus and midpiece centroids, um",
    "d_nucleus_flagellum_um": "distance between nucleus and flagellum centroids, um",
    "d_head_cell_um": "distance between head and brightfield-object centroids, um",
    "acrosome_anterior_um": "projection of (acrosome centroid - nucleus centroid) on the nucleus->midpiece axis, um (negative = ahead of the nucleus)",
    "head_midpiece_angle_deg": "angle between the head's and the midpiece's ellipse orientations, 0-90 degrees",
    "insertion_offset_um": "distance from the head pole nearest the midpiece to the nearest midpiece pixel, um",
    "acrosome_frac": "acrosome area / head area", "nucleus_frac": "nucleus area / head area",
    "midpiece_len_over_head": "midpiece skeleton path / head Feret max", "tail_len_over_head": "tail skeleton path / head Feret max",
    "midpiece_um2_per_um": "midpiece mask area / midpiece skeleton path (thickness proxy, since 0.6 um is under 2 px)",
    "residue_um2": "brightfield-object area outside head and tail masks (cytoplasmic residue proxy)",
}
QC_DOC = {
    "n_nuclei": "components of the nucleus mask", "n_objects": "components of the brightfield-object mask",
    "flagellum_edge_frac": "fraction of flagellum mask pixels within 2 px of the valid frame's border",
    "tail_end_edge_um": "distance from the tail path's edge-nearest end to the border, um",
    "focus_gradient_rms": "the brightfield (or first input) channel's gradient RMS in the cell mask, repeated here as the focus gate",
    "frame_fill": "brightfield-object pixels / valid frame pixels",
}


def colname(*parts):
    return ".".join(str(p).replace("/", "_") for p in parts)


def cell_mask(bf, valid, stains):
    """Brightfield object proxy (see module docstring)."""
    if bf is None:
        u = np.zeros_like(valid)
        for s in stains: u |= s
        return u & valid
    a = ndi.gaussian_filter(np.abs(bf), 1.0) > 1.5
    a = ndi.binary_closing(a, structure=np.ones((3, 3))) & valid
    a = ndi.binary_fill_holes(a)
    lab, n = ndi.label(a)
    if n > 1:
        sizes = ndi.sum(a, lab, range(1, n + 1)); a = lab == (1 + int(np.argmax(sizes)))
    for s in stains: a |= s
    return ndi.binary_fill_holes(a) & valid


def derive_masks(parts, bf, valid):
    """parts: {compartment: mask} for the markers present. Adds head, principal, tail, cell."""
    M = dict(parts)
    if "nucleus" in M or "acrosome" in M:
        h = np.zeros_like(valid)
        for k in ("nucleus", "acrosome"):
            if k in M: h |= M[k]
        h = ndi.binary_fill_holes(h); lab, n = ndi.label(h, structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndi.sum(h, lab, range(1, n + 1)); h = lab == (1 + int(np.argmax(sizes)))
        M["head"] = h
    if "flagellum" in M and "midpiece" in M:
        M["principal"] = M["flagellum"] & ~ndi.binary_dilation(M["midpiece"], iterations=1)
        M["tail"] = M["flagellum"] | M["midpiece"]
    elif "flagellum" in M:
        M["tail"] = M["flagellum"]
    M["cell"] = cell_mask(bf, valid, [m for k, m in parts.items()])
    return M


def cross_features(M, S, K, px_um):
    """Cross-mask geometry from the shape (S) and skeleton (K) dicts already computed."""
    out = {k: np.nan for k in X_DOC}
    def cen(k): return np.array([S[k]["centroid_y"], S[k]["centroid_x"]]) if k in S and S[k]["n_px"] > 0 else None
    def dist(a, b):
        ca, cb = cen(a), cen(b)
        return float(np.hypot(*(ca - cb))) * px_um if ca is not None and cb is not None else np.nan
    out["d_nucleus_acrosome_um"] = dist("nucleus", "acrosome"); out["d_nucleus_midpiece_um"] = dist("nucleus", "midpiece")
    out["d_nucleus_flagellum_um"] = dist("nucleus", "flagellum"); out["d_head_cell_um"] = dist("head", "cell")
    cn, cm, ca = cen("nucleus"), cen("midpiece"), cen("acrosome")
    if cn is not None and cm is not None and ca is not None and np.hypot(*(cm - cn)) > 0:
        axis = (cm - cn) / np.hypot(*(cm - cn)); out["acrosome_anterior_um"] = float((ca - cn) @ axis) * px_um
    if "head" in S and "midpiece" in S and S["head"]["n_px"] > 0 and S["midpiece"]["n_px"] > 0:
        d = abs(S["head"]["orientation_deg"] - S["midpiece"]["orientation_deg"]) % 180
        out["head_midpiece_angle_deg"] = float(min(d, 180 - d))
        ch = cen("head"); th = np.deg2rad(S["head"]["orientation_deg"]); half = S["head"]["major_um"] / px_um / 2
        poles = [ch + half * np.array([np.cos(th), np.sin(th)]), ch - half * np.array([np.cos(th), np.sin(th)])]
        my, mx = np.nonzero(M["midpiece"]); pts = np.stack([my, mx], 1)
        if len(pts):
            dists = [np.hypot(*(pts - p).T).min() for p in poles]; out["insertion_offset_um"] = float(min(dists)) * px_um
    if "head" in S and S["head"]["n_px"] > 0:
        for k, name in (("acrosome", "acrosome_frac"), ("nucleus", "nucleus_frac")):
            if k in S: out[name] = S[k]["n_px"] / float(S["head"]["n_px"])
        L = S["head"]["feret_max_um"]
        if L and np.isfinite(L) and L > 0:
            if "midpiece" in K and np.isfinite(K["midpiece"].get("path_um", np.nan)): out["midpiece_len_over_head"] = K["midpiece"]["path_um"] / L
            if "tail" in K and np.isfinite(K["tail"].get("path_um", np.nan)): out["tail_len_over_head"] = K["tail"]["path_um"] / L
    if "midpiece" in K and np.isfinite(K["midpiece"].get("path_um", np.nan)) and K["midpiece"]["path_um"] > 0:
        out["midpiece_um2_per_um"] = S["midpiece"]["area_um2"] / K["midpiece"]["path_um"]
    if "cell" in M:
        covered = np.zeros_like(M["cell"])
        for k in ("head", "tail"):
            if k in M: covered |= M[k]
        out["residue_um2"] = float((M["cell"] & ~covered).sum()) * px_um ** 2
    return out


REGION_UM = 3.0        # stain-image moments are taken within the brightfield object dilated by this much, the same region on both sides


def compartment_features(prefix, parts, chans, inputs, valid, px_um, docs, region=None):
    """All features for one object under one prefix ('pred' or 'ref'). parts: {compartment: mask};
    chans: {marker: image}; inputs: {role: image} or {}. Fills `docs` with column definitions.
    Returns (row, masks, shape dicts, skeleton dicts, (cell axis, centre) or None, region).
    `region`: where the stain-image moments are taken; default the brightfield object dilated
    by REGION_UM, and the pred side's region is passed to the ref side so both measure the
    same pixels."""
    row = {}
    M = derive_masks(parts, inputs.get("brightfield", inputs.get("lightloss")), valid)      # light loss is the A8's brightfield-like image
    S, K = {}, {}
    for k, m in M.items():
        S[k] = shp.shape_features(m & valid, px_um)
        for f, v in S[k].items(): row[colname(prefix, k, f)] = v; docs[colname(prefix, k, f)] = "%s mask: %s" % (k, shp.DOC[f])
        if k in SKELETON_MASKS:
            K[k] = skel.skeleton_features(m & valid, px_um, valid.shape)
            for f, v in K[k].items(): row[colname(prefix, k, "skel", f)] = v; docs[colname(prefix, k, "skel", f)] = "%s skeleton: %s" % (k, skel.DOC[f])
    for marker, img in chans.items():
        comp = COMPARTMENT.get(marker)
        for k in ([comp] if comp in M else []) + ["cell"]:
            if k not in M: continue
            for f, v in inten.intensity_features(img, M[k], valid).items():
                row[colname(prefix, marker, k, f)] = v; docs[colname(prefix, marker, k, f)] = "%s channel in the %s mask: %s" % (marker, k, inten.DOC[f])
    for role, img in inputs.items():
        for k in INPUT_MASKS:
            if k not in M: continue
            for f, v in inten.intensity_features(img, M[k], valid).items():
                row[colname("in", role, k, f)] = v; docs[colname("in", role, k, f)] = "%s input in the %s mask: %s" % (role, k, inten.DOC[f])
    names = list(chans)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = inten.similarity(chans[names[i]], chans[names[j]], M["cell"] & valid)
            for f, v in s.items(): row[colname(prefix, "sim", names[i] + "_" + names[j], f)] = v; docs[colname(prefix, "sim", names[i] + "_" + names[j], f)] = "Pearson %s of %s and %s inside the cell mask" % (f, names[i], names[j])
    for f, v in cross_features(M, S, K, px_um).items(): row[colname(prefix, "x", f)] = v; docs[colname(prefix, "x", f)] = X_DOC[f]
    # stain-image features, no thresholds: moments, profiles, cross-channel centroids, on the channels as images
    names_ = list(chans); imgs = np.stack([chans[n] for n in names_]) if names_ else None
    if region is None:
        region = ndi.binary_dilation(M["cell"], iterations=max(int(round(REGION_UM / px_um)), 1)) & valid
    if imgs is not None:
        axis, centre, info = stn.cell_axis(imgs, names_, region, px_um)
        sp = prefix + "s"                                              # 'preds' / 'refs': the stain-image family
        for f, v in info.items(): row[colname(sp, "cell", f)] = v; docs[colname(sp, "cell", f)] = stn.X_DOC[f]
        cents = {}
        for k, nm in enumerate(names_):
            feats = stn.channel_features(imgs[k], region, px_um, axis, centre)
            for f, v in feats.items(): row[colname(sp, nm, f)] = v; docs[colname(sp, nm, f)] = "%s image: %s" % (nm, stn.DOC[f])
            cents[nm] = (feats["cy"], feats["cx"])
        for i in range(len(names_)):
            for j in range(i + 1, len(names_)):
                a, b = cents[names_[i]], cents[names_[j]]
                d = float(np.hypot(a[0] - b[0], a[1] - b[1])) * px_um if np.isfinite(a[0]) and np.isfinite(b[0]) else np.nan
                along = float(((b[0] - a[0]) * axis[0] + (b[1] - a[1]) * axis[1]) * px_um) if np.isfinite(d) else np.nan
                row[colname(sp, "d", names_[i] + "_" + names_[j])] = d; docs[colname(sp, "d", names_[i] + "_" + names_[j])] = "distance between the intensity centroids of %s and %s, um" % (names_[i], names_[j])
                row[colname(sp, "along", names_[i] + "_" + names_[j])] = along; docs[colname(sp, "along", names_[i] + "_" + names_[j])] = "centroid of %s minus %s along the cell axis, um (positive = towards the tail)" % (names_[j], names_[i])
        axis_out = (axis, centre)
    else:
        axis_out = None
    return row, M, S, K, axis_out, region


def qc_features(M, K, row, valid, prefix="pred"):
    out = {k: np.nan for k in QC_DOC}
    out["n_nuclei"] = row.get(colname(prefix, "nucleus", "n_components"), 0)       # no nucleus channel (A8): zero nuclei, which passes the single-object gate
    out["n_objects"] = row.get(colname(prefix, "cell", "n_components"), np.nan)
    if "flagellum" in M and M["flagellum"].any():
        border = valid & ~ndi.binary_erosion(valid, iterations=2, border_value=0)
        out["flagellum_edge_frac"] = float((M["flagellum"] & border).sum() / M["flagellum"].sum())
    if "tail" in K and np.isfinite(K["tail"].get("end_y", np.nan)):
        ys, xs = np.nonzero(valid); ey, ex = K["tail"]["end_y"], K["tail"]["end_x"]
        out["tail_end_edge_um"] = float(min(ey - ys.min(), ys.max() - ey, ex - xs.min(), xs.max() - ex))
    focus_cols = [c for c in row if c.startswith("in.") and c.endswith(".cell.gradient_rms")]
    bf = colname("in", "brightfield", "cell", "gradient_rms")
    out["focus_gradient_rms"] = row.get(bf, row[focus_cols[0]] if focus_cols else np.nan)      # brightfield when the instrument has one, else the first input (A8: light loss)
    out["frame_fill"] = float(M["cell"].sum() / max(valid.sum(), 1))
    return out


def populations(values):
    """sperm / round / dim by the rule in eval.accuracy.groups_for, on any per-object DAPI quantity."""
    d = np.asarray(values, np.float64); g = np.array(["none"] * len(d), dtype=object); ok = np.isfinite(d)
    if ok.sum() < 10: return g
    med = np.nanmedian(d[ok & (d > np.nanpercentile(d[ok], 45))])
    g[ok & (d < 0.25 * med)] = "dim"; g[ok & (d >= 0.25 * med) & (d <= 4 * med)] = "sperm"; g[ok & (d > 4 * med)] = "round"
    return g


def denoise_measured(img, valid, how):
    """A measured stain channel with its read noise reduced: 'floor' removes the frame's noise
    floor (stain.denoise), 'smooth' first blurs with a 1-px Gaussian so single-pixel noise does
    not seed the threshold masks, then removes the floor. Pixels outside the valid frame stay 0."""
    img = np.asarray(img, dtype=np.float32)
    if how == "smooth":
        from scipy.ndimage import gaussian_filter
        v = (valid > 0).astype(np.float32)
        img = gaussian_filter(img * v, 1.0) / np.maximum(gaussian_filter(v, 1.0), 1e-3) * v     # normalised so the frame edge does not darken
    return stn.denoise(img, valid)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sperm_fluorescence features", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store"); ap.add_argument("--cache", default=None, help="default: the store's source cache")
    ap.add_argument("--classes", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="auto", choices=["auto", "ideas", "truth", "rule", "none"], help="reference masks: the store's IDEAS masks, the thresholded measured stain (masks.py rule), the SAME threshold rule the store used for the predictions applied to the measured stain (like with like), or none (auto: ideas if stored, else truth if stained, else none)")
    ap.add_argument("--daf", default=None, help="ImageStream .daf: validate reference features against the instrument's table and predicted against reference")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-ref", action="store_true", help="alias for --reference none")
    ap.add_argument("--no-pixels", action="store_true", help="skip the canonical aligned pixel stacks (<out>.pixels.npz)")
    ap.add_argument("--denoise-ref", default="none", choices=["none", "floor", "smooth"], help="denoise the MEASURED stain images before the reference masks and features: floor = subtract the frame's noise floor (median + 3 MAD, stain.denoise); smooth = Gaussian sigma 1 px then the floor. The predicted side is untouched, so the measured truth gets a fairer footing against the smooth predictions")
    ap.add_argument("--pixels-only", action="store_true", help="write only the aligned pixel stacks (cell axis from the virtual stains; no feature catalogue, no CSV): fast, for autoencoder training data")
    a = ap.parse_args(argv)
    st = open_store(a.store, [c.strip() for c in a.classes.split(",")] if a.classes else None)
    cache = a.cache or st.meta["source_cache"]
    ds, wells, scal = open_cache(cache, st.classes)
    cmeta = json.load(open(os.path.join(cache, "meta.json")))
    inst = cmeta["classes"][st.classes[0]]["instrument"]; px_um = float(cmeta["instruments"][inst].get("pixel_um") or 0.333)
    names = st.names; roles = list(ds.roles)
    stained = ds.avail is not None and bool(ds.avail.any())
    ref = "none" if a.no_ref else a.reference
    if ref == "auto": ref = "ideas" if any(x is not None for x in st.ims) else ("truth" if stained else "none")
    if ref == "ideas" and not any(x is not None for x in st.ims): raise SystemExit("the store has no IDEAS masks (predict --isx)")
    n = len(st) if not a.limit else min(a.limit, len(st))
    rows, docs = [], {}; pixels, pixels_ref = [], []
    t0 = time.time()
    if a.pixels_only:
        stem = os.path.splitext(a.out)[0]; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        for i in range(n):
            p, pm, m, im = st.item(i); x, y, mc = ds.item(i); valid = m > 0
            inputs = {r: x[j] for j, r in enumerate(roles)}
            parts = {COMPARTMENT[nm]: pm[k] for k, nm in enumerate(names) if nm in COMPARTMENT}
            M = derive_masks(parts, inputs.get("brightfield", inputs.get("lightloss")), valid)
            region = ndi.binary_dilation(M["cell"], iterations=max(int(round(REGION_UM / px_um)), 1)) & valid
            axis, centre, _ = stn.cell_axis(p, list(names), region, px_um)
            imgs = np.concatenate([p, x]); fill = [0.0] * len(names) + [float(np.median(np.concatenate([x[j][0], x[j][-1], x[j][:, 0], x[j][:, -1]]))) for j in range(len(roles))]
            pixels.append(stn.canonical_stack(imgs, valid, px_um, axis, centre, fill).astype(np.float16))
            if (i + 1) % 2000 == 0: print("  %d / %d  (%.0fs)" % (i + 1, n, time.time() - t0))
        np.savez(stem + ".pixels.npz", pixels=np.stack(pixels), index=np.arange(len(pixels)), channels=np.array(list(names) + list(roles)), px_um=stn.CANON_UM, side=stn.CANON_PX)
        print("wrote %s.pixels.npz: %d objects x %s (aligned, %.3g um/px; pixels only, %.0fs)" % (stem, len(pixels), "x".join(str(v) for v in pixels[0].shape), stn.CANON_UM, time.time() - t0))
        return 0
    for i in range(n):
        p, pm, m, im = st.item(i); x, y, mc = ds.item(i); valid = m > 0
        av = ds.avail[i] if ds.avail is not None else np.zeros(ds.cout, bool)
        inputs = {r: x[j] for j, r in enumerate(roles)}
        parts = {COMPARTMENT[nm]: pm[k] for k, nm in enumerate(names) if nm in COMPARTMENT}
        chans = {nm: p[k] for k, nm in enumerate(names)}
        row = {"id.class": str(st.class_of[i]), "id.index": i, "id.source_row": st.source_row(i), "id.well": str(wells[i]),
               "id.scalar": float(np.asarray(scal).reshape(len(scal), -1)[i, 0])}
        for k, nm in enumerate(names): row[colname("id", "avail", nm)] = int(av[k])
        r, M, S, K, axis_pred, region = compartment_features("pred", parts, chans, inputs, valid, px_um, docs); row.update(r)
        row.update({colname("qc", f): v for f, v in qc_features(M, K, row, valid).items()})
        if axis_pred is not None and not a.no_pixels:
            imgs = np.stack([p[k] for k in range(len(names))] + [x[j] for j in range(len(roles))])
            fill = [0.0] * len(names) + [float(np.median(np.concatenate([x[j][0], x[j][-1], x[j][:, 0], x[j][:, -1]]))) for j in range(len(roles))]
            pixels.append(stn.canonical_stack(imgs, valid, px_um, axis_pred[0], axis_pred[1], fill).astype(np.float16))
        if ref != "none" and av.any():
            if a.denoise_ref != "none":
                y = np.stack([denoise_measured(y[k], valid, a.denoise_ref) for k in range(len(y))])
            if ref == "ideas":
                rparts = {COMPARTMENT[nm]: im[k] & valid for k, nm in enumerate(names) if nm in COMPARTMENT and av[k]}
            elif ref == "rule":
                from ..eval.masks import pred_mask
                thr = st.meta["thresholds"]
                rparts = {COMPARTMENT[nm]: pred_mask(y[k], tuple(thr["%s:%s" % (inst, nm)]), valid.astype(np.float32)) for k, nm in enumerate(names) if nm in COMPARTMENT and av[k]}
            else:
                rparts = {COMPARTMENT[nm]: mask_from_image(y[k], valid, **RULE[inst]) for k, nm in enumerate(names) if nm in COMPARTMENT and av[k]}
            rchans = {nm: y[k] for k, nm in enumerate(names) if av[k]}
            r, _, _, _, axis_ref, _ = compartment_features("ref", rparts, rchans, {}, valid, px_um, docs, region=region); row.update(r)
            if axis_ref is not None and not a.no_pixels:
                imgs = np.stack([y[k] for k in range(len(names))]); fill = [0.0] * len(names)
                pixels_ref.append((i, stn.canonical_stack(imgs, valid, px_um, axis_ref[0], axis_ref[1], fill).astype(np.float16)))
        rows.append(row)
        if (i + 1) % 500 == 0: print("  %d / %d  (%.0fs)" % (i + 1, n, time.time() - t0))
    for f, d in QC_DOC.items(): docs[colname("qc", f)] = d
    # population labels, from the cache's DAPI scalar and from the predicted nucleus intensity
    g_scalar = populations([r["id.scalar"] for r in rows])
    key = colname("pred", "DAPI", "nucleus", "sum")
    g_pred = populations([r.get(key, np.nan) for r in rows]) if key in docs else np.array(["none"] * len(rows), dtype=object)
    for r, gs, gp in zip(rows, g_scalar, g_pred): r["id.pop_scalar"] = str(gs); r["id.pop_pred"] = str(gp)
    docs.update({"id.class": "cache class", "id.index": "index in the store (and cache) over the listed classes", "id.source_row": "object index in the source .cif, when known",
                 "id.well": "well / tube letter", "id.scalar": "per-event scalar from the cache (ISX: IDEAS DAPI intensity); NaN when absent",
                 "id.pop_scalar": "sperm / round / dim by the DAPI scalar (eval.accuracy.groups_for rule)", "id.pop_pred": "the same rule on the predicted DAPI sum in the nucleus mask"})
    for nm in names: docs[colname("id", "avail", nm)] = "1 if the object's instrument measured this marker"
    cols = [c for c in rows[0] if c.startswith("id.")] + [c for c in rows[0] if c.startswith("qc.")] + [c for c in rows[0] if c.startswith(("pred.", "in."))] + [c for c in rows[0] if c.startswith("ref.")]
    for r in rows:
        for c in r:
            if c not in cols: cols.append(c)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({c: ("" if (isinstance(v, float) and not np.isfinite(v)) else v) for c, v in r.items()})
    side = {"store": st.meta, "cache": os.path.abspath(cache), "classes": st.classes, "pixel_um": px_um, "reference": ref, "denoise_ref": a.denoise_ref, "n": len(rows),
            "columns": {c: docs.get(c, "") for c in cols}, "created": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(side, open(os.path.splitext(a.out)[0] + ".columns.json", "w"), indent=1)
    if pixels:
        stem = os.path.splitext(a.out)[0]
        np.savez(stem + ".pixels.npz", pixels=np.stack(pixels), index=np.arange(len(pixels)), channels=np.array(list(names) + list(roles)), px_um=stn.CANON_UM, side=stn.CANON_PX)
        if pixels_ref:
            np.savez(stem + ".pixels_ref.npz", pixels=np.stack([p for _, p in pixels_ref]), index=np.array([i for i, _ in pixels_ref]), channels=np.array(list(names)), px_um=stn.CANON_UM, side=stn.CANON_PX)
        print("wrote %s.pixels.npz: %d objects x %s (aligned, %.3g um/px)" % (stem, len(pixels), "x".join(str(v) for v in pixels[0].shape), stn.CANON_UM))
    print("wrote %s: %d objects x %d columns (%.0fs); reference = %s" % (a.out, len(rows), len(cols), time.time() - t0, ref))
    if a.daf:
        from .validate import validate
        validate(rows, a.daf, inst, names, px_um, os.path.splitext(a.out)[0] + ".validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
