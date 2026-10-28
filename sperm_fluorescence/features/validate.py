"""Validation step 1 of docs/morphology_framework.md: (a) which of our shape definitions
reproduce the instrument's own Area / Length / Width / Height / Aspect Ratio on the same
IDEAS masks (so the definitions are fixed by data, not by guessing the manual), and (b) how
the predicted-compartment features agree with the reference ones, cell by cell, as Pearson r
and Bland-Altman mean difference with 95 % limits."""
import json

import numpy as np

from ..models import registry
from .engine import COMPARTMENT, colname

CANDIDATES = {"Area": ["area_um2"], "Length": ["feret_max_um", "major_um", "bbox_h_um", "skel.path_um"], "Width": ["feret_min_um", "minor_um", "bbox_w_um", "thickness_max_um"],
              "Height": ["bbox_h_um", "feret_max_um"], "Aspect Ratio": ["aspect_ratio"]}
CHANNEL = {"Intensity_MC": "sum", "Gradient RMS": "gradient_rms", "Contrast": "contrast", "Bright Detail Intensity R3_MC": "bright_detail", "Max Pixel_MC": "max"}
COMPARE = ["area_um2", "feret_max_um", "feret_min_um", "major_um", "minor_um", "aspect_ratio", "ellipticity", "elongation", "rugosity", "regularity", "perimeter_um", "thickness_max_um"]
COMPARE_SKEL = ["path_um", "end_to_end_um", "tortuosity", "turning_total_deg", "curv_max_per_um", "n_sharp_bends"]
COMPARE_X = list(__import__("sperm_fluorescence.features.engine", fromlist=["X_DOC"]).X_DOC)


def _col(rows, c):
    return np.array([float(r.get(c, np.nan)) if r.get(c, "") not in ("", None) else np.nan for r in rows])


def r_and_ba(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10: return {"n": int(ok.sum()), "r": None}
    a, b = a[ok], b[ok]; d = a - b
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {"n": int(ok.sum()), "r": r, "mean_diff": float(d.mean()), "loa_lo": float(d.mean() - 1.96 * d.std()), "loa_hi": float(d.mean() + 1.96 * d.std()),
            "median_ratio": float(np.median(a[b != 0] / b[b != 0])) if (b != 0).any() else None}


def validate(rows, daf, inst, names, px_um, out_path):
    from ..io import isx_daf
    fnames, F, _ = isx_daf.read_features(daf); col = {n: i for i, n in enumerate(fnames)}
    src = np.array([r["id.source_row"] if r["id.source_row"] is not None else -1 for r in rows], int)
    ok_src = src >= 0
    stains = registry.instrument(inst)["stains"] if inst == "isx" else {}
    report = {"definitions": {}, "pred_vs_ref": {}}
    print("\n(a) our definitions against the instrument's table, on the IDEAS masks (r, median ours/theirs):")
    for marker, ch in stains.items():
        comp = COMPARTMENT.get(marker)
        if comp is None or marker not in names: continue
        mnum = "M%02d" % (ch + 1); chn = "Ch%02d" % (ch + 1)
        for feat, cands in CANDIDATES.items():
            key = "%s_%s" % (feat, mnum)
            if key not in col: continue
            theirs = np.where(ok_src, F[np.clip(src, 0, None), col[key]], np.nan)
            best = None
            for c in cands:
                ours = _col(rows, colname("ref", comp, c)); s = r_and_ba(ours, theirs)
                report["definitions"]["%s %s ~ ref.%s.%s" % (feat, mnum, comp, c)] = s
                if s.get("r") is not None and (best is None or s["r"] > best[1]): best = (c, s["r"], s.get("median_ratio"))
            if best: print("  %-12s %-8s best: %-16s r %.3f  ratio %.2f" % (feat, mnum, best[0], best[1], best[2] or float("nan")))
        for feat, c in CHANNEL.items():
            key = "%s_%s_%s" % (feat, mnum, chn)
            if key not in col: continue
            theirs = np.where(ok_src, F[np.clip(src, 0, None), col[key]], np.nan); ours = _col(rows, colname("ref", marker, comp, c)); s = r_and_ba(ours, theirs)
            report["definitions"]["%s ~ ref.%s.%s.%s" % (key, marker, comp, c)] = s
            if s.get("r") is not None: print("  %-30s ~ %-10s r %.3f" % (key, c, s["r"]))
    print("\n(b) predicted compartment features against the reference (IDEAS masks of the measured stain), cell by cell:")
    print("  %-34s %6s %7s %10s %22s" % ("feature", "n", "r", "mean diff", "95% limits"))
    comps = sorted({COMPARTMENT[nm] for nm in names if nm in COMPARTMENT} | {"head", "principal", "tail"})
    for comp in comps:
        for c in COMPARE + ["skel." + s for s in COMPARE_SKEL]:
            pc, rc = colname("pred", comp, c), colname("ref", comp, c)
            a, b = _col(rows, pc), _col(rows, rc); s = r_and_ba(a, b)
            if s.get("r") is None: continue
            report["pred_vs_ref"][colname(comp, c)] = s
            print("  %-34s %6d %7.3f %10.3f %10.3f .. %-10.3f" % (colname(comp, c), s["n"], s["r"], s["mean_diff"], s["loa_lo"], s["loa_hi"]))
    for c in COMPARE_X:
        a, b = _col(rows, colname("pred", "x", c)), _col(rows, colname("ref", "x", c)); s = r_and_ba(a, b)
        if s.get("r") is None: continue
        report["pred_vs_ref"][colname("x", c)] = s
        print("  %-34s %6d %7.3f %10.3f %10.3f .. %-10.3f" % (colname("x", c), s["n"], s["r"], s["mean_diff"], s["loa_lo"], s["loa_hi"]))
    json.dump(report, open(out_path, "w"), indent=1)
    print("wrote", out_path)
