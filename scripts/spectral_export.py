#!/usr/bin/env python
"""Export the raw spectral detector signal per event from A8 (or any) FCS files, as JSON,
small enough to copy to the laptop and plot (analysis/spectral_plot.py).

For every FCS: the names of the detector area columns ($PnN matching --pattern), per-
detector percentiles (p10, p50, p90) over all events, and a seeded random subsample of
events with those columns plus any --keep columns (unmixed scalars for gating later).

Standard library only.

    python scripts\\spectral_export.py "Z:\\...\\260709_Blair_Sperm_A8\\260709" --out spectral.json
    python scripts\\spectral_export.py sample_3S.acs --pattern "^(BL|VL|YL|RL)\\d+-A$" --out test.json
"""
import argparse, glob, json, os, random, re, sys, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcs_data  # noqa: E402
import fcs_probe  # noqa: E402

A8_PATTERN = r"^(ImgB|UV|V|B|YG|R)\d+-A$"
KEEP = ["CD45-A", "ACRV-1-A", "LDHC_AKAP4-A", "DAPI-A", "ImageFlag", "FSC-A", "SSC (Imaging)-A"]


def fcs_bytes(path):
    if path.lower().endswith(".fcs"):
        return open(path, "rb").read()
    with zipfile.ZipFile(path) as z:
        members = [i for i in z.infolist() if i.filename.lower().endswith(".fcs")]
        if len(members) != 1:
            raise IOError("%s: expected one .fcs member, found %d" % (path, len(members)))
        return z.read(members[0])


def percentile(sorted_vals, q):
    if not sorted_vals: return None
    k = (len(sorted_vals) - 1) * q; lo = int(k); hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def export_one(path, pattern, keep, nsample, seed):
    data = fcs_bytes(path)
    keywords = fcs_probe.read_text_segment(data, max_keywords=5000)
    params = fcs_probe.parameters(keywords)
    values, par, n = fcs_data.read_matrix(data, keywords)
    rx = re.compile(pattern)
    det = [(p["n"] - 1, p["name"]) for p in params if rx.match(p["name"].strip())]
    kept = [(p["n"] - 1, p["name"]) for p in params if p["name"].strip() in keep]
    cols = det + kept
    out = {"file": os.path.basename(path), "n_events": n, "detectors": [nm for _, nm in det], "keep": [nm for _, nm in kept],
           "voltage": {p["name"]: p["voltage"] for p in params if rx.match(p["name"].strip())}, "percentiles": {}, "sample": []}
    for j, nm in det:
        col = sorted(values[i * par + j] for i in range(n))
        out["percentiles"][nm] = {"p10": percentile(col, 0.10), "p50": percentile(col, 0.50), "p90": percentile(col, 0.90), "p99": percentile(col, 0.99)}
    rng = random.Random(seed)
    rows = sorted(rng.sample(range(n), min(nsample, n)))
    out["sample_columns"] = [nm for _, nm in cols]
    out["sample"] = [[float(values[i * par + j]) for j, _ in cols] for i in rows]
    out["sample_rows"] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="+", help=".fcs or .acs files, or directories holding them")
    ap.add_argument("--pattern", default=A8_PATTERN, help="regex on $PnN selecting the detector columns")
    ap.add_argument("--keep", nargs="*", default=KEEP, help="extra columns to carry in the subsample")
    ap.add_argument("--sample", type=int, default=400, help="events per file in the subsample")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = []
    for s in a.source:
        if os.path.isdir(s):
            for pat in ("*.fcs", "*.acs"): paths += sorted(glob.glob(os.path.join(s, "**", pat), recursive=True))
        else: paths.append(s)
    if not paths: ap.error("no .fcs/.acs found")
    result = {}
    for p in paths:
        try:
            r = export_one(p, a.pattern, a.keep, a.sample, a.seed)
        except Exception as e:  # noqa: BLE001
            print("SKIP %s: %s" % (p, e)); continue
        result[r["file"]] = r
        print("%-60s events %7d   detectors %3d   kept %s" % (r["file"], r["n_events"], len(r["detectors"]), r["keep"]))
    json.dump(result, open(a.out, "w"))
    print("wrote %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
