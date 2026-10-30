"""One table of a run's held-out accuracy, raw and denoised measured stains side by side: event r and
pixel r (from results.json for the raw full evaluation, or an accuracy summary.json), within-well
event r for the sperm well, and mask IoU (masks.json / masks_denoised_<mode>.json). Markdown out.

    python analysis/metrics_table.py runs/srv_a8_stems --accuracy figures/srv_a8_accuracy_denoised/summary.json --denoise smooth
    python analysis/metrics_table.py runs/srv_a8_stems --accuracy-raw figures/acc_none/summary.json --accuracy figures/acc_smooth/summary.json --denoise smooth --label "laptop sample"
"""
import argparse
import json
import os


def metrics_from(path):
    """{marker: {event_r, pixel_r, within}} from a results.json or an accuracy summary.json."""
    j = json.load(open(path)); res = j.get("metrics", j.get("results", {}))
    return {m: {"event_r": d.get("event_r"), "pixel_r": d.get("pixel_r"), "within": d.get("within_well_event_r") or {}} for m, d in res.items() if isinstance(d, dict) and "event_r" in d}, j.get("n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--denoise", default="smooth")
    ap.add_argument("--accuracy-raw", default=None, help="accuracy summary.json scored on raw stains (default: the run's results.json, the full evaluation)")
    ap.add_argument("--accuracy", default=None, help="accuracy summary.json scored with --denoise-measured")
    ap.add_argument("--label", default=None); ap.add_argument("--sperm-well", default="S")
    ap.add_argument("--denoised-only", action="store_true", help="one column set, scored against the denoised measured stains only (no raw columns)")
    a = ap.parse_args(argv)
    raw, n_raw = metrics_from(a.accuracy_raw or os.path.join(a.run, "results.json"))
    dn, n_dn = metrics_from(a.accuracy) if a.accuracy else ({}, None)
    def masks(path):
        if not os.path.exists(path): return {}, None
        j = json.load(open(path)); h = j["held_out"]
        return {k.split(":", 1)[1]: v for k, v in h.items()}, next(iter(h.values()))["n"]
    mr, nmr = masks(os.path.join(a.run, "masks.json")); md, nmd = masks(os.path.join(a.run, "masks_denoised_%s.json" % a.denoise))
    f = lambda d, k: ("%.2f" % d[k]) if d and d.get(k) is not None else "–"
    fw = lambda d: ("%.2f" % d["within"][a.sperm_well]) if d and d.get("within", {}).get(a.sperm_well) is not None else "–"
    fi = lambda d: ("%.2f" % d["iou"]) if d and d.get("iou") is not None else "–"
    if a.denoised_only:
        wells = sorted({w for m in dn for w in dn[m]["within"]}, key=lambda w: ["S", "P", "SP", "I", "sperm", "round", "dim"].index(w) if w in ["S", "P", "SP", "I", "sperm", "round", "dim"] else 99)
        wname = {"S": "sperm well", "P": "PBMC well", "SP": "mixed well", "I": "tube"}
        lines = ["| marker | event r | pixel r | " + " | ".join("event r, %s" % wname.get(w, w) for w in wells) + " | mask IoU |", "|---|" + "---:|" * (3 + len(wells))]
        for m in dn:
            cells = [f(dn[m], "event_r"), f(dn[m], "pixel_r")] + [("%.2f" % dn[m]["within"][w]) if dn[m]["within"].get(w) is not None else "–" for w in wells] + [fi(md.get(m))]
            lines.append("| %s | %s |" % (m, " | ".join(cells)))
        print("\n".join(lines)); print()
        print("%s: predictions scored against the measured stains denoised (%s); correlations on %s held-out cells, IoU on %s." % (a.label or os.path.basename(a.run.rstrip("/")), a.denoise, n_dn or "–", nmd or "–"))
        return 0
    head = "| marker | event r raw | event r denoised | pixel r raw | pixel r denoised | sperm-well event r raw | denoised | IoU raw | IoU denoised |"
    lines = [head, "|---|" + "---:|" * 8]
    for m in raw:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (m, f(raw[m], "event_r"), f(dn.get(m), "event_r"), f(raw[m], "pixel_r"), f(dn.get(m), "pixel_r"), fw(raw[m]), fw(dn.get(m)), fi(mr.get(m)), fi(md.get(m))))
    note = "%s: correlations on %s held-out cells (raw) / %s (denoised, %s); IoU on %s (raw) / %s (denoised) cells." % (
        a.label or os.path.basename(a.run.rstrip("/")), n_raw or "all", n_dn or "–", a.denoise, nmr or "–", nmd or "–")
    print("\n".join(lines)); print(); print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
