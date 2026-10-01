#!/usr/bin/env python
"""Cut a matched image + target subset from a BD FACSDiscover A8 export.

The A8 run folder holds, per sample, a bare `<pos>-<code>.fcs` at the root and an
`<pos>-<code>_images/<bucket>/<code>_<8 digits>.tiff` tree: one six-page float32 TIFF
per imaged event -- three label-free channels and three fluorescence channels,
pixel-registered. There is no event-id column in the FCS, so the join has to be the
FCS **row index** = the 8-digit filename number.

That is an inference, so this script tests it rather than trusting it. For the events
it draws, it decodes every page, sums it, and correlates each page-sum against each
`Total Intensity (...)` column in the FCS at row offsets -1, 0 and +1. If the join is
right, offset 0 wins by a wide margin and the page-to-channel mapping falls out of the
same matrix -- which also pins down page order without relying on the page names. If
offset 0 does not win, the build stops and says so.

Output layout:

    a8_bundle_seed0.zip
      MANIFEST.json              seed, drawn rows, columns, panel, page names, join report
      3S/targets.csv             one row per drawn event: markers + CellView image features
      3S/images/<row>.tiff       the six-page TIFF, copied byte-for-byte
      3P/ ...

Standard library only.

Examples
--------
  py -3 scripts\\make_a8_bundle.py <run-folder> --per-class 200 --out a8_bundle_seed0.zip
  py -3 scripts\\make_a8_bundle.py <run-folder> --per-class 0            # panel + page names only
  py -3 scripts\\make_a8_bundle.py --verify a8_bundle_seed0.zip
"""

import argparse
import array
import glob
import json
import math
import os
import random
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcs_data  # noqa: E402
import fcs_probe  # noqa: E402
import tiff_probe  # noqa: E402
from cytpix import human_bytes, run  # noqa: E402

CLASS_RE = re.compile(r"-(?P<cls>[123](?:S|P|SP))$", re.IGNORECASE)
INDEX_RE = re.compile(r"_(?P<idx>\d{6,10})\.tiff?$", re.IGNORECASE)
# Raw spectral detectors: 81 of them, -A and -H. Dropped from the target table by default.
SPECTRAL_RE = re.compile(r"^(?:UV|V|B|YG|R|ImgB)\d+ \(\d+\)-[AH]$")
TEXT_HEAD_BYTES = 1024 * 1024


# ----------------------------------------------------------------------------- FCS ----

def load_fcs(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    keywords = fcs_probe.read_text_segment(raw[:TEXT_HEAD_BYTES])
    params = fcs_probe.parameters(keywords)
    return raw, keywords, params


def labelled(params):
    out = []
    for r in params:
        label = r["label"].strip()
        if label and label.lower() != r["name"].strip().lower():
            out.append(r)
    return out


def choose_columns(params, all_columns):
    keep = []
    for r in params:
        if all_columns or not SPECTRAL_RE.match(r["name"].strip()):
            keep.append(r)
    return keep


# --------------------------------------------------------------------------- images ----

def find_images(run_folder, stem):
    """{row_index: path} for one sample's image tree."""
    root = os.path.join(run_folder, stem + "_images")
    out = {}
    for path in glob.iglob(os.path.join(root, "**", "*.tif*"), recursive=True):
        m = INDEX_RE.search(os.path.basename(path))
        if m:
            out[int(m.group("idx"))] = path
    return root, out


def read_pages(data):
    """Decode every page of an uncompressed float32 TIFF. Returns [(name, values)]."""
    info = tiff_probe.probe(data)
    little = info["byte_order"] == "little"
    pages = []
    for pg in info["pages"]:
        if pg.get("compression") not in ("none", 1, None):
            raise ValueError("compressed page (%s); this reader handles uncompressed float32 only" % pg.get("compression"))
        if tiff_probe.dtype_of(pg) != "float32":
            raise ValueError("page is %s, expected float32" % tiff_probe.dtype_of(pg))
        strips = pg["_strips"]
        buf = b"".join(data[o : o + c] for o, c in zip(strips["offsets"], strips["byte_counts"]))
        vals = array.array("f")
        vals.frombytes(buf[: len(buf) - len(buf) % 4])
        if little != (sys.byteorder == "little"):
            vals.byteswap()
        name = pg.get("pagename") or ""
        desc = pg.get("imagedescription") or ""
        if not name and desc and not desc.lstrip().startswith("{"):
            name = desc
        pages.append((name.strip(), vals, pg["width"], pg["height"]))
    return pages


def page_names(data):
    return [n for n, _, _, _ in read_pages(data)]


# ----------------------------------------------------------------------- join check ----

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def check_join(values, par, n_rows, params, images, rows, limit):
    """Correlate page sums against the FCS Total Intensity columns at row offsets -1/0/+1."""
    ti_cols = [(r["n"] - 1, r["name"]) for r in params if r["name"].startswith("Total Intensity (")]
    if not ti_cols:
        return {"skipped": "no 'Total Intensity (...)' columns in this FCS"}

    sample = rows[:limit]
    if not sample:
        return {"skipped": "no events to check"}
    sums = []  # per event: [page sums]
    names = None
    used = []
    for row in sample:
        with open(images[row], "rb") as fh:
            pages = read_pages(fh.read())
        if names is None:
            names = [p[0] or "page%d" % i for i, p in enumerate(pages)]
        sums.append([sum(p[1]) for p in pages])
        used.append(row)

    report = {"events_checked": len(used), "page_names": names, "by_offset": {}}
    for offset in (-1, 0, 1):
        matrix = []
        for pi in range(len(names)):
            xs = [s[pi] for s in sums]
            best = (float("-inf"), None)
            for ci, cname in ti_cols:
                ys = []
                for row in used:
                    r = row + offset
                    ys.append(values[r * par + ci] if 0 <= r < n_rows else float("nan"))
                pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
                if len(pairs) < 3:
                    continue
                r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
                if not math.isnan(r) and r > best[0]:
                    best = (r, cname)
            matrix.append({"page": names[pi], "best_column": best[1], "r": best[0] if best[1] else None})
        rs = [m["r"] for m in matrix if m["r"] is not None]
        report["by_offset"][str(offset)] = {"mean_r": sum(rs) / len(rs) if rs else None, "pages": matrix}

    scores = {k: v["mean_r"] for k, v in report["by_offset"].items() if v["mean_r"] is not None}
    best_offset = max(scores, key=scores.get) if scores else None
    report["best_offset"] = int(best_offset) if best_offset is not None else None
    report["verdict"] = (
        "JOIN CONFIRMED: filename index == FCS row index"
        if best_offset == "0" and scores["0"] >= 0.5
        else "JOIN NOT CONFIRMED (best offset %s, mean r %s)" % (best_offset, scores.get(best_offset))
    )
    return report


def print_join(rep):
    if "skipped" in rep:
        print("     join check skipped: %s" % rep["skipped"])
        return
    print("     join check on %d events -- page sum vs FCS 'Total Intensity' column, by row offset:" % rep["events_checked"])
    for off, block in rep["by_offset"].items():
        mr = block["mean_r"]
        print("       offset %2s: mean r = %s" % (off, "%.3f" % mr if mr is not None else "n/a"))
    print("     page -> channel, at the best offset (%s):" % rep["best_offset"])
    best = rep["by_offset"].get(str(rep["best_offset"]), {})
    for m in best.get("pages", []):
        print("       %-32s -> %-45s r=%s" % (m["page"][:32], (m["best_column"] or "?")[:45],
                                              "%.3f" % m["r"] if m["r"] is not None else "n/a"))
    print("     (LightLoss is extinction: its raw page sum is mostly background, so it is not expected to\n"
          "      match its own Total Intensity column; a fluorescence page matches another channel's column\n"
          "      where that marker is absent from the well, which is spillover, not a join error)")
    print("     => %s" % rep["verdict"])


# --------------------------------------------------------------------------- build ----

def fmt(v):
    if isinstance(v, float):
        if v != v:
            return ""
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(round(v, 6))
    return str(v)


def build(run_folder, args):
    fcs_paths = sorted(glob.glob(os.path.join(run_folder, "*.fcs")))
    samples = []
    for p in fcs_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = CLASS_RE.search(stem)
        if m and (not args.classes or m.group("cls").upper() in {c.upper() for c in args.classes}):
            samples.append((m.group("cls").upper(), stem, p))
    if not samples:
        raise IOError("no <pos>-<code>.fcs files found in %s" % run_folder)

    manifest = {"seed": args.seed, "per_class_requested": args.per_class, "classes": {}}
    out = None if args.dry_run else zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED)
    failed = False
    try:
        for cls, stem, fcs_path in samples:
            print("%-4s %s" % (cls, stem))
            raw, keywords, params = load_fcs(fcs_path)
            print("     %s" % fcs_probe.summarise(keywords))
            panel = labelled(params)
            print("     panel: " + ", ".join("%s=%s" % (r["name"], r["label"]) for r in panel))

            img_root, images = find_images(run_folder, stem)
            total = int(keywords.get("$TOT", "0") or 0)
            over = [i for i in images if i >= total]
            print("     images: %d under %s%s" % (len(images), os.path.basename(img_root),
                  "   (!) %d indices >= $TOT=%d" % (len(over), total) if over else ""))

            entry = {"source_fcs": os.path.basename(fcs_path), "events_in_fcs": total,
                     "images": len(images), "panel": [{"name": r["name"], "label": r["label"]} for r in panel]}

            first = next(iter(sorted(images.values())), None)
            if first:
                with open(first, "rb") as fh:
                    entry["page_names"] = page_names(fh.read())
                print("     pages: " + " | ".join(entry["page_names"]))

            if args.per_class <= 0:
                manifest["classes"][cls] = entry
                continue

            values, par, n_rows = fcs_data.read_matrix(raw, keywords)
            wp = next((r["n"] - 1 for r in params if r["name"].strip() == "WaveformPresent"), None)
            if wp is not None:
                flagged = sum(1 for i in range(n_rows) if values[i * par + wp])
                print("     WaveformPresent == 1 on %d rows (images: %d)%s"
                      % (flagged, len(images), "" if flagged == len(images) else "   (!) differs"))
                entry["waveform_present"] = flagged

            eligible = sorted(i for i in images if i < n_rows)
            if not eligible:
                print("     skipped: no images found%s"
                      % (" -- %s exists but is empty; is the zip still extracting?" % os.path.basename(img_root)
                         if os.path.isdir(img_root) else " -- no %s folder" % os.path.basename(img_root)))
                entry["skipped"] = "no images"
                manifest["classes"][cls] = entry
                continue
            rng = random.Random("%s|%s" % (args.seed, cls))
            drawn = sorted(rng.sample(eligible, args.per_class)) if args.per_class < len(eligible) else eligible
            print("     drew %d of %d eligible" % (len(drawn), len(eligible)))

            rep = check_join(values, par, n_rows, params, images, drawn, args.check)
            print_join(rep)
            entry["join"] = rep
            if "verdict" in rep and not rep["verdict"].startswith("JOIN CONFIRMED") and not args.force_join:
                failed = True
                print("     stopping: not writing a bundle on an unconfirmed join (--force-join to override)")
                manifest["classes"][cls] = entry
                break

            cols = choose_columns(params, args.all_columns)
            header = ["row"] + [r["label"].strip() or r["name"].strip() for r in cols]
            idx = [r["n"] - 1 for r in cols]
            lines = [",".join('"%s"' % h if "," in h else h for h in header)]
            for row in drawn:
                base = row * par
                lines.append(",".join([str(row)] + [fmt(values[base + j]) for j in idx]))
            entry.update({"rows": drawn, "columns": header, "images_written": 0})
            payload = 0
            if out:
                out.writestr("%s/targets.csv" % cls, "\n".join(lines) + "\n")
                if args.images:
                    for row in drawn:
                        with open(images[row], "rb") as fh:
                            data = fh.read()
                        out.writestr("%s/images/%08d.tiff" % (cls, row), data)
                        payload += len(data)
                    entry["images_written"] = len(drawn)
            print("     images: %s" % human_bytes(payload))
            manifest["classes"][cls] = entry

        if out and not failed:
            out.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
    except BaseException:
        failed = True
        raise
    finally:
        if out:
            out.close()
            if failed and os.path.exists(args.out):
                os.remove(args.out)
                print("     removed partial %s" % args.out)

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0
    if failed:
        print("\nno bundle written")
        return 1
    print("\nwrote %s -- %s" % (args.out, human_bytes(os.path.getsize(args.out))))
    print("next: python scripts/make_a8_bundle.py --verify %s" % args.out)
    return 0


def verify(path):
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if "MANIFEST.json" not in names:
            print("No MANIFEST.json -- not a bundle from make_a8_bundle.py?")
            return 1
        man = json.loads(zf.read("MANIFEST.json"))
        print("bundle: %s (%s)   seed %s   per-class %s\n" % (path, human_bytes(os.path.getsize(path)), man["seed"], man["per_class_requested"]))
        problems = []
        for cls, e in sorted(man["classes"].items()):
            rows = e.get("rows", [])
            if rows:
                csv = zf.read("%s/targets.csv" % cls).decode().strip().splitlines()
                if len(csv) - 1 != len(rows):
                    problems.append("%s: %d csv rows for %d drawn" % (cls, len(csv) - 1, len(rows)))
                missing = [r for r in rows if "%s/images/%08d.tiff" % (cls, r) not in names]
                if e.get("images_written") and missing:
                    problems.append("%s: %d drawn events have no image" % (cls, len(missing)))
            verdict = e.get("join", {}).get("verdict", "panel only")
            print("%-4s %4d events   %d columns   pages: %s\n     %s"
                  % (cls, len(rows), len(e.get("columns", [])), " | ".join(e.get("page_names", [])), verdict))
        if problems:
            print("\nFAILED"); [print("  " + p) for p in problems]; return 1
        print("\nOK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_folder", nargs="?", help="The A8 run folder holding <pos>-<code>.fcs and <pos>-<code>_images/")
    ap.add_argument("--verify", metavar="ZIP")
    ap.add_argument("--classes", nargs="+", metavar="CLS")
    ap.add_argument("--per-class", type=int, default=200, help="Events per sample (default 200; 0 = panel and page names only)")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--out", default="a8_bundle_seed0.zip")
    ap.add_argument("--check", type=int, default=150, help="Events per sample used for the join check (default 150)")
    ap.add_argument("--all-columns", action="store_true", help="Keep the 162 raw spectral detector columns too")
    ap.add_argument("--no-images", dest="images", action="store_false")
    ap.add_argument("--force-join", action="store_true", help="Write the bundle even if the join check fails")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite the output zip")
    args = ap.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    if not args.run_folder:
        ap.error("give the run folder (or --verify a bundle)")
    if os.path.exists(args.out) and not (args.dry_run or args.force):
        ap.error("%s already exists (pass --force to overwrite)" % args.out)
    return build(args.run_folder, args)


if __name__ == "__main__":
    sys.exit(run(main))
