#!/usr/bin/env python
"""Build one small zip holding a matched image + target subset, ready to copy off the server.

The .acs archives carry both halves -- the event images and the FCS the targets come from --
so a working subset can be cut from them in a single pass, with the pairing guaranteed by
construction rather than re-joined afterwards.

This is the fast path. The full `make_targets.py` run reads every event in every archive
and formats ~283,000 CSV rows; this reads the same FCS data but formats only the rows it
keeps, and copies only the images it drew. Pass --no-images for a targets-only bundle,
which is a few MB and takes about as long as reading the FCS members.

Output layout:

    bundle_seed0.zip
      MANIFEST.json
      2S/targets.csv          one row per drawn event
      2S/spillover.csv        the acquisition's compensation matrix
      2S/images/<event>.tif   the matching image, named by event id
      2P/...

Every drawn event has both its row and its image, and --verify re-checks that after the
copy. Standard library only.

Examples
--------
  python scripts/make_bundle.py <run-folder> --per-class 200 --out bundle_seed0.zip
  python scripts/make_bundle.py <run-folder> --no-images --out targets_only.zip
  python scripts/make_bundle.py <run-folder> --classes 2S 3S --per-class 500 --dry-run
  python scripts/make_bundle.py --verify bundle_seed0.zip
"""

import argparse
import glob
import json
import os
import posixpath
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcs_data  # noqa: E402
import fcs_probe  # noqa: E402
from cytpix_zips import IMAGE_EXTS, class_of, human_bytes, open_zip, run  # noqa: E402
from make_targets import ALWAYS, MORPHOLOGY, labelled_detectors  # noqa: E402

TEXT_HEAD_BYTES = 1024 * 1024


def acs_class(path):
    """Class token for an .acs, matching the convention used for the image zips."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return class_of(stem + ".zip")


def plan_archive(zf, args):
    """Choose the events to keep. Returns (event_ids, rows, header, spillover, stats)."""
    members = [i for i in zf.infolist() if not i.is_dir()]

    images = {}
    for info in members:
        if posixpath.splitext(info.filename)[1].lower() not in IMAGE_EXTS:
            continue
        stem = posixpath.splitext(posixpath.basename(info.filename))[0]
        if stem.isdigit():
            images[int(stem)] = info

    fcs_members = [i for i in members if i.filename.lower().endswith(".fcs")]
    if not fcs_members:
        raise IOError("no .fcs member in this archive")
    fcs_name = fcs_members[0].filename

    with zf.open(fcs_name) as fh:
        head = fh.read(TEXT_HEAD_BYTES)
    keywords = fcs_probe.read_text_segment(head)
    parameters = fcs_probe.parameters(keywords)

    stained = labelled_detectors(parameters)
    comp = fcs_probe.spillover(keywords)

    # --per-class 0 wants the compensation matrix and the panel, nothing else. Those live
    # in the TEXT segment, already read above, so the event data never has to come off the
    # share at all -- seconds per archive instead of decompressing tens of megabytes.
    if args.per_class <= 0:
        stats = {
            "fcs_member": fcs_name,
            "events_in_fcs": int(keywords.get("$TOT", "0") or 0),
            "images_in_archive": len(images),
            "events_with_an_image": None,
            "eligible": None,
            "drawn": 0,
            "detectors": [{"name": r["name"], "label": r["label"], "voltage": r["voltage"]} for r in stained],
        }
        return [], [], [], comp, images, stats

    raw = zf.read(fcs_name)
    values, par, n = fcs_data.read_matrix(raw, keywords)

    event_idx = fcs_data.column_index(parameters, "Event")
    flag_idx = fcs_data.column_index(parameters, "ImageFlag")
    if event_idx is None:
        raise IOError("this FCS has no 'Event' parameter; cannot key the join")

    # Row index for every event that has an image in this archive. Drawing from the
    # intersection means an event can never arrive with a row but no picture.
    row_of = {}
    for i in range(n):
        base = i * par
        if flag_idx is not None and not values[base + flag_idx]:
            continue
        row_of[int(values[base + event_idx])] = i

    eligible = sorted(set(row_of) & set(images))
    rng = random.Random("%s|%s" % (args.seed, acs_class(zf.filename) or zf.filename))
    drawn = sorted(rng.sample(eligible, args.per_class)) if args.per_class < len(eligible) else list(eligible)

    wanted = list(ALWAYS) + [r["name"] for r in stained]
    if args.morphology:
        wanted += MORPHOLOGY
    indices, header = [], []
    for name in wanted:
        idx = fcs_data.column_index(parameters, name)
        if idx is None or idx in indices:
            continue
        indices.append(idx)
        header.append(parameters[idx]["label"].strip() or parameters[idx]["name"].strip())

    rows = []
    for event in drawn:
        base = row_of[event] * par
        rows.append([values[base + j] for j in indices])

    stats = {
        "fcs_member": fcs_name,
        "events_in_fcs": n,
        "images_in_archive": len(images),
        "events_with_an_image": len(row_of),
        "eligible": len(eligible),
        "drawn": len(drawn),
        "detectors": [{"name": r["name"], "label": r["label"], "voltage": r["voltage"]} for r in stained],
    }
    return drawn, rows, header, comp, images, stats


def format_csv(header, rows):
    out = [",".join(header)]
    for row in rows:
        out.append(",".join(_fmt(v) for v in row))
    return "\n".join(out) + "\n"


def _fmt(v):
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(round(v, 6)) if isinstance(v, float) else str(v)


def format_spillover(comp):
    if not comp or not comp["matrix"]:
        return None
    lines = ["," + ",".join(comp["detectors"])]
    for name, row in zip(comp["detectors"], comp["matrix"]):
        lines.append(name + "," + ",".join(repr(v) for v in row))
    return "\n".join(lines) + "\n"


def build(paths, args):
    manifest = {
        "seed": args.seed,
        "per_class_requested": args.per_class,
        "includes_images": args.images,
        "classes": {},
    }
    out = None
    if not args.dry_run:
        out = zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED)

    try:
        for path in paths:
            cls = acs_class(path) or os.path.splitext(os.path.basename(path))[0]
            print("%-5s %s" % (cls, os.path.basename(path)))
            with open_zip(path) as zf:
                drawn, rows, header, comp, images, stats = plan_archive(zf, args)

                if stats["events_with_an_image"] is None:
                    print("       %d events, %d images -- panel and matrix only, event data not read"
                          % (stats["events_in_fcs"], stats["images_in_archive"]))
                else:
                    print("       %d events, %d imaged, drew %d"
                          % (stats["events_in_fcs"], stats["events_with_an_image"], stats["drawn"]))
                if stats["eligible"] is not None and stats["eligible"] < stats["events_with_an_image"]:
                    print("       note: %d events are flagged as imaged but have no image member"
                          % (stats["events_with_an_image"] - stats["eligible"]))

                payload = sum(images[e].file_size for e in drawn) if args.images else 0
                print("       image payload: %s" % human_bytes(payload))

                entry = dict(stats)
                entry["source"] = os.path.basename(path)
                entry["event_ids"] = drawn
                entry["columns"] = header
                manifest["classes"][cls] = entry

                if args.dry_run:
                    continue

                if header:
                    out.writestr("%s/targets.csv" % cls, format_csv(header, rows))
                matrix = format_spillover(comp)
                if matrix:
                    out.writestr("%s/spillover.csv" % cls, matrix)
                else:
                    print("       note: no usable $SPILLOVER matrix in this archive")

                if args.images:
                    for event in drawn:
                        info = images[event]
                        ext = posixpath.splitext(info.filename)[1].lower()
                        out.writestr("%s/images/%d%s" % (cls, event, ext), zf.read(info.filename))

        if args.dry_run:
            print("\ndry run -- nothing written")
            return 0

        out.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        if out is not None:
            out.close()

    print("\nwrote %s -- %s" % (args.out, human_bytes(os.path.getsize(args.out))))
    print("next: python scripts/make_bundle.py --verify %s" % args.out)
    return 0


def verify(path):
    with open_zip(path) as zf:
        names = {i.filename for i in zf.infolist() if not i.is_dir()}
        if "MANIFEST.json" not in names:
            print("No MANIFEST.json -- not a bundle from make_bundle.py?")
            return 1
        manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))

        print("bundle: %s (%s)" % (path, human_bytes(os.path.getsize(path))))
        print("seed: %s   per-class: %s   images: %s\n"
              % (manifest["seed"], manifest["per_class_requested"], manifest["includes_images"]))

        problems = []
        for cls, entry in sorted(manifest["classes"].items()):
            drawn = entry["event_ids"]
            csv_name = "%s/targets.csv" % cls
            if not drawn and csv_name not in names:
                print("%-5s matrix and panel only   spillover: %s"
                      % (cls, "yes" if "%s/spillover.csv" % cls in names else "NO"))
                continue
            if csv_name not in names:
                problems.append("%s: targets.csv missing" % cls)
                continue
            body = zf.read(csv_name).decode("utf-8").strip().splitlines()
            n_rows = len(body) - 1
            if n_rows != len(drawn):
                problems.append("%s: %d rows for %d drawn events" % (cls, n_rows, len(drawn)))

            missing_images = 0
            if manifest["includes_images"]:
                present = {n for n in names if n.startswith("%s/images/" % cls)}
                for event in drawn:
                    if not any(n.startswith("%s/images/%d." % (cls, event)) for n in present):
                        missing_images += 1
                if missing_images:
                    problems.append("%s: %d drawn events have no image" % (cls, missing_images))

            print("%-5s %4d events   %d columns   spillover: %s"
                  % (cls, len(drawn), len(entry["columns"]),
                     "yes" if "%s/spillover.csv" % cls in names else "NO"))

        if problems:
            print("\nFAILED")
            for p in problems[:20]:
                print("  %s" % p)
            return 1
        total_drawn = sum(len(e["event_ids"]) for e in manifest["classes"].values())
        if not total_drawn:
            print("\nOK -- compensation matrices and panel only; no events were drawn")
        else:
            print("\nOK -- all %d drawn events have a target row%s"
                  % (total_drawn, " and an image" if manifest["includes_images"] else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="*", help="Run folder holding the .acs archives, or individual .acs paths")
    ap.add_argument("--verify", metavar="ZIP", help="Check an existing bundle instead of building one")
    ap.add_argument("--classes", nargs="+", metavar="CLS", help="Class tokens to include (default: all found)")
    ap.add_argument("--per-class", type=int, default=200, help="Events to draw per acquisition (default: 200)")
    ap.add_argument("--seed", default="0", help="Sampling seed (default: 0)")
    ap.add_argument("--out", default="bundle_seed0.zip", help="Output zip (default: bundle_seed0.zip)")
    ap.add_argument("--no-images", dest="images", action="store_false",
                    help="Targets and spillover only -- a few MB, and much faster")
    ap.add_argument("--no-morphology", dest="morphology", action="store_false",
                    help="Omit the instrument's morphology and intensity columns")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be drawn, write nothing")
    ap.add_argument("--force", action="store_true", help="Overwrite the output zip if it exists")
    args = ap.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    if not args.source:
        ap.error("give a run folder (or --verify a bundle)")
    if os.path.exists(args.out) and not (args.dry_run or args.force):
        ap.error("%s already exists (pass --force to overwrite)" % args.out)

    paths = []
    for item in args.source:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.acs"))))
        else:
            paths.append(item)
    if args.classes:
        wanted = {c.upper() for c in args.classes}
        paths = [p for p in paths if (acs_class(p) or "").upper() in wanted]
    if not paths:
        ap.error("No .acs archives found under: %s" % ", ".join(args.source))

    return build(paths, args)


if __name__ == "__main__":
    sys.exit(run(main))
