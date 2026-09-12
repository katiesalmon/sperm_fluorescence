#!/usr/bin/env python
"""Survey the CytPix marker-replicate exports without extracting them.

Run this before sampling anything. It answers the open questions in
docs/task_brief.md -- how the four marker channels are actually stored, whether an
event is one file or several, what the channels are called, and whether there is a
per-event cytometer measurement to join against.

Reading the central directory of a 2.7 GB zip is fast; no image data is decompressed
unless you ask for it. --structure decodes the TIFF headers of a few files (not their
pixels), and --peek-metadata reads the head of the non-image sidecars.

Examples
--------
  python scripts/inspect_zip.py <images-dir>
  python scripts/inspect_zip.py <images-dir> --classes 2S 2P 2SP --structure 12 --peek-metadata
  python scripts/inspect_zip.py <images-dir> --classes --json survey_marker.json
"""

import argparse
import collections
import json
import os
import posixpath
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcs_probe  # noqa: E402
import tiff_probe  # noqa: E402
from cytpix_zips import (  # noqa: E402
    DEFAULT_CLASSES,
    IMAGE_EXTS,
    find_zips,
    group_events,
    grouping_summary,
    human_bytes,
    image_members,
    metadata_members,
    open_zip,
    run,
)

# Established for replicate 1 in the sperm_pbmc survey. Printed alongside the marker
# replicates so any difference is obvious rather than something you have to remember.
REPLICATE_1 = "30,000 events, one 248x248 uint8 TIF each (RGBA container, LZW), no sidecar"


def probe_structures(zf, members, limit):
    """Decode the TIFF headers of an evenly spread subset. Returns distinct structures."""
    if not members or limit <= 0:
        return {}
    step = max(1, len(members) // limit)
    picked = members[::step][:limit]

    shapes = collections.Counter()
    descriptions = collections.Counter()
    failures = []
    for info in picked:
        try:
            probed = tiff_probe.probe(zf.read(info.filename))
        except (tiff_probe.NotTiff, ValueError, KeyError) as exc:
            failures.append("%s: %s" % (info.filename, exc))
            continue
        shapes[tiff_probe.summarise(probed)] += 1
        for page in probed["pages"]:
            for key in ("pagename", "imagedescription"):
                if page.get(key):
                    descriptions[page[key][:120]] += 1
    return {
        "files_probed": len(picked),
        "structures": dict(shapes.most_common()),
        "page_descriptions": dict(descriptions.most_common(12)),
        "failures": failures[:5],
    }


def probe_metadata(zf, members, limit=4, head_bytes=65536):
    """Read the head of each small sidecar. FCS files get their TEXT segment parsed."""
    out = []
    for info in members[:limit]:
        entry = {"name": info.filename, "bytes": info.file_size}
        try:
            data = zf.open(info.filename).read(head_bytes)
        except (IOError, OSError, ValueError) as exc:
            entry["error"] = str(exc)
            out.append(entry)
            continue

        if data[:3] == b"FCS":
            try:
                keywords = fcs_probe.read_text_segment(data)
                entry["fcs"] = {
                    "summary": fcs_probe.summarise(keywords),
                    "parameters": fcs_probe.parameters(keywords),
                }
            except ValueError as exc:
                entry["error"] = "FCS header read failed: %s" % exc
        else:
            text = data[:2000].decode("utf-8", "replace")
            entry["head_lines"] = text.splitlines()[:6]
        out.append(entry)
    return out


def survey_zip(path, tree_depth, examples, structure, peek_metadata, channel_regex):
    with open_zip(path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]

        by_ext = collections.Counter()
        by_dir = collections.Counter()
        total_uncompressed = 0
        for info in infos:
            ext = os.path.splitext(info.filename)[1].lower() or "(none)"
            by_ext[ext] += 1
            total_uncompressed += info.file_size
            parts = posixpath.dirname(info.filename).split("/")
            prefix = "/".join(p for p in parts[:tree_depth] if p)
            by_dir[prefix or "(root)"] += 1

        images = image_members(zf)
        sidecars = metadata_members(zf)
        sizes = sorted(i.file_size for i in images)

        result = {
            "path": path,
            "entries": len(infos),
            "uncompressed_bytes": total_uncompressed,
            "extensions": dict(by_ext.most_common()),
            "folders": dict(by_dir.most_common(40)),
            "image_entries": len(images),
            "example_names": [i.filename for i in images[:examples]],
            "sidecars": [{"name": i.filename, "bytes": i.file_size} for i in sidecars[:examples]],
            "sidecar_count": len(sidecars),
        }
        if sizes:
            result["image_size_bytes"] = {
                "min": sizes[0],
                "median": sizes[len(sizes) // 2],
                "max": sizes[-1],
                "mean": int(sum(sizes) / len(sizes)),
            }
        if images:
            events = group_events(images, channel_regex)
            result["grouping"] = grouping_summary(events, channel_regex)
            ids = event_id_stats(events)
            if ids:
                result["event_ids"] = ids
        if structure:
            result["structure"] = probe_structures(zf, images, structure)
        if peek_metadata and sidecars:
            result["metadata_peek"] = probe_metadata(zf, sidecars)
        return result


def event_id_stats(events):
    """If event keys are bare integers, describe the id range.

    The CytPix images only a fraction of the events a run records -- the camera cannot
    keep up with the detectors -- and names each image by its event index. So a sparse
    integer id range is the signature of "these images are a subset of a larger event
    record", and max(id) is a lower bound on how many events that record holds. That
    number is how we recognise the right FCS when we find it.
    """
    ids = []
    for key in events:
        leaf = key.rsplit("/", 1)[-1]
        if leaf.isdigit():
            ids.append(int(leaf))
    if len(ids) < len(events) * 0.9:
        return None  # not integer-named; nothing to say

    ids.sort()
    span = ids[-1] - ids[0] + 1
    gaps = [ids[i + 1] - ids[i] for i in range(len(ids) - 1)]
    return {
        "count": len(ids),
        "min": ids[0],
        "max": ids[-1],
        "span": span,
        "density": len(ids) / float(span),
        "largest_gap": max(gaps) if gaps else 0,
        "median_gap": sorted(gaps)[len(gaps) // 2] if gaps else 0,
    }


def print_survey(cls, s):
    print("=" * 78)
    print("%s  %s" % (cls, os.path.basename(s["path"])))
    print("-" * 78)
    print(
        "  entries: %d   images: %d   sidecars: %d   uncompressed: %s"
        % (s["entries"], s["image_entries"], s["sidecar_count"], human_bytes(s["uncompressed_bytes"]))
    )
    print("  extensions: %s" % ", ".join("%s x%d" % (e, n) for e, n in s["extensions"].items()))

    folders = list(s["folders"].items())
    print("  folders: %s" % ", ".join("%s (%d)" % (f[:40], n) for f, n in folders[:6]))
    if len(folders) > 6:
        print("           ... and %d more" % (len(folders) - 6))

    if "image_size_bytes" in s:
        sz = s["image_size_bytes"]
        print(
            "  image size: min %s / median %s / max %s"
            % (human_bytes(sz["min"]), human_bytes(sz["median"]), human_bytes(sz["max"]))
        )

    if "grouping" in s:
        g = s["grouping"]
        print("\n  -- events --")
        print("  %d events from %d image files" % (g["events"], s["image_entries"]))
        print("  files per event: %s" % g["files_per_event"])
        print("  channel signatures:")
        for signature, n in g["channel_signatures"].items():
            print("    %-48s %d events" % (signature[:48], n))

    if "event_ids" in s:
        e = s["event_ids"]
        print("\n  -- event ids --")
        print(
            "  %d images numbered %d..%d  (span %d, density %.1f%%)"
            % (e["count"], e["min"], e["max"], e["span"], 100 * e["density"])
        )
        print("  gap between consecutive ids: median %d, largest %d" % (e["median_gap"], e["largest_gap"]))
        if e["density"] < 0.9:
            print(
                "  => these images are a SUBSET of a larger event record; the full run\n"
                "     holds at least %d events. An FCS for this run should report $TOT >= %d."
                % (e["max"], e["max"])
            )

    if "structure" in s:
        st = s["structure"]
        print("\n  -- image structure (%d files, headers only) --" % st["files_probed"])
        for shape, n in st["structures"].items():
            print("    %-56s x%d" % (shape, n))
        if st["page_descriptions"]:
            print("  page descriptions / names:")
            for desc, n in st["page_descriptions"].items():
                print("    %-56s x%d" % (desc[:56], n))
        for failure in st["failures"]:
            print("    could not read: %s" % failure)

    print("\n  example image entries:")
    for name in s["example_names"][:6]:
        print("    %s" % name)
    if s["sidecars"]:
        print("  sidecars:")
        for entry in s["sidecars"]:
            print("    %-56s %s" % (entry["name"][:56], human_bytes(entry["bytes"])))

    for entry in s.get("metadata_peek", []):
        print("\n  -- %s --" % entry["name"])
        if "error" in entry:
            print("    %s" % entry["error"])
        elif "fcs" in entry:
            print("    %s" % entry["fcs"]["summary"])
            for row in entry["fcs"]["parameters"]:
                print("      P%-3d %-16s %s" % (row["n"], row["name"], row["label"]))
        else:
            for line in entry.get("head_lines", []):
                print("    %s" % line[:100])
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="Directory holding the export zips, or individual zip paths")
    ap.add_argument(
        "--classes",
        nargs="*",
        default=None,
        metavar="CLS",
        help="Class tokens to survey (default: %s). Pass --classes with no values for all."
        % " ".join(DEFAULT_CLASSES),
    )
    ap.add_argument("--tree-depth", type=int, default=2, help="Folder depth to group by (default: 2)")
    ap.add_argument("--examples", type=int, default=10, help="Example entry names to print (default: 10)")
    ap.add_argument(
        "--structure",
        type=int,
        nargs="?",
        const=12,
        default=0,
        metavar="N",
        help="Read the TIFF headers of N files per zip (default 12 when given). No pixels are decoded.",
    )
    ap.add_argument(
        "--peek-metadata",
        action="store_true",
        help="Read the head of each sidecar; FCS files get their parameter table printed",
    )
    ap.add_argument(
        "--channel-regex",
        default=None,
        metavar="RE",
        help="Override channel auto-detection. Pattern with named groups `event` and optionally `channel`.",
    )
    ap.add_argument("--json", metavar="PATH", help="Also write the full survey as JSON")
    args = ap.parse_args(argv)

    if args.classes is None:
        classes = DEFAULT_CLASSES
    elif len(args.classes) == 0:
        classes = None  # every class found
    else:
        classes = args.classes

    zips = find_zips(args.source, classes)
    if not zips:
        ap.error("No CytPix zips found under: %s" % ", ".join(args.source))

    surveys = {}
    for cls in sorted(zips):
        surveys[cls] = survey_zip(
            zips[cls], args.tree_depth, args.examples, args.structure, args.peek_metadata, args.channel_regex
        )
        print_survey(cls, surveys[cls])

    print("for comparison, replicate 1 (brightfield only): %s" % REPLICATE_1)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(surveys, fh, indent=2, sort_keys=True)
        print("Wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(run(main))
