#!/usr/bin/env python
"""Turn an .acs (or a bare .fcs) into the per-event target table, as CSV.

This is the other half of the sample: make_sample.py draws the images, this draws the
numbers they are to be predicted from. One row per event, keyed by the event id that
names the image file.

By default it keeps only events that actually have an image ($PnN 'ImageFlag'), since an
event with no image is not a training pair -- roughly 70% of the record is dropped that
way, which is the point.

--verify-images checks the join rather than assuming it: it reads the image filenames out
of the archive and compares them against the event ids, reporting how many match. That is
the test of whether the filename really is the event index.

Standard library only.

Examples
--------
  python scripts/make_targets.py <run-folder> --out targets/
  python scripts/make_targets.py ..._3S.acs --verify-images --out targets/
  python scripts/make_targets.py ..._3S.acs --columns DAPI-A ACRV-1-PerCP-ef710-A --all-events
"""

import argparse
import glob
import os
import posixpath
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcs_data  # noqa: E402
import fcs_probe  # noqa: E402
from cytpix_zips import IMAGE_EXTS, human_bytes, run  # noqa: E402

# Columns worth having by default: the join key, the image flag, every detector the
# operator actually labelled, and the instrument's own morphology measurements -- which
# are a ready-made feature baseline, no image processing required.
ALWAYS = ["Event", "ImageFlag"]
MORPHOLOGY = [
    "TotalIntensity", "AverageIntensity", "StandardDeviationIntensity",
    "NumPixels", "AreaSquareMicrons", "PerimeterMicrons",
    "MajorDiameterMicrons", "MinorDiameterMicrons", "MinorMajorRatioPercent",
    "EccentricityPercent", "CircularityPercent", "ObjectCount", "ParticleCount",
    "IsOnBorder", "IsProcessed", "ConfidenceScore",
]


def load_fcs(path):
    """Return (raw bytes of the FCS, image filenames if the container has any)."""
    if path.lower().endswith(".fcs"):
        with open(path, "rb") as fh:
            return fh.read(), None

    with zipfile.ZipFile(path) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        fcs = [i for i in members if i.filename.lower().endswith(".fcs")]
        if not fcs:
            raise IOError("no .fcs member inside %s" % os.path.basename(path))
        if len(fcs) > 1:
            raise IOError(
                "%d .fcs members inside %s; pass the extracted file directly"
                % (len(fcs), os.path.basename(path))
            )
        images = [
            posixpath.splitext(posixpath.basename(i.filename))[0]
            for i in members
            if posixpath.splitext(i.filename)[1].lower() in IMAGE_EXTS
        ]
        return zf.read(fcs[0].filename), images


def labelled_detectors(parameters):
    """Detectors the operator gave a real label to -- i.e. the stained channels.

    An unlabelled detector carries its own name as its label ('YL2-A' labelled 'YL2-A'),
    which is the instrument filling in a blank, not a marker.
    """
    out = []
    for row in parameters:
        label = row["label"].strip()
        if label and label.lower() != row["name"].strip().lower():
            out.append(row)
    return out


def write_csv(path, header, rows):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(_fmt(v) for v in row) + "\n")


def _fmt(v):
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(round(v, 6))
    return str(v)


def process(path, args):
    print("=" * 78)
    print("%s (%s)" % (os.path.basename(path), human_bytes(os.path.getsize(path))))
    print("-" * 78)

    raw, archive_images = load_fcs(path)
    keywords = fcs_probe.read_text_segment(raw)
    parameters = fcs_probe.parameters(keywords)
    print("  %s" % fcs_probe.summarise(keywords))

    stained = labelled_detectors(parameters)
    print("  labelled detectors:")
    for row in stained:
        print("    %-12s %-26s voltage %s" % (row["name"], row["label"], row["voltage"] or "NA"))

    # Optional names may legitimately be absent on a given instrument configuration;
    # a name the caller asked for by hand should not vanish silently.
    wanted = [(name, True) for name in ALWAYS]
    if args.columns is None:
        wanted += [(r["name"], True) for r in stained]
    else:
        wanted += [(name, False) for name in args.columns]
    if args.morphology:
        wanted += [(name, True) for name in MORPHOLOGY]

    indices, header, absent = [], [], []
    for name, optional in wanted:
        idx = fcs_data.column_index(parameters, name)
        if idx is None:
            if optional:
                absent.append(name)
            else:
                print("  note: no parameter named %r -- skipped" % name)
            continue
        if idx in indices:
            continue
        indices.append(idx)
        header.append(parameters[idx]["label"].strip() or parameters[idx]["name"].strip())

    if absent:
        print("  not present on this instrument, omitted: %s" % ", ".join(absent))

    values, par, n = fcs_data.read_matrix(raw, keywords, args.max_events)
    print("  decoded %d events x %d parameters" % (n, par))

    flag_idx = fcs_data.column_index(parameters, "ImageFlag")
    event_idx = fcs_data.column_index(parameters, "Event")

    rows, imaged = [], 0
    event_ids = set()
    for i in range(n):
        base = i * par
        has_image = flag_idx is not None and values[base + flag_idx] != 0
        if has_image:
            imaged += 1
            if event_idx is not None:
                event_ids.add(int(values[base + event_idx]))
        if args.all_events or has_image or flag_idx is None:
            rows.append([values[base + j] for j in indices])

    if flag_idx is not None:
        print("  events with an image: %d of %d (%.1f%%)" % (imaged, n, 100.0 * imaged / n))

    if args.verify_images:
        if archive_images is None:
            print("  --verify-images needs the .acs, not a bare .fcs")
        elif event_idx is None:
            print("  --verify-images needs an 'Event' parameter; this file has none")
        else:
            names = set()
            unparsed = 0
            for stem in archive_images:
                if stem.isdigit():
                    names.add(int(stem))
                else:
                    unparsed += 1
            overlap = names & event_ids
            print("  -- join check --")
            print("    image files: %d (%d with non-numeric names)" % (len(names), unparsed))
            print("    events flagged as imaged: %d" % len(event_ids))
            print(
                "    ids present in both: %d  (%.1f%% of image files)"
                % (len(overlap), 100.0 * len(overlap) / max(1, len(names)))
            )
            if len(overlap) == len(names) == len(event_ids):
                print("    => JOIN CONFIRMED: image filename is the Event id, exactly.")
            elif len(overlap) > 0.95 * len(names):
                print("    => join holds for almost all images; inspect the remainder.")
            else:
                print("    => JOIN DOES NOT HOLD. The filename is not the Event id.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out, stem + ".csv")
        write_csv(out_path, header, rows)
        print("  wrote %s -- %d rows x %d columns (%s)"
              % (out_path, len(rows), len(header), human_bytes(os.path.getsize(out_path))))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="An .acs or .fcs file, several, or a directory holding them")
    ap.add_argument("--out", metavar="DIR", help="Write one CSV per input into DIR")
    ap.add_argument("--columns", nargs="+", metavar="NAME",
                    help="Parameters to keep, by detector name or label (default: every labelled detector)")
    ap.add_argument("--no-morphology", dest="morphology", action="store_false",
                    help="Omit the instrument's own morphology and intensity measurements")
    ap.add_argument("--all-events", action="store_true",
                    help="Keep every event, not only those with an image")
    ap.add_argument("--verify-images", action="store_true",
                    help="Check that image filenames match event ids (needs the .acs)")
    ap.add_argument("--max-events", type=int, default=None, help="Stop after this many events")
    args = ap.parse_args(argv)

    paths = []
    for item in args.source:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.acs"))))
            paths.extend(sorted(glob.glob(os.path.join(item, "*.fcs"))))
        else:
            paths.append(item)
    if not paths:
        ap.error("No .acs or .fcs files found under: %s" % ", ".join(args.source))

    for path in paths:
        process(path, args)
    return 0


if __name__ == "__main__":
    sys.exit(run(main))
