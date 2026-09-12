#!/usr/bin/env python
"""Build a small, reproducible sample zip from the full CytPix marker exports.

The sample is a random draw with a fixed seed, so the same command yields the same
events on any machine, and MANIFEST.json inside the output records exactly what was
drawn -- source zips, sizes, seed, channel signature, and every member with its CRC.

**The draw is over events, not files.** Where replicate 1 had one file per event, a
marker replicate may have one per channel, and an event whose DAPI frame is in the
sample but whose brightfield frame is not teaches nothing. So events are grouped first
(see cytpix_zips.split_channel), and an event travels as a whole or not at all.

By default only events carrying the *modal* channel signature are eligible -- the
combination of channels that most events in that zip have. That quietly excludes
half-written events and stray non-event files (contact sheets, mosaics) instead of
letting them into the sample. Pass --allow-partial-events to draw from everything.

Output layout:

    sample_2S_2P_seed0.zip
      MANIFEST.json
      2S/<original path inside the source zip>
      2P/<original path inside the source zip>

Examples
--------
  python scripts/make_sample.py <images-dir> --classes 2S 2P 2SP --per-class 300 --dry-run
  python scripts/make_sample.py <images-dir> --classes 2S 2P 2SP --per-class 300 --out sample_rep2_seed0.zip
  python scripts/make_sample.py <images-dir> --classes 2S --channel-regex '(?P<event>ev\\d+)_(?P<channel>\\w+)\\.tif'
"""

import argparse
import collections
import json
import os
import posixpath
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cytpix_zips import (  # noqa: E402
    DEFAULT_CLASSES,
    channels_of,
    find_zips,
    group_events,
    human_bytes,
    image_members,
    metadata_members,
    open_zip,
    run,
)

# A single event frame should be small. Anything far above this is likely a contact
# sheet or a full field-of-view mosaic; we skip it rather than bloat the sample.
DEFAULT_MAX_MEMBER_BYTES = 8 * 1024 * 1024

# Sidecars are copied whole. An FCS covering every event in the run is a few MB and is
# the label source if the marker signal is a cytometer measurement, so the cap is
# generous -- it exists to stop a stray raw-data dump, not to trim the FCS.
DEFAULT_MAX_METADATA_BYTES = 256 * 1024 * 1024


def signature_of(files, channel_regex):
    return "+".join(token or "(none)" for token in channels_of(files, channel_regex))


def pick(keys, count, seed, cls):
    """Deterministic sample of `count` event keys. Seeded per class so that changing
    --classes does not reshuffle the draw for the classes you kept."""
    rng = random.Random("%s|%s" % (seed, cls))
    if count >= len(keys):
        return list(keys)
    return sorted(rng.sample(keys, count))


def plan_class(zf, cls, args):
    """Choose this class's events. Returns (members_to_copy, manifest_entry, notes)."""
    images = image_members(zf)
    events = group_events(images, args.channel_regex)

    signatures = collections.Counter(
        signature_of(files, args.channel_regex) for files in events.values()
    )
    modal, modal_count = signatures.most_common(1)[0] if signatures else ("", 0)

    notes = []
    eligible = []
    wrong_signature = oversized = 0
    for key, files in events.items():
        if not args.allow_partial_events and signature_of(files, args.channel_regex) != modal:
            wrong_signature += 1
            continue
        if any(info.file_size > args.max_member_bytes for info in files):
            oversized += 1
            continue
        eligible.append(key)

    if wrong_signature:
        notes.append(
            "%d events did not match the modal channel signature '%s' and were excluded "
            "(--allow-partial-events to keep them)" % (wrong_signature, modal)
        )
    if oversized:
        notes.append(
            "%d events had a frame over %s and were excluded"
            % (oversized, human_bytes(args.max_member_bytes))
        )
    if len(signatures) > 1:
        notes.append(
            "channel signatures seen: %s"
            % ", ".join("%s x%d" % (s[:40], n) for s, n in signatures.most_common(4))
        )

    chosen = pick(eligible, args.per_class, args.seed, cls)
    if len(chosen) < args.per_class:
        notes.append(
            "only %d eligible events, fewer than --per-class %d" % (len(chosen), args.per_class)
        )

    members = []
    for key in chosen:
        members.extend(events[key])

    extras = []
    if args.include_metadata:
        extras = metadata_members(zf, args.max_metadata_bytes)

    entry = {
        "source_image_entries": len(images),
        "source_events": len(events),
        "modal_channel_signature": modal,
        "modal_signature_events": modal_count,
        "eligible_events": len(eligible),
        "events_sampled": len(chosen),
        "event_keys": chosen,
        "metadata_entries": len(extras),
        "members": [
            {"name": i.filename, "bytes": i.file_size, "crc": i.CRC} for i in members + extras
        ],
    }
    return members + extras, entry, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="Directory holding the export zips, or individual zip paths")
    ap.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        metavar="CLS",
        help="Class tokens to sample (default: %s)" % " ".join(DEFAULT_CLASSES),
    )
    ap.add_argument("--per-class", type=int, default=300, help="Events to draw per class (default: 300)")
    ap.add_argument("--seed", default="0", help="Sampling seed (default: 0)")
    ap.add_argument("--out", default=None, help="Output zip path (default: sample_<classes>_seed<seed>.zip)")
    ap.add_argument(
        "--channel-regex",
        default=None,
        metavar="RE",
        help="Override channel auto-detection. Pattern with named groups `event` and optionally `channel`.",
    )
    ap.add_argument(
        "--allow-partial-events",
        action="store_true",
        help="Draw from every event, not only those with the modal channel signature",
    )
    ap.add_argument(
        "--max-member-bytes",
        type=int,
        default=DEFAULT_MAX_MEMBER_BYTES,
        help="Exclude events with a frame larger than this (default: %d)" % DEFAULT_MAX_MEMBER_BYTES,
    )
    ap.add_argument(
        "--no-metadata",
        dest="include_metadata",
        action="store_false",
        help="Do not copy the FCS/CSV/XML sidecars (they are copied by default)",
    )
    ap.add_argument(
        "--max-metadata-bytes",
        type=int,
        default=DEFAULT_MAX_METADATA_BYTES,
        help="Skip sidecars larger than this (default: %d)" % DEFAULT_MAX_METADATA_BYTES,
    )
    ap.add_argument("--dry-run", action="store_true", help="Report what would be sampled, write nothing")
    ap.add_argument("--force", action="store_true", help="Overwrite the output zip if it exists")
    args = ap.parse_args(argv)

    classes = [c.upper() for c in args.classes]
    zips = find_zips(args.source, classes)

    out_path = args.out or "sample_%s_seed%s.zip" % ("_".join(classes), args.seed)
    if os.path.exists(out_path) and not (args.dry_run or args.force):
        ap.error("%s already exists (pass --force to overwrite)" % out_path)

    manifest = {
        "seed": args.seed,
        "per_class_requested": args.per_class,
        "max_member_bytes": args.max_member_bytes,
        "channel_regex": args.channel_regex,
        "allow_partial_events": args.allow_partial_events,
        "classes": {},
    }
    plan = []  # (cls, source_zip, [ZipInfo, ...])

    for cls in classes:
        src = zips[cls]
        with open_zip(src) as zf:
            members, entry, notes = plan_class(zf, cls, args)
            entry["source_zip"] = os.path.basename(src)
            entry["source_zip_bytes"] = os.stat(src).st_size
            manifest["classes"][cls] = entry
            plan.append((cls, src, members))

            print(
                "%-4s %-48s %6d events (%d files) -> sampling %d event%s, %d files"
                % (
                    cls,
                    os.path.basename(src)[:48],
                    entry["source_events"],
                    entry["source_image_entries"],
                    entry["events_sampled"],
                    "" if entry["events_sampled"] == 1 else "s",
                    len(members) - entry["metadata_entries"],
                )
            )
            print("     channels per event: %s" % (entry["modal_channel_signature"] or "(none)"))
            if entry["metadata_entries"]:
                print("     + %d sidecar entries" % entry["metadata_entries"])
            for note in notes:
                print("     note: %s" % note)

    projected = sum(m["bytes"] for c in manifest["classes"].values() for m in c["members"])
    print("\nprojected uncompressed payload: %s" % human_bytes(projected))

    if args.dry_run:
        print("dry run -- nothing written")
        return 0

    written = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
        for cls, src, members in plan:
            with open_zip(src) as zf:
                for info in members:
                    out.writestr(posixpath.join(cls, info.filename), zf.read(info.filename))
                    written += 1

    print(
        "\nwrote %s -- %d entries, %s on disk"
        % (out_path, written, human_bytes(os.path.getsize(out_path)))
    )
    print("next: python scripts/check_sample.py %s --extract-to data/samples" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(run(main))
