#!/usr/bin/env python
"""Verify (and optionally unpack) a sample zip built by make_sample.py.

Checks the sample is intact: every member listed in MANIFEST.json is present, CRCs
match, and -- the check that matters here -- every sampled event still carries its full
channel set. A sample that lost one channel of one event would otherwise show up much
later as a quietly wrong training pair.

Prints the sampling parameters so you can regenerate the exact same sample later.

Examples
--------
  python scripts/check_sample.py sample_2S_2P_2SP_seed0.zip
  python scripts/check_sample.py <zip> --extract-to data/samples
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cytpix_zips import human_bytes, open_zip, run, split_channel  # noqa: E402
import re  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample_zip", help="Sample zip produced by scripts/make_sample.py")
    ap.add_argument("--extract-to", metavar="DIR", help="Unpack into DIR/<sample-name>/ after verifying")
    args = ap.parse_args(argv)

    with open_zip(args.sample_zip) as zf:
        bad = zf.testzip()
        if bad is not None:
            print("CORRUPT: first bad entry is %s" % bad)
            return 1

        names = {i.filename for i in zf.infolist() if not i.is_dir()}
        crcs = {i.filename: i.CRC for i in zf.infolist() if not i.is_dir()}

        if "MANIFEST.json" not in names:
            print("No MANIFEST.json -- not a sample zip from make_sample.py?")
            return 1
        manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))

        print("sample: %s (%s)" % (args.sample_zip, human_bytes(os.path.getsize(args.sample_zip))))
        print(
            "seed: %s   per-class requested: %s   channel-regex: %s"
            % (manifest["seed"], manifest["per_class_requested"], manifest.get("channel_regex") or "(auto)")
        )
        print()

        missing, mismatched, short_events = [], [], []
        for cls, meta in sorted(manifest["classes"].items()):
            for member in meta["members"]:
                arcname = "%s/%s" % (cls, member["name"])
                if arcname not in names:
                    missing.append(arcname)
                elif crcs[arcname] != member["crc"]:
                    mismatched.append(arcname)

            # Every event should have arrived with the same number of frames it was
            # drawn with -- that is the whole reason the draw is event-level.
            expected = len(meta["modal_channel_signature"].split("+")) if meta["modal_channel_signature"] else 1
            drawn = set(meta["event_keys"])
            frames = meta["members"][: len(meta["members"]) - meta["metadata_entries"]]
            present = collections.Counter(
                _event_of(member["name"], manifest.get("channel_regex")) for member in frames
            )
            for key in sorted(drawn):
                if present.get(key, 0) != expected:
                    short_events.append(
                        "%s/%s has %d of %d frames" % (cls, key, present.get(key, 0), expected)
                    )
            for key in sorted(set(present) - drawn):
                short_events.append("%s/%s is present but was not in the drawn event list" % (cls, key))

            print(
                "%-4s %-44s %4d events x %d frames  (drawn from %d events)"
                % (
                    cls,
                    meta["source_zip"][:44],
                    meta["events_sampled"],
                    expected,
                    meta["source_events"],
                )
            )
            print("     channels: %s" % (meta["modal_channel_signature"] or "(single frame per event)"))

        by_class = collections.Counter(n.split("/")[0] for n in names if "/" in n)
        print("\nentries present per class: %s" % dict(sorted(by_class.items())))

        if missing or mismatched or short_events:
            print("\nFAILED")
            for arcname in missing[:20]:
                print("  missing: %s" % arcname)
            for arcname in mismatched[:20]:
                print("  crc mismatch: %s" % arcname)
            for note in short_events[:20]:
                print("  incomplete event: %s" % note)
            return 1

        total = sum(len(m["members"]) for m in manifest["classes"].values())
        print("\nOK -- all %d manifest entries present with matching CRCs, every event complete" % total)

        if args.extract_to:
            stem = os.path.splitext(os.path.basename(args.sample_zip))[0]
            dest = os.path.join(args.extract_to, stem)
            os.makedirs(dest, exist_ok=True)
            zf.extractall(dest)
            print("extracted to %s" % dest)
    return 0


def _event_of(name, channel_regex):
    """Which event a member belongs to, grouped exactly as make_sample.py grouped it.

    Recomputed rather than prefix-matched against the drawn keys: 'ev1' is a prefix of
    'ev10', so prefix matching would silently file one event's frames under another.
    """
    if channel_regex:
        match = re.compile(channel_regex).search(name)
        return match.group("event") if match else None
    return split_channel(name)[0]


if __name__ == "__main__":
    sys.exit(run(main))
