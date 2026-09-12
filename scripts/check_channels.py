#!/usr/bin/env python
"""Decide whether the RGBA channels of an export carry distinct images.

The marker-replicate zips hold one 248x248 uint8 TIF per event with four samples per
pixel -- structurally identical to replicate 1, where R == G == B and alpha was a
constant 255, i.e. grayscale in a four-channel container. If replicates 2 and 3 instead
put marker signal into those channels, this is where it would be, and it is the only
place left in these files for it to hide.

That question cannot be answered from the TIFF header, so this decodes pixels. It uses
the standard-library LZW decoder in tiff_read.py, so it still needs no install and can
run on the imaging server directly against the zips.

Examples
--------
  python scripts/check_channels.py <images-dir> --classes 2S 2P 2SP 3S 3P 3SP
  python scripts/check_channels.py <images-dir> --classes 1S 2S --events 40
  python scripts/check_channels.py data/samples/sample_rep2_seed0
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiff_read  # noqa: E402
from cytpix_zips import (  # noqa: E402
    DEFAULT_CLASSES,
    IMAGE_EXTS,
    find_zips,
    image_members,
    open_zip,
    run,
)

CHANNEL_NAMES = ["R", "G", "B", "A"]


def stats(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0, 0
    total = sum(values)
    mean = total / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5, min(values), max(values)


def analyse_image(data):
    """Per-channel statistics and cross-channel agreement for one event."""
    pixels, width, height, spp = tiff_read.read_page(data)
    channels = [list(tiff_read.channel(pixels, spp, i)) for i in range(spp)]

    result = {"width": width, "height": height, "spp": spp, "channels": []}
    for i, values in enumerate(channels):
        mean, sd, lo, hi = stats(values)
        result["channels"].append(
            {
                "name": CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else "S%d" % i,
                "mean": mean,
                "std": sd,
                "min": lo,
                "max": hi,
                "distinct_values": len(set(values)),
            }
        )

    # How far apart are the channels, pixel by pixel? Zero everywhere means duplicates.
    result["max_abs_diff"] = {}
    for i in range(spp):
        for j in range(i + 1, spp):
            worst = 0
            for a, b in zip(channels[i], channels[j]):
                d = a - b if a > b else b - a
                if d > worst:
                    worst = d
            result["max_abs_diff"]["%s-%s" % (CHANNEL_NAMES[i], CHANNEL_NAMES[j])] = worst
    return result


def verdict(per_image):
    """Summarise a class: are the channels duplicates, or do they carry separate images?"""
    if not per_image:
        return "no images decoded", {}

    spp = per_image[0]["spp"]
    if spp == 1:
        return "single channel -- nothing to compare", {}

    worst = {}
    for result in per_image:
        for pair, d in result["max_abs_diff"].items():
            worst[pair] = max(worst.get(pair, 0), d)

    colour_pairs = [p for p in worst if "A" not in p]
    alpha_pairs = [p for p in worst if "A" in p]
    colour_max = max((worst[p] for p in colour_pairs), default=0)

    alpha_constant = all(
        c["std"] == 0.0 for r in per_image for c in r["channels"] if c["name"] == "A"
    )

    if colour_max == 0 and alpha_constant:
        text = "GRAYSCALE IN AN RGBA CONTAINER -- R == G == B on every pixel, alpha constant."
    elif colour_max == 0:
        text = "R == G == B on every pixel, but alpha varies -- check what alpha holds."
    elif colour_max <= 2:
        text = (
            "colour channels differ by at most %d grey level -- compression or rounding, "
            "not separate images." % colour_max
        )
    else:
        text = (
            "COLOUR CHANNELS CARRY DIFFERENT DATA -- max difference %d grey levels. "
            "This is where the marker signal would be." % colour_max
        )
    del alpha_pairs
    return text, worst


def sources(args):
    """{label: [(name, bytes-reader), ...]} from either zips or an unpacked directory."""
    out = {}
    for path in args.source:
        if os.path.isdir(path):
            for sub in sorted(glob.glob(os.path.join(path, "*"))):
                if os.path.isdir(sub):
                    files = [
                        f
                        for f in sorted(glob.glob(os.path.join(sub, "*")))
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                        and not os.path.basename(f).startswith("._")
                    ]
                    if files:
                        out[os.path.basename(sub)] = ("dir", files)
    if out:
        return out
    return {cls: ("zip", path) for cls, path in find_zips(args.source, args.classes).items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="Export directory, individual zips, or an unpacked sample directory")
    ap.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        metavar="CLS",
        help="Class tokens to check (default: %s). Ignored for an unpacked sample." % " ".join(DEFAULT_CLASSES),
    )
    ap.add_argument("--events", type=int, default=12, help="Events to decode per class (default: 12)")
    ap.add_argument("--detail", action="store_true", help="Print per-event channel statistics")
    args = ap.parse_args(argv)

    for label, (kind, where) in sorted(sources(args).items()):
        print("=" * 78)
        print(label)
        print("-" * 78)

        results, failures = [], []
        if kind == "zip":
            with open_zip(where) as zf:
                members = image_members(zf)
                step = max(1, len(members) // args.events)
                for info in members[::step][: args.events]:
                    try:
                        results.append(analyse_image(zf.read(info.filename)))
                    except (tiff_read.Unsupported, ValueError) as exc:
                        failures.append("%s: %s" % (info.filename, exc))
        else:
            step = max(1, len(where) // args.events)
            for path in where[::step][: args.events]:
                try:
                    with open(path, "rb") as fh:
                        results.append(analyse_image(fh.read()))
                except (tiff_read.Unsupported, ValueError) as exc:
                    failures.append("%s: %s" % (os.path.basename(path), exc))

        if not results:
            print("  decoded nothing")
            for failure in failures[:5]:
                print("  %s" % failure)
            continue

        first = results[0]
        print("  decoded %d events  %dx%d  spp=%d" % (len(results), first["width"], first["height"], first["spp"]))

        for i in range(first["spp"]):
            name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else "S%d" % i
            means = [r["channels"][i]["mean"] for r in results]
            sds = [r["channels"][i]["std"] for r in results]
            distinct = [r["channels"][i]["distinct_values"] for r in results]
            print(
                "    %s  mean %7.2f (%.2f-%.2f)   sd %6.2f   distinct values %d-%d"
                % (name, sum(means) / len(means), min(means), max(means),
                   sum(sds) / len(sds), min(distinct), max(distinct))
            )

        text, worst = verdict(results)
        if worst:
            print("  worst pixel difference between channels, over all decoded events:")
            print("    %s" % "  ".join("%s=%d" % (p, d) for p, d in sorted(worst.items())))
        print("\n  => %s" % text)

        if args.detail:
            for n, r in enumerate(results):
                print("    event %d: %s" % (n, r["max_abs_diff"]))
        for failure in failures[:5]:
            print("  could not decode %s" % failure)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(run(main))
