#!/usr/bin/env python
"""Generate fake CytPix-shaped zips so the data scripts can be exercised without the real data.

For testing only -- never point it at the real export directory. The images are noise,
not cells; the point is to check that inspect/make/check behave, and in particular that
the event grouping survives whichever layout the marker exports turn out to use.

`--layout` picks the shape to rehearse against:

  per-channel   one TIF per channel per event:  <event>_DAPI.tif, <event>_BF.tif, ...
  multipage     one multi-page TIF per event, one page per channel
  single        one brightfield TIF per event plus an events.csv of per-channel
                intensities -- the shape if the marker signal is a cytometer
                measurement rather than an image

Example
-------
  python3 scripts/make_fixture.py /tmp/fixture/Images --layout per-channel
  python3 scripts/inspect_zip.py /tmp/fixture/Images --peek --structure
  python3 scripts/make_sample.py /tmp/fixture/Images --per-class 20 --out /tmp/sample.zip
"""

import argparse
import os
import random
import struct
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cytpix_zips import MARKERS, run  # noqa: E402

PREFIX = "260709_cytpix_export"
SIZE = 64  # smaller than the real 248 so fixtures stay quick to write

_SHORT, _LONG, _ASCII = 3, 4, 2


def _entry(tag, type_code, count, payload, extra_offset):
    """One 12-byte IFD entry. `payload` is inline if it fits in 4 bytes, else a pointer."""
    if len(payload) <= 4:
        payload = payload + b"\x00" * (4 - len(payload))
        return struct.pack("<HHI", tag, type_code, count) + payload, b""
    return struct.pack("<HHII", tag, type_code, count, extra_offset), payload


def tiff_bytes(pages):
    """Build a little-endian, uncompressed TIFF from `pages`.

    Each page is (width, height, samples_per_pixel, description, pixel_bytes).
    """
    ifd_size = 2 + 12 * 11 + 4
    ifd_offsets = [8 + i * ifd_size for i in range(len(pages))]
    cursor = 8 + ifd_size * len(pages)

    ifds, blobs = [], []
    for i, (width, height, spp, description, pixels) in enumerate(pages):
        desc = description.encode("utf-8") + b"\x00"
        entries, extras = [], []

        def add(tag, type_code, count, payload):
            nonlocal cursor
            entry, extra = _entry(tag, type_code, count, payload, cursor)
            entries.append(entry)
            if extra:
                extras.append(extra)
                cursor += len(extra) + (len(extra) % 2)  # IFD values are word-aligned

        add(256, _SHORT, 1, struct.pack("<H", width))
        add(257, _SHORT, 1, struct.pack("<H", height))
        add(258, _SHORT, spp, struct.pack("<%dH" % spp, *([8] * spp)))
        add(259, _SHORT, 1, struct.pack("<H", 1))
        add(262, _SHORT, 1, struct.pack("<H", 2 if spp >= 3 else 1))
        add(270, _ASCII, len(desc), desc)
        strip_entry_index = len(entries)
        entries.append(None)  # StripOffsets -- patched once the data offset is known
        add(277, _SHORT, 1, struct.pack("<H", spp))
        add(278, _SHORT, 1, struct.pack("<H", height))
        add(279, _LONG, 1, struct.pack("<I", len(pixels)))
        add(339, _SHORT, spp, struct.pack("<%dH" % spp, *([1] * spp)))

        entries[strip_entry_index] = struct.pack("<HHII", 273, _LONG, 1, cursor)
        extras.insert(0, pixels)  # keep pixel data adjacent to this page's other extras
        cursor += len(pixels)

        nxt = ifd_offsets[i + 1] if i + 1 < len(pages) else 0
        ifds.append(struct.pack("<H", 11) + b"".join(entries) + struct.pack("<I", nxt))
        blobs.append(extras)

    out = [b"II*\x00", struct.pack("<I", 8)] + ifds
    # Extras are emitted page by page, but pixel data was offset last within each page,
    # so put it back in the order the offsets were handed out.
    for extras in blobs:
        pixels = extras.pop(0)
        for extra in extras:
            out.append(extra + (b"\x00" if len(extra) % 2 else b""))
        out.append(pixels)
    return b"".join(out)


def noise(rng, width, height, spp):
    return bytes(rng.getrandbits(8) for _ in range(width * height * spp))


# Detector names as an Attune reports them, paired with the marker each one carries.
FCS_DETECTORS = [
    ("FSC-A", ""),
    ("SSC-A", ""),
    ("VL1-A", "DAPI"),
    ("BL1-A", "ACRV1"),
    ("BL2-A", "LDHC"),
    ("RL1-A", "Tomm20"),
]


def fcs_bytes(events):
    """A minimal but well-formed FCS 3.1 file: real HEADER and TEXT, empty DATA.

    Only the TEXT segment is ever read by the survey, and it is the part that carries
    the detector-to-marker mapping, so that is the part worth making realistic.
    """
    delimiter = "|"
    keywords = [
        ("$BEGINANALYSIS", "0"), ("$ENDANALYSIS", "0"),
        ("$BYTEORD", "1,2,3,4"), ("$DATATYPE", "F"), ("$MODE", "L"),
        ("$PAR", str(len(FCS_DETECTORS))), ("$TOT", str(events)),
        ("$CYT", "Attune CytPix"),
    ]
    for i, (name, label) in enumerate(FCS_DETECTORS, start=1):
        keywords.append(("$P%dN" % i, name))
        keywords.append(("$P%dR" % i, "1048576"))
        if label:
            keywords.append(("$P%dS" % i, label))

    body = delimiter + delimiter.join("%s%s%s" % (k, delimiter, v) for k, v in keywords)
    text = body.encode("ascii")
    start = 58
    end = start + len(text) - 1
    header = b"FCS3.1" + b"    " + b"".join(
        ("%8d" % n).encode("ascii") for n in (start, end, 0, 0, 0, 0)
    )
    return header + text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="Directory to write the fake zips into")
    ap.add_argument("--classes", nargs="+", default=["2S", "2P", "2SP"])
    ap.add_argument("--events", type=int, default=60, help="Fake events per class (default: 60)")
    ap.add_argument(
        "--layout",
        choices=["per-channel", "multipage", "single"],
        default="per-channel",
        help="Export shape to rehearse against (default: per-channel)",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    channels = ["BF"] + MARKERS_DISPLAY

    for cls in args.classes:
        path = os.path.join(args.out_dir, "%s_%s.zip" % (PREFIX, cls.upper()))
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(args.events):
                event = "event_%05d" % i
                if args.layout == "per-channel":
                    for channel in channels:
                        page = (SIZE, SIZE, 1, "channel=%s" % channel, noise(rng, SIZE, SIZE, 1))
                        zf.writestr("%s_%s.tif" % (event, channel), tiff_bytes([page]))
                elif args.layout == "multipage":
                    pages = [
                        (SIZE, SIZE, 1, "channel=%s" % channel, noise(rng, SIZE, SIZE, 1))
                        for channel in channels
                    ]
                    zf.writestr("%s.tif" % event, tiff_bytes(pages))
                else:
                    page = (SIZE, SIZE, 4, "brightfield RGBA", noise(rng, SIZE, SIZE, 4))
                    zf.writestr("%s.tif" % event, tiff_bytes([page]))

            if args.layout == "single":
                rows = ["event_id," + ",".join(MARKERS_DISPLAY)]
                for i in range(args.events):
                    rows.append(
                        "event_%05d," % i
                        + ",".join("%.1f" % (rng.random() * 5000) for _ in MARKERS_DISPLAY)
                    )
                zf.writestr("events.csv", "\n".join(rows) + "\n")
                zf.writestr("run.fcs", fcs_bytes(args.events))
            else:
                zf.writestr("metadata/events.csv", "event_id,area\nevent_00000,55\n")

            # An oversized entry, to exercise the --max-member-bytes guard.
            zf.writestr("mosaic_full.tif", b"II*\x00" + b"\x00" * (9 * 1024 * 1024))
        print("wrote %s" % path)
    return 0


# Display casing for the markers, matching what canonical_channel() returns.
MARKERS_DISPLAY = ["DAPI", "ACRV1", "LDHC", "Tomm20"]
assert {m.upper() for m in MARKERS_DISPLAY} == {m.upper() for m in MARKERS}


if __name__ == "__main__":
    sys.exit(run(main))
