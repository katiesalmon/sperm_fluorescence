#!/usr/bin/env python
"""Survey Amnis ImageStream files (.rif raw, .cif compensated, .daf analysis) in place.

The image files are TIFF containers: one page per channel per object, in sequence. This
reads the IFD chain for the first N pages -- no pixels -- and reports the page geometry,
dtype, any page names or descriptions, and the repeating period, which is the number of
pages per object (channels, possibly plus masks). The .daf and .ist files are reported by
header and any readable text near the front.

Standard library only; runs on the server against Z: without copying.

Example
-------
  py -3 scripts\\inspect_amnis.py "Z:\\...\\260813_Blair_Sperm_ISX" --pages 240
"""

import argparse
import collections
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiff_probe  # noqa: E402
from cytpix import human_bytes, run  # noqa: E402

IMAGE_EXTS = (".rif", ".cif")
TEXT_EXTS = (".daf", ".ist", ".bak", ".xml", ".txt")


def period(seq):
    """Smallest p such that seq repeats with period p (over the part we read)."""
    n = len(seq)
    for p in range(1, n // 2 + 1):
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return None


def survey_tiff(path, pages, head_bytes):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        data = fh.read(head_bytes) if head_bytes else fh.read()
    try:
        info = tiff_probe.probe(data, max_pages=pages)
    except tiff_probe.NotTiff as exc:
        print("  not a TIFF container: %s   header %s" % (exc, data[:16].hex()))
        return
    pg = info["pages"]
    print("  TIFF, %s-endian%s; read %d page(s)%s"
          % (info["byte_order"], ", BigTIFF" if info["bigtiff"] else "", len(pg),
             " (capped; pass --pages for more)" if len(pg) >= pages else " (end of chain reached)"))
    if not pg:
        return
    def one(v):  # a tag read as a list (per-sample values) is hashed by its first element
        return v[0] if isinstance(v, list) and v else v
    sig = [(one(p.get("width")), one(p.get("height")), one(p.get("samples_per_pixel")),
            tiff_probe.dtype_of(p), one(p.get("compression"))) for p in pg]
    shapes = collections.Counter(sig)
    print("  page geometries (w x h, spp, dtype, compression):")
    for (w, h, spp, dt, comp), n in shapes.most_common(8):
        print("    %5sx%-5s spp=%s %-8s %-8s x%d" % (w, h, spp, dt, comp, n))
    # period on the coarse signature (dtype + dims class), since object crops vary in width
    coarse = [(p.get("samples_per_pixel"), tiff_probe.dtype_of(p), p.get("height")) for p in pg]
    per = period(coarse)
    print("  repeating period over (spp, dtype, height): %s   <- pages per object, if stable" % per)
    names = collections.Counter()
    for p in pg[:48]:
        for k in ("pagename", "imagedescription"):
            if p.get(k):
                names[("%s: %s" % (k, p[k][:140]))] += 1
    if names:
        print("  page names / descriptions, first 48 pages:")
        for k, n in names.most_common(16):
            print("    x%-3d %s" % (n, k))
    else:
        print("  no page names or descriptions in the first %d pages" % min(48, len(pg)))
    widths = [one(p.get("width")) for p in pg]; heights = [one(p.get("height")) for p in pg]
    print("  width range %s..%s   height range %s..%s" % (min(widths), max(widths), min(heights), max(heights)))

    # ImageStream layout hypothesis: one page per object with every channel tiled side by
    # side, so the page width is a multiple of the channel count; a uint16 page (image)
    # is followed by a uint8 page (mask) of the same size.
    real = [(w, h, dt) for (w, h, _, dt, _) in sig if w and w > 1]
    for nch in (12, 6):
        hit = sum(1 for w, _, _ in real if w % nch == 0)
        print("  widths divisible by %2d: %d of %d real pages%s" % (
            nch, hit, len(real), "   -> %d channels tiled, each %d..%d px wide" % (
                nch, min(w for w, _, _ in real) // nch, max(w for w, _, _ in real) // nch) if hit == len(real) and real else ""))
    pairs = sum(1 for a, b in zip(sig, sig[1:]) if a[3] == "uint16" and b[3] == "uint8" and a[:2] == b[:2])
    print("  uint16 page followed by a same-size uint8 page: %d times in %d pages   <- image + mask pairs" % (pairs, len(sig)))
    comps = sorted({c for (_, _, _, _, c) in sig if c is not None})
    print("  compression codes: %s%s" % (comps, "   (30817/30818 are Amnis-private, not a public codec)"
                                          if any(c in (30817, 30818) for c in comps) else ""))


def survey_text(path, nbytes=4096):
    with open(path, "rb") as fh:
        head = fh.read(nbytes)
    print("  header: %s" % head[:16].hex())
    if head[:2] == b"PK":
        print("  zip container")
    text = re.findall(rb"[\x20-\x7e]{12,}", head)
    for t in text[:12]:
        print("    %s" % t[:120].decode("ascii", "replace"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="A directory, or .rif/.cif/.daf/.ist files")
    ap.add_argument("--pages", type=int, default=240, help="Pages to walk per image file (default 240)")
    ap.add_argument("--head-mb", type=int, default=0, help="Read only the first N MB of each image file (0 = whole file)")
    args = ap.parse_args(argv)

    paths = []
    for item in args.source:
        if os.path.isdir(item):
            for ext in IMAGE_EXTS + TEXT_EXTS:
                paths += sorted(glob.glob(os.path.join(item, "*" + ext)))
        else:
            paths.append(item)
    if not paths:
        ap.error("nothing found under %s" % ", ".join(args.source))

    for path in paths:
        print("=" * 78)
        print("%s  (%s)" % (os.path.basename(path), human_bytes(os.path.getsize(path))))
        print("-" * 78)
        if path.lower().endswith(IMAGE_EXTS):
            survey_tiff(path, args.pages, args.head_mb * 1024 * 1024)
        else:
            survey_text(path)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(run(main))
