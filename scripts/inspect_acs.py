#!/usr/bin/env python
"""Look inside an .acs archive and pull out just the cytometry data.

An ACS (Archival Cytometry Standard) file is a zip container holding FCS data files
alongside a table of contents and, often, the images they were acquired with. The
CytPix run's .acs files are ~3.8 GB each -- larger than the image zips they correspond
to -- so they very likely carry both. Copying all nine to a laptop is ~34 GB; the FCS
inside is a few MB.

So this does two things: says what is in the archive, and extracts only the parts worth
moving. The FCS TEXT segment is parsed in place, which needs to decompress only the head
of the member, so listing is fast even on a multi-gigabyte archive.

Standard library only -- it runs on the server without an install, like everything else
under scripts/.

Examples
--------
  python scripts/inspect_acs.py <run-folder>
  python scripts/inspect_acs.py 260709_..._2P.acs --extract-fcs fcs_out
  python scripts/inspect_acs.py <run-folder> --extract-fcs fcs_out --quiet
"""

import argparse
import collections
import glob
import os
import posixpath
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcs_probe  # noqa: E402
from cytpix_zips import IMAGE_EXTS, human_bytes, run  # noqa: E402

FCS_EXTS = {".fcs"}
# Enough to reach the end of any reasonable TEXT segment without decompressing the events.
TEXT_HEAD_BYTES = 1024 * 1024


def categorise(name):
    ext = os.path.splitext(name)[1].lower()
    if ext in FCS_EXTS:
        return "fcs"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in (".xml", ".json", ".txt", ".html"):
        return "metadata"
    return "other"


def open_archive(path):
    if not zipfile.is_zipfile(path):
        with open(path, "rb") as fh:
            head = fh.read(16)
        raise IOError(
            "%s is not a zip container (first bytes: %s).\n"
            "If those bytes are all zero the file has not actually been copied -- a "
            "Remote Desktop redirected folder shows server-side metadata as a sparse "
            "stub. Run this on the machine that holds the real file."
            % (os.path.basename(path), head.hex() or "(empty)")
        )
    return zipfile.ZipFile(path)


def describe_fcs(zf, info):
    """Parse one FCS member's TEXT segment without decompressing its event data."""
    with zf.open(info.filename) as fh:
        head = fh.read(TEXT_HEAD_BYTES)
    keywords = fcs_probe.read_text_segment(head)
    return keywords, fcs_probe.parameters(keywords)


def inspect(path, max_fcs, quiet):
    print("=" * 78)
    print("%s  (%s)" % (os.path.basename(path), human_bytes(os.path.getsize(path))))
    print("-" * 78)

    with open_archive(path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        by_kind = collections.Counter()
        bytes_by_kind = collections.Counter()
        for info in infos:
            kind = categorise(info.filename)
            by_kind[kind] += 1
            bytes_by_kind[kind] += info.file_size

        print("  %d entries" % len(infos))
        for kind in ("fcs", "image", "metadata", "other"):
            if by_kind[kind]:
                print(
                    "    %-10s %6d entries  %s uncompressed"
                    % (kind, by_kind[kind], human_bytes(bytes_by_kind[kind]))
                )

        fcs_members = [i for i in infos if categorise(i.filename) == "fcs"]
        if not fcs_members:
            print("\n  no .fcs member found")
            if not quiet:
                print("  entries:")
                for info in infos[:20]:
                    print("    %-58s %s" % (info.filename[:58], human_bytes(info.file_size)))
            return []

        print()
        for info in fcs_members[:max_fcs]:
            print("  -- %s  (%s) --" % (info.filename, human_bytes(info.file_size)))
            try:
                keywords, params = describe_fcs(zf, info)
            except ValueError as exc:
                print("     could not read TEXT segment: %s" % exc)
                continue
            print("     %s" % fcs_probe.summarise(keywords))
            for key in ("$CYT", "$DATE", "$BTIM", "$SRC", "$FIL"):
                if keywords.get(key):
                    print("     %-8s %s" % (key, keywords[key]))
            print("     parameters:")
            for row in params:
                label = row["label"] or ""
                print("       P%-3d %-14s %s" % (row["n"], row["name"], label))
            print()
        if len(fcs_members) > max_fcs:
            print("  ... and %d more FCS members" % (len(fcs_members) - max_fcs))
        return fcs_members


def extract(path, dest, keep_metadata):
    """Write out the FCS members (and optionally the TOC/metadata), nothing else."""
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(dest, stem)
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    total = 0
    with open_archive(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            kind = categorise(info.filename)
            if kind == "fcs" or (keep_metadata and kind == "metadata"):
                target = os.path.join(out_dir, posixpath.basename(info.filename))
                with zf.open(info.filename) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                written += 1
                total += info.file_size
    print("  extracted %d entries (%s) to %s" % (written, human_bytes(total), out_dir))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="+", help="An .acs file, several, or a directory holding them")
    ap.add_argument("--extract-fcs", metavar="DIR", help="Write the FCS members into DIR/<archive-name>/")
    ap.add_argument("--no-metadata", action="store_false", dest="keep_metadata",
                    help="Extract only .fcs, not the TOC/XML alongside it")
    ap.add_argument("--max-fcs", type=int, default=4, help="FCS members to describe per archive (default: 4)")
    ap.add_argument("--quiet", action="store_true", help="Skip the entry listing when no FCS is found")
    args = ap.parse_args(argv)

    paths = []
    for item in args.source:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.acs"))))
        else:
            paths.append(item)
    if not paths:
        ap.error("No .acs files found under: %s" % ", ".join(args.source))

    failures = 0
    for path in paths:
        try:
            inspect(path, args.max_fcs, args.quiet)
            if args.extract_fcs:
                extract(path, args.extract_fcs, args.keep_metadata)
        except (IOError, OSError) as exc:
            print("error: %s" % exc)
            failures += 1
        print()
    return 1 if failures == len(paths) else 0


if __name__ == "__main__":
    sys.exit(run(main))
