"""Shared helpers for reading the CytPix .acs archives.

Standard library only, Python 3.8+, so everything under scripts/ runs on the imaging
server with no install.

This module used to carry a lot more: machinery for locating the `Images\\*.zip` export,
grouping files into events, and recognising per-channel filename tokens. All of that was
built before the exports were surveyed, against a guess that a marker replicate would
store one image file per channel per event. The data said otherwise -- one file per event,
flat, with the marker signal in a separate FCS -- so that code is gone. It is in git
history if the guess ever turns out to be right for a future export.
"""

import os
import re
import zipfile

# Archives are named like  <prefix>_2SP.acs  /  <prefix>_2SP.zip. We match the trailing
# class token so a different date or prefix still resolves.
CLASS_RE = re.compile(r"_(?P<cls>[123](?:S|P|SP))\.(?:acs|zip)$", re.IGNORECASE)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def run(main, argv=None):
    """Entry-point wrapper: report expected failures as one clear line, not a traceback."""
    import sys

    try:
        return main(argv)
    except (ValueError, IOError, OSError) as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130


def class_of(path):
    """Return the class token ('2S', '2P', '2SP', ...) for an archive path, or None."""
    match = CLASS_RE.search(os.path.basename(path))
    return match.group("cls").upper() if match else None


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0


def open_zip(path):
    """Open a zip with a clearer error than zipfile's default for a bad or partial file."""
    if not os.path.isfile(path):
        raise IOError("Not a file: %s" % path)
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise IOError(
            "Could not read %s as a zip: %s\n"
            "A zip's index sits at the end of the file, so a partial copy fails here. "
            "Check the file is the size the server reports." % (path, exc)
        )
