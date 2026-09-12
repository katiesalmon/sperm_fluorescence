"""Shared helpers for locating and reading the CytPix export zips.

Standard library only, Python 3.8+, so it runs anywhere with no installs.

This is the sperm_pbmc module extended for the marker replicates. The difference that
matters: in replicate 1 one file was one event, so sampling files and sampling events
were the same thing. Here an event may span several files -- one per channel -- and a
sample that splits an event across the boundary is useless for learning the mapping
between them. So everything below is written in terms of *events*, and the file-per-
event case is just the degenerate one.
"""

import collections
import os
import re
import zipfile

# Export files are named like
#   <some_export_prefix>_2SP.zip
# We match on the trailing class token so a different date/prefix still resolves.
CLASS_RE = re.compile(r"_(?P<cls>[123](?:S|P|SP))\.zip$", re.IGNORECASE)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Sidecars worth carrying into a sample. If the marker signal turns out to live in the
# cytometer's per-event measurements rather than in images, it arrives in one of these.
METADATA_EXTS = {".fcs", ".csv", ".tsv", ".xml", ".txt", ".json", ".ome"}

# Replicate 1 is brightfield only. 2 and 3 carry the fluorescent markers and are what
# this repo is about.
DEFAULT_CLASSES = ["2S", "2P", "2SP", "3S", "3P", "3SP"]

# The antigen markers under study, plus the labels a brightfield channel tends to carry.
# Used only to *recognise* a channel token in a filename -- the real vocabulary comes
# from the survey, and --channel-regex overrides all of this when the export disagrees.
MARKERS = ["DAPI", "ACRV1", "LDHC", "TOMM20"]
BRIGHTFIELD_TOKENS = ["BF", "BRIGHTFIELD", "BRIGHT", "TRANS", "PHASE", "DIC"]

# Channel tokens as they show up in the wild: a marker name, a brightfield name, or a
# positional index (Ch1, C2, ch_3). Anchored, so it only ever matches a whole token.
_CHANNEL_TOKEN_RE = re.compile(
    r"^(?:(?P<named>%s)|(?:ch|c)[_-]?(?P<index>\d{1,2}))$"
    % "|".join(MARKERS + BRIGHTFIELD_TOKENS),
    re.IGNORECASE,
)

# A channel token is joined onto an event id by one of these: <event>_DAPI, <event>-Ch2.
# The token itself may contain one too (ch_2), so candidate splits are tried right to
# left and the first one that yields a *recognised* token wins.
_SEPARATORS = "._-"


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
    """Return the class token ('2S', '2P', '2SP', ...) for a zip path, or None."""
    match = CLASS_RE.search(os.path.basename(path))
    return match.group("cls").upper() if match else None


def find_zips(source, classes=None):
    """Resolve `source` (a directory or a list of zip paths) to {class: zip_path}.

    Raises if a requested class is missing or ambiguous, so a typo fails loudly
    instead of silently producing a sample with two classes in it.
    """
    if isinstance(source, str):
        source = [source]

    candidates = []
    for item in source:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith(".zip"):
                    candidates.append(os.path.join(item, name))
        else:
            candidates.append(item)

    found = {}
    for path in candidates:
        cls = class_of(path)
        if cls is None:
            continue
        if cls in found:
            raise ValueError(
                "Two zips resolve to class %s:\n  %s\n  %s" % (cls, found[cls], path)
            )
        found[cls] = path

    if classes is None:
        return found

    wanted = {}
    for cls in classes:
        cls = cls.upper()
        if cls not in found:
            raise ValueError(
                "No zip found for class %s. Saw: %s"
                % (cls, ", ".join(sorted(found)) or "(none)")
            )
        wanted[cls] = found[cls]
    return wanted


def image_members(zf):
    """Image entries in an open ZipFile, sorted by name for deterministic sampling."""
    return _members_with_ext(zf, IMAGE_EXTS)


def metadata_members(zf, max_bytes=None):
    """Non-image entries -- the FCS / CSV / XML sidecars, if the export ships any."""
    members = [
        info
        for info in zf.infolist()
        if not info.is_dir()
        and os.path.splitext(info.filename)[1].lower() not in IMAGE_EXTS
        and not os.path.basename(info.filename).startswith("._")
        and "__MACOSX/" not in info.filename
    ]
    if max_bytes is not None:
        members = [i for i in members if i.file_size <= max_bytes]
    members.sort(key=lambda info: info.filename)
    return members


def _members_with_ext(zf, exts):
    members = [
        info
        for info in zf.infolist()
        if not info.is_dir()
        and os.path.splitext(info.filename)[1].lower() in exts
        # Skip macOS resource-fork junk that sneaks into zips made on a Mac.
        and not os.path.basename(info.filename).startswith("._")
        and "__MACOSX/" not in info.filename
    ]
    members.sort(key=lambda info: info.filename)
    return members


def split_channel(name):
    """Split an image path into (event_key, channel_token).

    Handles the three layouts an export plausibly uses:

        <event>.tif                 -> ('<event>', None)          one file per event
        <event>_DAPI.tif            -> ('<event>', 'DAPI')        channel in the filename
        DAPI/<event>.tif            -> ('<event>', 'DAPI')        channel as a folder

    `channel_token` is None when nothing channel-shaped is recognised, which is also the
    correct answer for a multi-channel TIF -- there the channels are inside the file, and
    the file already is the event. Anything the auto-detection gets wrong is overridden
    with --channel-regex once the survey has shown the real naming.
    """
    directory, base = os.path.split(name.replace("\\", "/"))
    stem = os.path.splitext(base)[0]

    parts = [p for p in directory.split("/") if p]
    if parts and _CHANNEL_TOKEN_RE.match(parts[-1]):
        rest = "/".join(parts[:-1])
        return (posix_join(rest, stem), canonical_channel(parts[-1]))

    for i in range(len(stem) - 1, 0, -1):
        if stem[i] not in _SEPARATORS:
            continue
        event, token = stem[:i], stem[i + 1 :]
        if token and _CHANNEL_TOKEN_RE.match(token):
            return (posix_join(directory, event), canonical_channel(token))

    return (posix_join(directory, stem), None)


def canonical_channel(token):
    """Normalise a channel token so DAPI/dapi/Dapi and Ch2/ch_2/C2 each collapse to one."""
    match = _CHANNEL_TOKEN_RE.match(token)
    if match is None:
        return token
    if match.group("named"):
        named = match.group("named").upper()
        # Tomm20 is a protein, not an acronym; keep the conventional casing.
        return "Tomm20" if named == "TOMM20" else named
    return "Ch%d" % int(match.group("index"))


def posix_join(directory, leaf):
    return "%s/%s" % (directory, leaf) if directory else leaf


def group_events(members, channel_regex=None):
    """Group image members into {event_key: [ZipInfo, ...]}, in sorted event order.

    `channel_regex` is an optional pattern with a named group `event` and, optionally,
    `channel`; it replaces the auto-detection when the export names things unexpectedly.
    """
    compiled = re.compile(channel_regex) if channel_regex else None

    events = collections.OrderedDict()
    for info in members:
        if compiled is None:
            key, _channel = split_channel(info.filename)
        else:
            match = compiled.search(info.filename)
            if match is None:
                raise ValueError(
                    "--channel-regex did not match %s\n"
                    "Every image entry has to match, or events would be silently dropped."
                    % info.filename
                )
            key = match.group("event")
        events.setdefault(key, []).append(info)

    for files in events.values():
        files.sort(key=lambda info: info.filename)
    return collections.OrderedDict(sorted(events.items()))


def channels_of(files, channel_regex=None):
    """The channel tokens present for one event's files, in filename order."""
    compiled = re.compile(channel_regex) if channel_regex else None
    tokens = []
    for info in files:
        if compiled is None:
            _key, channel = split_channel(info.filename)
        else:
            match = compiled.search(info.filename)
            channel = match.groupdict().get("channel") if match else None
        tokens.append(channel)
    return tokens


def grouping_summary(events, channel_regex=None):
    """Describe how the grouping came out, for the survey to print.

    Returns files-per-event counts and the channel token sets seen, which together say
    whether the export is one file per event or one file per channel per event.
    """
    per_event = collections.Counter(len(files) for files in events.values())
    signatures = collections.Counter()
    for files in events.values():
        tokens = channels_of(files, channel_regex)
        signatures["+".join(t or "(none)" for t in tokens)] += 1
    return {
        "events": len(events),
        "files_per_event": dict(sorted(per_event.items())),
        "channel_signatures": dict(signatures.most_common(10)),
    }


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0


def open_zip(path):
    """Open a zip with a clearer error than zipfile's default for a bad path."""
    if not os.path.isfile(path):
        raise IOError("Not a file: %s" % path)
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise IOError("Could not read %s as a zip: %s" % (path, exc))
