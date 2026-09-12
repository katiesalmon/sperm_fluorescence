"""Read the TEXT segment of an FCS file with the standard library only.

If the marker signal turns out to be a cytometer measurement rather than an image, the
FCS sidecar is the label source, and its TEXT segment says so directly: $PAR parameters,
each with a detector name ($PnN, e.g. VL1-A) and a human label ($PnS, e.g. DAPI). That
mapping -- detector to antigen -- is the first thing we need and the hardest to guess.

Only the HEADER and TEXT segments are read. The event data is left alone.
"""

DEFAULT_MAX_KEYWORDS = 400


def read_text_segment(data, max_keywords=DEFAULT_MAX_KEYWORDS):
    """Parse an FCS HEADER + TEXT segment from `data` (bytes). Returns a keyword dict.

    Raises ValueError if this does not look like an FCS file.
    """
    if len(data) < 58 or not data[:3] == b"FCS":
        raise ValueError("not an FCS file (magic was %r)" % data[:6])

    version = data[:6].decode("ascii", "replace").strip()
    try:
        text_start = int(data[10:18])
        text_end = int(data[18:26])
    except ValueError:
        raise ValueError("unreadable FCS header offsets")

    if text_start <= 0 or text_end <= text_start:
        raise ValueError("FCS header gives an empty TEXT segment")
    if text_end >= len(data):
        raise ValueError(
            "TEXT segment ends at byte %d but only %d bytes were read" % (text_end, len(data))
        )

    segment = data[text_start : text_end + 1]
    delimiter = segment[:1]
    body = segment[1:]

    # A doubled delimiter is an escaped literal one. Rare; handled so it cannot
    # silently split a value in half.
    fields = body.replace(delimiter * 2, b"\x00ESC\x00").split(delimiter)
    fields = [f.replace(b"\x00ESC\x00", delimiter) for f in fields]

    keywords = {"$FCSVERSION": version}
    for i in range(0, len(fields) - 1, 2):
        key = fields[i].decode("utf-8", "replace").strip()
        if not key:
            continue
        keywords[key.upper()] = fields[i + 1].decode("utf-8", "replace").strip()
        if len(keywords) >= max_keywords:
            break
    return keywords


def parameters(keywords):
    """The per-parameter table: [{'n', 'name' ($PnN), 'label' ($PnS), 'range' ($PnR)}, ...].

    $PnN is the detector (VL1-A, BL2-H); $PnS is whatever the operator typed when setting
    up the run, which is where an antigen name like ACRV1 would appear.
    """
    try:
        count = int(keywords.get("$PAR", "0"))
    except ValueError:
        count = 0
    rows = []
    for n in range(1, count + 1):
        rows.append(
            {
                "n": n,
                "name": keywords.get("$P%dN" % n, ""),
                "label": keywords.get("$P%dS" % n, ""),
                "range": keywords.get("$P%dR" % n, ""),
            }
        )
    return rows


def summarise(keywords):
    """One line: version, event count, parameter count."""
    return "%s  $TOT=%s events  $PAR=%s parameters" % (
        keywords.get("$FCSVERSION", "?"),
        keywords.get("$TOT", "?"),
        keywords.get("$PAR", "?"),
    )
