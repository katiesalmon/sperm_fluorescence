"""Read the TEXT segment of an FCS file with the standard library only.

If the marker signal turns out to be a cytometer measurement rather than an image, the
FCS sidecar is the label source, and its TEXT segment says so directly: $PAR parameters,
each with a detector name ($PnN, e.g. VL1-A) and a human label ($PnS, e.g. DAPI). That
mapping -- detector to antigen -- is the first thing we need and the hardest to guess.

Only the HEADER and TEXT segments are read. The event data is left alone.
"""

# A guard against a malformed segment, not a budget. The TEXT segment is bounded by the
# header offsets, so it cannot grow without limit -- and capping it low is a real hazard:
# writers emit keywords alphabetically, so $P1B ... $P99V all sort *before* $PAR and
# $TOT. A cap of a few hundred truncates mid-parameter-block and silently loses the two
# keywords that matter most. Hitting this limit is reported, never absorbed.
DEFAULT_MAX_KEYWORDS = 100000


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
    truncated = False
    for i in range(0, len(fields) - 1, 2):
        key = fields[i].decode("utf-8", "replace").strip()
        if not key:
            continue
        keywords[key.upper()] = fields[i + 1].decode("utf-8", "replace").strip()
        if len(keywords) >= max_keywords:
            truncated = True
            break
    if truncated:
        keywords["$TRUNCATED"] = "yes -- stopped at %d keywords" % max_keywords
    return keywords


def parameters(keywords):
    """The per-parameter table, one row per detector.

    $PnN is the detector (VL1-A, BL2-H); $PnS is whatever the operator typed when setting
    up the run, which is where an antigen name like ACRV1 would appear. $PnV is the PMT
    voltage -- the acquisition gain, which is worth reading directly: if it differs
    between replicates, the fluorescence targets are not on a common scale and the
    held-out-replicate evaluation has to account for it.
    """
    count = _int(keywords.get("$PAR"))
    if count is None:
        # $PAR missing (or truncated away) -- fall back to counting $PnN keywords.
        count = 0
        while "$P%dN" % (count + 1) in keywords:
            count += 1

    rows = []
    for n in range(1, count + 1):
        rows.append(
            {
                "n": n,
                "name": keywords.get("$P%dN" % n, ""),
                "label": keywords.get("$P%dS" % n, ""),
                "range": keywords.get("$P%dR" % n, ""),
                "voltage": keywords.get("$P%dV" % n, ""),
                "gain": keywords.get("$P%dG" % n, ""),
            }
        )
    return rows


def _int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def spillover(keywords):
    """Describe the compensation matrix, if the file carries one.

    Spectral spillover between detectors puts correlated signal into channels that should
    be independent. A per-marker model would learn that correlation and we would read it
    as biology, so whether the exported values are compensated is a question to settle
    before training, not after.
    """
    for key in ("$SPILLOVER", "SPILL", "$COMP"):
        raw = keywords.get(key)
        if not raw:
            continue
        head = [h.strip() for h in raw.split(",")]
        n = _int(head[0]) if head else None
        if not n:
            return {"keyword": key, "detectors": [], "size": None, "matrix": []}
        detectors = head[1 : 1 + n]
        flat = head[1 + n : 1 + n + n * n]
        matrix = []
        for row in range(n):
            values = flat[row * n : (row + 1) * n]
            if len(values) != n:
                matrix = []  # incomplete; report the names but not a half matrix
                break
            matrix.append([_float(v) for v in values])
        return {"keyword": key, "detectors": detectors, "size": n, "matrix": matrix}
    return None


def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def summarise(keywords):
    """One line: version, event count, parameter count."""
    line = "%s  $TOT=%s events  $PAR=%s parameters" % (
        keywords.get("$FCSVERSION", "?"),
        keywords.get("$TOT", "?"),
        keywords.get("$PAR", "?"),
    )
    if keywords.get("$TRUNCATED"):
        line += "   [TEXT PARSE TRUNCATED: %s]" % keywords["$TRUNCATED"]
    return line
