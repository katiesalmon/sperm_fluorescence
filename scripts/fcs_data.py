"""Read the DATA segment of an FCS file with the standard library only.

fcs_probe.py reads the TEXT segment, which says what the columns are. This reads the
numbers. Together they turn an FCS into the target table: one row per event, one column
per detector and per instrument-computed feature.

Narrow on purpose -- list mode, one bit width across all parameters, float or integer --
which covers what the Attune writes. Anything else raises Unsupported rather than
returning plausible nonsense.
"""

import array
import sys


class Unsupported(ValueError):
    pass


def data_offsets(data, keywords):
    """Where the DATA segment starts and ends.

    The HEADER's offsets are eight ASCII characters wide, so a file over ~100 MB writes
    zeros there and puts the real offsets in $BEGINDATA / $ENDDATA. Prefer the keywords
    when they disagree with a zeroed header.
    """
    start = end = 0
    if len(data) >= 42:
        try:
            start = int(data[26:34])
            end = int(data[34:42])
        except ValueError:
            start = end = 0

    for key, setter in (("$BEGINDATA", "start"), ("$ENDDATA", "end")):
        raw = keywords.get(key, "").strip()
        if raw.isdigit():
            value = int(raw)
            if setter == "start" and (start == 0 or value):
                start = value
            elif setter == "end" and (end == 0 or value):
                end = value

    if start <= 0 or end <= start:
        raise Unsupported("could not locate the DATA segment (start=%s end=%s)" % (start, end))
    return start, end


def _typecode(keywords, par):
    datatype = keywords.get("$DATATYPE", "F").upper()
    widths = set()
    for i in range(1, par + 1):
        raw = keywords.get("$P%dB" % i, "")
        if not raw.isdigit():
            raise Unsupported("$P%dB missing or non-numeric" % i)
        widths.add(int(raw))
    if len(widths) != 1:
        raise Unsupported("mixed parameter bit widths: %s" % sorted(widths))
    bits = widths.pop()

    if datatype == "F" and bits == 32:
        return "f", bits
    if datatype == "D" and bits == 64:
        return "d", bits
    if datatype == "I" and bits in (16, 32):
        return ("H" if bits == 16 else "I"), bits
    raise Unsupported("$DATATYPE %s at %d bits is not supported" % (datatype, bits))


def read_matrix(data, keywords, max_events=None):
    """Decode the event matrix. Returns (values, n_parameters, n_events).

    `values` is a flat array; event i parameter j is values[i * n_parameters + j].
    Parameters are 1-indexed in FCS keywords and 0-indexed here.
    """
    par = int(keywords.get("$PAR", "0") or 0)
    total = int(keywords.get("$TOT", "0") or 0)
    if par <= 0 or total <= 0:
        raise Unsupported("$PAR=%s $TOT=%s -- cannot size the matrix" % (par, total))
    if keywords.get("$MODE", "L").upper() != "L":
        raise Unsupported("only list mode ($MODE L) is supported, got %s" % keywords.get("$MODE"))

    typecode, bits = _typecode(keywords, par)
    width = bits // 8

    start, end = data_offsets(data, keywords)
    available = (end - start + 1) // (width * par)
    n = min(total, available) if max_events is None else min(total, available, max_events)
    if n <= 0:
        raise Unsupported("DATA segment holds no complete events")

    values = array.array(typecode)
    values.frombytes(bytes(data[start : start + n * par * width]))

    little = keywords.get("$BYTEORD", "1,2,3,4").strip().startswith("1")
    if little != (sys.byteorder == "little"):
        values.byteswap()
    return values, par, n


def column_index(parameters, name):
    """Index of a parameter by detector name ($PnN) or label ($PnS), case-insensitive."""
    wanted = name.strip().lower()
    for row in parameters:
        if row["name"].strip().lower() == wanted:
            return row["n"] - 1
    for row in parameters:
        if row["label"].strip().lower() == wanted:
            return row["n"] - 1
    return None
