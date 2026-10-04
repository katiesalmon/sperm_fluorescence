"""Read the per-object features IDEAS stores in a .daf.

A .daf is an XML header (populations, gates, the UDF feature list) followed by a binary
block: a short header, then one column per feature -- an int32 feature index and
`objcount` float64 values. Features come out in the order of the <UDF> elements.
"""

import re
import struct

import numpy as np


def read_features(path):
    """Returns (names, values) with values of shape (objects, features), float64."""
    raw = open(path, "rb").read()
    end = raw.find(b"</Assay>") + 8
    xml = raw[:end].decode("utf-8", "replace")
    names = [n for n, _ in re.findall(r'<UDF name="([^"]+)" type="(\w+)"', xml)]
    n_obj = int(re.search(r'objcount="(\d+)"', xml).group(1))
    blk = raw[end:]
    n_feat = struct.unpack_from("<i", blk, 7)[0]
    assert n_feat == len(names), (n_feat, len(names))
    assert struct.unpack_from("<i", blk, 11)[0] == n_obj
    off = 15
    cols = []
    for k in range(n_feat):
        idx = struct.unpack_from("<i", blk, off)[0]
        if idx != k:
            raise ValueError("feature %d has index %d at offset %d" % (k, idx, off))
        cols.append(np.frombuffer(blk, "<f8", n_obj, off + 4))
        off += 4 + 8 * n_obj
    return names, np.stack(cols, axis=1), xml
