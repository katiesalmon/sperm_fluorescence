"""Read Amnis ImageStream .cif / .rif files directly: no IDEAS, no Bio-Formats, no C++.

The container is a TIFF with one page per object: every in-use channel tiled side by side
in a single strip, a uint16 image page (compression 30817) followed by a uint8 mask page
(30818) of the same size. Both codecs are documented by the open-source FlowSight reader
(Bio-Formats; ported to C++ in nmichiels/cifDataset) and are simple enough to vectorise:

  30817  a stream of nibbles, low nibble of each byte first. Each value is built from
         3-bit groups (nibble & 7) at increasing shifts; nibble bit 3 set means "more
         groups follow"; bit 2 of the final group is the sign, extended upward. The first
         value is discarded. Values are 2-D prediction residuals:
             this[x] = d + last[x] + this[x-1] - last[x-1]   (x > 0)
             this[0] = d + last[0]
  30818  pairs of bytes (value, run_length - 1), filling the flattened image in order.

All 12 channels are written, used or not: slot k is Ch(k+1). Verified on ALL_1_400_1 by
rendering objects 0, 2 and 6 and matching them to the IDEAS gallery (Ch02 ring, Ch03
ring-plus-spot, Ch07 blob, Ch11 spot). The nominally unused channels (Ch04-06, 08, 10,
12) carry spillover and -- in Ch06 -- side scatter, and should not be assumed empty.
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import tiff_probe  # noqa: E402

GREYSCALE, BITMASK = 30817, 30818


def decode_nibbles(buf, count):
    """Decode `count` signed values from the 30817 nibble stream in `buf` (bytes)."""
    b = np.frombuffer(buf, dtype=np.uint8)
    nib = np.empty(2 * len(b), dtype=np.uint8)
    nib[0::2] = b & 0x0F          # low nibble first
    nib[1::2] = b >> 4
    cont = (nib & 0x8) != 0
    # group boundaries: a value ends at a nibble whose continuation bit is clear
    ends = np.flatnonzero(~cont)
    if len(ends) < count:
        raise ValueError("nibble stream holds %d values, need %d" % (len(ends), count))
    ends = ends[:count]
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1
    pos = np.arange(ends[-1] + 1) - np.repeat(starts, lengths)   # position within group
    contrib = (nib[: ends[-1] + 1] & 0x7).astype(np.int64) << (3 * pos)
    vals = np.add.reduceat(contrib, starts)
    neg = (nib[ends] & 0x4) != 0
    vals[neg] -= 1 << (3 * lengths[neg])
    return vals


def decode_greyscale(data, page, nchannels):
    """(nchannels, h, w) int16 for one image page."""
    strips = page["_strips"]
    buf = b"".join(data[o:o + c] for o, c in zip(strips["offsets"], strips["byte_counts"]))
    W, H = page["width"], page["height"]
    # The C++ port discards a leading value; this IDEAS version writes none -- the stream
    # holds exactly W*H residuals and the first is the top-left pixel.
    d = decode_nibbles(buf, W * H).reshape(H, W)
    out = np.zeros((H, W), dtype=np.int64)
    last = np.zeros(W, dtype=np.int64)
    for y in range(H):
        # this[x] = this[x-1] + (d[x] + last[x] - last[x-1]);  this[0] = d[0] + last[0]
        step = d[y].copy()
        step[1:] += last[1:] - last[:-1]
        step[0] += last[0]
        row = np.cumsum(step)
        out[y] = row
        last = row
    # Signed: a compensated .cif holds slightly negative pixels in two's complement, and
    # reading them as uint16 (~65500) overflows float16 downstream. Raw .rif data is
    # non-negative either way.
    out = (out & 0xFFFF).astype(np.uint16).view(np.int16)
    return np.stack(np.hsplit(out, nchannels))


def decode_mask(data, page, nchannels):
    """(nchannels, h, w) uint8 for one mask page."""
    strips = page["_strips"]
    buf = b"".join(data[o:o + c] for o, c in zip(strips["offsets"], strips["byte_counts"]))
    b = np.frombuffer(buf, dtype=np.uint8)
    values, runs = b[0::2], b[1::2].astype(np.int64) + 1
    flat = np.repeat(values, runs)
    W, H = page["width"], page["height"]
    if len(flat) != W * H:
        raise ValueError("mask RLE expands to %d, expected %d" % (len(flat), W * H))
    return np.stack(np.hsplit(flat.reshape(H, W), nchannels))


class CIF:
    """An opened .cif/.rif: objects as (image, mask) page pairs."""

    def __init__(self, path, nchannels=None, channel_names=None):
        with open(path, "rb") as fh:
            self.data = fh.read()
        info = tiff_probe.probe(self.data, max_pages=10 ** 7)
        pages = [p for p in info["pages"] if (p.get("width") or 0) > 1]
        self.images = [p for p in pages if p["_strips"]["compression_code"] == GREYSCALE]
        self.masks = [p for p in pages if p["_strips"]["compression_code"] == BITMASK]
        self.xml = self._xml()
        self.nchannels = nchannels or self._infer_channels()
        self.channel_names = channel_names or ["ch%d" % i for i in range(self.nchannels)]

    def _xml(self):
        m = re.search(rb"<\?xml.*?</rifAssayDB>", self.data[:8_000_000], re.S)
        return m.group(0).decode("utf-8", "replace") if m else ""

    def _infer_channels(self):
        # An ImageStream X Mk II always writes all 12 channels, used or not -- the
        # "unused" ones carry spillover and scatter. Verified by eye on this file: a
        # 6-way split showed two cells per slot. Fall back only if 12 does not divide.
        widths = np.array([p["width"] for p in self.images])
        for n in (12, 6, 4, 3, 2, 1):
            if np.all(widths % n == 0):
                return n
        return 1

    def __len__(self):
        return len(self.images)

    def image(self, i):
        return decode_greyscale(self.data, self.images[i], self.nchannels)

    def mask(self, i):
        return decode_mask(self.data, self.masks[i], self.nchannels)

    def shape(self, i):
        p = self.images[i]
        return self.nchannels, p["height"], p["width"] // self.nchannels


ISX_CHANNELS = ["Ch%02d" % i for i in range(1, 13)]
# '?' = the PE / PerCP-eF710 assignment is unconfirmed between the template and the gates;
# see docs/task_brief.md. Ch04/05/08/10/12 hold spillover only; Ch06 is side scatter.
ISX_MARKERS = {"Ch01": "BF", "Ch02": "LDHC_AF488", "Ch03": "ACRV1_PE?", "Ch06": "SSC",
               "Ch07": "DAPI", "Ch09": "BF2", "Ch11": "TOMM20_PerCP?"}
LABEL_FREE = [0, 8, 5]            # Ch01, Ch09, Ch06
TARGETS = [1, 2, 6, 10]           # Ch02, Ch03, Ch07, Ch11
