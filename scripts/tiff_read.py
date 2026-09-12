"""Decode the pixels of a CytPix export TIF using only the standard library.

The export is LZW-compressed, which `tifffile` cannot decode without `imagecodecs`.
That is fine on a laptop and awkward on the imaging server, where the question
"do the RGBA channels actually differ from each other" needs answering *before*
anyone decides whether installing a scientific Python stack there is worth it.

So: a small TIFF LZW decoder, per the TIFF 6.0 spec, including the horizontal
differencing predictor. Deliberately narrow -- 8-bit, chunky (planar=1), LZW or
uncompressed. Anything else raises Unsupported rather than returning wrong pixels.

For real work use tifffile + imagecodecs; this exists so one specific question can
be answered with no install.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import tiff_probe

CLEAR_CODE = 256
EOI_CODE = 257
FIRST_CODE = 258


class Unsupported(ValueError):
    pass


def lzw_decode(data):
    """TIFF-flavour LZW. Returns a bytearray."""
    out = bytearray()
    table = None
    next_code = FIRST_CODE
    width = 9
    previous = None

    bitpos = 0
    nbits = len(data) * 8

    def take():
        nonlocal bitpos
        if bitpos + width > nbits:
            return EOI_CODE
        value = 0
        for _ in range(width):
            byte = data[bitpos >> 3]
            value = (value << 1) | ((byte >> (7 - (bitpos & 7))) & 1)
            bitpos += 1
        return value

    while True:
        code = take()
        if code == EOI_CODE:
            break
        if code == CLEAR_CODE:
            table = {i: bytes([i]) for i in range(256)}
            next_code = FIRST_CODE
            width = 9
            previous = None
            continue
        if table is None:
            # A well-formed stream opens with a clear code. Tolerate one that does not.
            table = {i: bytes([i]) for i in range(256)}
            next_code = FIRST_CODE
            width = 9

        if code in table:
            entry = table[code]
        elif code == next_code and previous is not None:
            entry = previous + previous[:1]
        else:
            raise Unsupported("bad LZW code %d (next would be %d)" % (code, next_code))

        out += entry
        if previous is not None:
            table[next_code] = previous + entry[:1]
            next_code += 1
            # TIFF bumps the code width one code early -- at 511, not 512.
            if next_code == 511:
                width = 10
            elif next_code == 1023:
                width = 11
            elif next_code == 2047:
                width = 12
        previous = entry

    return out


def unpredict(rows, width, spp):
    """Undo horizontal differencing (Predictor 2), in place, per row."""
    stride = width * spp
    for start in range(0, len(rows), stride):
        for i in range(start + spp, start + stride):
            rows[i] = (rows[i] + rows[i - spp]) & 0xFF
    return rows


def read_page(data, page_index=0):
    """Decode one page. Returns (pixels, width, height, samples_per_pixel).

    `pixels` is a flat bytearray in row-major, interleaved order:
    pixel (x, y) sample s is at ((y * width) + x) * spp + s.
    """
    info = tiff_probe.probe(data)
    if page_index >= len(info["pages"]):
        raise Unsupported("page %d requested, file has %d" % (page_index, len(info["pages"])))
    page = info["pages"][page_index]

    bits = page.get("bits_per_sample")
    if isinstance(bits, list):
        if len(set(bits)) != 1:
            raise Unsupported("mixed bit depths per sample: %s" % bits)
        bits = bits[0]
    if bits != 8:
        raise Unsupported("only 8-bit samples are supported, got %s" % bits)

    strips = page["_strips"]
    if strips["planar"] != 1:
        raise Unsupported("only chunky (planar=1) data is supported")

    code = strips["compression_code"]
    if code == 1:
        decode = lambda chunk: bytearray(chunk)  # noqa: E731
    elif code == 5:
        decode = lzw_decode
    else:
        raise Unsupported("compression %s is not supported" % page.get("compression"))

    width, height, spp = page["width"], page["height"], page["samples_per_pixel"]
    offsets, counts = strips["offsets"], strips["byte_counts"]
    if not offsets or len(offsets) != len(counts):
        raise Unsupported("strip offsets and byte counts disagree")

    rows_per_strip = strips["rows_per_strip"] or height
    pixels = bytearray()
    for i, (offset, count) in enumerate(zip(offsets, counts)):
        chunk = decode(data[offset : offset + count])
        if strips["predictor"] == 2:
            rows_here = min(rows_per_strip, height - i * rows_per_strip)
            chunk = unpredict(chunk, width, spp)
            del rows_here
        elif strips["predictor"] not in (1, None):
            raise Unsupported("predictor %s is not supported" % strips["predictor"])
        pixels += chunk

    expected = width * height * spp
    if len(pixels) < expected:
        raise Unsupported("decoded %d bytes, expected %d" % (len(pixels), expected))
    return pixels[:expected], width, height, spp


def channel(pixels, spp, index):
    """One channel as a flat bytes object, in pixel order."""
    return bytes(pixels[index::spp])
