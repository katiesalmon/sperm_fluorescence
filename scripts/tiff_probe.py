"""Read a TIFF's structure without decoding any pixels, using only the standard library.

The survey has to answer "how many channels are in this file, and what are they called"
before we can decide how to sample -- and it has to answer it on the imaging server,
where installing packages is a whole conversation. Everything needed is in the header
and the IFD chain: page count, dimensions, bit depth, samples per pixel, compression,
and the ImageDescription tag, which is where OME-TIFF and ImageJ record channel names.

No pixel data is touched, so LZW (which tifffile cannot decode without imagecodecs)
is irrelevant here.
"""

import struct

# Only the tags that tell us something about how the image is laid out.
TAGS = {
    254: "NewSubfileType",
    256: "ImageWidth",
    257: "ImageLength",
    258: "BitsPerSample",
    259: "Compression",
    262: "PhotometricInterpretation",
    270: "ImageDescription",
    277: "SamplesPerPixel",
    285: "PageName",
    339: "SampleFormat",
}

COMPRESSION = {1: "none", 5: "LZW", 7: "JPEG", 8: "deflate", 32773: "PackBits"}
PHOTOMETRIC = {0: "min-is-white", 1: "min-is-black", 2: "RGB", 3: "palette", 4: "mask"}
SAMPLE_FORMAT = {1: "uint", 2: "int", 3: "float"}

# type code -> (struct char, byte width)
TYPES = {
    1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 5: ("I", 4),
    6: ("b", 1), 7: ("B", 1), 8: ("h", 2), 9: ("i", 4), 10: ("i", 4),
    11: ("f", 4), 12: ("d", 8), 13: ("I", 4), 16: ("Q", 8), 17: ("q", 8), 18: ("Q", 8),
}

MAX_PAGES = 512  # a guard against a malformed or circular IFD chain


class NotTiff(ValueError):
    pass


def probe(data, max_description=400):
    """Describe the TIFF in `data` (bytes). Raises NotTiff if it is not one.

    Returns {'byte_order', 'bigtiff', 'pages': [ {...}, ... ]}, one dict per page.
    """
    if len(data) < 8:
        raise NotTiff("too short to be a TIFF (%d bytes)" % len(data))

    magic = data[:4]
    if magic[:2] == b"II":
        endian = "<"
    elif magic[:2] == b"MM":
        endian = ">"
    else:
        raise NotTiff("no TIFF byte-order mark (got %s)" % data[:4].hex())

    version = struct.unpack(endian + "H", data[2:4])[0]
    if version == 42:
        bigtiff = False
        offset = struct.unpack(endian + "I", data[4:8])[0]
    elif version == 43:
        bigtiff = True
        offsize, pad = struct.unpack(endian + "HH", data[4:8])
        if offsize != 8 or pad != 0:
            raise NotTiff("unsupported BigTIFF offset size %d" % offsize)
        offset = struct.unpack(endian + "Q", data[8:16])[0]
    else:
        raise NotTiff("unknown TIFF version %d" % version)

    pages = []
    seen = set()
    while offset and offset not in seen and len(pages) < MAX_PAGES:
        seen.add(offset)
        page, offset = _read_ifd(data, offset, endian, bigtiff, max_description)
        pages.append(page)

    return {"byte_order": "little" if endian == "<" else "big", "bigtiff": bigtiff, "pages": pages}


def _read_ifd(data, offset, endian, bigtiff, max_description):
    if bigtiff:
        count = struct.unpack_from(endian + "Q", data, offset)[0]
        entry_size, entries_at, inline_max = 20, offset + 8, 8
    else:
        count = struct.unpack_from(endian + "H", data, offset)[0]
        entry_size, entries_at, inline_max = 12, offset + 2, 4

    fields = {}
    for i in range(count):
        at = entries_at + i * entry_size
        if at + entry_size > len(data):
            break
        if bigtiff:
            tag, type_code, n = struct.unpack_from(endian + "HHQ", data, at)
            value_at = at + 12
        else:
            tag, type_code, n = struct.unpack_from(endian + "HHI", data, at)
            value_at = at + 8
        if tag not in TAGS or type_code not in TYPES:
            continue
        fields[TAGS[tag]] = _read_value(
            data, endian, type_code, n, value_at, inline_max, max_description
        )

    next_at = entries_at + count * entry_size
    if bigtiff:
        nxt = struct.unpack_from(endian + "Q", data, next_at)[0] if next_at + 8 <= len(data) else 0
    else:
        nxt = struct.unpack_from(endian + "I", data, next_at)[0] if next_at + 4 <= len(data) else 0
    return _describe(fields), nxt


def _read_value(data, endian, type_code, n, value_at, inline_max, max_description):
    fmt, width = TYPES[type_code]
    total = n * width
    if total > inline_max:
        pointer_fmt = "Q" if inline_max == 8 else "I"
        where = struct.unpack_from(endian + pointer_fmt, data, value_at)[0]
    else:
        where = value_at
    if where + total > len(data):
        return None  # value lives past the bytes we were handed

    if type_code == 2:  # ASCII
        text = data[where : where + total].split(b"\x00")[0]
        return text.decode("utf-8", "replace")[:max_description]
    if type_code in (5, 10):  # RATIONAL: numerator/denominator pairs, not worth unpacking
        return None

    values = struct.unpack_from("%s%d%s" % (endian, n, fmt), data, where)
    return values[0] if n == 1 else list(values)


def _describe(fields):
    page = {
        "width": fields.get("ImageWidth"),
        "height": fields.get("ImageLength"),
        "samples_per_pixel": fields.get("SamplesPerPixel", 1),
        "bits_per_sample": fields.get("BitsPerSample"),
        "compression": COMPRESSION.get(fields.get("Compression"), fields.get("Compression")),
        "photometric": PHOTOMETRIC.get(
            fields.get("PhotometricInterpretation"), fields.get("PhotometricInterpretation")
        ),
    }
    fmt = fields.get("SampleFormat")
    if isinstance(fmt, list):
        fmt = fmt[0]
    page["sample_format"] = SAMPLE_FORMAT.get(fmt, "uint" if fmt is None else fmt)
    for key in ("PageName", "ImageDescription"):
        if fields.get(key):
            page[key.lower()] = fields[key]
    return page


def dtype_of(page):
    """A numpy-style dtype string for a page, e.g. 'uint8' or 'uint16'."""
    bits = page.get("bits_per_sample")
    if isinstance(bits, list):
        bits = bits[0]
    if bits is None:
        return "?"
    return "%s%d" % (page.get("sample_format") or "uint", bits)


def summarise(info):
    """One line describing the file: pages, geometry, dtype, compression."""
    pages = info["pages"]
    if not pages:
        return "no IFDs"
    first = pages[0]
    shapes = {(p.get("width"), p.get("height"), p.get("samples_per_pixel")) for p in pages}
    geometry = "%sx%s" % (first.get("width"), first.get("height"))
    if len(shapes) > 1:
        geometry += " (+%d other page shapes)" % (len(shapes) - 1)
    return "%d page%s  %s  spp=%s  %s  %s" % (
        len(pages),
        "" if len(pages) == 1 else "s",
        geometry,
        first.get("samples_per_pixel"),
        dtype_of(first),
        first.get("compression"),
    )
