"""Were the wells separate physical acquisitions, or one acquisition split by a gate? Reads each
FCS file's header and TEXT segment ($BTIM, $ETIM, $DATE, $TOT, $CYT, $FIL, plate Row / Column,
the instrument's own export keywords) and the DATA segment's Time, Row, Column and Sorted
parameters, and prints one line per file. Standard library only; FCS 3.0 / 3.1 / 3.2, float or
double data, either byte order, as the FACSDiscover A8 and the CytPix write it.

    python scripts/fcs_provenance.py path\\to\\export\\*.fcs
    python scripts/fcs_provenance.py path\\to\\export        # every .fcs under a directory

Separate acquisitions have distinct $BTIM/$ETIM, their own event counts, and Time columns that
each start near zero and do not interleave; a gated split of one acquisition shares $BTIM and
has Time ranges that overlap. Time is in units of $TIMESTEP seconds when that keyword is present.
"""
import array
import glob
import os
import struct
import sys


def read_text(fh):
    hdr = fh.read(58); ver = hdr[:6].decode("ascii", "replace")
    tb, te = int(hdr[10:18]), int(hdr[18:26])
    fh.seek(tb); text = fh.read(te - tb + 1).decode("latin-1")
    delim = text[0]; parts = text[1:].split(delim)
    kw = {}
    for i in range(0, len(parts) - 1, 2):
        if parts[i].strip(): kw[parts[i].strip().upper()] = parts[i + 1]
    db, de = int(hdr[26:34] or 0), int(hdr[34:42] or 0)
    if db == 0 and "$BEGINDATA" in kw: db, de = int(kw["$BEGINDATA"]), int(kw["$ENDDATA"])
    return ver, kw, db, de


def read_columns(fh, kw, db, de, wanted):
    """The named parameters' values for every event, as lists; wanted = {name_lower: label}."""
    n_par, n_tot = int(kw["$PAR"]), int(kw["$TOT"]); dtype = kw.get("$DATATYPE", "F").upper()
    names = [kw.get("$P%dN" % (i + 1), "") for i in range(n_par)]
    idx = {names[i].lower(): i for i in range(n_par)}
    cols = {k: idx[k] for k in wanted if k in idx}
    if not cols or dtype not in ("F", "D"): return {}, names
    code = "f" if dtype == "F" else "d"; width = 4 if dtype == "F" else 8
    fh.seek(db); raw = fh.read(min(de - db + 1, n_par * n_tot * width))
    a = array.array(code); a.frombytes(raw[: (len(raw) // width) * width])
    if kw.get("$BYTEORD", "1,2,3,4").startswith("4") != (sys.byteorder == "big"): a.byteswap()
    n = len(a) // n_par
    return {k: a[j::n_par][:n] for k, j in cols.items()}, names


def describe(path):
    with open(path, "rb") as fh:
        ver, kw, db, de = read_text(fh)
        data, names = read_columns(fh, kw, db, de, {"time": "Time", "row": "Row", "column": "Column", "sorted": "Sorted"})
    step = float(kw.get("$TIMESTEP", 0) or 0)
    out = {"file": os.path.basename(path), "fcs": ver.strip(), "cyt": kw.get("$CYT", "?"), "date": kw.get("$DATE", "?"), "btim": kw.get("$BTIM", "?"), "etim": kw.get("$ETIM", "?"),
           "tot": int(kw.get("$TOT", 0)), "par": int(kw.get("$PAR", 0)), "fil": kw.get("$FIL", ""), "timestep": step}
    for k in ("WELL ID", "WELL_ID", "TUBE NAME", "TUBENAME", "$SRC", "SAMPLE ID", "PLATE NAME", "PLATE ID", "EXPERIMENT NAME", "$SMNO", "GUID", "CREATOR", "$INST", "$OP"):
        if k in kw: out[k.lower()] = kw[k]
    if "time" in data:
        t = data["time"]; lo, hi = min(t), max(t)
        out["time_min"], out["time_max"] = lo, hi; out["time_span_s"] = (hi - lo) * step if step else None
        # monotone? a gated subset of one run is still monotone; two runs concatenated are not
        drops = sum(1 for i in range(1, len(t)) if t[i] < t[i - 1]); out["time_drops"] = drops
    for k in ("row", "column", "sorted"):
        if k in data: out[k] = sorted(set(round(v) for v in data[k]))[:12]
    return out


def main(argv):
    paths = []
    for a in argv[1:]:
        if os.path.isdir(a): paths += sorted(glob.glob(os.path.join(a, "**", "*.fcs"), recursive=True))
        else: paths += sorted(glob.glob(a))
    if not paths: print(__doc__); return 2
    rows = [describe(p) for p in paths]
    rows.sort(key=lambda r: (r["date"], r["btim"]))
    print("%-16s %-10s %-9s %-9s %9s %5s  %-22s %-22s %-10s %-8s %s" % ("file", "date", "$BTIM", "$ETIM", "$TOT", "$PAR", "Time min..max", "span / drops", "row", "column", "sorted / well keywords"))
    for r in rows:
        tm = "%.0f..%.0f" % (r["time_min"], r["time_max"]) if "time_min" in r else "no Time column"
        sp = ("%.0f s" % r["time_span_s"] if r.get("time_span_s") is not None else "?") + " / %s" % r.get("time_drops", "?")
        extra = " ".join("%s=%s" % (k, v) for k, v in r.items() if k not in ("file", "fcs", "cyt", "date", "btim", "etim", "tot", "par", "fil", "timestep", "time_min", "time_max", "time_span_s", "time_drops", "row", "column"))
        print("%-16s %-10s %-9s %-9s %9d %5d  %-22s %-22s %-10s %-8s %s" % (r["file"][:16], r["date"], r["btim"][:9], r["etim"][:9], r["tot"], r["par"], tm, sp, r.get("row", "-"), r.get("column", "-"), extra[:120]))
    print("\n%s, FCS %s. Separate runs: distinct $BTIM, each Time starting near 0, no overlap. A gated split: same $BTIM, overlapping Time." % (rows[0]["cyt"], rows[0]["fcs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
