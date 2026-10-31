"""Survey a new A8 export before anything is prepared from it: what is in the folder and the
sample zip(s), and which assumptions of the current pipeline (configs/instruments/a8.toml,
data/prepare_a8.py, io/a8.py) it would break. Standard library only, read-only.

    python scripts\\survey_a8_export.py "Z:\\Blair_Main\\2026\\081226\\260812_Blair_Sperm_A8\\260812"
    python scripts\\survey_a8_export.py "...\\260812\\B5-All Primary 1_002.zip"          # one sample zip
    python scripts\\survey_a8_export.py <folder> --tiffs 200 --out survey_260812.txt     # inspect more images, save the report

For every .fcs: cytometer, acquisition time, events, the fluorescence parameters and the operator's
labels, the imaging detectors, plate position. For every zip (and .tiff folder): image count,
event-index range, TIFF pages (count, names, size, dtype) from a sample of files. Then the checklist
of pipeline assumptions, each marked OK / CHANGE / CHECK with what to do.
"""
import glob
import os
import re
import struct
import sys
import zipfile
from collections import Counter

# ---- what the pipeline assumes today -------------------------------------------------------------
EXPECTED_PAGES = ["LightLoss (Imaging)", "FSC", "SSC (Imaging)", "Imaging_BP/534/46/LP/505", "Imaging_BP/598/60/LP/570", "Imaging_BP/788/225/LP/675"]
EXPECTED_MARKERS = {"LDHC_AKAP4": "Alexa Fluor 488", "CD45": "PE", "ACRV-1": "PerCP-e710*"}
CLASS_RE = re.compile(r"-(?P<cls>[123](?:S|P|SP))$", re.IGNORECASE)          # prepare_a8.CLASS_RE: sample stem must end in -2S / -3P / -3SP ...
INDEX_RE = re.compile(r"_(?P<idx>\d{6,10})\.tiff?$", re.IGNORECASE)          # prepare_a8.INDEX_RE: <stem>_<event index>.tiff
IMAGE_SHAPE = "104 wide x 57-85 high, float32, 6 pages"


# ---- FCS -----------------------------------------------------------------------------------------
def fcs_text(data):
    hdr = data[:58]; tb, te = int(hdr[10:18]), int(hdr[18:26]); text = data[tb:te + 1].decode("latin-1")
    d = text[0]; parts = text[1:].split(d); kw = {}
    for i in range(0, len(parts) - 1, 2):
        if parts[i].strip(): kw[parts[i].strip().upper()] = parts[i + 1]
    return hdr[:6].decode("ascii", "replace").strip(), kw


def fcs_summary(name, data):
    ver, kw = fcs_text(data); n = int(kw.get("$PAR", 0))
    params = [(kw.get("$P%dN" % i, ""), kw.get("$P%dS" % i, "")) for i in range(1, n + 1)]
    fluor = [(a, b) for a, b in params if a.endswith("-A") and not any(a.startswith(p) for p in ("FSC", "SSC", "Time", "Img", "UV", "V", "B", "YG", "R", "DAPI")) and not re.match(r"^(UV|V|B|YG|R)\d+-", a)]
    fluor = [(a, b) for a, b in params if re.match(r"^(Alexa|PE|PerCP|APC|FITC|BV|BUV|DAPI|Pacific|eFluor|Brilliant|Super Bright|Spark|CD|LDHC|ACRV|TOMM)", a) and a.endswith("-A")]
    imaging = [a for a, _ in params if a.startswith("ImgB") and a.endswith("-A")]
    scatter_img = [a for a, _ in params if "(Imaging)" in a and " (" not in a.split("(Imaging)")[0]]
    features = sorted({a.split(" (")[0] for a, _ in params if " (" in a and a.endswith(")")})
    well = kw.get("WELL ID") or kw.get("WELL_ID") or kw.get("TUBE NAME") or ""
    return {"file": name, "fcs": ver, "cyt": kw.get("$CYT", "?"), "date": kw.get("$DATE", "?"), "btim": kw.get("$BTIM", "?"), "etim": kw.get("$ETIM", "?"),
            "tot": int(kw.get("$TOT", 0)), "par": n, "fluor": fluor, "imaging": imaging, "features": features, "well": well,
            "img_labels": sorted({m.group(1) for a, _ in params for m in [re.search(r"\(([^()/]+)\)$", a)] if m and a.split(" (")[0] in ("Size", "Total Intensity")})}


# ---- TIFF (minimal: page count, names, size, dtype) ---------------------------------------------
TYPES = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 11: ("f", 4), 12: ("d", 8), 16: ("Q", 8)}


def tiff_pages(data):
    bo = "<" if data[:2] == b"II" else ">"
    magic = struct.unpack(bo + "H", data[2:4])[0]
    big = magic == 43
    off = struct.unpack(bo + ("Q" if big else "I"), data[8:16] if big else data[4:8])[0]
    pages = []
    while off and len(pages) < 64:
        if big: n = struct.unpack(bo + "Q", data[off:off + 8])[0]; p = off + 8; esz = 20
        else: n = struct.unpack(bo + "H", data[off:off + 2])[0]; p = off + 2; esz = 12
        tags = {}
        for _ in range(n):
            e = data[p:p + esz]
            if big: tag, typ, cnt = struct.unpack(bo + "HHQ", e[:12]); val = e[12:20]
            else: tag, typ, cnt = struct.unpack(bo + "HHI", e[:8]); val = e[8:12]
            fmt, sz = TYPES.get(typ, ("B", 1)); total = sz * cnt
            if total > (8 if big else 4):
                vo = struct.unpack(bo + ("Q" if big else "I"), val)[0]; raw = data[vo:vo + total]
            else: raw = val[:total]
            if typ == 2: tags[tag] = raw.split(b"\0")[0].decode("latin-1", "replace")
            elif fmt in ("B", "H", "I", "Q", "f", "d") and cnt <= 4: tags[tag] = struct.unpack(bo + fmt * cnt, raw[:sz * cnt])
            p += esz
        bits = tags.get(258, (0,))[0]; sf = tags.get(339, (1,))[0]
        dtype = {(32, 3): "float32", (64, 3): "float64", (16, 1): "uint16", (8, 1): "uint8", (16, 2): "int16"}.get((bits, sf), "%d-bit sf%d" % (bits, sf))
        pages.append({"w": tags.get(256, (0,))[0], "h": tags.get(257, (0,))[0], "dtype": dtype, "name": tags.get(285, ""), "desc": tags.get(270, "")[:60]})
        off = struct.unpack(bo + ("Q" if big else "I"), data[p:p + (8 if big else 4)])[0]
    return pages


# ---- the survey ----------------------------------------------------------------------------------
def survey_images(entries, read, n_tiff, label):
    """entries: list of image member names; read(name) -> bytes."""
    tiffs = [e for e in entries if e.lower().endswith((".tif", ".tiff"))]
    idx = [int(m.group("idx")) for e in tiffs for m in [INDEX_RE.search(os.path.basename(e))] if m]
    stems = Counter(INDEX_RE.sub("", os.path.basename(e)) for e in tiffs)
    out = {"label": label, "n_images": len(tiffs), "n_other": len(entries) - len(tiffs), "index_min": min(idx) if idx else None, "index_max": max(idx) if idx else None,
           "n_indexed": len(idx), "stems": stems.most_common(5), "pages": Counter(), "names": Counter(), "sizes": Counter(), "dtypes": Counter(), "bad": 0}
    step = max(1, len(tiffs) // n_tiff)
    for e in tiffs[::step][:n_tiff]:
        try:
            pg = tiff_pages(read(e))
        except Exception:
            out["bad"] += 1; continue
        out["pages"][len(pg)] += 1; out["names"][tuple(p["name"] for p in pg)] += 1
        out["sizes"][(pg[0]["w"], tuple(sorted({p["h"] for p in pg})))] += 1; out["dtypes"][tuple(sorted({p["dtype"] for p in pg}))] += 1
    return out


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    n_tiff = int(argv[argv.index("--tiffs") + 1]) if "--tiffs" in argv else 60
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else None
    if not args: print(__doc__); return 2
    target = args[0]
    folder = target if os.path.isdir(target) else os.path.dirname(target)
    lines = []; say = lambda s="": (lines.append(s), print(s))
    say("A8 export survey: %s" % target); say("=" * 100)
    # 1. folder listing
    say("\n[1] folder %s" % folder)
    exts = Counter(os.path.splitext(f)[1].lower() for f in os.listdir(folder)) if os.path.isdir(folder) else Counter()
    say("    files by type: %s" % dict(exts))
    fcs_files = sorted(glob.glob(os.path.join(folder, "*.fcs"))); zips = sorted(glob.glob(os.path.join(folder, "*.zip"))); cvws = sorted(glob.glob(os.path.join(folder, "*.cvw")))
    img_dirs = sorted(d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d))
    if not os.path.isdir(target) and target.lower().endswith(".zip"): zips = [target] + [z for z in zips if os.path.abspath(z) != os.path.abspath(target)]
    say("    .fcs: %d   .zip: %d   .cvw: %d   subfolders: %d" % (len(fcs_files), len(zips), len(cvws), len(img_dirs)))
    for p in fcs_files + zips + cvws: say("      %-60s %10.1f MB" % (os.path.basename(p), os.path.getsize(p) / 1e6))
    # 2. FCS
    fcs = []
    say("\n[2] FCS files")
    for p in fcs_files:
        with open(p, "rb") as fh: data = fh.read(4 * 1024 * 1024)
        f = fcs_summary(os.path.basename(p), data); fcs.append(f)
        say("    %s" % f["file"]); say("      %s, FCS %s, %s %s-%s, %s events, %d parameters, well %r" % (f["cyt"], f["fcs"], f["date"], f["btim"], f["etim"], "{:,}".format(f["tot"]), f["par"], f["well"]))
        say("      fluorescence: %s" % ", ".join("%s = %s" % (a, b or "(no label)") for a, b in f["fluor"]))
        say("      imaging detectors: %s   imaged-marker labels: %s" % (f["imaging"], f["img_labels"]))
        stem = os.path.splitext(f["file"])[0]; m = CLASS_RE.search(stem)
        say("      sample stem %r -> class %s" % (stem, m.group("cls").upper() if m else "NONE (does not match -<1-3><S|P|SP>)"))
    # 3. images
    say("\n[3] images")
    surveys = []
    for z in zips:
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            say("    %s: not a zip" % os.path.basename(z)); continue
        names = [n for n in zf.namelist() if not n.endswith("/")]
        inner_zips = [n for n in names if n.lower().endswith(".zip")]
        s = survey_images(names, zf.read, n_tiff, os.path.basename(z)); s["inner_zips"] = inner_zips[:5]; s["members_sample"] = names[:3]; surveys.append(s)
    for d in img_dirs:
        names = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True) if os.path.isfile(p)]
        if any(n.lower().endswith((".tif", ".tiff")) for n in names):
            s = survey_images(names, lambda p: open(p, "rb").read(), n_tiff, os.path.basename(d) + "/"); surveys.append(s)
    for s in surveys:
        say("    %s" % s["label"]); say("      %s images (%s other members%s); event index %s..%s on %d files; name stems %s" % ("{:,}".format(s["n_images"]), s["n_other"], (", nested zips: %s" % s["inner_zips"]) if s.get("inner_zips") else "", s["index_min"], s["index_max"], s["n_indexed"], s["stems"]))
        if s["pages"]:
            say("      pages per TIFF: %s   dtypes: %s" % (dict(s["pages"]), [list(k) for k in s["dtypes"]]))
            for nm, c in s["names"].most_common(3): say("      page names (%d files): %s" % (c, list(nm)))
            say("      width x heights: %s" % [(w, list(h), c) for (w, h), c in s["sizes"].most_common(3)])
        if s["bad"]: say("      %d files could not be parsed as TIFF" % s["bad"])
        if s.get("members_sample"): say("      first members: %s" % s["members_sample"])
    # 4. checklist
    say("\n[4] what the current pipeline assumes, and whether this export fits")
    def check(status, what, do=""): say("    %-6s %s%s" % (status, what, ("\n           -> " + do) if do else ""))
    # samples / classes
    stems = [os.path.splitext(f["file"])[0] for f in fcs]
    cls = [CLASS_RE.search(s_) for s_ in stems]
    if fcs and all(cls): check("OK", "sample names carry the -<replicate><S|P|SP> code the reader keys on: %s" % [m.group("cls").upper() for m in cls])
    elif fcs: check("CHANGE", "sample names do not match prepare_a8.CLASS_RE (-2S / -3P / -3SP): %s" % stems, "add a --classes map (file stem -> class code) or a new CLASS_RE for this naming; decide what the replicate digit and S/P/SP mean here")
    else: check("CHECK", "no .fcs in the folder: the reader needs one per sample for event rows and scalars", "find the FCS export for this run, or point the survey at the folder that has it")
    kinds = {m.group("cls").upper().lstrip("123") for m in cls if m}
    if fcs and not {"S", "P"} <= kinds: check("CHANGE", "no pure sperm + pure PBMC wells: spillover estimation (io.a8.spillover) uses AF488-dominant and PE-dominant wells, and the well-based validation (pure wells as population labels) needs them",
                                               "either acquire/locate single-stain or pure-population wells, or fix the spillover matrix from the July run and validate against the measured stains alone")
    if fcs and not any(m and m.group("cls")[0] == "1" for m in cls): check("CHECK", "no replicate-1 (unstained) control in the folder", "the unstained control is optional for training; needed only for the autofluorescence floor")
    if fcs and len({m.group("cls")[0] for m in cls if m}) < 2: check("CHANGE", "fewer than two replicates: training and the held-out test are split by replicate digit", "a single acquisition needs an event-wise split (e.g. by acquisition half) and the leakage caveat that brings; results would not be comparable to the July replicate split")
    # panel
    for f in fcs:
        labels = {b.replace("-A", "").upper(): a for a, b in f["fluor"]}
        present = {m for m in EXPECTED_MARKERS if any(m.upper() in k for k in labels)}
        missing = set(EXPECTED_MARKERS) - present; extra = [b for a, b in f["fluor"] if not any(m.upper() in (b or "").upper() for m in EXPECTED_MARKERS)]
        if not missing and not extra: check("OK", "%s: panel labels match the a8.toml stains (LDHC_AKAP4 / CD45 / ACRV-1)" % f["file"])
        else: check("CHANGE", "%s: panel differs from a8.toml: missing %s, new %s" % (f["file"], sorted(missing) or "none", extra or "none"), "edit [stains] in configs/instruments/a8.toml (marker -> page), configs/markers/*.toml for any new marker, PAGES in io/a8.py; the stems model has one output head per marker so it is retrained, not reused")
        fl = [a for a, _ in f["fluor"]]
        exp_fl = set(EXPECTED_MARKERS.values())
        if fl and not exp_fl <= set(x.replace("-A", "") for x in fl): check("CHECK", "%s: fluorophores %s (July: AF488, PE, PerCP-eF710)" % (f["file"], fl), "a changed fluorophore changes the imaging filter page and the spillover triangle (AF488 -> PE -> PerCP order)")
        if len(f["imaging"]) != 3: check("CHANGE", "%s: %d imaging fluorescence detectors (July: ImgB1-3)" % (f["file"], len(f["imaging"])), "a different page count: update PAGES / [stains] / [roles] and the 6-page assumption in io/a8.py and prepare_a8.py")
    # images
    for s in surveys:
        if not s["n_images"]: check("CHECK", "%s holds no TIFFs (members: %s)" % (s["label"], s.get("members_sample")), "if the images are in a nested zip or another folder, run the survey on that; the reader globs <stem>_images/**/*.tif*"); continue
        pc = s["pages"].most_common(1)[0][0] if s["pages"] else None
        if pc == 6: check("OK", "%s: 6 pages per TIFF" % s["label"])
        elif pc: check("CHANGE", "%s: %d pages per TIFF (reader assumes 6: LightLoss, FSC, SSC, 3 fluorescence)" % (s["label"], pc), "map pages in a8.toml [roles]/[stains] and io/a8.py PAGES")
        if s["names"]:
            nm = list(s["names"].most_common(1)[0][0])
            if nm == EXPECTED_PAGES: check("OK", "%s: page names identical to July" % s["label"])
            else: check("CHANGE" if len(nm) != 6 or nm[:3] != EXPECTED_PAGES[:3] else "CHECK", "%s: page names differ from July: %s" % (s["label"], nm), "filter bands name the fluorescence pages; match each to a marker and update [stains]; label-free pages must stay LightLoss / FSC / SSC in that order or [roles] moves")
        if s["sizes"]:
            (w, hs), _ = s["sizes"].most_common(1)[0]
            check("OK" if w == 104 else "CHECK", "%s: images %d wide, heights %s (July: %s)" % (s["label"], w, list(hs), IMAGE_SHAPE), "" if w == 104 else "a different frame size changes pixel_um / zoom_to_reference checks and the 48 x 48 feature stacks only if the pixel pitch changed; confirm the CellView magnification")
        if s["dtypes"] and list(s["dtypes"].most_common(1)[0][0]) != ["float32"]: check("CHECK", "%s: pixel type %s (July: float32, background-subtracted)" % (s["label"], list(s["dtypes"].most_common(1)[0][0])), "integer pages are not background-subtracted: the noise-floor and scaling steps assume zero-centred fluorescence")
        if s["n_indexed"] != s["n_images"]: check("CHANGE", "%s: %d of %d TIFF names lack the _<6-10 digit index>.tiff pattern" % (s["label"], s["n_images"] - s["n_indexed"], s["n_images"]), "INDEX_RE in prepare_a8.py must match this naming to join images to FCS rows")
        if s["n_images"] and s["n_images"] < 5000: check("CHECK", "%s: only %s images" % (s["label"], "{:,}".format(s["n_images"])), "July had 20,000 per well; fewer images mean fewer training objects per class and wider error bars")
        if s["stems"] and len(s["stems"]) > 1: check("CHECK", "%s: more than one name stem inside: %s" % (s["label"], s["stems"]), "one zip per sample is what the reader expects; several stems in one zip need splitting by stem")
    # clustering side
    say("\n[5] for the unsupervised clustering (features + populate)")
    say("    - The feature engine reads marker names from the store (predict), so a changed panel changes the compartment map: features/engine.py COMPARTMENT")
    say("      maps LDHC/AKAP4 -> flagellum/tail, CD45 -> cd45 body, ACRV1 -> acrosome/head. A new marker needs a compartment entry and, if nuclear, qc.n_nuclei switches on.")
    say("    - Without pure wells there is no well-based check (confusion vs wells, the 'carried-over sperm' story); the measured-side clustering and")
    say("      --measured-inputs become the only truths, plus any expert annotation. Say so in the plan before running.")
    say("    - A mixed 'All Primary' well is the realistic input: k = 2 should still split sperm / leukocyte if both are present; if the sample is")
    say("      sperm-only, k = 2 lands on the tail-configuration split instead and the sorting abstract's leukocyte gate cannot be tested here.")
    say("    - populate, galleries, confusion matrices and the grid all key on cluster ids and marker names only; they run unchanged once a store exists.")
    if out_path:
        open(out_path, "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\nreport saved to", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
