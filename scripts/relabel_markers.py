"""Swap two marker names in existing caches and runs, in place. Standard library only.

    python scripts/relabel_markers.py --swap ACRV1 TOMM20 data/isx_roles runs/abl_isx_dropout runs/srv_isx_stems figures/srv_isx_accuracy

Why: the ImageStream caches prepared before 2026-10-27 named Ch03 "ACRV1" and Ch11 "TOMM20",
and it is the other way round (docs/results.md, docs/task_brief.md). The image columns are
right, only the names are wrong, so a cache's meta.json and a run's results.json, ablate.json,
masks.json, denoised.json, log.csv and best_<marker>.pt files are relabelled rather than
regenerated. Each directory gets a relabel.txt and is never relabelled twice with the same swap.
"""
import os
import re
import sys
import time


def swap_text(text, a, b):
    # a name counts when it is not glued to other letters or digits: "ACRV1_pixel_r", "isx:ACRV1",
    # "best_ACRV1.pt" and "\"ACRV1\"" all match, "ACRV10" would not
    ra, rb = re.compile(r"(?<![A-Za-z0-9/])%s(?![A-Za-z0-9/])" % re.escape(a)), re.compile(r"(?<![A-Za-z0-9/])%s(?![A-Za-z0-9/])" % re.escape(b))
    text = ra.sub("\0SWAP\0", text)
    text = rb.sub(a, text)
    return text.replace("\0SWAP\0", b)


def relabel_dir(d, a, b):
    note = os.path.join(d, "relabel.txt")
    tag = "swap %s <-> %s" % tuple(sorted((a, b)))
    if os.path.exists(note) and tag in open(note).read():
        print("%s: already relabelled (%s), skipped" % (d, tag)); return 0
    changed = []
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if fn.endswith((".json", ".csv", ".txt")) and fn != "relabel.txt":
            s = open(p, encoding="utf-8").read(); t = swap_text(s, a, b)
            if t != s:
                open(p, "w", encoding="utf-8").write(t); changed.append(fn)
    fa, fb = a.replace("/", "_"), b.replace("/", "_")
    for ext in (".pt", ".npz", ".npy", ".png"):
        pa, pb = os.path.join(d, "best_%s%s" % (fa, ext)), os.path.join(d, "best_%s%s" % (fb, ext))
        if os.path.exists(pa) or os.path.exists(pb):
            tmp = pa + ".swap"
            if os.path.exists(pa): os.rename(pa, tmp)
            if os.path.exists(pb): os.rename(pb, pa)
            if os.path.exists(tmp): os.rename(tmp, pb)
            changed.append("best_%s%s <-> best_%s%s" % (fa, ext, fb, ext))
    with open(note, "a") as f:
        f.write("%s  %s  (%s)\n" % (time.strftime("%Y-%m-%d %H:%M"), tag, ", ".join(changed) or "nothing matched"))
    print("%s: %s" % (d, ", ".join(changed) or "nothing matched"))
    return len(changed)


def main(argv):
    if len(argv) < 5 or argv[1] != "--swap":
        print(__doc__); return 2
    a, b, dirs = argv[2], argv[3], argv[4:]
    for d in dirs:
        if not os.path.isdir(d):
            print("%s: not a directory" % d); return 1
        relabel_dir(d, a, b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
