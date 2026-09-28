#!/usr/bin/env python
"""Summarise a directory tree without listing every file.

For a folder full of exports the useful facts are structural: which directories exist,
how many files of which kind each holds, how big, what the names look like -- and for
TIFFs, what is inside the header. A raw `tree` of 40,000 files answers none of that
legibly. This prints one block per directory and stays short even on a large tree.

Nothing is read except TIFF headers (--probe), so it is fast on a network share.

Standard library only.

Examples
--------
  py -3 scripts\\survey_tree.py "Z:\\Blair_Main\\2026\\260709_Blair_Sperm_A8\\260709"
  py -3 scripts\\survey_tree.py <root> --probe 3 --json tree_A8.json
  py -3 scripts\\survey_tree.py <root> --depth 3 --max-children 20
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiff_probe  # noqa: E402
from cytpix import human_bytes, run  # noqa: E402

TIFF_EXTS = {".tif", ".tiff"}


def survey(root, depth, examples, probe, max_children):
    """Walk `root`, returning a nested dict: one node per directory."""
    def node(path, level):
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except (PermissionError, OSError) as exc:
            return {"path": path, "error": str(exc)}

        files, dirs = [], []
        for e in entries:
            try:
                (dirs if e.is_dir(follow_symlinks=False) else files).append(e)
            except OSError:
                continue

        by_ext = collections.Counter()
        total = 0
        for f in files:
            by_ext[os.path.splitext(f.name)[1].lower() or "(none)"] += 1
            try:
                total += f.stat(follow_symlinks=False).st_size
            except OSError:
                pass

        out = {
            "path": path,
            "n_files": len(files),
            "n_dirs": len(dirs),
            "bytes": total,
            "extensions": dict(by_ext.most_common()),
            "examples": [f.name for f in files[:examples]],
        }

        tifs = [f for f in files if os.path.splitext(f.name)[1].lower() in TIFF_EXTS]
        if probe and tifs:
            step = max(1, len(tifs) // probe)
            shapes = collections.Counter()
            descs = collections.Counter()
            for f in tifs[::step][:probe]:
                try:
                    # Whole file, not just the head: many writers put the IFD at the
                    # end, and the probe follows the header's offset to it.
                    with open(f.path, "rb") as fh:
                        data = fh.read()
                    info = tiff_probe.probe(data)
                    shapes[tiff_probe.summarise(info)] += 1
                    for pg in info["pages"]:
                        for k in ("pagename", "imagedescription"):
                            if pg.get(k):
                                descs[pg[k][:100]] += 1
                except (OSError, tiff_probe.NotTiff, ValueError, KeyError) as exc:
                    shapes["unreadable: %s" % str(exc)[:50]] += 1
            out["tiff"] = {"structures": dict(shapes), "descriptions": dict(descs.most_common(6))}

        if depth is None or level < depth:
            out["children"] = [node(d.path, level + 1) for d in dirs]
        else:
            out["children_unexpanded"] = [d.name for d in dirs]
        return out

    return node(root, 0)


def totals(n, acc=None):
    acc = acc if acc is not None else {"files": 0, "bytes": 0, "dirs": 0, "ext": collections.Counter()}
    if "error" in n:
        return acc
    acc["files"] += n["n_files"]
    acc["bytes"] += n["bytes"]
    acc["dirs"] += 1
    acc["ext"].update(n["extensions"])
    for c in n.get("children", []):
        totals(c, acc)
    return acc


def print_tree(n, indent, max_children, root_len):
    pad = "  " * indent
    name = n["path"][root_len:].lstrip("\\/") or os.path.basename(n["path"]) or n["path"]
    if "error" in n:
        print("%s%s/   [unreadable: %s]" % (pad, name, n["error"]))
        return

    exts = ", ".join("%s x%d" % (e, c) for e, c in list(n["extensions"].items())[:6])
    if len(n["extensions"]) > 6:
        exts += ", …"
    print("%s%s/   %d dirs, %d files, %s%s"
          % (pad, name, n["n_dirs"], n["n_files"], human_bytes(n["bytes"]),
             ("   [%s]" % exts) if exts else ""))
    if n["examples"]:
        print("%s    e.g. %s" % (pad, ", ".join(n["examples"])))
    if "tiff" in n:
        for shape, c in n["tiff"]["structures"].items():
            print("%s    tiff: %s   (x%d probed)" % (pad, shape, c))
        for d, c in n["tiff"]["descriptions"].items():
            print("%s    desc: %s" % (pad, d))

    children = n.get("children", [])
    for c in children[:max_children]:
        print_tree(c, indent + 1, max_children, root_len)
    if len(children) > max_children:
        rest = children[max_children:]
        print("%s  … %d more directories (%s files, %s)"
              % (pad, len(rest), f"{sum(totals(r)['files'] for r in rest):,}",
                 human_bytes(sum(totals(r)["bytes"] for r in rest))))
    if n.get("children_unexpanded"):
        print("%s  … %d subdirectories not expanded (raise --depth): %s"
              % (pad, len(n["children_unexpanded"]), ", ".join(n["children_unexpanded"][:8])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Directory to survey")
    ap.add_argument("--depth", type=int, default=None, help="Levels to descend (default: all)")
    ap.add_argument("--examples", type=int, default=4, help="Example filenames per directory (default: 4)")
    ap.add_argument("--probe", type=int, default=3, help="TIFF headers to read per directory (default: 3; 0 to skip)")
    ap.add_argument("--max-children", type=int, default=12, help="Subdirectories to print per level before collapsing (default: 12)")
    ap.add_argument("--json", metavar="PATH", help="Also write the full survey as JSON")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise IOError("Not a directory: %s" % root)

    tree = survey(root, args.depth, args.examples, args.probe, args.max_children)
    print("root: %s\n" % root)
    print_tree(tree, 0, args.max_children, len(root))

    t = totals(tree)
    print("\ntotal: %d directories, %s files, %s" % (t["dirs"], f"{t['files']:,}", human_bytes(t["bytes"])))
    print("by extension: " + ", ".join("%s x%s" % (e, f"{c:,}") for e, c in t["ext"].most_common(12)))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(tree, fh, indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(run(main))
