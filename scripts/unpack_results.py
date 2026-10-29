"""Unpack a server_results_<timestamp>.zip into the repo: runs\\srv_* and figures\\srv_* land in
runs/ and figures/ so every laptop command can read them. Standard library only.

    python scripts/unpack_results.py ~/Downloads/server_results_20261026_2209.zip

A zip's data/<store>/ files land in data/srv_<store>/ so they never overwrite the laptop's own store
(2026-10-29: a server meta.json replaced data/a8_vstain/meta.json). Zips written by Windows PowerShell's Compress-Archive name their members with backslashes,
which `unzip` on macOS keeps as literal characters in file names; this converts them.
"""
import os
import sys
import zipfile


def main(argv):
    if len(argv) != 2:
        print(__doc__); return 2
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with zipfile.ZipFile(argv[1]) as z:
        n = 0
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/") or name.startswith("/") or ".." in name.split("/"):
                continue
            if name.startswith("data/") and not name.startswith("data/srv_"):
                name = "data/srv_" + name[len("data/"):]      # the server's store files (meta.json, embeddings) must not overwrite the laptop store of the same name
            if "/" not in name:     # commit.txt, environment.txt: keep them with the runs, not at the repo root
                name = "runs/%s/%s" % (os.path.splitext(os.path.basename(argv[1]))[0], name)
            dest = os.path.join(root, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(z.read(info))
            n += 1
    for note in ("commit.txt", "environment.txt"):
        p = os.path.join(root, "runs", os.path.splitext(os.path.basename(argv[1]))[0], note)
        if os.path.exists(p):
            print("%s: %s" % (note, open(p, encoding="utf-16" if open(p, "rb").read(2) == b"\xff\xfe" else "utf-8").read().strip()))
    print("unpacked %d files into %s (runs/srv_*, figures/srv_*)" % (n, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
