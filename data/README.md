# data/

Nothing in here is committed. `.gitignore` blocks `data/*` outright, and blocks `*.zip`,
`*.tif`, `*.tiff` and `*.fcs` anywhere in the tree as a backstop.

| Path | What lands here |
| --- | --- |
| `data/samples/` | Sample zips unpacked by `scripts/check_sample.py --extract-to data/samples` |
| `data/raw/` | A full extract, if one is ever staged locally |

A sample is described by the `MANIFEST.json` inside it -- source zips, seed, per-class
event counts, channel signature, and every member with its CRC. If you need different
images, regenerate with a new `--seed` rather than hand-picking files.
