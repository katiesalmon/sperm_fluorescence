# Sampling the marker exports

How to see what is in the marker-replicate zips and pull a small working subset onto a
laptop. Everything under `scripts/` is **standard library Python 3.8+** — no environment
setup, which is the point: the survey has to run on the imaging server, where installing
packages is a whole conversation.

`<images-dir>` is the directory holding the nine CytPix export zips. The scripts accept
either that directory — they resolve class tokens from the filenames — or individual zip
paths. Substitute `py -3` for `python` if `python` is not on PATH.

---

## 1. Survey the zips — do this first

```bash
python scripts/inspect_zip.py <images-dir> --classes 2S 2P 2SP 3S 3P 3SP --structure --peek-metadata --json survey_marker.json
```

Reads the zip index, the TIFF *headers* of a dozen files per zip, and the head of each
sidecar. No pixels are decoded, so this is seconds per file even on a 3.5 GB zip.

It answers the open questions in [task_brief.md](task_brief.md), and three parts of the
output matter more than the rest:

- **`-- events --`** — how many events, how many files each, and the channel signature.
  `files per event: {5: 30000}` with signature `ACRV1+BF+DAPI+LDHC+Tomm20` means one file
  per channel. `{1: 30000}` with signature `(none)` means the channels are inside the
  file, or are not images at all.
- **`-- image structure --`** — pages, samples-per-pixel, dtype, compression, and any
  `ImageDescription` / `PageName`. A 5-page TIF is a multi-channel event; `spp=4 uint8`
  is the RGBA-grayscale container replicate 1 used.
- **the FCS parameter table**, if there is an FCS sidecar. `--peek-metadata` parses its
  TEXT segment and prints `$PnN` (detector, e.g. `VL1-A`) against `$PnS` (whatever the
  operator typed, e.g. `DAPI`). That mapping is the hardest thing to guess and the most
  expensive thing to guess wrong.

**Do not skip to step 2.** The right `--per-class` depends on how many files an event is.

## 2. Dry-run the sample

```bash
python scripts/make_sample.py <images-dir> --classes 2S 2P 2SP --per-class 300 --dry-run
```

Reports what would be drawn, the channel signature it grouped on, and the projected
payload. Writes nothing. Check the "channels per event" line matches what the survey
showed before going further.

If the auto-detection got the grouping wrong — the printed signature is `(none)` when you
know there are per-channel files, or the event count is 5× what it should be — override
it:

```bash
python scripts/make_sample.py <images-dir> --channel-regex '(?P<event>.+?)_(?P<channel>DAPI|ACRV1|LDHC|Tomm20|BF)\.tif$' --dry-run
```

Every image entry must match the pattern, or it fails loudly rather than dropping events.

## 3. Build the sample

```bash
python scripts/make_sample.py <images-dir> --classes 2S 2P 2SP --per-class 300 --out sample_rep2_seed0.zip
```

Output layout:

```
sample_rep2_seed0.zip
  MANIFEST.json
  2S/<original path inside the source zip>
  2P/...
  2SP/...
```

Worth knowing:

- **The draw is over events, not files.** An event travels with all of its channels or
  not at all — a sample holding an event's DAPI frame but not its brightfield frame
  teaches nothing. This is the main difference from the `sperm_pbmc` sampler.
- **Only events with the modal channel signature are eligible** by default. That quietly
  excludes half-written events and stray non-event files (contact sheets, mosaics)
  instead of letting them into the sample. `--allow-partial-events` turns it off; the
  counts of what was excluded are always printed.
- **The draw is seeded** (`--seed`, default `0`) and seeded *per class*, so re-running
  gives the same events, and changing which classes you pass does not reshuffle the ones
  you kept.
- **Sidecars are copied by default** (`--no-metadata` to stop it). An FCS covering the
  whole run is a few MB and may be the label source — worth carrying even though the
  sample is 300 events.
- **Entries over 8 MB exclude their whole event** (`--max-member-bytes`).
- **`MANIFEST.json`** records the source zips, the seed, the channel signature, the drawn
  event keys, and every member with its CRC. Keep the filename, which encodes the classes
  and seed.

## 4. Verify and unpack

```bash
python scripts/check_sample.py sample_rep2_seed0.zip --extract-to data/samples
```

CRC-checks every member against the manifest **and** confirms every drawn event arrived
with its full frame count, so a truncated copy fails loudly here rather than showing up
later as a quietly wrong training pair. Files land in
`data/samples/<sample-name>/<CLASS>/...`, which is gitignored.

## 5. Getting it onto the laptop

Same as `sperm_pbmc`: the scripts are written here, run on the server, and the resulting
sample zip is copied back by hand. A 300-event, 5-channel sample is on the order of
100–400 MB — small enough to move over a share or a stick.

Then on the laptop:

```bash
python scripts/check_sample.py sample_rep2_seed0.zip --extract-to data/samples
```

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

The venv is only needed for the analysis layer. Verification and unpacking need nothing.

---

## Rehearsing without the real data

`scripts/make_fixture.py` writes fake zips with the same naming and shape, in each of the
three layouts the export might use, so the whole sequence can be run anywhere:

```bash
python3 scripts/make_fixture.py /tmp/fixture/Images --layout per-channel && python3 scripts/inspect_zip.py /tmp/fixture/Images --classes 2S 2P 2SP --structure --peek-metadata && python3 scripts/make_sample.py /tmp/fixture/Images --classes 2S 2P 2SP --per-class 20 --out /tmp/sample.zip --force && python3 scripts/check_sample.py /tmp/sample.zip
```

`--layout multipage` puts the channels inside one TIF; `--layout single` gives one
brightfield TIF per event plus an `events.csv` and a valid `run.fcs` carrying a
detector-to-marker table — the shape if the marker signal is a cytometer measurement
rather than an image. The images are noise with a real TIFF header: useful for checking
the plumbing, useless for anything else.

---

## Rules of thumb

- **No image data in git.** `data/` is gitignored in full; `.gitignore` also blocks
  `*.zip`, `*.tif`, `*.tiff` and `*.fcs` anywhere in the tree as a backstop.
- **Samples are described by their manifest,** not by memory. A result on a sample should
  always be traceable back to the seed and event list that produced it.
- **Regenerating beats copying.** Need different events? Re-run with a new `--seed` or a
  larger `--per-class` rather than hand-picking files.
- **Take the data root as an argument.** Code written against `data/samples/...` should
  run unchanged against a full extract.
- **Never assume channel order.** It comes from a filename, a page description, or the
  FCS table — never from position. Getting ACRV1 and Tomm20 the wrong way round would
  produce a model that trains fine and means nothing.
