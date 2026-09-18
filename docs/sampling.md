# Getting data off the server

Everything under `scripts/` is **standard library Python 3.8+** — no environment setup.
That constraint exists so the survey and the sampler can be run by whoever has access to
the data, rather than whoever can get packages installed on the imaging server.

`<run-folder>` below is the directory holding the nine `.acs` archives:

```
Z:\Blair_Main\2026\260709_Blair_Sperm_Cytpix\
```

Substitute `py -3` for `python` if `python` is not on PATH.

---

## Everything is in the .acs

This is the thing to know. Each archive is a plain zip holding, for one acquisition:

| Member | What it is |
| --- | --- |
| `<n>.tif` x 30,000 | The event images — **248 x 248 uint16, single channel, uncompressed** |
| `<name>.fcs` | Every recorded event, ~100-400k of them, 59 parameters each |
| `<uuid>.masks.zip` | One JSON per event: the instrument's segmentation |
| `Toc1.xml` | Archive table of contents |

**Use this, not the `Images\*.zip` export.** That export holds the same events
downconverted to 8-bit in an RGBA container, throws away about two bits of the sensor's
~10-bit range, and carries no measurements or masks. The tooling for reading it has been
removed; it is in git history if it is ever needed.

The images are a **subset** of the recorded events — between 7% and 33% — because the
camera was set to image only events inside a `DAPI+` gate, up to 30,000 per acquisition
(`capture_settings.xml`, carried in every bundle). Each image is named by its event index,
which is the join key into the FCS.

---

## 1. Survey an archive

```bash
py -3 scripts\inspect_acs.py <run-folder>
```

Prints, per archive: the member breakdown, `$TOT`, the acquisition keywords, the
detector-to-antigen table (`$PnN` → `$PnS`) with PMT voltages, the spillover matrix size,
and what fraction of events were imaged. Reads only the FCS TEXT segment, so it is fast
even on a 5 GB archive.

## 2. Cut a bundle to work on locally

```bash
py -3 scripts\make_bundle.py <run-folder> --per-class 200 --out bundle_seed0.zip
```

```
bundle_seed0.zip
  MANIFEST.json
  2S/targets.csv           one row per drawn event
  2S/spillover.csv         that acquisition's compensation matrix
  2S/images/<event>.tif    the matching image
  2S/masks/<event>.json    the instrument's segmentation for that event
  2P/ ... 3SP/
```

Three modes, in increasing cost:

| Command | What you get | Cost |
| --- | --- | --- |
| `--per-class 0` | Compensation matrices and the panel only | Seconds — reads only the FCS TEXT segment |
| `--per-class 200 --no-images` | Also a 200-row target sample per acquisition | Reads the FCS event data (~370 MB across nine) |
| `--per-class 200` | Also the matching images and masks | Adds ~120 KB per image; 200 x 9 is roughly 220 MB |

Worth knowing:

- **Events are drawn from the intersection** of "has a row flagged as imaged" and "has an
  image member", so a half-pair cannot be drawn. The pairing is guaranteed by
  construction rather than re-joined afterwards.
- **The draw is seeded** (`--seed`, default `0`) and seeded per acquisition, so re-running
  gives the same events and changing `--classes` does not reshuffle the ones you kept.
- **Every detector in `$SPILLOVER` is kept**, not just the four labelled ones.
  Compensation is a change of basis across all of them, so a table holding only the
  stained channels cannot be compensated afterwards. `--stained-only` overrides this and
  is almost always the wrong choice.
- **`MANIFEST.json`** records the seed, the drawn event ids, the column list, the panel
  with voltages, and the image format.

## 3. Verify after the copy

```bash
python3 scripts/make_bundle.py --verify bundle_seed0.zip
```

Re-checks that every drawn event still has a target row, an image and a mask. **Do this
before trusting a transfer** — a zip's index lives at the end of the file, so a partial
copy is not a slightly-short bundle, it is an unopenable one.

## 4. The full measurement table, when a sample is not enough

`make_bundle.py` draws from imaged events only. To export every recorded event —
including the ~70% with no image, for characterising the population:

```bash
py -3 scripts\make_targets.py <run-folder> --all-events --out targets
```

---

## Transfer

There is no network path from the laptop to the share, so files are copied by hand
through the Remote Desktop session.

**Dragging a file into an application window does not copy it.** macOS Remote Desktop
creates a sparse placeholder with the right logical size and zero bytes of content; the
file looks present and is empty. Copy into a Finder window and let it finish, then check:

```bash
du -h <file>
```

If that disagrees with `ls -l`, the copy is incomplete.

---

## Rules of thumb

- **No data in git.** `data/` is gitignored in full; `.gitignore` also blocks `*.zip`,
  `*.tif`, `*.tiff`, `*.acs` and `*.fcs` anywhere in the tree as a backstop.
- **Samples are described by their manifest,** not by memory. A result on a sample should
  be traceable back to the seed and event list that produced it.
- **Regenerating beats copying.** Need different events? Re-run with a new `--seed`.
- **Take the data root as an argument** rather than hardcoding a path, so the same code
  runs over a bundle or a full archive unchanged.
- **Never assume channel order.** It comes from the FCS `$PnN`/`$PnS` table. Getting ACRV1
  and CD45 the wrong way round would train fine and mean nothing.
