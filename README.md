# sperm_fluorescence

Predicting the **fluorescence signal in each antigen marker channel — DAPI, ACRV1, LDHC,
Tomm20 — from the raw brightfield CytPix event image**, with the marker channel itself
supplying the training target. No one labels anything: the label is measured at the same
instant as the input.

Right now this repo contains **only the data-handling layer** — tools to see what is
inside the marker-replicate export zips and to pull a small, reproducible sample out of
them. The modeling decision is deliberately not made yet, because it depends on a fact
about the data we do not have: see [docs/task_brief.md](docs/task_brief.md) for the task
and [docs/approach.md](docs/approach.md) for the research and the recommended sequence.

Sibling repo: **`sperm_pbmc`**, which classifies replicate-1 brightfield events as
straight sperm / curled sperm / PBMC. Two findings from it constrain this project and are
carried forward in the task brief — the acquisitions are separable by background noise
alone, and the single-population wells are impure.

## Where this is up to

The survey has been run (2026-09-12). It found all six marker zips to be **flat, one
248 × 248 uint8 RGBA LZW TIF per event, no sidecars — structurally identical to replicate
1**, which rules out both of the layouts the approach doc had costed. Full numbers in
[docs/task_brief.md](docs/task_brief.md).

`check_channels.py` then decoded 96 events and found `R == G == B` on every pixel with
alpha constant — **brightfield in a four-channel container, no marker signal at all**.

The cytometry data turned out to be alongside, not inside: nine `.acs` archives in the run
folder, one per acquisition. An ACS is a zip holding FCS data, and the FCS is the target
variable. Next step reads it without moving 34 GB of archive:

```bash
python scripts/inspect_acs.py <run-folder> --extract-fcs fcs_out
```

That prints `$TOT`, the acquisition keywords, and the `$PnN` → `$PnS` table mapping each
detector (`VL1-A`, `BL1-A`, …) to the antigen on it — then writes out the FCS members
alone, a few MB, to copy back.

## Then

```bash
python scripts/make_sample.py <images-dir> --classes 2S 2P 2SP --per-class 300 --dry-run
```

```bash
python scripts/make_sample.py <images-dir> --classes 2S 2P 2SP --per-class 300 --out sample_rep2_seed0.zip
```

```bash
python scripts/check_sample.py sample_rep2_seed0.zip --extract-to data/samples
```

Full detail, including the guards worth knowing about, is in
[docs/sampling.md](docs/sampling.md).

## Why sample

The six marker exports are ~16 GB zipped. Iterating against all of it is slow, and the
first questions — are the channels registered, does ACRV1 signal actually sit on the
acrosome — are answered by looking at a few hundred events. So the working pattern is the
same as `sperm_pbmc`: survey, sample, verify, build against the sample, then run the
finished pipeline over the full zips on the server.

**The sampler draws events, not files.** This is the one substantive difference from the
`sperm_pbmc` version. If an event is stored as five files — brightfield plus four markers —
then a sample holding an event's DAPI frame but not its brightfield frame teaches nothing.
Events are grouped first and travel whole, and `check_sample.py` fails if any drawn event
arrives short a frame.

Sampling is seeded, and every sample zip carries a `MANIFEST.json` recording the source
zips, the seed, the channel signature, and the exact events drawn — so a result on a
sample is reproducible, and a sample can be regenerated rather than passed around.

## Layout

| Path | What it is |
| --- | --- |
| `docs/task_brief.md` | The task, the marker biology, the data inventory, and the open questions the survey must answer |
| `docs/approach.md` | **The research**: prior art, what performance to expect per marker, the two branches, evaluation design, and the sequence to run |
| `docs/sampling.md` | How to survey the exports and build a working subset |
| `docs/server-setup.md` | What has to be installed where (short answer: nothing, for sampling) |
| `scripts/inspect_zip.py` | Read-only survey — entry counts, event grouping, TIFF structure, FCS parameter table |
| `scripts/make_sample.py` | Deterministic random sample of N **events** per class into one small zip |
| `scripts/check_sample.py` | Verify a sample zip against its manifest, confirm no event lost a channel, and unpack |
| `scripts/inspect_acs.py` | Read the FCS inside an `.acs` archive in place, and extract just the FCS. Standard library |
| `scripts/check_channels.py` | Decode pixels and report whether the RGBA channels carry distinct images. Standard library |
| `scripts/tiff_read.py` | TIFF LZW decoder, so the above needs no install on the server. Standard library |
| `scripts/make_fixture.py` | Fake CytPix-shaped zips in all three candidate layouts, to exercise the above without the real data |
| `scripts/cytpix_zips.py` | Shared helpers: locating zips, grouping files into events, channel tokens |
| `scripts/tiff_probe.py` | TIFF header reader — pages, dtype, channels, descriptions. Standard library |
| `scripts/fcs_probe.py` | FCS TEXT-segment reader — the `$PnN` → `$PnS` detector-to-antigen map. Standard library |
| `analysis/` | Analysis layer (needs `requirements.txt`). Empty until there is a sample to analyse |
| `data/samples/` | Unpacked sample images — **gitignored** |
| `notebooks/` | Exploration notebooks |

Nothing under `data/` is ever committed.

## Conventions

- Class codes follow the export filenames: `S` = sperm alone, `P` = PBMCs alone,
  `SP` = mixture. The leading digit is the replicate. **Replicates 2 and 3 carry the
  markers and are what this repo works on**; replicate 1 is brightfield only and belongs
  to `sperm_pbmc`.
- **Replicate is the held-out axis.** Train on `2*`, test on `3*`. Random event-level
  splits share an acquisition's gain, focus and noise floor, and will overstate
  everything.
- Everything under `scripts/` is standard library Python 3.8+, so it runs on the imaging
  server with no install. Keep it that way — that constraint is why the survey can be run
  by whoever has access to the data rather than whoever has admin rights.
- Code that runs over the sample takes its data root as an argument rather than
  hardcoding a path, so the same code runs unchanged over a full extract.
- Never assume channel order. It comes from a filename, a page description, or the FCS
  table.
