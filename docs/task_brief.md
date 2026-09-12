# Task brief

## Goal

Given the raw (brightfield) CytPix event image, predict the fluorescence signal in each
antigen marker channel:

| Marker | Antigen | Where it sits in a sperm cell | What a brightfield image can plausibly see |
| --- | --- | --- | --- |
| **DAPI** | DNA | nucleus — i.e. the head | the head: the largest, highest-contrast structure in the frame |
| **ACRV1** | intra-acrosomal protein SP-10 | acrosome — the cap over the anterior head | the acrosomal cap margin; a subtle intensity step within the head |
| **LDHC** | testis-specific lactate dehydrogenase C | principal piece of the flagellum, weaker in midpiece and head | the tail |
| **Tomm20** | mitochondrial outer-membrane import receptor | midpiece — the mitochondrial sheath | the thickened segment immediately behind the head |

One model per marker, as four separate heads or four separate models — that decision
belongs in [approach.md](approach.md), not here.

The four markers are worth noticing as a set: they tile the sperm from front to back —
acrosome, nucleus, midpiece, principal piece. Each one localises to a compartment with a
distinct brightfield correlate. That is the reason to expect this to work at all, and it
is also the yardstick: a marker whose predicted signal does not land on its own
compartment is not being predicted, however good its correlation looks.

## Why "self-supervised"

The label comes from the paired fluorescence acquisition rather than from a person. No
one draws a box or types a class; the target is measured at the same instant as the input.

Two distinct things get called self-supervised, and the write-up will be clearer if we
keep them apart from the start:

- **Free-label cross-modal supervision** — the actual task here. Architecturally it is
  ordinary supervised regression; what is free is the *labels*, not the supervision.
  Published work in this area calls it *virtual staining* or *in silico labeling*.
- **Self-supervised representation learning proper** — masked autoencoding, DINO,
  contrastive pretraining — where the objective is constructed from unlabeled images
  alone. This has a real role here as a *pretraining stage*, and we have ~180,000
  unlabeled brightfield events across replicate 1 to do it with.

Both are on the table. They are not the same technique and should not share a name in
the methods section.

## Design of the experiment

**Training source.** Paired events from the marker replicates (`2*`, `3*`), where the
brightfield frame and the marker signal describe the same event at the same moment.

**The held-out axis that matters is the replicate, not the event.** Random event-level
splits will overstate performance, because events from one acquisition share a gain
setting, a focus, and a noise floor. Replicate 2 → replicate 3 (and back) is the split
that answers "does this generalise", and it is available for free.

**The `S` / `P` / `SP` wells are a second, independent check.** PBMCs have no acrosome,
no midpiece and no flagellum. A model that predicts ACRV1, LDHC or Tomm20 signal onto a
PBMC is reading something other than the biology — and the `2P` / `3P` wells supply
thousands of those negatives without anyone labeling anything.

## What we already know, from replicate 1

Established by the survey in the `sperm_pbmc` repo, and worth carrying over because it
sets expectations for what the marker replicates will look like:

| | |
| --- | --- |
| Layout | Flat — no subfolders. One TIF per event, named `<event-id>.tif`. |
| Events per zip | 30,000 in each of `1S`, `1P`, `1SP` |
| Image | 248 × 248, single page, uint8 |
| Channels | Stored RGBA, but R == G == B and alpha constant — 8-bit grayscale in a 4-channel container |
| Compression | LZW (needs `imagecodecs` alongside `tifffile`) |
| Metadata sidecar | None in replicate 1 |

**Two findings from replicate 1 that constrain this project directly:**

1. **Acquisitions are separable by background alone.** A classifier given only 40 × 40
   corner patches — containing no cell whatsoever — told `1S` from `1P` at AUC 0.992,
   because the two wells have different noise floors (background σ 5.5 vs 10.7). Any
   model here that is allowed to see raw, un-normalised background can learn the
   acquisition instead of the biology. For a regression target that varies by
   acquisition gain, that is not a hypothetical.
2. **The single-population wells are impure.** Semen normally contains round cells, so
   a sperm well is a sperm-and-round-cell well. Well identity is a prior, never a label.

## Data inventory

Nine zips, ~24 GB total, exported 2026-07-09. The six marker zips are what this repo is
about; replicate 1 is brightfield only and stays in `sperm_pbmc`.

| File | Size | Replicate | Contents | In scope |
| --- | --- | --- | --- | --- |
| `..._1P.zip` | 2.75 GB | 1 | PBMCs alone | no — brightfield only |
| `..._1S.zip` | 2.35 GB | 1 | sperm alone | no — brightfield only |
| `..._1SP.zip` | 2.60 GB | 1 | sperm + PBMC mixture | no — brightfield only |
| `..._2P.zip` | 2.77 GB | 2 | PBMCs alone + fluorescence | **yes** |
| `..._2S.zip` | 2.34 GB | 2 | sperm alone + fluorescence | **yes** |
| `..._2SP.zip` | 3.55 GB | 2 | mixture + fluorescence | **yes** |
| `..._3P.zip` | 2.77 GB | 3 | PBMCs alone + fluorescence | **yes** |
| `..._3S.zip` | 2.34 GB | 3 | sperm alone + fluorescence | **yes** |
| `..._3SP.zip` | 2.42 GB | 3 | mixture + fluorescence | **yes** |

**The sizes are the first open question.** `2S` is 2.34 GB against `1S`'s 2.35 GB —
essentially identical. If replicate 2 carried four extra image channels per event, it
should be several times larger. Either the marker channels are not stored as images in
these zips, or they compress far better than brightfield, or the event count differs.
The survey settles it; do not assume.

## What the survey found (2026-09-12)

`inspect_zip.py --structure --peek-metadata` over all six marker zips, from
`Z:\Blair_Main\2026\260709_Blair_Sperm_Cytpix\Images\`:

| | `2S` | `2P` | `2SP` | `3S` | `3P` | `3SP` |
| --- | --- | --- | --- | --- | --- | --- |
| Events | 30,000 | 30,000 | **42,874** | 30,000 | 30,000 | 30,000 |
| Files per event | 1 | 1 | 1 | 1 | 1 | 1 |
| Mean file size | 78.0 KB | 92.2 KB | 82.7 KB | 78.0 KB | 92.2 KB | 80.7 KB |
| Sidecars | 0 | 0 | 0 | 0 | 0 | 0 |

Every zip: **flat, one `<event-id>.tif` per event, 248 × 248, 1 page, spp=4, uint8, LZW,
no `ImageDescription`, no sidecar of any kind.**

**That is structurally identical to replicate 1.** Both candidate branches in
[approach.md](approach.md) are ruled out as stated: the markers are not separate files,
not extra pages, and there is no FCS or CSV in these zips to join against.

Three things follow from the numbers themselves:

1. **The RGBA container is the only place left in these files for marker signal.** If
   replicate 1's `R == G == B, alpha constant` no longer holds in `2*` / `3*`, the
   fluorescence is in those channels. `scripts/check_channels.py` decodes pixels and
   settles it with no install.
2. **The file sizes argue it is not.** A 248 × 248 LZW TIF holding grayscale replicated
   across RGBA with constant alpha measures ~83 KB on synthetic cell-like content; three
   independent channels plus alpha measures ~175 KB, four independent ~197 KB. The real
   files are 78–92 KB. They carry roughly *one* channel's worth of entropy. This is
   indirect but quantitative, and it points at these zips being brightfield only.
3. **`S` files (78 KB) are consistently smaller than `P` files (92 KB), in both
   replicates.** That is `sperm_pbmc` Finding 1 showing up in file sizes: the PBMC wells
   have a higher background noise floor (σ 10.7 vs 5.5), noise does not compress, so the
   files are bigger. File size is tracking acquisition noise, not cell content — a
   reminder of how strongly the acquisition confound is present in this data.

### Open questions, revised

1. **Where is the fluorescence?** If the RGBA channels are duplicates, the marker signal
   is not in `Images\` at all. The obvious place to look is the run folder one level up,
   `Z:\Blair_Main\2026\260709_Blair_Sperm_Cytpix\` — an Attune writes its per-event
   detector measurements as FCS, and that export would be a sibling of `Images\`, not
   inside it. **Listing that directory is the next decisive step.**
2. **Why does `2SP` have 42,874 events when every other zip has exactly 30,000?** 30,000
   is a round number and looks like a collection cap; 42,874 does not. Either that
   acquisition was configured differently or the export was assembled differently.
3. **What is the 5,656-byte file in `2SP`?** Every other file in every zip is 73–105 KB.
   A 5.7 KB LZW frame at this size is nearly blank. Worth decoding before it lands in a
   training set — and `--max-member-bytes` will not catch it, since that guard is for
   files that are too *large*.
4. **Do `2*` and `3*` reuse event ids?** Filenames are bare integers (`10.tif`,
   `100002.tif`), so ids certainly collide across zips. Any index built for training has
   to be keyed on (class, event-id), never the id alone.
5. **Are the replicates the same panel?** Still open, and it now matters more: if the
   fluorescence lives in a separate FCS export, the detector-to-antigen mapping comes
   from that file's `$PnN` / `$PnS` keywords, and it may differ between replicates.

## Status

- [x] Repo + event-aware sampling tools scaffolded
- [x] Tooling rehearsed against fixtures in all three candidate layouts
- [x] Marker zips surveyed on the server — see above
- [ ] **RGBA channels checked for distinct signal** (`scripts/check_channels.py`) — next step
- [ ] **Run folder listed** to find the fluorescence export, if it is not in the images
- [ ] Working sample built and pulled back to the laptop
- [ ] Approach chosen (see [approach.md](approach.md))
