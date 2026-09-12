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

## Open questions the survey must answer

Run `scripts/inspect_zip.py` on the server before deciding anything below it.

1. **Is the marker signal an image at all?** The CytPix images brightfield; an Attune's
   fluorescence detection is by PMT. The marker signal may be a *per-event intensity* in
   an FCS sidecar rather than a picture. `--peek-metadata` parses the FCS TEXT segment
   and prints the `$PnN` → `$PnS` table, which maps detector (`VL1-A`) to antigen
   (`DAPI`) — the single most useful thing the survey can return.
2. **If images: one file per channel, or one multi-channel file?** `--structure` reads
   the TIFF headers (no pixels) and reports pages, samples-per-pixel and any
   `ImageDescription`, which is where OME-TIFF and ImageJ record channel names.
3. **Which channel is which?** Four markers plus brightfield is five channels. Ordering
   has to come from the file itself — a name, a page description, or the FCS table —
   never from an assumption about channel order.
4. **What bit depth?** Replicate 1 was uint8. Fluorescence intensity spanning four
   decades in uint8 would be badly quantised, and if it is uint16 the loading path
   differs from `sperm_pbmc`'s.
5. **Is the event count still 30,000 per zip, and do 2* and 3* image the same events?**
   They are separate acquisitions of separate aliquots, so almost certainly not — but if
   event ids collide between replicates, a naive split would leak.
6. **Are the frames registered to each other?** For a per-channel layout, whether the
   marker frame is pixel-aligned with the brightfield frame decides whether the task is
   image-to-image at all, or whether it collapses to predicting summary intensities.

## Status

- [x] Repo + event-aware sampling tools scaffolded
- [x] Tooling rehearsed against fixtures in all three candidate layouts
- [ ] Marker zips surveyed on the server — **next step**
- [ ] Sampling parameters chosen from the survey
- [ ] Working sample built and pulled back to the laptop
- [ ] Approach chosen (see [approach.md](approach.md) for the decision tree)
