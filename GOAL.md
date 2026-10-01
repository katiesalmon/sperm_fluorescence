# Goal

**Predict how brightly a sperm cell stains, from a picture of it that was never stained.**

Two instruments imaged the same samples on the same day. The Attune CytPix photographs
each event in brightfield and measures its fluorescence as four numbers. The BD
FACSDiscover A8 images each event label-free *and* in three fluorescence channels,
pixel-registered. We want a model that sees only the label-free picture and predicts
the stain — as a number for all four markers, and as a picture for the three the A8
imaged. The labels are free: no one annotates anything, because the
cytometer already recorded the answer for every imaged event.

## The four targets

| Marker | What it reports | Where it lives on a sperm | Ground truth available |
| --- | --- | --- | --- |
| DAPI | DNA content | nucleus — the head | scalar only (both instruments) |
| ACRV1 | acrosomal integrity | the cap over the anterior head | scalar + **per-pixel image** (A8) |
| LDHC + AKAP4 | flagellar proteins, pooled on one fluor | the principal piece — the tail | scalar + **per-pixel image** (A8) |
| CD45 | pan-leukocyte | absent from sperm entirely | scalar + **per-pixel image** (A8) |

## Why it matters

If brightfield carries the signal, a routine unstained image yields compartment-level
information that currently costs a stain, a wash, and a fluorochrome. If it does not, that
is worth knowing precisely — "morphology predicts DNA content and leukocyte identity but
not acrosomal integrity" is a real result, and the one the evidence currently points at.

## What success looks like

A per-marker table of correlations between predicted and measured intensity, that:

1. is **held out by replicate** — trained on replicate 2, tested on replicate 3;
2. uses **compensated** targets, since the raw values carry heavy spectral spillover;
3. **beats the morphology floor** already measured (`docs/results.md`): DAPI 0.78,
   CD45 0.77, LDHC 0.38, ACRV1 0.21;
4. **survives the negative controls** — predicted ACRV1 and LDHC on a PBMC should be near
   zero, predicted DAPI should not, and a permuted-target model should sit at zero.

Anything that clears all four is a finding. A marker that fails is also a finding, as long
as it fails for a stated reason.

## What this is not

- Not a classifier. The sibling repo `sperm_pbmc` sorts events into cell types; this
  predicts a continuous measurement.
- Not *only* scalar prediction any more. The CytPix recorded four numbers per event and
  no fluorescence picture; the BD FACSDiscover A8 run of the same samples recorded
  per-pixel fluorescence images for three of the four markers. Image-to-image virtual
  staining is in scope for those three; DAPI stays scalar-only.
- Not about Tomm20. It was in the original brief but is not in the experiment.

## Where things stand

The data is understood, the targets are extractable, and the morphology floor is measured.
The next step is a network on pixels with compensated targets.

- [docs/task_brief.md](docs/task_brief.md) — what the data turned out to be
- [docs/approach.md](docs/approach.md) — prior art and the sequence to run
- [docs/results.md](docs/results.md) — what has actually been measured
