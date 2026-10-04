# Goal

**Predict how brightly a sperm cell stains, from a picture of it that was never stained.**

The BD FACSDiscover A8 images each event label-free — extinction, forward scatter and
side scatter, pixel-registered — and in three fluorescence channels at the same instant.
We want a model that sees only the label-free pages and predicts the stain: as a picture
for the three markers the A8 imaged, and as a number for all four.

## The four targets

| Marker | What it reports | Where it lives on a sperm | Ground truth |
| --- | --- | --- | --- |
| LDHC + AKAP4 | flagellar proteins, pooled on AF488 | the principal piece — the tail | **per-pixel image** + unmixed scalar |
| CD45 | pan-leukocyte, on PE | absent from sperm entirely | **per-pixel image** + unmixed scalar |
| ACRV1 | acrosomal integrity, on PerCP-eF710 | the cap over the anterior head | **per-pixel image** + unmixed scalar |
| DAPI | DNA content | nucleus — the head | unmixed scalar only (violet-excited; not imaged) |

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
   zero, predicted CD45 on a sperm should be near zero, and a permuted-target model should
   sit at zero. (Unstained replicate 1 is *not* a control: a correct model predicts the
   stain there too.)

Anything that clears all four is a finding. A marker that fails is also a finding, as long
as it fails for a stated reason.

## What this is not

- Not a classifier. The sibling repo `sperm_pbmc` sorts events into cell types; this
  predicts a stain.
- Not about the Attune CytPix. Those archives were surveyed first and taught us the
  experimental design — replicate 1 unstained, the panel, the batch effects — but the
  samples were run for the A8, and the A8 is the dataset. The CytPix findings stay in
  the docs as history.
- Not about Tomm20. It was in the original brief and is not in the panel.

## Where things stand

Full-scale virtual staining has run, both replicate folds. Within cell type — the number
that cannot be earned by recognising the cell — LDHC/AKAP4 is predicted from label-free
scatter at r ≈ 0.72, CD45 at ≈ 0.55 within PBMCs, and ACRV1 at ≈ 0.25 within sperm:
small but real, and probably acrosome extent rather than acrosomal state. The next run
fixes the loss weighting and checkpoint selection that both disadvantaged ACRV1.

- [docs/task_brief.md](docs/task_brief.md) — what the data turned out to be
- [docs/results.md](docs/results.md) — what has actually been measured
- [docs/paths.md](docs/paths.md) and [docs/framework.md](docs/framework.md) — approaches
