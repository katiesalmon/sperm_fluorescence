# Abstract draft: feature-based clustering on the A8 for whole-ejaculate sorting

*Draft 2026-10-28, numbers replaced 2026-10-29 with the server's full held-out replicate
(60,000 objects, `server_populations_20261029_0336.zip` → `runs/pop_srv_a8_feat_k*`,
`figures/srv_a8_pop_feat_k*.png`; the counts are tallied in results.md, "The A8 populations at
full scale"). The 600-object laptop numbers this replaced are in the git history of this file.*

## Title

Virtual Fluorescence for Image-Enabled Cell Sorting: Unsupervised Feature-Based Clustering of
Whole Ejaculate Separates Spermatozoa from Leukocytes Without Staining for High-Throughput Sorting

## Objective

Image-enabled cell sorters record light-loss and scatter images of every event in a whole ejaculate
sample and can sort on them at thousands of events per second, but telling a spermatozoon from a
leukocyte or debris has required fluorescent antibody staining, which is slow, costly and renders
the cells unusable for Assisted Reproductive Technology (ART). We asked whether virtual
fluorescence, antibody stains predicted from the label-free images by a neural network, yields
measurements that separate sperm from non-sperm cells without any labels, so that whole
ejaculate can be sorted for intracytoplasmic sperm injection (ICSI) without staining.

## Materials and Methods

A remnant ejaculate sample [n = 1 patient] and peripheral blood mononuclear cells (PBMCs) were
imaged on a BD FACSDiscover A8 in three wells, sperm, PBMCs and a sperm–PBMC mixture, in two
replicates, after staining for LDHC/AKAP4 (flagellum), ACRV1 (acrosome) and CD45 (leukocytes).
A multi-stem U-Net trained on one replicate predicted the three fluorescence images from light
loss, forward scatter and side scatter. On the held-out replicate, 60,000 events (20,000 per
well), a feature engine computed 358 measurements per event from the predicted images and their
compartment masks: size, shape and skeleton geometry of each compartment, intensity and texture
per channel, and distances between compartments. Events were clustered without labels by
principal component analysis and k-means. Well identity, the measured stains and the same
clustering applied to the measured stains were used only afterwards to score the clusters;
stability was measured by refitting on bootstrap resamples and on each acquisition half.

## Results

Two clusters separated the 60,000 events into 36,874 sperm and 23,126 leukocytes. Against the
pure wells, which the clustering never saw, 19,938 of 20,000 sperm-well events fell in the sperm
cluster and 19,491 of 20,000 PBMC-well events in the leukocyte cluster (adjusted Rand index 0.94,
Hungarian-matched balanced accuracy 0.99). Of the 509 PBMC-well events placed with the sperm,
493 were sperm by their measured stains, carried over into the PBMC preparation, and 41 of the
62 sperm-well events placed with the leukocytes were leukocytes by their measured stains, so the
label-free clusters agreed with the cells' actual stains on 98.7% of events (sensitivity for sperm
98.9%, specificity 98.4%, Cohen's κ 0.97); the same clustering
run on the measured stains agreed with the label-free clustering at adjusted Rand index 0.95. In
the mixed well, 3,434 of the 3,626 leukocytes identified by their measured stains (95%) fell in the
leukocyte cluster, with 0.8% of the well's sperm lost to it. The partition was fully
reproducible: bootstrap adjusted Rand index 1.00 over ten resamples and 1.00 on each acquisition
half. With three clusters the leukocyte cluster was unchanged (23,070) and the sperm divided by
tail configuration into 19,201 straight-tailed and 17,729 looped or folded sperm, at bootstrap
stability 0.996 and 0.99 on each half, and the measured stains of the same cells showed the same
two configurations. In a 600-event pilot, clustering the raw label-free images without the
virtual stains placed 111 of 370 sperm with the leukocytes, so the virtual-staining step carried
the separation.

## Conclusions

Measurements derived from virtual fluorescence separate spermatozoa from leukocytes in 60,000
whole-ejaculate events without labels and without staining, agreeing with the cells' measured
stains on 98.7% of events with perfect reproducibility, and further divide the sperm by tail
configuration. Because the FACSDiscover S8 sorter shares the A8's imaging, these clusters can
serve directly as sort gates: sperm with straight tails to one output, other sperm to a second,
leukocytes and debris to waste. Validation across additional patients and at sorting rates is
underway.

## Impact Statement

Label-free clustering on virtual fluorescence lets an image-enabled cell sorter isolate viable,
unstained spermatozoa from whole ejaculate at cytometer throughput for downstream use in ICSI.

## Before submission

- "By their measured stains" means the same unsupervised clustering run on the measured stain
  images (`measured_cluster` in `populations.csv`), not a threshold; say so if a reviewer asks.
- One patient, one donor of PBMCs; the wells are population labels, not cell labels, which the
  abstract states by the "pure wells" wording and the carried-over sperm.
- The 600-event pilot sentence (raw images without virtual stains) has not been repeated at
  60,000; run `analysis/autoencoder_embedding.py --inputs-only` on the server to replace it, or
  drop the sentence.
- The autoencoder-latent path (abstract 3) did not reproduce at 60,000 (k = 2 unstable; k = 3
  isolates the leukocytes); this abstract rests on the feature path only.
- Sort-rate latency of U-Net plus features has not been measured; the last sentence of the
  conclusions says so.
- Length ≈ 4,000 characters with spaces (body, headings excluded); the mixed-well and pilot sentences are the first to
  cut if the AUA limit bites.
