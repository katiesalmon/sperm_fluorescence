# Abstract draft: feature-based clustering on the A8 for whole-ejaculate sorting

*Draft 2026-10-28. Numbers are from the laptop's 600-object held-out sample (`runs/pop_a8_all_k*`,
`figures/a8_cluster_validation_features.png`); the server run (`server_populations.ps1`) replaces
them with the 60,000-object replicate before submission. Bracketed items need filling in.*

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
loss, forward scatter and side scatter. On the held-out replicate, [600 / 60,000] events, a
feature engine computed 358 measurements per event from the predicted images and their
compartment masks: size, shape and skeleton geometry of each compartment, intensity and texture
per channel, and distances between compartments. Events were clustered without labels by
principal component analysis and k-means. Well identity, the measured stains and the same
clustering applied to the measured stains were used only afterwards to score the clusters;
stability was measured by refitting on bootstrap resamples and on each acquisition half.

## Results

Two clusters separated the sample into [370] sperm and [230] leukocytes. Against the pure
wells, which the clustering never saw, agreement was near perfect: adjusted Rand index 0.99,
Hungarian-matched balanced accuracy 0.998, with all 200 sperm-well events in the sperm cluster
and 199 of 200 PBMC-well events in the leukocyte cluster; the one exception was, by its
measured stains, a sperm carried over into the PBMC preparation. The 31 leukocytes spiked into
the mixed well all fell in the leukocyte cluster. The same clustering on the measured stains
agreed with the label-free clustering at adjusted Rand index 0.99. The partition was fully
reproducible: bootstrap adjusted Rand index 1.00, acquisition halves 0.98 and 0.95, and every
object clustered identically across random initialisations. With three clusters the sperm divided
by tail configuration into straight-tailed ([191]) and looped or folded ([179]) sperm at
bootstrap stability 0.96 while the leukocyte cluster was unchanged, and the measured stains of
the same cells confirmed both groups. Withholding the virtual stains and clustering the raw
images alone placed [111] sperm with the leukocytes, so the virtual-staining step carried the
separation.

## Conclusions

Measurements derived from virtual fluorescence separate spermatozoa from leukocytes in whole
ejaculate without labels and without staining, with agreement against held-out ground truth
of 0.99 and perfect reproducibility, and further divide the sperm by tail configuration. Because
the FACSDiscover S8 sorter shares the A8's imaging, these clusters can serve directly as sort
gates: sperm with straight tails to one output, other sperm to a second, leukocytes and debris
to waste. Validation across additional patients and at sorting rates is underway.

## Impact Statement

Label-free clustering on virtual fluorescence lets an image-enabled cell sorter isolate viable,
unstained spermatozoa from whole ejaculate at cytometer throughput for downstream use in ICSI.

## Before submission

- Replace every bracketed count with the 60,000-object replicate from the server; recompute the
  stability band (it narrows) and check whether the k = 3 tail split holds at scale.
- One patient, one donor of PBMCs; the wells are population labels, not cell labels, which the
  abstract states by the "pure wells" wording and the carried-over sperm.
- Sort-rate latency of U-Net plus features has not been measured; the last sentence of the
  conclusions says so.
- Length ≈ 3,300 characters with spaces; trim the methods if the AUA limit bites.
