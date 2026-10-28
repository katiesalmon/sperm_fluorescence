# Whole-ejaculate morphology from virtual fluorescence: the framework

*Rewritten 2026-10-28 around a feature engine and population analysis, after the survey in
`morphology_research.md`. The first draft was a rule engine on compartment masks; the survey
showed that fixed rules are the arm of the field that agrees worst with manual scoring, that
IDEAS itself analyses populations by gating a large generic feature catalogue, and that
unsupervised clustering of label-free imaging-cytometry features with stains used only as
post-hoc truth is an established design (Zhang 2023) that nobody has applied to sperm. The
"What exists" and "Facts" sections are measured; the rest is the proposal.*

## The idea

The four ImageStream markers are the compartments morphology is scored on: DAPI the nucleus,
ACRV1 the acrosome (together the head), TOMM20 the midpiece, LDHC/AKAP4 the principal piece.
The stems model predicts them from label-free images at r = 0.77–0.98 within sperm, with
masks at IoU 0.60–0.73 against the instrument's own. So an unstained acquisition can be given
the same per-object feature table the instrument would have computed on a stained one. The
analysis then proceeds the way flow cytometry always has: a large feature vector per object,
populations found by gating and clustering in that space, labels attached to populations, and
clinical quantities read off as counts over labelled populations. Hand-written morphology rules
do not decide anything; they survive only as a sanity check on the population named "ideal".

```
.cif (unstained or stained)
   └─ predict     predicted stain images per object ─────────────┐
   └─ masks       compartment masks from the predictions         ├─ virtual-stain store
   └─ features    the catalogue: generic features × masks × channels, plus learned embeddings ─ features table
        ├─ populate   gates, PCA/UMAP, clustering, galleries per population
        ├─ label      labels for populations: stains (free), technician tags, named clusters
        └─ report     WHO quantities as counts over labelled populations; per-sample report
   validate    features vs IDEAS on the stained tube; cluster stability; post-hoc stains; patients
```

## What exists

| piece | state |
|---|---|
| Predicted stain images | `predict` writes them for a whole cache to the virtual-stain store (`data/store.py`): predicted images, predicted masks from the run's calibrated thresholds, IDEAS masks of the measured channels, indexed like the cache; `prepare --unstained` makes the cache for an unstained .cif |
| Compartment masks | `masks`: per-channel threshold (`rel` 0.10–0.15 of the object's peak), IoU 0.60–0.73 vs IDEAS, 87–100 % detected |
| Feature engine | `features` (`sperm_fluorescence/features/`): the catalogue below over a store, CSV + column definitions, reference features on the IDEAS masks, and `--daf` validation (built 2026-10-28) |
| Population labels | `accuracy.groups_for`: sperm / round / DAPI-dim by the DAPI scalar (ISX); wells on the A8; sperm-vs-PBMC AUC 0.95 on predicted CD45 (A8) |
| Object selection by rule | `panel.sperm_shaped` (nucleus area, flagellum-to-nucleus distance), `slides.pick`: first features, already used |
| IDEAS's own features | the `.daf` feature block (`analysis/isx_daf.py`): Area, Length, Width, Aspect Ratio, Circularity, Major/Minor axis, intensities, per channel mask, all 20,000 objects of the stained tube. Decoded intensities match IDEAS at r = 1.000 |
| Rendering | `render`, `slides`, `panel`: galleries by rule, one display range per marker |

## Facts that shape the design

1. **Tails fit the frame.** The LDHC stain touches the frame edge on 1 % of held-out objects
   (frames are object-sized, median 80 × 64 px = 27 × 21 µm), so tail features are
   computable, but a 45 µm tail in a 27 µm frame means tails are folded or coiled in these
   fixed cells. Whether that is fixation, flow or the sample is the first measurement.
2. **Resolution.** 0.333 µm/px at 60×. WHO head width limits (2.5–3.2 µm) are 7.5–9.6 px
   apart; midpiece width (0.5–0.7 µm) is under 2 px and not measurable as a width. Sub-pixel
   fits and calibration against IDEAS Length/Width on the same cells are mandatory.
3. **60× is required for the acrosome** (our ablation: 0.30 → 0.96 with darkfield; the boar
   data: acrosome learnable only at 60×). The bovine group chose 40× to fit whole cells; a 40×
   acquisition is a measured trade-off for tails, not a default.
4. **Domain shift is expected.** The model has seen only fixed, stained cells; the bovine
   model lost 6–8 points between fresh and frozen. One unstained acquisition of the same
   sample measures ours.
5. **The clinical reference is noisy.** Automated morphology agrees poorly with manual
   scoring, and manual scoring varies between technicians. Cell-level agreement of
   measurements and population-level stability are stronger validations than matching one
   manual percentage.
6. **One sample so far**, split by acquisition order.

## 1. `predict` and the virtual-stain store (built 2026-10-28)

`prepare` makes the cache (stained: classes 2I/3I; `--unstained`: one class 1I with zero
targets and nothing marked available), and `predict` runs a model over it and writes the
store: per object the predicted channels in cache units, a predicted mask per channel from
the run's calibrated threshold rule, the IDEAS masks of the measured channels when the
`.cif` is given, and the cache's own index arrays, so `open_store` and `open_cache` over the
same classes index identically. Inputs and measured channels stay in the cache; the feature
engine opens both. One caveat for the features stage: the `rel` mask rule is relative to
each object's own peak, so every object gets a non-empty mask, including objects with no
stain; mask presence is not a detection and the intensity features must carry that.

## 2. The feature engine (built 2026-10-28; `sperm_fluorescence/features/`)

Status: `features` computes everything below except the learned embeddings, lobe count and
symmetry, and writes one CSV row per object with a JSON of column definitions beside it.
Masks: nucleus, acrosome, head, midpiece, principal piece, flagellum, tail, cell (a
brightfield proxy). Reference columns (`ref.*`) are the same features on the IDEAS masks of
the measured stains, so validation step 1 is a column comparison; `--daf` runs it.

The catalogue is deliberately generic and large, after IDEAS: nothing in it is a sperm
defect. Features are computed for every (mask, channel) pair that makes sense, and the table
is the hand-off to everything else.

**Masks** (from `masks`, plus derived ones by the same generic operators IDEAS offers):
nucleus (DAPI), acrosome (ACRV1), head = nucleus ∪ acrosome refined by the brightfield outline,
midpiece (TOMM20), principal piece (LDHC/AKAP4 minus midpiece), whole tail (midpiece ∪
principal piece), whole cell (brightfield object mask), and the skeletons of midpiece, tail
and whole cell. Derivations are limited to threshold, erode/dilate, fill, largest component,
skeleton, and boolean combinations, each with its parameters recorded.

**Per-mask shape features** (the IDEAS and CASA-Morph sets, formulas fixed in the config):
area, perimeter, Feret length and width (0°, 30°, …, 150°), major and minor axis of the fitted
ellipse (sub-pixel, from moments), aspect ratio, ellipticity L/W, elongation (L − W)/(L + W),
rugosity 4πA/P², regularity πLW/4A, circularity, compactness, thickness min/max, lobe count,
symmetry 2/3/4, centroid, bounding box, and for skeletons: path length, end-to-end distance,
tortuosity (path / end-to-end), maximum and mean curvature along the path, number of
sharp bends, branch count, endpoint count, self-proximity (coiling), and the curvature profile
resampled to a fixed number of arc-length points.

**Cross-mask features:** delta centroid between any two masks (the stain-geometry measurement
that fixed the channel swap), angle between the major axes of head and midpiece (insertion
alignment), insertion offset of the midpiece from the head's posterior pole, acrosome area ÷
head area, nucleus area ÷ head area, midpiece length ÷ head length, tail length ÷ head length,
TOMM20 area ÷ midpiece skeleton length (the midpiece-thickness proxy), area of brightfield
object outside every stain mask (cytoplasmic residue proxy).

**Per-channel intensity and texture features** on each mask, for predicted and (when
present) measured channels: integrated intensity, max pixel, mean, background, bright detail
intensity, gradient RMS (focus), contrast, standard deviation, Haralick entropy and
homogeneity; and between channels: similarity and bright-detail similarity.

**Learned embeddings**, as further columns: the pooled bottleneck of the stems U-Net (what the
model needed to paint the stains), and a convolutional autoencoder trained on the label-free
crops alone (the Zhang 2023 design), each as a vector. These are the columns clustering tends
to find structure in that engineered features miss, and the engineered columns are how such
structure is read back.

**Quality features** that later become gates: brightfield gradient RMS, object count per
frame, number of nuclei, stain-touches-edge fraction, skeleton endpoints near the frame edge,
and the population-from-DAPI feature (predicted DAPI area and intensity, since the DAPI scalar
does not exist on an unstained run).

The table has one row per object and the feature definitions as a versioned header. Scikit-
image is the only new dependency.

## 3. The population engine (built 2026-10-28; `sperm_fluorescence/population/`)

Status: `populate` does the gates, feature-set selection, PCA, k-means / GMM / HDBSCAN,
halves-and-bootstrap stability, discriminating features and galleries described below;
UMAP falls back to t-SNE until umap-learn is installed. Gates are pandas queries, so a gate
is a line of text in the summary, never code.

Three operations on the feature table, all producing *populations* (sets of object ids with a
name and a provenance):

- **Gates.** Thresholds and polygons on one or two features, drawn on data, saved as a
  template and applied in batch; hierarchical, like IDEAS. The first gates are the quality
  gates (in focus, single object, whole cell in frame, not overlapped) and the coarse
  populations (sperm / round cells / debris / free heads / free tails), and every gate's loss
  is reported.
- **Reduction and clustering.** Standardise, PCA (Kaiser criterion on the engineered
  features, the established morphometry recipe), UMAP for display, then clustering: GMM or
  k-means for the conventional three-cluster head analysis, HDBSCAN for structure whose count
  is unknown (tails). Run on three feature spaces separately and compared: engineered features
  only, each embedding only, and all together.
- **Galleries.** Every population gets a panel of its nearest-to-centre and most-peripheral
  members drawn with the existing rendering, and a table of the features that separate it
  from the rest (the Feature Finder idea: rank features by discriminant ratio against the
  complement). A population without a readable gallery is not named.

**Stability** is the test that a population is real: the same clustering on the two halves
of the tube, on bootstrap resamples, and later across patients, with adjusted Rand index
and per-cluster persistence reported. The cluster count is conventional in the literature;
stability is what stands in for a principled choice.

## 4. Labels

Labels attach to populations, never to thresholds, in descending order of trust:

1. **Stains**, free and objective. Measured CD45 (A8) labels leukocytes; a DAPI mask with no
   flagellum mask is a free head; a flagellum mask with no nucleus is a free tail; two nuclei
   in one object is an aggregate; the DAPI scalar labels round cells and debris. On a stained
   tube these labels are the post-hoc truth for every unsupervised population, exactly as in
   Zhang 2023; on an unstained run they are what the predicted channels reproduce.
2. **Technician tags.** A few hundred cells per category, tagged from the ImageStream images
   in the WHO decision order (residue → tail → midpiece → head), so that each tag is one of
   the four regional flags. A classifier on the feature table (a shallow tree or logistic
   model, so its thresholds are readable) learns the gate; Feature Finder-style ranking says
   which features carried it.
3. **Named clusters.** An unsupervised population gets a name only after gallery review, with
   the reviewer and date recorded. These rows carry the least trust and are the ones the
   stability test is for.

A categorical CNN on technician-labelled brightfield crops (the bovine and boar design, 5–8
classes plus "multiple" and "debris") is kept as a parallel arm, because it is the one the
IFC literature has validated, and the vendor's Amnis AI module is run once as a baseline.
Disagreement between the CNN and the feature-space populations is itself a finding.

## 5. `report`: WHO quantities as a mapping layer

The report is the WHO 6th-edition structure, read off labelled populations. Per sperm: a
1-plus-4 record, ideal or abnormal plus the four regional flags (head, neck/midpiece, tail,
excess residual cytoplasm). Per sample: % ideal, %H, %NM, %T, %C (summing above 100 if
multiple entry is right), TZI = total defects ÷ abnormal sperm (1.00–4.00), MAI and SDI as
extras, free heads and free tails tallied separately (reported when over 20 per 100), and
non-sperm objects counted separately.

**The mapping** is a versioned config with one row per population: the population, the WHO
flag it carries, the label source (stain / technician / named cluster), the evidence (gallery
path), the reviewer. A population can carry more than one flag; several populations can carry
the same one. No thresholds live here, only correspondences; when the clustering changes the
config is re-derived and the galleries re-audited.

```toml
[[map]]
population = "tail_hdbscan_3"       # from populate
flag       = "T"                    # H | NM | T | C
source     = "named_cluster"        # stain | technician | named_cluster
evidence   = "figures/populations/tail_hdbscan_3.png"
reviewer   = "…"
date       = 2026-11-…
note       = "coiled > 360°, self-proximity high, tortuosity > 3"
```

**Decision order.** Each object passes the WHO gate first (is it a sperm; can it be
classified: in focus, single, whole, not overlapped), and failures leave the denominator but
enter a gate-loss table. Then the flags are read in the manual's order, residue, tail,
midpiece, head; ideal is the absence of all four; borderline populations map to a defect
because the manual says borderline is abnormal, which fixes the tie in a direction that
cannot inflate the normal count.

**Sanity check, not rule.** The manual's reference dimensions (head length 3.7–4.7 µm,
width 2.5–3.2, L/W 1.3–1.8, acrosome 40–70 % of head area, midpiece ≈ one head length, tail ≈
ten) are compared once with the measurements of the population mapped to ideal. A miss points
at either a biased feature or a misnamed population, and the provenance says which.

**What the report states beyond the numbers:** the sample size and its sampling error (200
sperm is the manual's minimum; 1,500 distinguishes 3 % from 5 %), the share of each flag that
came from stains, technician gates or named clusters, and the gate losses (the bovine IFC work
kept 24 % of events).

**The A8 version** has no head or tail morphology; its report is the non-morphology rows:
sperm / leukocyte / debris from predicted CD45 and LDHC with darkfield texture, free heads and
tails if the masks support them, non-sperm counts. That is the "other cells and specific
defects" part of the WHO report and the part a sorter needs.

## 6. Validation, in the order it can be done

1. **Now, feature against feature.** On the stained tube, every feature the catalogue shares
   with IDEAS, computed on predicted masks, against the `.daf` value on the real masks, cell
   by cell: Bland–Altman with acceptance limits from the reference's own repeatability (the
   QART design), per feature. This says which measurements survive virtual staining at 60×.
2. **Now, the conventional result.** PCA plus clustering on the head features should give
   three subpopulations of roughly a third each, as in every human morphometry paper. If it
   does not, something upstream is wrong before anything new is claimed.
3. **Now, post-hoc stains.** Unsupervised populations on label-free features scored against
   the stain-derived labels (round cells, free heads, aggregates; on the A8, leukocytes): the
   balanced accuracy Zhang reported for blood cells is the comparator.
4. **Now, stability.** Tube halves and bootstraps.
5. **One acquisition.** The same sample unstained (and unstained-but-fixed): feature
   distributions and population proportions against the stained run. This retires or
   confirms the domain-shift risk; the fix if it fails is training on unstained brightfield.
6. **Patients.** Ten normospermic and ten teratozoospermic ejaculates with manual morphology on
   record: % ideal and TZI from the report against the lab's numbers, judged against the
   known spread between technicians, and population proportions across patients.
7. **Technician audit** of every mapped population's gallery.

## Data to acquire

- The same sample unstained on the ImageStream (step 5): the single most informative run.
- The same sample at 40× and 60×, to settle coiling versus frame and acrosome loss.
- Patients with routine semen analysis and manual morphology on record.
- A few hundred ImageStream images tagged by a technician in the WHO decision order.

## Risks, ranked

1. Domain shift from fixed-stained to unstained cells (testable with one run).
2. Clusters that are artefacts of fixation, focus or flow orientation rather than biology;
   stability and the post-hoc stains are the defence.
3. Resolution at the head and midpiece; vacuoles and midpiece width are probably out of reach.
4. Cluster naming is judgement; galleries and recorded provenance are the only control.
5. n = 1 for everything so far.

## Build order

1. `predict` and the store (unblocks everything, including the unstained run).
2. The feature engine on the stained tube, with validation step 1 as its first result.
3. `populate` with galleries and stability; validation steps 2–4.
4. `label` and `report` with the mapping config; the conventional three-cluster head result and
   the stain-derived rows first, technician tags when they exist.
5. The autoencoder embedding and the CNN arm.
6. The unstained acquisition, then patients.
