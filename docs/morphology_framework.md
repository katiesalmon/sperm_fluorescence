# Whole-ejaculate morphology from virtual fluorescence: the framework

*Plan, 2026-10-28. Builds on the ImageStream stems model (`runs/srv_isx_stems`: LDHC/AKAP4,
TOMM20, DAPI, ACRV1 from brightfield ×2 + darkfield at 60×, 0.333 µm/px) and the A8 model.
Nothing here is built yet; the sections "What exists" and "Facts that shape the design" are
measured, the rest is the proposal.*

## The idea in one paragraph

The four ImageStream markers are exactly the compartments WHO morphology is scored on: DAPI is
the nucleus, ACRV1 the acrosome (together the head), TOMM20 the midpiece, LDHC/AKAP4 the
principal piece. The model predicts each of them from label-free images at r = 0.77–0.98 within
sperm and at mask IoU 0.60–0.73 against the instrument's own masks. So for every object in an
unstained whole-ejaculate acquisition we can get compartment masks, measure them, and score the
sample three ways: with WHO-style rules (per-cell defects, % normal forms, a deformity index),
with unsupervised structure (clusters of head and tail shape), and at the population level
(sperm / round cells / debris, feature distributions, scatter populations). The stain is never
applied, so the same aliquot stays usable.

## What exists

| piece | state |
|---|---|
| Predicted stain images per object | `evaluate` returns them at native size; nothing writes them to disk yet |
| Compartment masks from predictions | `masks` command: per-channel threshold (`rel` 0.10–0.15 of the object's own peak), scored against IDEAS masks, IoU 0.60–0.73, 87–100 % detected |
| Population labels | `accuracy.groups_for`: sperm / round / DAPI-dim from the DAPI scalar (ISX); wells on the A8 |
| Object selection by rule | `slides.pick`, `panel.choose`, `panel.sperm_shaped` (nucleus area, flagellum-to-nucleus distance): the first two morphology features, already in use |
| IDEAS features for the stained tube | the `.daf` feature block (`analysis/isx_daf.py`): Area, Length, Width, Aspect Ratio, Circularity, Major/Minor axis and the rest, per channel mask, for all 20,000 objects |
| Geometry from stains | `isx_ch03_vs_ch11` analysis: stain centroid along the head→tail axis; the measurement that fixed the channel swap |
| Rendering | `render`, `slides`, `panel`: galleries by rule, one display range per marker |

## Facts that shape the design

1. **Tails fit the frame.** On the held-out half of the tube, the LDHC stain touches the frame
   edge on 1 % of objects (frames are object-sized, median 80 × 64 px = 27 × 21 µm). Principal
   piece length is therefore measurable, but a 27 µm frame holding a 45 µm tail means tails are
   folded or coiled in these fixed cells, which is what the panels show. Whether that is
   fixation, the flow cell, or the sample is the first thing to measure (tail length
   distribution, straight versus coiled).
2. **Resolution is marginal for WHO head thresholds.** WHO head width 2.5–3.5 µm is 7.5–10.5 px;
   the thresholds that separate normal from tapered or small heads are 2–3 px apart. Sub-pixel
   fits (ellipse to the DAPI+ACRV1 mask, not pixel counts) and calibration against IDEAS's own
   Length/Width on the same cells are required, and head-vacuole criteria (brightfield, not a
   stain) are probably out of reach at 0.333 µm/px.
3. **WHO morphology is defined on smears.** Fixed, flattened, Papanicolaou-stained cells scored
   at 1000× by eye. Cells in flow are in suspension, randomly oriented, imaged over a few µm
   depth of field, and not flattened. The numbers will not be WHO's numbers. The aim is
   *WHO-informed* criteria whose % normal forms tracks the andrology lab's manual result across
   patients, not a replication of the smear protocol.
4. **The model has only seen fixed, stained cells.** Every brightfield image it was trained on is
   of a fixed, permeabilised, antibody-stained cell. An unstained live aliquot may look different
   in brightfield (no fixation shrinkage, motility blur). The domain shift is the biggest risk in
   the whole plan, and it is testable with one acquisition: the same sample run unstained.
5. **Darkfield is required for the acrosome** (0.30 → 0.96). Brightfield-only instruments (CytPix)
   get head (DAPI, partly), midpiece and tail, not the acrosome, so no acrosome-area criterion.
6. **One sample so far**, split by acquisition order. Nothing population-level can be validated
   until there are several patients with manual morphology results.

## The pipeline

```
.cif (unstained or stained)
   └─ predict      predicted stain images per object  ──┐      (new; evaluate + writer)
   └─ segment      compartment masks + skeletons + QC   ├─ virtual-stain store (npz per class, like the cache)
   └─ features     one row per object, WHO-named measurements ─ features.csv / parquet
        ├─ who       rules → defects per sperm, % normal, TZI / SDI, galleries per defect
        ├─ cluster   feature-space and embedding clustering, UMAP, cluster galleries
        └─ population  sperm / round / debris counts, scatter populations, distributions, a per-sample report
   validate       the same features from the measured stains and from IDEAS, on the stained tube
```

### 1. `predict`: virtual stains for every object

Run a trained model over a `.cif` and write, per object, the predicted channels at native
size plus the label-free inputs it used, in the same ragged layout as the cache, so every
downstream command reads one store. For a stained acquisition the measured channels ride
along so validation is a column comparison. For an unstained acquisition the stain columns
are empty and only the predictions exist. (Half of this is `evaluate`; the writer and the
"cif without stain channels" path through `prepare` are new.)

### 2. `segment`: from stain images to geometry

Per object, from the four predicted channels:

- **Head** = DAPI mask ∪ ACRV1 mask, refined against the brightfield outline (the dark head
  in brightfield is the sharpest edge we have). Ellipse fit → length, width, area, aspect
  ratio, ellipticity residual (how un-oval), symmetry about the long axis.
- **Acrosome** = ACRV1 mask ∩ head: area fraction of the head (WHO: 40–70 %), position on the
  anterior pole (the geometry already measured: centroid 0.8 µm ahead of the nucleus).
- **Midpiece** = TOMM20 mask: skeleton length, mean width, straightness, angle between its axis
  and the head's long axis (axial insertion), insertion offset from the head's posterior pole,
  excess area beside the axis (cytoplasmic droplet proxy; brightfield can confirm).
- **Principal piece** = LDHC/AKAP4 mask minus the midpiece: skeleton path (`skimage.morphology.
  skeletonize` + longest-path), length, width profile, number of branches (double tails),
  breaks (components), curvature along the path, tortuosity (path length / end-to-end
  distance), coiled flag (path self-proximity), sharp-bend count.
- **QC flags**: in frame, single object (one nucleus), in focus (brightfield gradient energy),
  tail complete (path ends away from the frame edge), population (sperm / round / debris by
  predicted DAPI area and intensity, since the DAPI *scalar* will not exist on an unstained run).

Mask thresholds come from `masks.py` (already calibrated against IDEAS). Skeletons and fits
are scikit-image / numpy; no new dependency beyond scikit-image, which `requirements.txt`
does not yet list.

### 3. `features`: one row per object

A flat table with QC flags, population, and every measurement above in µm, plus the
intensity sums already used for event r. Also IDEAS-style descriptors (Area, Major/Minor
axis, Circularity, Aspect Ratio) computed the same way IDEAS defines them, so that on a
stained tube the predicted-mask features can be set against the `.daf`'s own columns
cell by cell. This table is the hand-off point: everything after it is statistics.

### 4. `who`: rules, defects, indices

A TOML config (`configs/morphology/who6.toml`) holding thresholds per criterion, with the
WHO 6th-edition strict values as the starting point and each threshold marked as
"calibrated" once it has been set against manual scores:

- head: length, width, length/width ratio, ellipticity residual, acrosome fraction, amorphous
  (residual above threshold), tapered, pyriform (width asymmetry front/back), round, small
- midpiece: length relative to head, width, bent (angle), asymmetric insertion (offset),
  thick / thin, residual cytoplasm
- principal piece: short (length < k × head length), coiled, bent (sharp-bend count),
  irregular width, double, broken
- outputs per sperm: a defect list; per sample: % normal forms, defect prevalences,
  **TZI** (defects per abnormal sperm) and **SDI** (defects per sperm, Aziz), plus a
  gallery per defect class drawn by `panel`-style rendering so a technician can audit each
  rule on real cells.

Calibration needs manual WHO scoring of the same samples (the lab's routine result, and ideally
a few hundred cells scored by a technician from the ImageStream brightfield images themselves,
so that the rule set is tuned to what the instrument shows rather than to a smear).

### 5. `cluster`: unsupervised structure, tails first

Two feature spaces, both worth trying:

- **Measured features**: the `features` rows (standardised) → UMAP → HDBSCAN. Interpretable;
  clusters are described by which features separate them.
- **Shape descriptors of the tail skeleton**: curvature sampled along the arc length (resampled
  to a fixed number of points), or elliptic Fourier descriptors of the path, rotation- and
  scale-normalised. This is the direct answer to "unsupervised clustering of tail morphology":
  straight, hooked, coiled, hairpin, double emerge as clusters without naming them first.
- **Encoder embeddings**: the stems U-Net bottleneck already encodes what the model needed to
  paint the tail; pooled bottleneck vectors cluster for free and can be compared with the
  feature-space clusters.

Each cluster gets a gallery (the same rendering as the panels) and a feature summary. Stability
across the two halves of the tube and across patients is the test that a cluster is real.

### 6. `population`: the sample report

Per acquisition: object counts by population (sperm / round cells / debris; on the A8 also
CD45-positive leukocytes), % normal forms with the WHO rule set, defect prevalences, TZI/SDI,
feature distributions (head length, tail length, tortuosity), a darkfield-versus-DAPI (ISX) or
FSC-versus-SSC (A8) scatter coloured by predicted marker sums, which is the "population
scatter" of the goals list, and the cluster composition. One HTML or PDF per sample, and a CSV
row per sample for cross-patient comparison.

## Validation, in the order it can be done

1. **Now, on the stained tube** (no new data): features from predicted masks versus features
   from measured masks versus IDEAS's `.daf` columns, cell by cell. Per-feature r and bias.
   This says which measurements survive virtual staining at 60× and which do not (expect head
   length and tail length to be fine, head width and midpiece width marginal).
2. **Now**: tail length and coiling distribution on the stained tube, to decide whether the
   fixed cells' tails are usable for principal-piece criteria at all.
3. **One acquisition**: the same ejaculate unstained (and unstained-but-fixed, to separate
   fixation from staining). Compare the feature distributions from predicted masks between
   stained and unstained runs. If they match, the domain-shift risk is retired; if not, the
   model needs unstained brightfield in training (a self-supervised or paired-aliquot scheme).
4. **Several patients with manual morphology**: % normal forms and TZI from the pipeline
   against the lab's manual result; the rule thresholds are calibrated here, on half the
   patients, and tested on the other half. This is the clinical claim and it cannot be made
   from one sample.
5. **Technician audit** of defect galleries: precision of each rule on a few hundred cells.

## Data to acquire

- The same sample unstained, on the ImageStream (step 3 above). This is the single most
  informative acquisition available.
- A tail-capture test: the same sample at 40× (0.5 µm/px, wider frame) versus 60×, to see
  whether coiling is real or a frame effect, and whether 40× keeps the acrosome.
- Patients: normospermic and teratozoospermic ejaculates with routine semen analysis and
  manual morphology on record; ten of each is the minimum for calibrating and testing rules.
- A few hundred ImageStream brightfield images scored by a technician into WHO categories.

## Risks, ranked

1. Domain shift from fixed-stained to unstained cells (testable with one run).
2. Fixed cells' tails coiled, so principal-piece criteria measure fixation, not the sperm.
3. Resolution at the head: WHO width thresholds are 2–3 px apart; vacuoles likely unmeasurable.
4. WHO numbers will not transfer; the deliverable is agreement with manual scoring across
   patients, which needs patients.
5. n = 1 for everything so far.

## Build order

1. `predict` + the store (a day): unblocks everything, including the unstained run.
2. `segment` + `features` on the stained tube, with the predicted-versus-measured-versus-IDEAS
   comparison (validation step 1) as the first result.
3. `who` with uncalibrated WHO thresholds and defect galleries, so the lab can react to real
   cells before any calibration data exists.
4. `cluster` on tails.
5. `population` report, then the A8 version (sperm / leukocyte / debris counts and the FSC–SSC
   scatter; no morphology there beyond head and tail presence).
