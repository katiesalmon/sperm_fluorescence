# Results

Measured findings, newest first. Everything here is reproducible from the scripts named
alongside it. The plan these test is in [approach.md](approach.md); what the data turned
out to be is in [task_brief.md](task_brief.md).

---

## ImageStream pilot — all four markers predicted within nucleated sperm at r ≈ 0.92–0.98

*2026-10-04 · `analysis/prepare_isx.py`, `train_a8.py` (width 32, 20 epochs, `--channel-scale`,
laptop MPS, 21 min), `eval_subset.py`. One sample, 20,000 objects; trained on the first
10,000 by acquisition order, tested on the second 10,000. Inputs Ch01 + Ch09 (two
brightfield cameras) + Ch06 (side scatter); targets the IDEAS-compensated Ch02/Ch03/Ch07/Ch11.*

| within nucleated sperm (6,241 test objects) | pixel r | event r | morphology floor (42 IDEAS features) |
| --- | ---: | ---: | ---: |
| LDHC/AKAP4 | 0.947 | **0.972** | 0.932 |
| ACRV1 | 0.950 | **0.981** | 0.912 |
| DAPI | 0.923 | **0.922** | 0.811 |
| TOMM20 | 0.942 | **0.976** | 0.719 |

Whole-sample numbers are within 0.01 of these for the three sperm markers; DAPI drops to
0.40 on the DAPI-low fragments, as it should — they have little DNA to predict.

**This is a different regime from the A8.** There, ACRV1 within sperm reached 0.25 from
57,000 training events; here it is 0.98 from 10,000, and the prediction sheets show the
acrosomal cap drawn on the anterior head from epoch 3 onward. Four things differ, and
the comparison has to keep them apart:

1. **Contrast.** 60× brightfield with a phase-like halo on two cameras plus side scatter,
   against the A8's extinction channel. The acrosome is *visible* in the ISX Ch01 panel;
   it was not in LightLoss. This is the explanation the phase-microscopy literature
   predicts, and the one that matters scientifically.
2. **Target quality.** The ISX stains are bright, 12-bit, compensated and masked; the A8
   pages were tiny float values with heavy noise. A noisy target caps achievable r
   regardless of the input. "Higher quality images" was true on both sides of the pair.
3. **The split.** First half versus second half of one tube, minutes apart — no batch
   shift at all. The A8's held-out replicate carried real staining and gain differences.
   The A8's own within-replicate random split gave LDHC 0.66 against 0.38 across, so
   part of the gap is the evaluation, not the instrument. A same-split A8 run is the
   control this comparison needs.
4. **A shared per-object factor.** The four stains inter-correlate at 0.48 within
   nucleated sperm, so something per-cell — focus, in-frame fraction — scales all of
   them, and it is label-free-visible. It cannot explain 0.98 on each marker alone, but
   the marker-*ratio* numbers (e.g. ACRV1 relative to DAPI) are the version of the result
   that is immune to it, and should be reported alongside.

The 38.5% DAPI-low population is why the morphology floor read 0.96 on the whole sample:
telling fragment from sperm is most of that. Within nucleated sperm it is still high
(0.72–0.93), and the network clears it on every marker, by the most on TOMM20 (+0.26)
and DAPI (+0.11).

---

## Full-scale virtual staining — 57,000 training events per fold, RTX A6000

*2026-10-04 · `analysis/train_a8.py`, width-48 U-Net, 40 epochs, ~1 h per fold. Checkpoint
chosen on validation L1 within the training replicate; the test replicate is never used
for selection. Outputs in `runs/fold_2to3/` and `runs/fold_3to2/` on the server.*

### The headline table

| | fold A: train 2 → test 3 | | | fold B: train 3 → test 2 | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| marker | pixel r | event r | **within-type r** | pixel r | event r | **within-type r** |
| LDHC/AKAP4 (within sperm) | 0.715 | 0.729 | **0.727** | 0.713 | 0.720 | **0.708** |
| CD45 (within PBMC) | 0.512 | 0.756 | **0.567** | 0.523 | 0.737 | **0.525** |
| ACRV1 (within sperm) | 0.505 | 0.601 | **0.282** | 0.557 | 0.572 | **0.227** |

*Within-type r is event r computed inside one well type, where recognising the cell earns
nothing. It is the number to quote.*

### What scale changed, and a correction

**ACRV1 within sperm is not zero.** At 600 training events it read 0.02 / −0.00 and I
wrote "zero is zero". At 57,000 it reads **0.28 / 0.23**, replicated across folds on
20,000 test sperm each — small, but far from chance. The pilot was underpowered to see
r ≈ 0.25, and the earlier claim was overconfident. What the signal *is* remains the
question: ACRV1 is intra-acrosomal, so its total scales with acrosome size, and acrosome
size is visible in scatter. An r of 0.25 — six percent of variance — is consistent with
the model reading acrosome *extent*, not acrosomal *state*. The optical-limit conclusion
softens from "nothing" to "a sliver, probably geometric".

**LDHC/AKAP4 is the strong result**: 0.73 / 0.71 within sperm, up from 0.58 / 0.61 in
the pilot. Half the variance in flagellar stain intensity is predictable from label-free
scatter. **CD45 within PBMCs** is 0.57 / 0.53 — essentially unchanged from the pilot, so
that is probably its ceiling: CD45 level varies by leukocyte subtype, which morphology
only partly resolves.

### Controls

| | fold A pred / true P:S | fold B pred / true P:S | want |
| --- | --- | --- | --- |
| ACRV1 | 0.00 / 0.02 | 0.20 / 0.02 | ≪ 1 ✓ |
| CD45 | 17.98 / 7.54 | 8.49 / 9.09 | ≫ 1 ✓ (fold A overshoots: sperm predictions near zero inflate the ratio) |
| LDHC/AKAP4 | 1.07 / **0.53** | 0.44 / **1.07** | ≪ 1 — see below |

LDHC's control reverses between folds *in the truth*: replicate 3 has sperm > PBMC
(0.53, the biology), replicate 2 has PBMC > sperm (1.07, the antibody binding PBMCs
non-specifically). Each model faithfully reproduces its training replicate and so fails
on the other. This is now established on 60,000 events per replicate, not inferred.
**Replicate 3 is the LDHC reference; replicate 2's LDHC channel should not be trained on
without saying so.**

### Training dynamics, and the fix they point at

Validation L1 plateaued by epoch 13–25 and drifted up after; train L1 kept falling — mild
overfit to the training replicate. Test ACRV1 within-sperm r **peaked at 0.30 around
epochs 15–20 and declined to 0.26** by epoch 40, so selection on validation L1 is not
selecting for the metric we care about.

A second thing in the loss: a single shared target scale lets the brightest channel
dominate. AF488 is ~3× PerCP, so ACRV1 received roughly a fifth of the gradient. Two
flags added for the next run — `--channel-scale` (one scale per channel) and
`--select-on acrv1_sperm` (checkpoint on within-sperm ACRV1 r, measured on the validation
slice of the training replicate, never on test). Neither changes the defaults, so these
results stay reproducible.

---

## Pilot image-to-image model — the stain is predictable from label-free pages

*2026-10-01 · `analysis/pilot_floor.py` and `analysis/pilot_unet.py` on
`data/a8_bundle_seed0.zip` — 600 training events, 600 test, held out by replicate.
A feasibility check on a laptop (74 s per fold on MPS), not a result; the full run is
120,000 events on the server GPU.*

Input: the three label-free pages, normalised per event. Target: the three fluorescence
pages after the triangular unmix. Model: a 0.5M-parameter U-Net with no global pooling,
L1 loss on a validity mask. The **floor** is a ridge regression from a 5 × 5 label-free
patch to the centre pixel — the cheapest model that could possibly work.

### Fold A: train replicate 2 → test replicate 3

| marker | floor pixel r | **U-Net pixel r** | floor event r | **U-Net event r** | pred P:S | true P:S | want |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LDHC/AKAP4 | 0.364 | **0.607** | 0.155 | **0.540** | 1.45 | 0.72 | ≪ 1 |
| CD45 | 0.387 | **0.506** | 0.567 | **0.873** | 9.35 | 8.04 | ≫ 1 |
| ACRV1 | 0.262 | **0.388** | −0.346 | **0.513** | 0.35 | 0.20 | ≪ 1 |

### Fold B: train replicate 3 → test replicate 2

| marker | U-Net pixel r | U-Net event r | pred P:S | true P:S | want |
| --- | ---: | ---: | ---: | ---: | --- |
| LDHC/AKAP4 | 0.623 | 0.613 | **0.61** | 0.96 | ≪ 1 |
| CD45 | 0.532 | 0.880 | 10.69 | 9.96 | ≫ 1 |
| ACRV1 | 0.407 | 0.457 | 0.37 | 0.25 | ≪ 1 |

*P:S = median predicted page sum on PBMC-well events ÷ sperm-well events, in the stained
test replicate. The floor scores ~1.12 on all three — a local patch cannot tell a PBMC
from a sperm.*

### What it says

**ACRV1 is *not* read from label-free pages — the head is.** The cross-type event r of
0.51 / 0.46 looked like the project's headroom. It is not. Computed **within sperm only**,
where recognising the cell earns nothing, ACRV1 event r is **0.017 and −0.001** across
the two folds — while true ACRV1 on sperm spans 11–14× between the 5th and 95th
percentile, so there is a real intact-versus-reacted spread to explain and the model
explains none of it. The spot on the head is a prior ("sperm have acrosomes"), not a
reading. This is the optical limit the phase-microscopy literature predicted, now shown
with pixel-registered ground truth rather than inferred. At 600 training events a true
within-sperm r of ~0.3 would have been visible; zero is zero. The full run will say
whether weak signal appears with 60,000 — `train_a8.py` now reports within-well r, and
for ACRV1 that is the only number that matters.

| within one well type, event r | sperm only | PBMC only | mixture |
| --- | ---: | ---: | ---: |
| LDHC/AKAP4 | **0.58 / 0.61** | 0.35 / 0.51 | 0.62 / 0.66 |
| CD45 | 0.65 / 0.04 | **0.53 / 0.59** | 0.88 / 0.83 |
| ACRV1 | **0.02 / −0.00** | 0.14 / 0.47 | 0.38 / 0.29 |

*(fold A / fold B)* LDHC carries genuine within-sperm signal — the tail is visible in FSC
and SSC and its extent predicts the stain. CD45 carries within-PBMC signal. ACRV1 carries
none within sperm. Across cell types all three look predictable; only two are.

**CD45 is the easy one, as everywhere.** Event r 0.87–0.88 and the right cell-type
behaviour (P:S 9–11 against 8–10 true). The model has learned "round cell".

**LDHC fails the control in fold A and passes it in fold B — and that is the staining,
not the model.** Trained on replicate 2, the model puts *more* LDHC on PBMCs than sperm
(P:S 1.45). Replicate 2's LDHC/AKAP4 antibody binds PBMCs non-specifically — the Attune
scalars showed it, the A8 contact sheet shows AF488 blobs on PBMCs, and the fold-A
model reproduces its training data faithfully. Trained on the cleaner replicate 3 it gets
the direction right (0.61). The "≪ 1" expectation is biology; replicate 2's staining
violates it. **Treat replicate 3 as the reference for LDHC, and report both folds.**

**Every number clears the floor.** Pixel r up by 0.12–0.25, event r up by 0.3–0.85,
and the cell-type controls go from uniformly uninformative to mostly correct.

### Known pipeline defects, to fix before the full run

- A bright band along the padded edge in some predicted LDHC frames (visible on the
  sheet). Padding events to 80 rows leaves a boundary the network learns; crop
  predictions to the valid region, or pad with reflection rather than zeros.
- The CellView horizontal scan-line streak is present in the targets and will be learned
  as signal at scale. Mask it or model it.
- 600 events is far too few to speak to generalisation; the ~0.1 spread between folds is
  mostly sample size.

---

## First look at the A8 images — the stain pictures, and what they say

*2026-10-01 · `analysis/a8_io.py` on `data/a8_bundle_seed0.zip` (8 samples x 200 events, join
verified on the server; see [task_brief.md](task_brief.md))*

### The data, numerically

| | |
| --- | --- |
| Stack | `(6, h, 104)` float32; h in 57–77 (mode 73) — flow axis varies with the event's transit |
| LightLoss | background 0.129, cells **darker** (extinction); range ~0–0.17 |
| FSC, SSC, fluorescence | **already background-subtracted** by the instrument: background ≈ 0, ~46% of pixels negative (noise), cell signal positive |
| Scale | tiny float units: AF488 p99 ≈ 0.10, PE 0.04, PerCP 0.03; SSC up to 5.5 |

Two consequences for any loss: no log transform (half the pixels are negative), and no
further background subtraction.

**`tifffile.imread` returns page 1 only.** BD writes a `{"shape": [h, w]}` description on
every page, which tifffile reads as six one-page series. Iterate `TiffFile(...).pages`.
`a8_io.read_stack` does; nothing else in the repo should touch these files directly.

### Replicate 1 is unstained on the A8 too

Median background-subtracted fluorescence page sums: `1S` and `1P` read 0–2 on every
channel against 28–38 (AF488) for replicates 2 and 3. Same experimental design as the
CytPix run. **Usable paired events: the six stained samples, 120,000.** Replicate 1's
40,000 are **not** a negative control for image-to-image work — a correct virtual-stain
model should predict the stain on an unstained cell, since the label-free pages cannot
know the tube was not stained. The controls are cell-type ones inside the stained
replicates: ACRV1 and LDHC low on PBMCs, CD45 low on sperm. Replicate 1 is extra
label-free input, nothing more.

### The fluorescence pages are raw filter channels — and the spillover is estimable

Not from an unconstrained fit: the unmixed scalars are collinear by cell type (sperm carry
AF488 *and* PerCP, PBMCs carry PE), so regressing page sums on all three returns R² of
0.8–0.9 and coefficients that are physically impossible (PerCP-eF710 does not put 90% of
its light through a 534 nm filter). Physics supplies the constraint: emission spills
**red-ward only**, so the mixing is lower-triangular with three coefficients, each
estimable from a well where the upstream fluorophore dominates:

| | coefficient | r | from |
| --- | ---: | ---: | --- |
| AF488 → PE | **0.202** | 0.965 | S wells (sperm carry no CD45) |
| AF488 → PerCP | 0.023 | 0.205 | S wells |
| PE → PerCP | 0.166 | 0.618 | P wells, AF488's share removed first |

Condition number 1.30 — inverting it per pixel is benign. `a8_io.spillover` and
`a8_io.unmix` implement this. Estimated under a physical constraint, not from single-stain
controls, which the experiment does not have; say so when it is used.

### What the pictures show

Contact sheet: `analysis/out/a8_pages.png`.

- **ACRV1 is a compact spot on the sperm head.** The acrosomal cap, imaged. This is the
  spatial ground truth the CytPix could never provide, on the marker that was least
  predictable from scalars.
- **LDHC/AKAP4 lights the whole flagellum**, plus the head. Principal piece, as the
  biology says.
- **The spillover is visible and the triangular unmix removes it.** On sperm the raw PE
  page shows the same tail as AF488; after unmixing the tail is gone and the page is
  noise. On PBMCs the unmixed PE page stays bright — genuine CD45.
- **The instrument's own unmixed CD45 scalar goes strongly negative on sperm** (−2e5 to
  −6e5), the same over-subtraction signature the Attune matrix showed. For scalar targets
  that means `asinh`; for images, the image-level triangular unmix looks cleaner than the
  instrument's spectral one.
- **A horizontal streak runs through the cell's row** in several fluorescence pages — a
  CellView scan-line artefact. Worth masking or modelling; it will otherwise be learned.

### The design decision this forces

**Use all three label-free pages as input, not LightLoss alone.** LightLoss shows a sperm
head crisply but the tail only faintly, and barely resolves a PBMC at all — a sperm head
is condensed chromatin and extinguishes strongly; a lymphocyte does not. FSC and SSC show
the tail clearly and the PBMC brightly, and they are every bit as label-free. The
image-to-image model's input is `(LightLoss, FSC, SSC)` → `(AF488, PE, PerCP)` unmixed.
Whether LightLoss *alone* suffices is then an ablation, and it is the ablation that bears
on transfer to the CytPix, which has nothing but extinction.

---

## What is actually inside an .acs, and what it changes

*2026-09-13 · `analysis/explore_masks.py`, `scripts/inspect_acs.py`, on a local copy of `3SP.acs`*

### The images are 16-bit, and the `Images\` export is a downconversion

| | `Images\*.zip` | inside the `.acs` |
| --- | --- | --- |
| Format | 248 x 248, **uint8**, RGBA container, LZW | 248 x 248, **uint16**, single channel, uncompressed |
| Per file | 74-105 KB (varies with noise) | 123,250 B (constant) |

Decoded pixel values run ~145-1011 with 686 distinct levels pooled, so the sensor is
~10-bit and the `.acs` preserves it. The `Images\` export throws away about two bits.
**Source images from the `.acs`.** A standard-library decoder for both widths was written and verified byte-for-byte
against PIL on the real archives; it has since been removed as unused (git history).

### The instrument ships its segmentation

Each archive holds a `*.masks.zip` — one JSON per event, keyed by the same event id:

```
[{"version":"1.0"}, {"masks":[{"pixelIndexes":{"indexes":[...]}}]}, {"masks":[...]}]
```

Flat pixel indices into the 248 x 248 frame, in two layers, and both are now identified
against the FCS columns on 594 events:

| Layer | Is | Evidence |
| --- | --- | --- |
| 1 | **Objects** | mask count == `ObjectCount` on 594/594; pixel count == `NumPixels` |
| 2 | **Particles** | mask count == `ParticleCount` on 594/594; lies entirely inside layer 1 |

Layer 2 is the instrument's particle detection — a small region (~7% of the object),
slightly darker than the rest but *not* the darkest part, so not a nucleus proxy. Both
layers are brightfield segmentation products. **There are no per-stain masks in the
archive**: all 30,000 mask files have exactly these two layers, every entry carries only
`pixelIndexes`, and neither XML in the archive names a channel, stain or layer. Rendered,
the object mask tracks the flagellum correctly on tailed cells, so the segmentation step
`sperm_pbmc` built by hand is already done here.

### Compensation: the export is raw, and one coefficient is wrong

The `$SPILLOVER` matrix is 15 x 15 and well-conditioned (condition number 2.8). Whether
the exported `-A` values were pre- or post-compensation could not be read off the matrix,
so it was tested against biology: **PBMCs have no acrosome, so ACRV1 on a CD45-high round
cell must be ~0.**

| | ACRV1 median on CD45-high events | corr(ACRV1, CD45) | events with a negative channel |
| --- | ---: | ---: | ---: |
| as exported | 10,608 | 0.940 | 4.5% |
| **inv(S) applied** | **91** | 0.578 | 88.5% |
| S applied | 22,589 | 0.988 | 0.1% |

**The export is raw and `inv(S)` is the correction.** That settles it: the ACRV1 signal on
PBMCs was spillover, and compensation removes it almost exactly.

But one coefficient does not survive scrutiny. `BL1-A -> YL1-A = 1.0001` says AF488 lands
in the PE detector at 100% of its own-detector intensity. Compensating with it drives the
median CD45 in `3SP` to **-15,408**: most events there are sperm, which are LDHC-high and
genuinely CD45-negative, so a ~100% subtraction of BL1 from YL1 overshoots. Compensated
data legitimately spreads below zero, but not by that much.

**Two consequences for modeling:**

1. **Compensate, then use an `asinh` (biexponential) transform, not `log1p`.** 88.5% of
   compensated events carry a negative in some channel, which `log1p(clip(x, 0))` would
   flatten to zero. That is the standard flow-cytometry transform for exactly this reason.
2. **The morphology baseline below was run on uncompensated targets.** Its ACRV1 number
   (0.209) is partly measuring PE spillover, and has to be rerun compensated before it
   means anything. DAPI and CD45 are far less affected — VL1 has almost no spillover in or
   out — but the rerun is the number to quote.

**Open with the lab:** is `BL1-A -> YL1-A = 1.0001` real, or a compensation setup error?
It is the single largest coefficient in the matrix and it governs the CD45 channel.

---

## Morphology baseline — how much of each marker is predictable without a network

*2026-09-12 · `python analysis/feature_baseline.py data/targets --train 2 --test 3`*

This is step 3 of [approach.md](approach.md), and it ran without a single image: the
CytPix computes shape, intensity and texture per event, so "does what the cell looks like
predict what it stains for" is answerable from the target tables alone.

Spearman correlation of predicted vs measured intensity. **within** is a random split
inside replicate 2 — what a naive evaluation would report. **across** is trained on
replicate 2 and tested on replicate 3 — the honest number. **null** is permuted-target
models, three runs, reported as a range.

### All wells pooled (`S`, `P`, `SP`) — 102,874 train / 90,000 test

| marker | features | within | across | null range | |
| --- | --- | ---: | ---: | --- | --- |
| DAPI | shape only | 0.791 | **0.776** | −0.095 .. 0.101 | solid |
| DAPI | shape + intensity | 0.835 | **0.798** | −0.094 .. −0.024 | solid |
| CD45 | shape only | 0.803 | **0.767** | −0.197 .. 0.085 | solid |
| CD45 | shape + intensity | 0.831 | **0.797** | −0.113 .. −0.029 | solid |
| LDHC | shape only | 0.660 | **0.384** | −0.030 .. 0.038 | solid, but see below |
| LDHC | shape + intensity | 0.716 | **0.425** | −0.023 .. −0.002 | solid, but see below |
| ACRV1 | shape only | 0.359 | **0.209** | −0.070 .. 0.118 | weak |
| ACRV1 | shape + intensity | 0.408 | **0.284** | −0.074 .. 0.161 | weak |

### The `SP` mixture only — the control that rules out acquisition leakage

Every event here comes from a single acquisition per replicate, so nothing can be
explained by "which well was this". 42,874 train / 30,000 test.

| marker | across (shape only) | across (+ intensity) |
| --- | ---: | ---: |
| LDHC | **0.668** | 0.727 |
| DAPI | **0.616** | 0.628 |
| CD45 | **0.544** | 0.565 |
| ACRV1 | **0.306** | 0.348 |

Morphology still predicts every channel inside one acquisition. The pooled numbers are
not an artefact of the model learning which well an event came from.

### What this says

**1. The ordering is the one the literature predicts.** DeepIFC's rule — a marker is
predictable exactly to the extent that what it binds has a visible correlate — gives
DAPI ≈ CD45 ≫ LDHC ≫ ACRV1, and that is the observed order. DAPI and CD45 both come down
to "is this a round cell or a sperm", which morphology answers easily. ACRV1 is acrosomal
*integrity*, a biochemical state, and it is the one that barely moves. That was the
prediction in [approach.md](approach.md) §2 and it held.

**2. The replicate split costs a lot, and that is the point.** LDHC reports **0.660** on a
random split and **0.384** across replicates — a 42% relative drop. Anyone evaluating this
on a naive split would have believed the higher number. The drop is concentrated in the
`P` wells, which is where replicate 2's LDHC channel misbehaves (see
[task_brief.md](task_brief.md)): restricted to the mixture, LDHC holds up at 0.704 → 0.668.

**3. Brightfield intensity is not carrying the result.** Adding `TotalIntensity`,
`AverageIntensity` and `StandardDeviationIntensity` moves every marker by only +0.02 to
+0.06. Given `sperm_pbmc` Finding 1 — that background noise alone separated acquisitions
at AUC 0.992 — the worry was that intensity features would dominate. They do not. The
signal is in shape.

**4. Run the null more than once.** A single permutation put the LDHC null at 0.207, which
looked like serious leakage. Three permutations put the range at −0.03 .. 0.04. A
permuted-label model does not predict a constant — it emits an arbitrary function of the
features, and when the features are this informative, an arbitrary function of them can
correlate with the truth by luck. One permutation is not a control.

### What it means for the next step

These are **floors**. A few dozen instrument-computed features, no learned representation,
no pixels. A CNN on the images has to beat 0.78 on DAPI and 0.77 on CD45 to be worth
having — and the interesting target is ACRV1, where there is the most headroom and the
least reason to expect success.

**Caveat carried forward:** these targets are **uncompensated**. ACRV1 correlates with
CD45 at ρ 0.97–0.98 in the PBMC wells, so part of what "ACRV1 0.209" measures is spillover
from PE. The number could move in either direction after compensation, and the compensated
rerun is the one to quote.
